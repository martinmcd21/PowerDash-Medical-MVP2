"""
PowerDash Medical — Internal Medical Affairs AI Workbench (MVP)
----------------------------------------------------------------
Core principles:
- Stateless: no persistence beyond the session (no DB, no file storage of user content)
- No authentication, no analytics
- Streamlit-only UI (no JS injection, no st.components, no private Streamlit APIs)
- ABPI Code–compliant drafting enforced across the suite (feature, not disclaimer)
- Drafting support only; medical/legal/regulatory review required
- References: ONLY from user-provided material

Tech:
- Python 3.10+
- Streamlit
- OpenAI API via OPENAI_API_KEY env var (OpenAI Responses API)
- Optional PDF export via reportlab (in-memory only)
"""

from __future__ import annotations

import os
import re
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import streamlit as st

# Optional PDF export (in-memory)
PDF_AVAILABLE = False
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import mm
    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False

# OpenAI official SDK (Responses API)
# Docs: https://platform.openai.com/docs/overview?lang=python
# Structured Outputs: https://platform.openai.com/docs/guides/structured-outputs
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # handled gracefully


# -----------------------------
# UI CONSTANTS / BRANDING
# -----------------------------
APP_TITLE = "PowerDash Medical"
PRIMARY_BLUE = "#2563eb"

PERSISTENT_BADGES = [
    "✅ ABPI Code Compliant",
    "🩺 Medical Affairs Drafting Support Only",
    "🔎 Medical review required",
    "🛡️ No data retention",
]

DEFAULT_MODEL = "gpt-5.2"


# -----------------------------
# ABPI COMPLIANCE INJECTION
# -----------------------------
ABPI_GLOBAL_RULES = """
You are an internal Medical Affairs drafting assistant for UK & Ireland.
ABPI Code compliance is mandatory and must be actively enforced in your output.

NON-NEGOTIABLE RULES (apply to ALL outputs):
- Non-promotional scientific/medical intent at all times.
- Do NOT include promotional language, inducements, calls to action, or marketing copy.
- Do NOT make comparative, superiority, "best-in-class", or exaggerated efficacy/safety claims.
- Do NOT encourage or describe off-label use. If asked, explain you cannot support off-label promotion.
- Do NOT provide prescriptive clinical advice or patient-specific treatment recommendations.
- Maintain balanced, factual, evidence-led tone; include appropriate uncertainty and limitations.
- Clearly distinguish scientific exchange vs promotion; avoid language that could be construed as promotion.
- References MUST come ONLY from user-provided material. NEVER invent citations.
- Output is a DRAFT only and requires medical/legal/regulatory review before use.
- If evidence is insufficient, explicitly say so and suggest what evidence is needed (without fabricating).

SAFETY / COMPLIANCE RESPONSE STYLE:
- Conservative UK/I Medical Affairs tone.
- Use neutral phrasing (e.g., "may", "is associated with", "evidence suggests") where appropriate.
- Prefer structured, review-friendly writing.
"""

DRAFT_ONLY_BANNER_TEXT = (
    "ABPI Code Compliant • Medical Affairs Drafting Support Only • Medical review required • No data retention"
)


# -----------------------------
# GLOBAL SAFETY (KEYWORD + PII)
# -----------------------------
# Requirement: Phone number detection must be removed entirely -> we do NOT detect phone numbers.

AE_KEYWORDS = [
    "adverse event", "adverse reaction", "side effect", "suspected adverse", "aer",
    "pharmacovigilance", "pv case", "safety case", "reportable event", "serious adverse",
    "medwatch", "yellow card", "icsr", "fatal", "hospitalisation", "hospitalization"
]

# Patient-identifiable patterns (simple heuristics; conservative)
EMAIL_REGEX = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)

# NHS number: 10 digits, often spaced (e.g., 943 476 5919). We detect 10 digits optionally spaced.
NHS_REGEX = re.compile(r"\b(\d\s*){10}\b")

# DOB patterns: dd/mm/yyyy, dd-mm-yyyy, yyyy-mm-dd
DOB_REGEX = re.compile(
    r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b"
)

# "Patient name" heuristic: explicit field labels only (we do NOT attempt general name detection)
PATIENT_NAME_LABEL_REGEX = re.compile(
    r"\b(patient\s*name|name\s*of\s*patient|patient:\s*[A-Z])", re.IGNORECASE
)


def detect_safety_issues(text: str) -> Dict[str, Any]:
    """
    Returns dict with flags and matched evidence. Conservative; blocks on:
    - AE / PV triggers
    - Explicit patient-identifiable data triggers (NHS number, email, DOB, explicit "patient name" labels)
    """
    if not text:
        return {"blocked": False, "reasons": [], "matches": {}}

    lower = text.lower()

    ae_hits = [kw for kw in AE_KEYWORDS if kw in lower]
    email_hits = EMAIL_REGEX.findall(text)
    nhs_hit = bool(NHS_REGEX.search(text))
    dob_hit = bool(DOB_REGEX.search(text))
    patient_name_hit = bool(PATIENT_NAME_LABEL_REGEX.search(text))

    reasons = []
    matches: Dict[str, Any] = {}

    if ae_hits:
        reasons.append("Potential adverse event / pharmacovigilance content detected.")
        matches["ae_keywords"] = ae_hits[:10]

    if email_hits:
        reasons.append("Potential patient-identifiable data detected (email address).")
        matches["emails"] = list(set(email_hits))[:5]

    if nhs_hit:
        reasons.append("Potential patient-identifiable data detected (NHS number pattern).")
        matches["nhs_number"] = True

    if dob_hit:
        reasons.append("Potential patient-identifiable data detected (date of birth pattern).")
        matches["dob"] = True

    if patient_name_hit:
        reasons.append("Potential patient-identifiable data detected (explicit patient name label).")
        matches["patient_name_label"] = True

    blocked = len(reasons) > 0
    return {"blocked": blocked, "reasons": reasons, "matches": matches}


def render_blocked(block: Dict[str, Any]) -> None:
    """
    Centralised safety blocking renderer (required).
    """
    st.error("Generation blocked for safety/compliance reasons.", icon="⛔")
    for r in block.get("reasons", []):
        st.write(f"- {r}")

    st.info(
        "Next steps:\n"
        "- Remove any adverse event / pharmacovigilance details and route through your PV process.\n"
        "- Remove any patient-identifiable information (NHS number, email, DOB, explicit patient name fields).\n"
        "- Re-run with fully de-identified, non-case-specific information.\n",
        icon="ℹ️",
    )

    # Optional: show matched hints (do not echo sensitive strings beyond minimal)
    matches = block.get("matches", {})
    if matches:
        with st.expander("What was detected (high-level)", expanded=False):
            st.json(matches)


# -----------------------------
# OPENAI JSON OUTPUT HANDLING
# -----------------------------
def _extract_first_json_object(text: str) -> Optional[str]:
    """
    Fallback extraction: finds the first top-level JSON object in text.
    """
    if not text:
        return None

    # quick path
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text

    # brace matching
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                return candidate
    return None


def robust_json_loads(raw: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Robust JSON parsing fallback:
    - Try direct json.loads
    - If fails, extract first {...} and parse
    Returns (parsed, parse_note)
    """
    if not raw:
        return None, "Empty model output."

    try:
        return json.loads(raw), "Parsed as full JSON."
    except Exception:
        candidate = _extract_first_json_object(raw)
        if candidate:
            try:
                return json.loads(candidate), "Parsed by extracting first JSON object."
            except Exception as e:
                return None, f"JSON parsing failed after extraction: {e}"
        return None, "No JSON object found in output."


# -----------------------------
# SHARED GENERATION ENGINE
# -----------------------------
def get_openai_client() -> Optional[Any]:
    """
    Creates OpenAI client from environment variable OPENAI_API_KEY.
    """
    if OpenAI is None:
        return None
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    return OpenAI()


def make_json_schema_response_format(name: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Responses API 'text.format' json_schema format object (strict).
    """
    return {
        "type": "json_schema",
        "name": name,
        "strict": True,
        "schema": schema,
    }


def generate_json(
    *,
    model: str,
    tool_name: str,
    system_prompt: str,
    user_payload: Dict[str, Any],
    json_schema: Dict[str, Any],
    temperature: float = 0.2,
    max_output_tokens: int = 1800,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    Shared generation function:
    - Injects ABPI rules into every tool system prompt
    - Enforces JSON-only via Structured Outputs (json_schema) when supported
    - Robust fallback parsing if output deviates
    Returns: (parsed_json_or_none, meta)
    """
    client = get_openai_client()
    if client is None:
        return None, {
            "ok": False,
            "error": "OPENAI_API_KEY missing or OpenAI SDK not installed.",
            "raw": "",
            "parse_note": "",
        }

    # Assemble instructions: ABPI + tool-specific
    instructions = (
        ABPI_GLOBAL_RULES.strip()
        + "\n\n"
        + f"TOOL CONTEXT: {tool_name}\n"
        + system_prompt.strip()
        + "\n\n"
        + "You MUST output JSON only and MUST follow the provided JSON Schema exactly.\n"
        + "Do not include markdown fences, commentary, or extra keys.\n"
    )

    # User input: strictly structured
    input_items = [
        {
            "role": "user",
            "content": (
                "Use ONLY the information in this JSON payload. "
                "Do NOT add external facts or references.\n\n"
                + json.dumps(user_payload, ensure_ascii=False, indent=2)
            ),
        }
    ]

    # Try Structured Outputs (json_schema). Per OpenAI docs, supported in Responses API via text.format.
    # If the model rejects the format, we retry with json_object and stricter instructions.
    raw_text = ""
    parse_note = ""
    try:
        resp = client.responses.create(
            model=model,
            instructions=instructions,
            input=input_items,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            text={"format": make_json_schema_response_format(f"{tool_name}_schema", json_schema)},
        )
        raw_text = getattr(resp, "output_text", "") or ""
        parsed, parse_note = robust_json_loads(raw_text)
        if parsed is not None:
            return parsed, {"ok": True, "raw": raw_text, "parse_note": parse_note}
        return None, {"ok": False, "raw": raw_text, "parse_note": parse_note, "error": "Model output was not valid JSON."}
    except Exception as e:
        # Fallback: JSON mode (json_object) + schema embedded in instructions
        try:
            fallback_instructions = instructions + "\n\nJSON SCHEMA (must match):\n" + json.dumps(json_schema, indent=2)
            resp = client.responses.create(
                model=model,
                instructions=fallback_instructions,
                input=input_items,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                text={"format": {"type": "json_object"}},
            )
            raw_text = getattr(resp, "output_text", "") or ""
            parsed, parse_note = robust_json_loads(raw_text)
            if parsed is not None:
                return parsed, {
                    "ok": True,
                    "raw": raw_text,
                    "parse_note": f"{parse_note} (fallback json_object due to: {e})",
                }
            return None, {
                "ok": False,
                "raw": raw_text,
                "parse_note": parse_note,
                "error": f"Failed to parse JSON in fallback mode. Root error: {e}",
            }
        except Exception as e2:
            return None, {
                "ok": False,
                "error": f"OpenAI request failed (primary and fallback). Primary: {e} | Fallback: {e2}",
                "raw": raw_text,
                "parse_note": parse_note,
            }


# -----------------------------
# EXPORT HELPERS (TXT / PDF)
# -----------------------------
def as_pretty_text(obj: Dict[str, Any]) -> str:
    """
    Display/export helper: stable, review-friendly formatting.
    """
    return json.dumps(obj, ensure_ascii=False, indent=2)


def make_txt_download(label: str, text: str, filename: str) -> None:
    st.download_button(
        label=label,
        data=text.encode("utf-8"),
        file_name=filename,
        mime="text/plain",
        use_container_width=True,
    )


def make_pdf_bytes(title: str, text: str) -> Optional[bytes]:
    """
    In-memory PDF generation (no file storage).
    """
    if not PDF_AVAILABLE:
        return None

    from io import BytesIO

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # Simple header
    c.setTitle(title)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(20 * mm, height - 20 * mm, title)
    c.setFont("Helvetica", 9)
    c.drawString(20 * mm, height - 27 * mm, DRAFT_ONLY_BANNER_TEXT)

    # Body (naive line wrap)
    c.setFont("Helvetica", 9)
    x = 20 * mm
    y = height - 35 * mm
    max_width = width - 40 * mm
    line_height = 4.5 * mm

    def wrap_line(line: str, max_chars: int = 110) -> List[str]:
        if len(line) <= max_chars:
            return [line]
        chunks = []
        current = ""
        for word in line.split(" "):
            if len(current) + len(word) + 1 <= max_chars:
                current = (current + " " + word).strip()
            else:
                chunks.append(current)
                current = word
        if current:
            chunks.append(current)
        return chunks

    for raw_line in text.splitlines():
        for line in wrap_line(raw_line):
            if y < 20 * mm:
                c.showPage()
                c.setFont("Helvetica", 9)
                y = height - 20 * mm
            c.drawString(x, y, line)
            y -= line_height

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.read()


def make_pdf_download(label: str, title: str, text: str, filename: str) -> None:
    if not PDF_AVAILABLE:
        st.caption("PDF export unavailable (reportlab not installed).")
        return
    pdf_bytes = make_pdf_bytes(title, text)
    if pdf_bytes:
        st.download_button(
            label=label,
            data=pdf_bytes,
            file_name=filename,
            mime="application/pdf",
            use_container_width=True,
        )


# -----------------------------
# TOOL DEFINITIONS (PROMPTS + SCHEMAS)
# -----------------------------
@dataclass(frozen=True)
class ToolSpec:
    key: str
    name: str
    emoji: str
    category: str  # "Core Medical Affairs Tools" | "Additional Tools"
    description: str
    system_prompt: str
    schema: Dict[str, Any]


def schema_base() -> Dict[str, Any]:
    """
    Shared schema fragments via composition (simple dict merges in place).
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["abpi_compliance", "draft_notice"],
        "properties": {
            "abpi_compliance": {
                "type": "object",
                "additionalProperties": False,
                "required": ["status", "checks"],
                "properties": {
                    "status": {"type": "string", "enum": ["compliant_draft"]},
                    "checks": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 3,
                    },
                },
            },
            "draft_notice": {
                "type": "string",
                "description": "Short fixed notice that this is drafting support only and requires review.",
            },
        },
    }


TOOLS: Dict[str, ToolSpec] = {}

# 1) Scientific Narrative Generator
TOOLS["scientific_narrative"] = ToolSpec(
    key="scientific_narrative",
    name="Scientific Narrative Generator",
    emoji="📄",
    category="Core Medical Affairs Tools",
    description="Create a balanced scientific narrative and short-form variants using only user-provided publications/notes.",
    system_prompt="""
Generate a conservative UK/I Medical Affairs scientific narrative pack.
Requirements:
- Use ONLY the provided publications/positioning notes.
- Avoid promotional tone; no calls to action; no superiority/comparative claims.
- Present balanced disease and science overview; include uncertainties and limitations.
- If the provided evidence is thin, clearly state gaps and what would be needed.
""".strip(),
    schema={
        **schema_base(),
        "required": ["abpi_compliance", "draft_notice", "core_scientific_narrative", "disease_state_overview", "short_form_variants", "references_used"],
        "properties": {
            **schema_base()["properties"],
            "core_scientific_narrative": {"type": "string"},
            "disease_state_overview": {"type": "string"},
            "short_form_variants": {
                "type": "object",
                "additionalProperties": False,
                "required": ["msl_conversation", "internal_training", "congress_discussions"],
                "properties": {
                    "msl_conversation": {"type": "string"},
                    "internal_training": {"type": "string"},
                    "congress_discussions": {"type": "string"},
                },
            },
            "references_used": {
                "type": "array",
                "description": "List ONLY citations explicitly present in the user-provided text, quoted/identified succinctly.",
                "items": {"type": "string"},
            },
        },
        "additionalProperties": False,
        "type": "object",
    },
)

# 2) MSL Briefing Pack Generator
TOOLS["msl_briefing_pack"] = ToolSpec(
    key="msl_briefing_pack",
    name="MSL Briefing Pack Generator",
    emoji="🧠",
    category="Core Medical Affairs Tools",
    description="Generate a non-promotional MSL briefing pack: objectives, key messages, hypotheses, guides, Q&A, do/don’t, follow-ups.",
    system_prompt="""
Create an MSL briefing pack for scientific exchange.
Requirements:
- Non-promotional, UK/I Medical Affairs tone.
- Q&A must not include off-label promotion or prescriptive clinical advice.
- Include Do/Don't guidance aligned to ABPI Code principles.
- Use ONLY provided evidence/notes; if insufficient, state gaps.
""".strip(),
    schema={
        **schema_base(),
        "required": ["abpi_compliance", "draft_notice", "objectives", "key_scientific_messages", "stakeholder_hypotheses",
                    "discussion_guide", "anticipated_qa", "do_dont_guidance", "follow_up_actions", "references_used"],
        "properties": {
            **schema_base()["properties"],
            "objectives": {"type": "array", "items": {"type": "string"}},
            "key_scientific_messages": {"type": "array", "items": {"type": "string"}},
            "stakeholder_hypotheses": {"type": "array", "items": {"type": "string"}},
            "discussion_guide": {"type": "array", "items": {"type": "string"}},
            "anticipated_qa": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["question", "answer"],
                    "properties": {
                        "question": {"type": "string"},
                        "answer": {"type": "string"},
                    },
                },
            },
            "do_dont_guidance": {"type": "object",
                                "additionalProperties": False,
                                "required": ["do", "dont"],
                                "properties": {
                                    "do": {"type": "array", "items": {"type": "string"}},
                                    "dont": {"type": "array", "items": {"type": "string"}},
                                }},
            "follow_up_actions": {"type": "array", "items": {"type": "string"}},
            "references_used": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": False,
        "type": "object",
    },
)

# 3) Medical Information Response Generator
TOOLS["med_info_response"] = ToolSpec(
    key="med_info_response",
    name="Medical Information Response Generator",
    emoji="📚",
    category="Core Medical Affairs Tools",
    description="Draft ABPI-compliant medical information responses using ONLY user-provided evidence. Blocks on AE or patient-identifiable data.",
    system_prompt="""
Draft a Medical Information response for UK or Ireland.
Requirements:
- Audience-specific (HCP / Pharmacist / Payer) with appropriate level of detail.
- Provide:
  1) Long-form written response
  2) Short verbal response
  3) References list ONLY from user-provided evidence
- No promotional language; no off-label encouragement; no prescriptive advice.
- If question implies off-label/promotion, respond with a compliant boundary statement and provide on-label, factual info only if supported by evidence provided.
""".strip(),
    schema={
        **schema_base(),
        "required": ["abpi_compliance", "draft_notice", "long_form_response", "short_verbal_response", "references_used", "limitations"],
        "properties": {
            **schema_base()["properties"],
            "long_form_response": {"type": "string"},
            "short_verbal_response": {"type": "string"},
            "references_used": {"type": "array", "items": {"type": "string"}},
            "limitations": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": False,
        "type": "object",
    },
)

# 4) Congress & Advisory Board Planner
TOOLS["congress_adboard_planner"] = ToolSpec(
    key="congress_adboard_planner",
    name="Congress & Advisory Board Planner",
    emoji="🎤",
    category="Core Medical Affairs Tools",
    description="Create compliant agendas, discussion guides, question banks, and insight capture frameworks.",
    system_prompt="""
Plan a congress activity or advisory board in a compliant Medical Affairs manner.
Requirements:
- No promotional intent or inducements.
- Focus on scientific exchange and insight generation.
- Provide agenda, discussion guide, question bank, and an insight capture framework.
- Use only user-provided context; avoid invented data.
""".strip(),
    schema={
        **schema_base(),
        "required": ["abpi_compliance", "draft_notice", "agenda", "discussion_guide", "question_bank", "insight_capture_framework"],
        "properties": {
            **schema_base()["properties"],
            "agenda": {"type": "array", "items": {"type": "string"}},
            "discussion_guide": {"type": "array", "items": {"type": "string"}},
            "question_bank": {"type": "array", "items": {"type": "string"}},
            "insight_capture_framework": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": False,
        "type": "object",
    },
)

# 5) Insight Capture & Thematic Analysis
TOOLS["insight_thematic_analysis"] = ToolSpec(
    key="insight_thematic_analysis",
    name="Insight Capture & Thematic Analysis",
    emoji="📊",
    category="Additional Tools",
    description="Session-only: group insights into themes, separate signal vs noise, and create an executive-ready summary (no persistence).",
    system_prompt="""
Analyse user-provided insight notes (session-only).
Requirements:
- Group into themes with short rationale.
- Identify signal vs noise conservatively (state uncertainty).
- Provide an executive-ready summary with risks/opportunities framed non-promotional.
- Use only the text provided; no external assumptions.
""".strip(),
    schema={
        **schema_base(),
        "required": ["abpi_compliance", "draft_notice", "themes", "signal_vs_noise", "executive_summary", "open_questions"],
        "properties": {
            **schema_base()["properties"],
            "themes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["theme", "supporting_points"],
                    "properties": {
                        "theme": {"type": "string"},
                        "supporting_points": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
            "signal_vs_noise": {
                "type": "object",
                "additionalProperties": False,
                "required": ["signal", "noise", "uncertainties"],
                "properties": {
                    "signal": {"type": "array", "items": {"type": "string"}},
                    "noise": {"type": "array", "items": {"type": "string"}},
                    "uncertainties": {"type": "array", "items": {"type": "string"}},
                },
            },
            "executive_summary": {"type": "string"},
            "open_questions": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": False,
        "type": "object",
    },
)

# 6) Medical Affairs Executive Summary Generator
TOOLS["exec_summary"] = ToolSpec(
    key="exec_summary",
    name="Medical Affairs Executive Summary Generator",
    emoji="📈",
    category="Additional Tools",
    description="Leadership-ready summary: themes, risks, opportunities, and recommended next steps (non-promotional, evidence-led).",
    system_prompt="""
Create a leadership-ready Medical Affairs executive summary.
Requirements:
- Balanced, conservative tone; avoid promotional framing.
- Clearly list themes, risks, opportunities, and recommended next steps.
- Recommendations must be process/insight oriented, not promotional calls to action.
- Use only user-provided information.
""".strip(),
    schema={
        **schema_base(),
        "required": ["abpi_compliance", "draft_notice", "summary", "themes", "risks", "opportunities", "recommended_next_steps", "limitations"],
        "properties": {
            **schema_base()["properties"],
            "summary": {"type": "string"},
            "themes": {"type": "array", "items": {"type": "string"}},
            "risks": {"type": "array", "items": {"type": "string"}},
            "opportunities": {"type": "array", "items": {"type": "string"}},
            "recommended_next_steps": {"type": "array", "items": {"type": "string"}},
            "limitations": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": False,
        "type": "object",
    },
)

# 7) Compliance & Governance Summary (static)
TOOLS["compliance_governance"] = ToolSpec(
    key="compliance_governance",
    name="Compliance & Governance Summary",
    emoji="🔒",
    category="Additional Tools",
    description="Static page explaining stateless design, ABPI intent, drafting-only scope, review requirement, and guardrail limitations.",
    system_prompt="(static page; no LLM call)",
    schema={"type": "object", "properties": {}},
)

# 8) SOP Drafting Tool
TOOLS["sop_drafting"] = ToolSpec(
    key="sop_drafting",
    name="Medical Affairs SOP Drafting Tool",
    emoji="📑",
    category="Additional Tools",
    description="Draft a conservative SOP in ABPI-aware regulatory tone (draft only).",
    system_prompt="""
Draft a Medical Affairs SOP in a conservative UK/I regulatory tone.
Requirements:
- Structured SOP sections: Purpose, Scope, Definitions, Roles & Responsibilities, Procedure, Documentation, Training, Compliance, Version control.
- Non-promotional; avoid product marketing language.
- Use only the user-provided process requirements and context.
""".strip(),
    schema={
        **schema_base(),
        "required": ["abpi_compliance", "draft_notice", "sop_title", "sop_document", "assumptions", "open_questions"],
        "properties": {
            **schema_base()["properties"],
            "sop_title": {"type": "string"},
            "sop_document": {"type": "string"},
            "assumptions": {"type": "array", "items": {"type": "string"}},
            "open_questions": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": False,
        "type": "object",
    },
)


# -----------------------------
# STREAMLIT APP SHELL / STYLES
# -----------------------------
def inject_css() -> None:
    """
    Streamlit-stable CSS injection (no JS, no overlays, no click interception).
    Styles tool tiles and sidebar badges.
    """
    st.markdown(
        f"""
<style>
/* Wider layout */
.block-container {{
  padding-top: 1.2rem;
  padding-bottom: 2rem;
}}

/* Tile button style: we style the Streamlit button itself */
div.stButton > button {{
  background: {PRIMARY_BLUE};
  color: white;
  border: 0;
  border-radius: 10px;
  padding: 0.9rem 0.9rem;
  width: 100%;
  text-align: left;
  font-weight: 700;
}}
div.stButton > button:hover {{
  filter: brightness(0.95);
}}
div.stButton > button:active {{
  filter: brightness(0.9);
}}

/* Small caption-like muted text */
.pdm-muted {{
  color: rgba(255,255,255,0.9);
  font-weight: 500;
  font-size: 0.85rem;
  line-height: 1.2rem;
  margin-top: 0.25rem;
}}
.pdm-badge {{
  display: inline-block;
  background: rgba(37,99,235,0.15);
  border: 1px solid rgba(37,99,235,0.35);
  border-radius: 999px;
  padding: 0.25rem 0.6rem;
  margin: 0.15rem 0.15rem 0.15rem 0;
  font-size: 0.8rem;
}}
</style>
        """,
        unsafe_allow_html=True,
    )


def sidebar_persistent_ui(selected_model: str) -> str:
    st.sidebar.title(APP_TITLE)

    st.sidebar.markdown("### Status")
    for b in PERSISTENT_BADGES:
        st.sidebar.markdown(f"<span class='pdm-badge'>{b}</span>", unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model")
    model = st.sidebar.text_input(
        "OpenAI model (runtime)",
        value=selected_model,
        help="Change the OpenAI model at runtime. Default is gpt-5.2.",
    )
    st.sidebar.caption(f"Current model in use: **{model}**")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Navigation")
    return model


def tool_nav() -> str:
    """
    Sidebar navigation (mirrors PowerDash HR layout):
    - Home
    - Core Medical Affairs Tools
    - Additional Tools
    """
    options = [
        ("home", "🏠 Home"),
        ("core_header", "— Core Medical Affairs Tools —"),
        ("scientific_narrative", "📄 Scientific Narrative Generator"),
        ("msl_briefing_pack", "🧠 MSL Briefing Pack Generator"),
        ("med_info_response", "📚 Medical Information Response Generator"),
        ("congress_adboard_planner", "🎤 Congress & Advisory Board Planner"),
        ("additional_header", "— Additional Tools —"),
        ("insight_thematic_analysis", "📊 Insight Capture & Thematic Analysis"),
        ("exec_summary", "📈 Medical Affairs Executive Summary Generator"),
        ("sop_drafting", "📑 Medical Affairs SOP Drafting Tool"),
        ("compliance_governance", "🔒 Compliance & Governance Summary"),
    ]

    labels = [label for _, label in options]
    keys = [key for key, _ in options]

    choice_label = st.sidebar.radio("Go to", labels, index=0)
    choice_key = keys[labels.index(choice_label)]

    # headers route to home
    if choice_key in ("core_header", "additional_header"):
        return "home"
    return choice_key


def render_header() -> None:
    st.markdown(f"#### {DRAFT_ONLY_BANNER_TEXT}")
    st.caption(
        "This internal MVP is **stateless** and does not retain your content beyond the session refresh. "
        "All outputs are **drafts** and must be reviewed per your medical/legal/regulatory process."
    )


# -----------------------------
# HOME PAGE (TOOL TILES)
# -----------------------------
def render_home(tool_choice_key: Optional[str] = None) -> Optional[str]:
    st.title(APP_TITLE)
    render_header()

    st.markdown("### Tool Suite")
    st.write("Select a tool using the sidebar, or click a tile below.")

    # Tile grid
    core = [t for t in TOOLS.values() if t.category == "Core Medical Affairs Tools" and t.key != "compliance_governance"]
    additional = [t for t in TOOLS.values() if t.category == "Additional Tools"]

    def tile(tool: ToolSpec) -> Optional[str]:
        label = f"{tool.emoji} {tool.name}"
        if st.button(label, key=f"tile_{tool.key}", use_container_width=True):
            return tool.key
        st.markdown(f"<div class='pdm-muted'>{tool.description}</div>", unsafe_allow_html=True)
        st.write("")  # spacing
        return None

    st.subheader("Core Medical Affairs Tools")
    c1, c2 = st.columns(2)
    picked = None
    for i, tool in enumerate(core):
        with (c1 if i % 2 == 0 else c2):
            res = tile(tool)
            if res:
                picked = res

    st.subheader("Additional Tools")
    a1, a2 = st.columns(2)
    for i, tool in enumerate(additional):
        with (a1 if i % 2 == 0 else a2):
            res = tile(tool)
            if res:
                picked = res

    return picked


# -----------------------------
# TOOL UIs
# -----------------------------
def export_section(tool_key: str, title: str, parsed: Dict[str, Any]) -> None:
    """
    Export/copy section (Streamlit-native):
    - st.code() provides built-in copy button
    - Download .txt
    - Optional PDF download (reportlab)
    """
    st.markdown("### Output (JSON draft)")
    pretty = as_pretty_text(parsed)
    st.code(pretty, language="json")

    # filenames are randomised and session-only; no content is stored server-side by this code
    suffix = uuid.uuid4().hex[:8]
    txt_name = f"{tool_key}_{suffix}.txt"
    pdf_name = f"{tool_key}_{suffix}.pdf"

    col1, col2 = st.columns(2)
    with col1:
        make_txt_download("⬇️ Download .txt", pretty, txt_name)
    with col2:
        make_pdf_download("⬇️ Download PDF", title, pretty, pdf_name)


def run_generation_flow(
    *,
    model: str,
    tool: ToolSpec,
    user_payload: Dict[str, Any],
) -> None:
    """
    Executes:
    - Safety scan (AE/PII)
    - Generation call
    - Displays output + export
    """
    # Global safety scan on the full concatenated payload text
    concat_text = json.dumps(user_payload, ensure_ascii=False)
    block = detect_safety_issues(concat_text)
    if block["blocked"]:
        render_blocked(block)
        return

    with st.spinner(f"Generating ABPI-compliant draft using {model}..."):
        parsed, meta = generate_json(
            model=model,
            tool_name=tool.name,
            system_prompt=tool.system_prompt,
            user_payload=user_payload,
            json_schema=tool.schema,
        )

    if not meta.get("ok") or parsed is None:
        st.error("Generation failed.", icon="⚠️")
        st.write(meta.get("error", "Unknown error"))
        with st.expander("Debug (raw model output)", expanded=False):
            st.code(meta.get("raw", ""), language="text")
            st.caption(meta.get("parse_note", ""))
        return

    st.success("Draft generated (review required).", icon="✅")
    st.caption(meta.get("parse_note", ""))
    export_section(tool.key, f"{APP_TITLE} — {tool.name}", parsed)


def tool_scientific_narrative(model: str) -> None:
    tool = TOOLS["scientific_narrative"]
    st.header(f"{tool.emoji} {tool.name}")
    render_header()

    with st.form("scientific_narrative_form"):
        therapy_area = st.text_input("Therapy area")
        product = st.text_input("Product / Molecule")
        indication = st.text_input("Indication")
        moa = st.text_area("Mechanism of Action", height=120)
        pubs = st.text_area("Key publications (paste text / citations)", height=180)
        positioning = st.text_area("Internal positioning notes (non-promotional)", height=180)
        submitted = st.form_submit_button("Generate draft", use_container_width=True)

    if submitted:
        payload = {
            "therapy_area": therapy_area,
            "product_or_molecule": product,
            "indication": indication,
            "mechanism_of_action": moa,
            "key_publications_user_provided": pubs,
            "internal_positioning_notes": positioning,
            "country_default": "UK & Ireland",
        }
        run_generation_flow(model=model, tool=tool, user_payload=payload)


def tool_msl_briefing_pack(model: str) -> None:
    tool = TOOLS["msl_briefing_pack"]
    st.header(f"{tool.emoji} {tool.name}")
    render_header()

    with st.form("msl_briefing_pack_form"):
        therapy_area = st.text_input("Therapy area")
        product = st.text_input("Product / Molecule")
        indication = st.text_input("Indication")
        objectives_context = st.text_area("Context / objectives (what is the scientific exchange purpose?)", height=140)
        evidence = st.text_area("Evidence / key publications provided by user", height=200)
        constraints = st.text_area("Internal constraints / boundaries (e.g., on-label scope)", height=120)
        submitted = st.form_submit_button("Generate draft", use_container_width=True)

    if submitted:
        payload = {
            "therapy_area": therapy_area,
            "product_or_molecule": product,
            "indication": indication,
            "context_objectives": objectives_context,
            "user_provided_evidence": evidence,
            "boundaries": constraints,
            "country_default": "UK & Ireland",
        }
        run_generation_flow(model=model, tool=tool, user_payload=payload)


def tool_med_info_response(model: str) -> None:
    tool = TOOLS["med_info_response"]
    st.header(f"{tool.emoji} {tool.name}")
    render_header()

    st.warning(
        "Guardrail: generation will be blocked if adverse event / PV content or patient-identifiable data is detected.",
        icon="🛑",
    )

    with st.form("med_info_response_form"):
        country = st.selectbox("Country", ["UK", "Ireland"], index=0)
        audience = st.selectbox("Audience", ["HCP", "Pharmacist", "Payer"], index=0)
        question = st.text_area("Medical question (no patient case details)", height=140)
        evidence = st.text_area("Evidence provided by user (paste only approved excerpts/citations)", height=220)
        submitted = st.form_submit_button("Generate draft", use_container_width=True)

    if submitted:
        payload = {
            "country": country,
            "audience": audience,
            "medical_question": question,
            "user_provided_evidence": evidence,
        }
        run_generation_flow(model=model, tool=tool, user_payload=payload)


def tool_congress_adboard_planner(model: str) -> None:
    tool = TOOLS["congress_adboard_planner"]
    st.header(f"{tool.emoji} {tool.name}")
    render_header()

    with st.form("congress_adboard_planner_form"):
        meeting_type = st.selectbox("Type", ["Congress activity", "Advisory Board", "Hybrid"], index=0)
        goals = st.text_area("Objectives (scientific exchange / insight goals)", height=140)
        audience = st.text_input("Intended participants (e.g., specialty mix, stakeholder types)")
        constraints = st.text_area("Constraints (compliance boundaries, topics in/out of scope)", height=140)
        context = st.text_area("Scientific context / evidence (user-provided only)", height=220)
        submitted = st.form_submit_button("Generate draft", use_container_width=True)

    if submitted:
        payload = {
            "meeting_type": meeting_type,
            "objectives": goals,
            "participants": audience,
            "constraints": constraints,
            "user_provided_context": context,
            "country_default": "UK & Ireland",
        }
        run_generation_flow(model=model, tool=tool, user_payload=payload)


def tool_insight_thematic_analysis(model: str) -> None:
    tool = TOOLS["insight_thematic_analysis"]
    st.header(f"{tool.emoji} {tool.name}")
    render_header()

    st.caption("Session-only: nothing is saved beyond the current session/refresh.")

    with st.form("insight_thematic_analysis_form"):
        notes = st.text_area("Paste insight notes (de-identified; no patient data)", height=260)
        framing = st.text_area("Optional framing (therapy area, stakeholder type, meeting context)", height=140)
        submitted = st.form_submit_button("Analyse insights", use_container_width=True)

    if submitted:
        payload = {
            "insight_notes": notes,
            "context_framing": framing,
            "country_default": "UK & Ireland",
        }
        run_generation_flow(model=model, tool=tool, user_payload=payload)


def tool_exec_summary(model: str) -> None:
    tool = TOOLS["exec_summary"]
    st.header(f"{tool.emoji} {tool.name}")
    render_header()

    with st.form("exec_summary_form"):
        source_text = st.text_area("Paste source material (insights, notes, draft outputs — de-identified)", height=260)
        leader_audience = st.selectbox("Leadership audience", ["UK/I Medical Lead", "Country Medical Director", "Regional/Global MA Leadership"], index=0)
        focus = st.text_area("Focus areas (optional)", height=140)
        submitted = st.form_submit_button("Generate executive summary", use_container_width=True)

    if submitted:
        payload = {
            "source_material": source_text,
            "leadership_audience": leader_audience,
            "focus_areas": focus,
            "country_default": "UK & Ireland",
        }
        run_generation_flow(model=model, tool=tool, user_payload=payload)


def tool_sop_drafting(model: str) -> None:
    tool = TOOLS["sop_drafting"]
    st.header(f"{tool.emoji} {tool.name}")
    render_header()

    with st.form("sop_drafting_form"):
        sop_title = st.text_input("SOP title")
        purpose = st.text_area("Purpose and scope (what should this SOP cover?)", height=160)
        process_requirements = st.text_area("Process requirements (steps, approvals, roles, timelines)", height=240)
        governance = st.text_area("Governance requirements (training, documentation, audit readiness)", height=160)
        submitted = st.form_submit_button("Draft SOP", use_container_width=True)

    if submitted:
        payload = {
            "sop_title": sop_title,
            "purpose_scope": purpose,
            "process_requirements": process_requirements,
            "governance_requirements": governance,
            "country_default": "UK & Ireland",
        }
        run_generation_flow(model=model, tool=tool, user_payload=payload)


def tool_compliance_governance_static() -> None:
    st.header("🔒 Compliance & Governance Summary")
    render_header()

    st.markdown("### What this internal MVP is")
    st.write(
        "- A **private internal** Medical Affairs drafting workbench.\n"
        "- A foundation to iterate on workflows and prompts.\n"
        "- **Not** a public-facing product.\n"
    )

    st.markdown("### Stateless design (No data retention)")
    st.write(
        "- No databases.\n"
        "- No file storage of user content.\n"
        "- No authentication and no analytics.\n"
        "- Session-only operation; outputs are generated on demand and export happens **in-memory**.\n"
    )

    st.markdown("### ABPI Code–compliant intent (enforced)")
    st.write(
        "The generation engine injects ABPI-aligned rules into **every** tool prompt:\n"
        "- Non-promotional scientific intent\n"
        "- No inducements/calls to action\n"
        "- No superiority/comparative claims\n"
        "- No off-label encouragement\n"
        "- No prescriptive clinical advice\n"
        "- Balanced, factual tone with uncertainty\n"
        "- **References only from user-provided material**\n"
        "- Drafting support only; review required\n"
    )

    st.markdown("### Drafting support only / review requirement")
    st.write(
        "- Outputs are **drafts** intended to accelerate internal Medical Affairs work.\n"
        "- All content requires medical/legal/regulatory review before use.\n"
    )

    st.markdown("### Guardrail limitations (honest constraints)")
    st.write(
        "- Safety detection is keyword/pattern based (conservative but not perfect).\n"
        "- The tool blocks if it detects potential adverse event / PV content or explicit patient-identifiable data.\n"
        "- The system does not validate scientific accuracy beyond the evidence you provide.\n"
        "- The system will not create references; it will only list references present in your input.\n"
    )


# -----------------------------
# MAIN ROUTER
# -----------------------------
def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    inject_css()

    # Sidebar: model selector + persistent badges + navigation
    selected_model = st.session_state.get("selected_model", DEFAULT_MODEL)
    model = sidebar_persistent_ui(selected_model)
    st.session_state["selected_model"] = model

    page = tool_nav()

    # Home tiles (also allow tile click to navigate by setting session state)
    if page == "home":
        picked = render_home()
        if picked:
            # Navigate to selected tool within the same session
            st.session_state["nav_to"] = picked
            st.rerun()

        nav_to = st.session_state.pop("nav_to", None)
        if nav_to:
            page = nav_to

    # Show selected model clearly in main area too
    st.caption(f"Model in use: **{model}**")

    # Route
    if page == "scientific_narrative":
        tool_scientific_narrative(model)
    elif page == "msl_briefing_pack":
        tool_msl_briefing_pack(model)
    elif page == "med_info_response":
        tool_med_info_response(model)
    elif page == "congress_adboard_planner":
        tool_congress_adboard_planner(model)
    elif page == "insight_thematic_analysis":
        tool_insight_thematic_analysis(model)
    elif page == "exec_summary":
        tool_exec_summary(model)
    elif page == "sop_drafting":
        tool_sop_drafting(model)
    elif page == "compliance_governance":
        tool_compliance_governance_static()
    else:
        render_home()


if __name__ == "__main__":
    main()
