import json
import os
import re
import textwrap
from dataclasses import dataclass
from datetime import datetime
from html import escape as html_escape
from io import BytesIO
from typing import Any, Dict, Optional, Tuple, List

import streamlit as st

# Optional PDF export (safe, server-side)
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
except Exception:
    canvas = None  # reportlab not installed


# -----------------------------
# App constants / configuration
# -----------------------------
APP_TITLE = "PowerDash Medical"
PRIMARY_BLUE = "#2563eb"
DEFAULT_MODEL = "gpt-5.2"

TOOL_HOME = "Home"
TOOL_SCI_NARRATIVE = "📄 Scientific Narrative Generator"
TOOL_MSL_BRIEF = "🧠 MSL Briefing Pack Generator"
TOOL_MI_RESPONSE = "📚 Medical Information Response Generator"
TOOL_CONGRESS_PLANNER = "🎤 Congress & Advisory Board Planner"

TOOL_INSIGHTS = "📊 Insight Capture & Thematic Analysis"
TOOL_EXEC_SUMMARY = "📈 Medical Affairs Executive Summary Generator"
TOOL_COMPLIANCE = "🔒 Compliance & Governance Summary"
TOOL_SOP = "📑 Medical Affairs SOP Drafting Tool"

CORE_TOOLS = [
    TOOL_SCI_NARRATIVE,
    TOOL_MSL_BRIEF,
    TOOL_MI_RESPONSE,
    TOOL_CONGRESS_PLANNER,
]
ADDITIONAL_TOOLS = [
    TOOL_INSIGHTS,
    TOOL_EXEC_SUMMARY,
    TOOL_COMPLIANCE,
    TOOL_SOP,
]

ALL_PAGES = [TOOL_HOME] + CORE_TOOLS + ADDITIONAL_TOOLS


# -----------------------------
# ABPI compliance (injected)
# -----------------------------
ABPI_CORE_INSTRUCTIONS = """
You are assisting UK & Ireland Medical Affairs. Your purpose is drafting support only.
You MUST comply with the ABPI Code of Practice. This is a core requirement.

ABPI COMPLIANCE RULES (MANDATORY):
- Non-promotional intent at all times; scientific exchange only.
- No promotional language, inducements, marketing phrasing, or calls to action.
- No comparative, superiority, or "best-in-class" claims.
- No encouragement of off-label use; do not propose unapproved indications/dosing/populations.
- No prescriptive clinical advice; do not instruct clinicians what to do.
- Balanced, factual, evidence-led tone with appropriate uncertainty and limitations.
- Clear separation of evidence vs interpretation; avoid overstatement.
- References MUST come ONLY from user-provided material. Do NOT invent citations.
- Output MUST be a draft and require medical/legal/regulatory review.

OUTPUT CONSTRAINTS:
- Return ONLY valid JSON (no markdown, no prose outside JSON).
- Use the exact JSON keys requested.
- If user-provided evidence is insufficient, say so explicitly and ask for what is missing (within JSON fields).
""".strip()


# -----------------------------
# Safety blocking
# -----------------------------
@dataclass
class SafetyResult:
    blocked: bool
    reasons: List[str]
    guidance: str


AE_KEYWORDS = [
    # Keep intentionally simple + conservative; keyword-based only (as requested)
    "adverse event",
    "adverse reaction",
    "side effect",
    "serious adverse",
    "sae",
    "suspected adverse",
    "pharmacovigilance",
    "pv case",
    "pregnancy exposure",
    "overdose",
    "fatal",
    "hospitalis",  # matches hospitalisation/hospitalization
    "anaphyl",
    "rash",
    "death",
]

# Patient-identifiable data detection (explicitly includes: NHS number, email, DOB, patient name)
# NOTE: Phone number detection MUST BE REMOVED entirely (per requirement) => no phone regex here.

EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# NHS numbers are 10 digits, often spaced as 3-3-4; we match "NHS" near digits OR a 10-digit pattern with optional spaces
NHS_REGEX = re.compile(r"\bNHS\b.{0,20}\b(\d[\d ]{8,}\d)\b", re.IGNORECASE)
TEN_DIGIT_LIKE = re.compile(r"\b\d{10}\b")
DOB_REGEX = re.compile(
    r"\b(DOB|date of birth)\b|\b\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}\b",
    re.IGNORECASE,
)
PATIENT_NAME_REGEX = re.compile(r"\b(patient name|patient:)\b", re.IGNORECASE)


def run_safety_checks(text: str) -> SafetyResult:
    """
    Global safety:
    - Block if AE/PV content detected
    - Block if patient-identifiable content detected (NHS number, email, DOB, patient name)
    - Phone detection intentionally omitted (per requirement)
    """
    if not text or not text.strip():
        return SafetyResult(False, [], "")

    t = text.lower()
    reasons: List[str] = []

    # AE / PV keyword scan
    for kw in AE_KEYWORDS:
        if kw in t:
            reasons.append("Potential adverse event / pharmacovigilance content detected.")
            break

    # Patient-identifiable scans
    if EMAIL_REGEX.search(text):
        reasons.append("Email address detected (potential patient-identifiable data).")

    if NHS_REGEX.search(text) or TEN_DIGIT_LIKE.search(text):
        # Ten digits alone can be many things, but we must be conservative.
        reasons.append("Potential NHS number / identifier detected (potential patient-identifiable data).")

    if DOB_REGEX.search(text):
        reasons.append("Date of birth / DOB detected (potential patient-identifiable data).")

    if PATIENT_NAME_REGEX.search(text):
        reasons.append("Patient name indicator detected (potential patient-identifiable data).")

    blocked = len(reasons) > 0

    guidance = ""
    if blocked:
        guidance = (
            "This tool cannot be used with adverse event / pharmacovigilance content or patient-identifiable data.\n\n"
            "Next steps:\n"
            "- Remove any AE/PV details and route through your established PV process if relevant.\n"
            "- Remove identifiers (NHS number, email, DOB, names) and re-submit with fully de-identified, aggregated information.\n"
            "- If you need an MI response, provide only non-identifiable question context and user-supplied evidence excerpts."
        )

    return SafetyResult(blocked, reasons, guidance)


def render_blocked(result: SafetyResult) -> None:
    """Centralised blocked rendering (per requirement)."""
    st.error("Generation blocked for safety/compliance reasons.")
    for r in result.reasons:
        st.write(f"- {r}")
    st.info(result.guidance)


# -----------------------------
# OpenAI helper (shared engine)
# -----------------------------
def _extract_json_fallback(raw: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Robust-ish JSON fallback:
    - Try direct json.loads
    - If it fails, attempt to extract substring between first '{' and last '}' and parse that
    Returns (parsed_json_or_none, raw_used_for_attempt).
    """
    if not raw:
        return None, ""

    raw_stripped = raw.strip()

    try:
        return json.loads(raw_stripped), raw_stripped
    except Exception:
        pass

    first = raw_stripped.find("{")
    last = raw_stripped.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = raw_stripped[first : last + 1]
        try:
            return json.loads(candidate), candidate
        except Exception:
            return None, candidate

    return None, raw_stripped


def generate_json_with_openai(
    *,
    model: str,
    tool_system_prompt: str,
    user_payload: Dict[str, Any],
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """
    Shared generation engine:
    - Injects ABPI compliance instructions into every prompt
    - Requests JSON-only output
    - Uses OpenAI API key from environment variable OPENAI_API_KEY
    - Attempts Responses API first; falls back to Chat Completions if needed
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

    # Import inside function to avoid import-time failures in some deployments
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("openai package is not installed or could not be imported.") from e

    client = OpenAI(api_key=api_key)

    system = (ABPI_CORE_INSTRUCTIONS + "\n\n" + tool_system_prompt).strip()
    user_content = json.dumps(user_payload, ensure_ascii=False)

    # Preferred: Responses API (OpenAI Python SDK v1+)
    # We ask for JSON output. If the SDK/model doesn't support strict JSON schema,
    # we still robustly parse with fallback.
    raw_text = ""

    # 1) Try Responses API
    try:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            temperature=temperature,
            # "json_object" is broadly supported; if unavailable, it will error and we'll fall back.
            response_format={"type": "json_object"},
        )

        # SDKs vary slightly; prefer output_text when available
        raw_text = getattr(resp, "output_text", "") or ""
        if not raw_text and hasattr(resp, "output"):
            # Attempt to reconstruct from output blocks
            chunks = []
            for item in resp.output:
                if getattr(item, "type", "") == "message":
                    for c in getattr(item, "content", []):
                        if getattr(c, "type", "") in ("output_text", "text"):
                            chunks.append(getattr(c, "text", ""))
            raw_text = "\n".join([c for c in chunks if c])

    except Exception:
        raw_text = ""

    # 2) Fall back to Chat Completions
    if not raw_text:
        try:
            chat = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_object"},
            )
            raw_text = chat.choices[0].message.content or ""
        except Exception as e:
            raise RuntimeError(f"OpenAI request failed (Responses + Chat Completions). Details: {e}") from e

    parsed, attempted = _extract_json_fallback(raw_text)
    if parsed is None:
        # Return a consistent JSON envelope even on parsing failures
        return {
            "status": "error",
            "error": "Model did not return valid JSON.",
            "raw_model_output": attempted[:8000],  # keep bounded
        }
    return parsed


# -----------------------------
# Export helpers
# -----------------------------
def make_txt_download(filename_base: str, text: str) -> Tuple[str, bytes]:
    safe_base = re.sub(r"[^A-Za-z0-9._-]+", "_", filename_base).strip("_") or "powerdash_medical"
    filename = f"{safe_base}.txt"
    return filename, text.encode("utf-8")


def make_pdf_bytes(title: str, text: str) -> Optional[bytes]:
    if canvas is None:
        return None

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # Simple, robust PDF renderer (monospace-ish layout)
    margin = 40
    y = height - margin
    line_height = 12

    c.setTitle(title)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, title[:110])
    y -= 22

    c.setFont("Helvetica", 10)
    wrapped_lines: List[str] = []
    for paragraph in text.splitlines():
        if not paragraph.strip():
            wrapped_lines.append("")
            continue
        wrapped_lines.extend(textwrap.wrap(paragraph, width=105))

    for line in wrapped_lines:
        if y <= margin:
            c.showPage()
            c.setFont("Helvetica", 10)
            y = height - margin
        c.drawString(margin, y, line[:150])
        y -= line_height

    c.save()
    buffer.seek(0)
    return buffer.getvalue()


def render_output_block(title: str, obj: Any) -> str:
    """
    Standardised rendering:
    - Convert JSON to pretty string for copying and downloads
    """
    if isinstance(obj, (dict, list)):
        return json.dumps(obj, indent=2, ensure_ascii=False)
    return str(obj)


def render_exports(filename_base: str, content_text: str) -> None:
    """
    Export UI:
    - Copy via st.code() built-in copy
    - Download .txt
    - Optional PDF download
    """
    st.subheader("Copy / Download")

    # Copy: Streamlit-native copy button in code widget
    st.code(content_text, language="json")

    txt_name, txt_bytes = make_txt_download(filename_base, content_text)
    st.download_button(
        label="Download .txt",
        data=txt_bytes,
        file_name=txt_name,
        mime="text/plain",
        use_container_width=True,
    )

    pdf_bytes = make_pdf_bytes(filename_base, content_text)
    if pdf_bytes is not None:
        pdf_name = re.sub(r"[^A-Za-z0-9._-]+", "_", filename_base).strip("_") or "powerdash_medical"
        st.download_button(
            label="Download PDF",
            data=pdf_bytes,
            file_name=f"{pdf_name}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        st.caption("PDF export unavailable (reportlab not installed).")


# -----------------------------
# UI: CSS (no JS, no overlays)
# -----------------------------
def inject_css() -> None:
    """
    Streamlit-safe CSS:
    - Styles st.button to look like blue tool tiles
    - No JS, no components, no click-intercept overlays
    """
    st.markdown(
        f"""
        <style>
        /* Make buttons fill container */
        div.stButton > button {{
            width: 100%;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.25);
            background: {PRIMARY_BLUE};
            color: white;
            padding: 0.9rem 1rem;
            font-weight: 700;
            text-align: left;
        }}
        div.stButton > button:hover {{
            filter: brightness(1.06);
            border-color: rgba(255,255,255,0.45);
        }}
        div.stButton > button:active {{
            transform: translateY(1px);
        }}

        /* Slightly tighter main container */
        .block-container {{
            padding-top: 1.6rem;
        }}

        /* Sidebar title spacing */
        section[data-testid="stSidebar"] .block-container {{
            padding-top: 1.2rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,  # CSS only; no JS; no HTML overlays
    )


# -----------------------------
# Prompts per tool (system)
# -----------------------------
SCI_NARRATIVE_SYSTEM = """
You will generate a scientific narrative and variants for Medical Affairs use (UK & Ireland default).
Return JSON with these EXACT keys:

{
  "core_scientific_narrative": "string",
  "disease_state_overview": "string",
  "short_form_variants": {
    "msl_conversations": "string",
    "internal_training": "string",
    "congress_discussions": "string"
  },
  "evidence_traceability": {
    "used_sources": ["string"],
    "gaps_or_limitations": ["string"]
  }
}

Rules:
- Use ONLY user-provided publications/notes as sources.
- If evidence is limited, state limitations explicitly in "gaps_or_limitations".
- Maintain a conservative, factual tone; avoid promotional phrasing.
""".strip()

MSL_BRIEF_SYSTEM = """
Generate a non-promotional MSL briefing pack draft (UK & Ireland default).
Return JSON with these EXACT keys:

{
  "objectives": ["string"],
  "key_scientific_messages": ["string"],
  "stakeholder_hypotheses": ["string"],
  "discussion_guide": ["string"],
  "anticipated_qa": [
    {"question": "string", "answer": "string", "evidence_basis": "string"}
  ],
  "do_dont_guidance": {
    "do": ["string"],
    "dont": ["string"]
  },
  "follow_up_actions": ["string"],
  "evidence_traceability": {
    "used_sources": ["string"],
    "gaps_or_limitations": ["string"]
  }
}

Rules:
- No calls to action; follow-up actions must be operational/medical and non-promotional (e.g., "share requested paper").
- Answers must not introduce references not present in the user evidence.
""".strip()

MI_RESPONSE_SYSTEM = """
Draft a Medical Information response (UK or Ireland).
Return JSON with these EXACT keys:

{
  "long_form_written_response": "string",
  "short_verbal_response": "string",
  "reference_list_user_provided_only": ["string"],
  "uncertainties_and_limitations": ["string"],
  "compliance_checks": {
    "non_promotional": "string",
    "off_label_avoidance": "string",
    "no_prescriptive_advice": "string",
    "references_user_provided_only": "string"
  }
}

Rules:
- Blocked content (AE/PV or patient-identifiable) is handled outside the model. If evidence is insufficient, say so.
- Do NOT hallucinate references; the reference list must be drawn ONLY from the user evidence text.
""".strip()

CONGRESS_SYSTEM = """
Create a compliant Congress & Advisory Board plan for Medical Affairs scientific exchange.
Return JSON with these EXACT keys:

{
  "agenda": ["string"],
  "discussion_guide": ["string"],
  "question_bank": ["string"],
  "insight_capture_framework": ["string"],
  "compliance_notes": ["string"]
}

Rules:
- Non-promotional; focus on scientific exchange and insights gathering.
""".strip()

INSIGHTS_SYSTEM = """
You will group session-only insights into themes, separate signal vs noise, and produce an executive-ready summary.
Return JSON with these EXACT keys:

{
  "themes": [
    {"theme": "string", "supporting_insights": ["string"], "confidence": "string"}
  ],
  "signal_vs_noise": {
    "signal": ["string"],
    "noise_or_low_confidence": ["string"]
  },
  "executive_ready_summary": "string",
  "recommended_next_steps": ["string"]
}

Rules:
- Base outputs ONLY on the user-provided insights text.
- Be explicit about uncertainty and confidence.
""".strip()

EXEC_SUMMARY_SYSTEM = """
Create a leadership-ready Medical Affairs executive summary (UK & Ireland default tone).
Return JSON with these EXACT keys:

{
  "summary": "string",
  "themes": ["string"],
  "risks": ["string"],
  "opportunities": ["string"],
  "recommended_next_steps": ["string"],
  "assumptions_and_limits": ["string"]
}

Rules:
- Use only user-provided inputs; do not add external facts.
""".strip()

SOP_SYSTEM = """
Draft a conservative Medical Affairs SOP in ABPI-aware regulatory tone.
Return JSON with these EXACT keys:

{
  "sop_title": "string",
  "purpose": "string",
  "scope": "string",
  "definitions": ["string"],
  "roles_and_responsibilities": ["string"],
  "procedure": ["string"],
  "documentation_and_records": ["string"],
  "compliance_and_quality_checks": ["string"],
  "version_control_stub": {
    "version": "string",
    "effective_date": "string",
    "owner": "string",
    "review_cycle": "string"
  }
}

Rules:
- SOP should be generic and conservative; no promotional language.
""".strip()


# -----------------------------
# Page routing helpers
# -----------------------------
def set_page(page: str) -> None:
    st.session_state["page"] = page


def get_page() -> str:
    return st.session_state.get("page", TOOL_HOME)


def sidebar() -> Tuple[str, str]:
    st.sidebar.title(APP_TITLE)

    # Persistent compliance badges / statements (as required)
    st.sidebar.success("✅ ABPI Code Compliant")
    st.sidebar.info("🩺 Medical Affairs Drafting Support Only")
    st.sidebar.warning("🔎 Medical review required")
    st.sidebar.caption("🧼 No data retention")

    st.sidebar.divider()

    # Model selection (new requirement)
    st.sidebar.subheader("Model")
    model = st.sidebar.text_input("OpenAI model (runtime)", value=st.session_state.get("model", DEFAULT_MODEL))
    model = model.strip() or DEFAULT_MODEL
    st.session_state["model"] = model
    st.sidebar.caption(f"Current model in use: **{html_escape(model)}**")

    st.sidebar.divider()

    # Navigation
    st.sidebar.subheader("Navigation")
    # Use radio for stable navigation; tiles also available on home
    page = st.sidebar.radio(
        "Go to",
        options=ALL_PAGES,
        index=ALL_PAGES.index(get_page()) if get_page() in ALL_PAGES else 0,
        label_visibility="collapsed",
    )
    st.session_state["page"] = page

    return page, model


# -----------------------------
# Home (tiles)
# -----------------------------
def home_page() -> None:
    st.title(APP_TITLE)
    st.caption("ABPI Code Compliant • Medical Affairs Drafting Support Only • Medical review required • No data retention")

    st.write(
        "This internal MVP is **stateless**: it does not persist your inputs or outputs beyond the current session refresh. "
        "All outputs are **drafts** and must be reviewed through your medical/legal/regulatory process."
    )

    st.subheader("Tool Suite")
    st.caption("Select a tool using the sidebar, or click a tile below.")

    st.markdown("### Core Medical Affairs Tools")
    col1, col2 = st.columns(2, gap="large")

    with col1:
        if st.button(TOOL_SCI_NARRATIVE, key="tile_sci"):
            set_page(TOOL_SCI_NARRATIVE)
            st.rerun()
        if st.button(TOOL_MI_RESPONSE, key="tile_mi"):
            set_page(TOOL_MI_RESPONSE)
            st.rerun()

    with col2:
        if st.button(TOOL_MSL_BRIEF, key="tile_msl"):
            set_page(TOOL_MSL_BRIEF)
            st.rerun()
        if st.button(TOOL_CONGRESS_PLANNER, key="tile_congress"):
            set_page(TOOL_CONGRESS_PLANNER)
            st.rerun()

    st.markdown("### Additional Tools")
    col3, col4 = st.columns(2, gap="large")

    with col3:
        if st.button(TOOL_INSIGHTS, key="tile_insights"):
            set_page(TOOL_INSIGHTS)
            st.rerun()
        if st.button(TOOL_COMPLIANCE, key="tile_compliance"):
            set_page(TOOL_COMPLIANCE)
            st.rerun()

    with col4:
        if st.button(TOOL_EXEC_SUMMARY, key="tile_exec"):
            set_page(TOOL_EXEC_SUMMARY)
            st.rerun()
        if st.button(TOOL_SOP, key="tile_sop"):
            set_page(TOOL_SOP)
            st.rerun()


# -----------------------------
# Tool pages
# -----------------------------
def page_scientific_narrative(model: str) -> None:
    st.header(TOOL_SCI_NARRATIVE)
    st.caption("Create a balanced scientific narrative and variants using only user-provided publications/notes.")

    with st.form("form_sci_narrative", clear_on_submit=False):
        therapy_area = st.text_input("Therapy area", placeholder="e.g., Oncology")
        product = st.text_input("Product / Molecule", placeholder="e.g., [Molecule name]")
        indication = st.text_input("Indication", placeholder="e.g., [Indication]")
        moa = st.text_area("Mechanism of Action", height=120, placeholder="Describe MoA in neutral scientific terms.")
        pubs = st.text_area("Key publications (user pasted)", height=200, placeholder="Paste citations/excerpts you want used.")
        notes = st.text_area("Internal positioning notes", height=160, placeholder="Non-promotional scientific positioning notes.")

        generate = st.form_submit_button("Generate draft", use_container_width=True)

    if generate:
        combined = "\n".join([therapy_area, product, indication, moa, pubs, notes])
        safety = run_safety_checks(combined)
        if safety.blocked:
            render_blocked(safety)
            return

        payload = {
            "therapy_area": therapy_area,
            "product_or_molecule": product,
            "indication": indication,
            "mechanism_of_action": moa,
            "key_publications_user_provided": pubs,
            "internal_positioning_notes": notes,
            "country_default": "UK & Ireland",
            "intent": "Medical Affairs drafting support only; non-promotional scientific exchange",
        }

        with st.spinner("Generating (JSON-only)…"):
            result = generate_json_with_openai(
                model=model,
                tool_system_prompt=SCI_NARRATIVE_SYSTEM,
                user_payload=payload,
                temperature=0.2,
            )

        st.subheader("Draft output (JSON)")
        out_text = render_output_block("Scientific Narrative", result)
        render_exports("scientific_narrative", out_text)


def page_msl_brief(model: str) -> None:
    st.header(TOOL_MSL_BRIEF)
    st.caption("Generate a non-promotional MSL briefing pack: objectives, messages, hypotheses, guides, Q&A, do/don’t.")

    with st.form("form_msl_brief", clear_on_submit=False):
        context = st.text_area(
            "Context (therapy area, product/molecule, indication, scenario)",
            height=160,
            placeholder="Provide neutral scientific context for the briefing pack.",
        )
        evidence = st.text_area(
            "Evidence / publications (user provided only)",
            height=220,
            placeholder="Paste the evidence excerpts/citations you want used (no external sourcing).",
        )
        audience = st.text_input("Primary stakeholder type (optional)", placeholder="e.g., Respiratory specialist, Pharmacist, Payer")
        objectives_hint = st.text_area("Objectives (optional hints)", height=120, placeholder="Optional bullet hints for objectives.")

        generate = st.form_submit_button("Generate draft", use_container_width=True)

    if generate:
        combined = "\n".join([context, evidence, audience, objectives_hint])
        safety = run_safety_checks(combined)
        if safety.blocked:
            render_blocked(safety)
            return

        payload = {
            "context": context,
            "stakeholder_type": audience,
            "objective_hints": objectives_hint,
            "user_provided_evidence": evidence,
            "country_default": "UK & Ireland",
            "intent": "Non-promotional scientific exchange; MSL field medical support",
        }

        with st.spinner("Generating (JSON-only)…"):
            result = generate_json_with_openai(
                model=model,
                tool_system_prompt=MSL_BRIEF_SYSTEM,
                user_payload=payload,
                temperature=0.2,
            )

        st.subheader("Draft output (JSON)")
        out_text = render_output_block("MSL Briefing Pack", result)
        render_exports("msl_briefing_pack", out_text)


def page_mi_response(model: str) -> None:
    st.header(TOOL_MI_RESPONSE)
    st.caption("Draft ABPI-compliant MI responses using ONLY user-provided evidence. Blocks AE / patient-identifiable data.")

    with st.form("form_mi", clear_on_submit=False):
        country = st.selectbox("Country", options=["UK", "Ireland"], index=0)
        audience = st.selectbox("Audience", options=["HCP", "Pharmacist", "Payer"], index=0)
        question = st.text_area("Medical question", height=140, placeholder="Provide the MI question in neutral terms.")
        evidence = st.text_area(
            "Evidence provided by user (required)",
            height=240,
            placeholder="Paste only the evidence you want cited/used. Do not include patient identifiers or AE details.",
        )

        generate = st.form_submit_button("Generate draft", use_container_width=True)

    if generate:
        combined = "\n".join([country, audience, question, evidence])
        safety = run_safety_checks(combined)
        if safety.blocked:
            render_blocked(safety)
            return

        payload = {
            "country": country,
            "audience": audience,
            "medical_question": question,
            "user_provided_evidence": evidence,
            "intent": "Medical Information drafting support only; non-promotional; no prescriptive advice",
        }

        with st.spinner("Generating (JSON-only)…"):
            result = generate_json_with_openai(
                model=model,
                tool_system_prompt=MI_RESPONSE_SYSTEM,
                user_payload=payload,
                temperature=0.2,
            )

        st.subheader("Draft output (JSON)")
        out_text = render_output_block("MI Response", result)
        render_exports("medical_information_response", out_text)


def page_congress_planner(model: str) -> None:
    st.header(TOOL_CONGRESS_PLANNER)
    st.caption("Create compliant agendas, discussion guides, question banks, and insight capture frameworks.")

    with st.form("form_congress", clear_on_submit=False):
        meeting_type = st.selectbox("Type", options=["Congress", "Advisory Board", "Hybrid"], index=0)
        topic = st.text_input("Topic / focus area", placeholder="e.g., Disease state updates, unmet need discussion")
        objectives = st.text_area("Objectives (non-promotional)", height=140, placeholder="State objectives focused on scientific exchange.")
        attendees = st.text_area("Attendee types / roles (optional)", height=100, placeholder="e.g., KOLs, pharmacists, payers")
        evidence = st.text_area("User-provided evidence / notes", height=220, placeholder="Paste notes/publications to ground the plan.")

        generate = st.form_submit_button("Generate draft", use_container_width=True)

    if generate:
        combined = "\n".join([meeting_type, topic, objectives, attendees, evidence])
        safety = run_safety_checks(combined)
        if safety.blocked:
            render_blocked(safety)
            return

        payload = {
            "type": meeting_type,
            "topic": topic,
            "objectives": objectives,
            "attendee_types": attendees,
            "user_provided_notes_and_evidence": evidence,
            "country_default": "UK & Ireland",
            "intent": "Scientific exchange planning; non-promotional",
        }

        with st.spinner("Generating (JSON-only)…"):
            result = generate_json_with_openai(
                model=model,
                tool_system_prompt=CONGRESS_SYSTEM,
                user_payload=payload,
                temperature=0.2,
            )

        st.subheader("Draft output (JSON)")
        out_text = render_output_block("Congress / Ad Board Plan", result)
        render_exports("congress_adboard_plan", out_text)


def page_insights(model: str) -> None:
    st.header(TOOL_INSIGHTS)
    st.caption("Session-only insight grouping, thematic analysis, signal vs noise, and executive-ready summary.")

    with st.form("form_insights", clear_on_submit=False):
        raw_insights = st.text_area(
            "Paste session insights (no persistence beyond refresh)",
            height=260,
            placeholder="Paste bullet notes from MSL insights / congress notes / advisory boards.",
        )
        context = st.text_area("Optional context (therapy area / objectives)", height=120, placeholder="Optional context to frame themes.")
        generate = st.form_submit_button("Analyse", use_container_width=True)

    if generate:
        combined = "\n".join([raw_insights, context])
        safety = run_safety_checks(combined)
        if safety.blocked:
            render_blocked(safety)
            return

        payload = {
            "insights_text": raw_insights,
            "context": context,
            "intent": "Non-promotional insight synthesis for Medical Affairs",
            "stateless_notice": "No persistence beyond refresh",
        }

        with st.spinner("Analysing (JSON-only)…"):
            result = generate_json_with_openai(
                model=model,
                tool_system_prompt=INSIGHTS_SYSTEM,
                user_payload=payload,
                temperature=0.2,
            )

        st.subheader("Draft output (JSON)")
        out_text = render_output_block("Insight Analysis", result)
        render_exports("insight_thematic_analysis", out_text)


def page_exec_summary(model: str) -> None:
    st.header(TOOL_EXEC_SUMMARY)
    st.caption("Leadership-ready summary: themes, risks, opportunities, and recommended next steps (draft).")

    with st.form("form_exec_summary", clear_on_submit=False):
        input_notes = st.text_area(
            "Inputs (notes, insights, data excerpts — user provided only)",
            height=260,
            placeholder="Paste only content you want included (no external facts will be added).",
        )
        audience = st.selectbox("Leadership audience", options=["Medical Director", "UK/Ireland Leadership", "Cross-functional Leadership"], index=0)
        objective = st.text_input("Objective (optional)", placeholder="e.g., Summarise insights from Q4 scientific exchange")
        generate = st.form_submit_button("Generate draft", use_container_width=True)

    if generate:
        combined = "\n".join([input_notes, audience, objective])
        safety = run_safety_checks(combined)
        if safety.blocked:
            render_blocked(safety)
            return

        payload = {
            "leadership_audience": audience,
            "objective": objective,
            "user_provided_inputs": input_notes,
            "intent": "Medical Affairs executive summary drafting support; non-promotional",
        }

        with st.spinner("Generating (JSON-only)…"):
            result = generate_json_with_openai(
                model=model,
                tool_system_prompt=EXEC_SUMMARY_SYSTEM,
                user_payload=payload,
                temperature=0.2,
            )

        st.subheader("Draft output (JSON)")
        out_text = render_output_block("Executive Summary", result)
        render_exports("medical_affairs_exec_summary", out_text)


def page_compliance() -> None:
    st.header(TOOL_COMPLIANCE)
    st.caption("Static explanation of stateless design, ABPI intent, drafting-only scope, and guardrails.")

    st.subheader("Stateless by design")
    st.write(
        "- This MVP does **not** use databases.\n"
        "- It does **not** store user inputs or outputs.\n"
        "- It does **not** write user content to files.\n"
        "- Outputs exist only in the current Streamlit session and disappear on refresh/reload."
    )

    st.subheader("ABPI Code–compliant intent")
    st.write(
        "- The suite is designed for **non-promotional** Medical Affairs drafting support.\n"
        "- Drafts are designed to support **scientific exchange** and internal Medical Affairs workflows.\n"
        "- The model is instructed to avoid promotional language, superiority claims, and off-label encouragement."
    )

    st.subheader("Drafting support only")
    st.write(
        "- Outputs are **drafts** and require **medical/legal/regulatory review** before use.\n"
        "- The tools do not replace Medical, Legal, or Regulatory judgement."
    )

    st.subheader("Guardrails & limitations")
    st.write(
        "- Simple keyword-based detection blocks generation if **adverse event / pharmacovigilance** content is detected.\n"
        "- Generation is blocked if **patient-identifiable data** is detected (NHS number, email, DOB, patient name indicators).\n"
        "- **Phone number detection is intentionally not implemented** (removed entirely as specified).\n"
        "- References must be derived **only** from user-provided evidence; the model is instructed not to invent citations.\n"
        "- Like all LLMs, the model may be incomplete or overly confident; review is mandatory."
    )


def page_sop(model: str) -> None:
    st.header(TOOL_SOP)
    st.caption("Draft a conservative Medical Affairs SOP in ABPI-aware regulatory tone (draft only).")

    with st.form("form_sop", clear_on_submit=False):
        sop_title = st.text_input("SOP title", placeholder="e.g., Medical Information Response Drafting (AI-assisted)")
        purpose = st.text_area("Purpose (what does this SOP govern?)", height=120)
        scope = st.text_area("Scope (what is included/excluded?)", height=120)
        roles = st.text_area("Roles (optional hints)", height=120, placeholder="e.g., Medical signatory, MI lead, reviewer")
        procedure_notes = st.text_area("Procedure notes / steps (optional hints)", height=200, placeholder="Provide the workflow steps you want included.")
        generate = st.form_submit_button("Generate draft", use_container_width=True)

    if generate:
        combined = "\n".join([sop_title, purpose, scope, roles, procedure_notes])
        safety = run_safety_checks(combined)
        if safety.blocked:
            render_blocked(safety)
            return

        payload = {
            "sop_title_hint": sop_title,
            "purpose_hint": purpose,
            "scope_hint": scope,
            "roles_hint": roles,
            "procedure_hints": procedure_notes,
            "country_default": "UK & Ireland",
            "intent": "Conservative SOP drafting support; ABPI-aware; non-promotional",
        }

        with st.spinner("Generating (JSON-only)…"):
            result = generate_json_with_openai(
                model=model,
                tool_system_prompt=SOP_SYSTEM,
                user_payload=payload,
                temperature=0.2,
            )

        st.subheader("Draft output (JSON)")
        out_text = render_output_block("SOP Draft", result)
        render_exports("medical_affairs_sop_draft", out_text)


# -----------------------------
# Main app
# -----------------------------
def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🧬", layout="wide")
    inject_css()

    page, model = sidebar()

    # Top-level persistent header band (mirrors screenshot intent)
    if page != TOOL_HOME:
        st.title(APP_TITLE)
        st.caption("ABPI Code Compliant • Medical Affairs Drafting Support Only • Medical review required • No data retention")

    if page == TOOL_HOME:
        home_page()
    elif page == TOOL_SCI_NARRATIVE:
        page_scientific_narrative(model)
    elif page == TOOL_MSL_BRIEF:
        page_msl_brief(model)
    elif page == TOOL_MI_RESPONSE:
        page_mi_response(model)
    elif page == TOOL_CONGRESS_PLANNER:
        page_congress_planner(model)
    elif page == TOOL_INSIGHTS:
        page_insights(model)
    elif page == TOOL_EXEC_SUMMARY:
        page_exec_summary(model)
    elif page == TOOL_COMPLIANCE:
        page_compliance()
    elif page == TOOL_SOP:
        page_sop(model)
    else:
        st.error("Unknown page.")


if __name__ == "__main__":
    main()
