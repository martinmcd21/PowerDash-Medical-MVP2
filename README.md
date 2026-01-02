# PowerDash Medical (Internal MVP)

PowerDash Medical is a **private internal** Medical Affairs AI drafting workbench, inspired by the PowerDash HR suite.
It is designed as a **multi-tool suite** (not a single app), with a shared layout and a shared LLM generation engine.

## Core principles
- **Stateless:** no persistence beyond the session.
- **No databases, no storage of user content.**
- **No authentication, no analytics.**
- **Streamlit-only** (Python-first).
- Conservative **UK & Ireland** Medical Affairs tone.
- Drafting support only; **medical/legal/regulatory review required**.

## ABPI Code compliance (mandatory)
ABPI-aligned constraints are injected into **every** generation call:
- Non-promotional intent at all times
- No promotional language, inducements, or calls to action
- No comparative/superiority/"best-in-class" claims
- No encouragement of off-label use
- No prescriptive clinical advice
- Balanced, factual, evidence-led tone with uncertainty
- Clear distinction between scientific exchange and promotion
- **References must come only from user-provided material**
- Outputs are **drafts only** and require review

## Safety guardrails
The app blocks generation if it detects:
- Potential adverse event / pharmacovigilance content (keyword-based)
- Patient-identifiable data patterns:
  - NHS number pattern
  - Email address
  - Date of birth pattern
  - Explicit "patient name" field labels

> Phone number detection is intentionally **not implemented** (per requirements).

## Tools included
### Core Medical Affairs Tools
1. 📄 Scientific Narrative Generator  
2. 🧠 MSL Briefing Pack Generator  
3. 📚 Medical Information Response Generator (blocks on AE/PII)
4. 🎤 Congress & Advisory Board Planner  

### Additional Tools
5. 📊 Insight Capture & Thematic Analysis (session-only)
6. 📈 Medical Affairs Executive Summary Generator
7. 🔒 Compliance & Governance Summary (static)
8. 📑 Medical Affairs SOP Drafting Tool

## Model selection
- A sidebar control allows changing the OpenAI model at runtime.
- Default: **gpt-5.2**
- The selected model is passed into the shared generation engine (no hard-coding inside tool logic).

## Export
All tools support:
- Copy via Streamlit-native `st.code()` (built-in copy button)
- Download `.txt`
- Optional PDF download (in-memory) using ReportLab

## Setup
1. Create and activate a virtual environment
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
