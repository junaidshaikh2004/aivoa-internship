"""
Standalone LangGraph pipeline: complaint text in, structured JSON out.

This is an isolation test, not the real app. It proves the AI logic works
before we wire anything else up:
  1. An LLM turns messy complaint text into the structured fields our
     `complaints` table expects (extraction node).
  2. A second LLM call looks at that structured data and adds an AI risk
     assessment on top (risk assessment node).
  3. A third, separate flow takes an already-extracted complaint plus a
     short correction message and returns ONLY the fields that actually
     changed (correction node).
  4. A PDF can be used as the input source instead of typed text, by
     extracting its text before handing it to the same extraction node.

LangGraph concepts used here, in plain terms:
  - State: one shared dict-like object (a TypedDict) that flows through
    a graph. Every node reads it and can add/update fields.
  - Node: a plain Python function shaped as (state) -> dict of updates.
    LangGraph merges whatever a node returns back into the shared state.
  - Edge: wiring that says "after node A, run node B."

Two graphs are built:
  - `graph`: extract -> assess_risk -> END (the main intake pipeline)
  - `correction_graph`: apply_correction -> END (its own graph, since a
    correction happens on a different input shape, at a different time,
    not always right after intake)

All three LLM calls use Groq's strict JSON-schema structured output mode,
so responses are guaranteed to match the schema we hand them (not just
"asked to return JSON").
"""

import json
import os
from typing import TypedDict

from dotenv import load_dotenv
from groq import Groq
from langgraph.graph import StateGraph, END
from pypdf import PdfReader

load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])

EXTRACTION_MODEL = "openai/gpt-oss-20b"
RISK_MODEL = "qwen/qwen3.8-27b"


# ---------------------------------------------------------------------------
# State: the shared structure that flows through the main intake graph.
# ---------------------------------------------------------------------------
class ComplaintState(TypedDict, total=False):
    raw_text: str

    # Filled in by the extraction node
    product_type: str
    complaint_source: str
    customer_name: str
    product_name: str
    product_strength_grade: str
    batch_lot_number: str
    affected_quantity: str
    manufacturing_date: str
    expiry_date: str
    originating_site_block: str
    impacted_npm: str
    complaint_category: str
    complaint_description: str

    # Filled in by the risk assessment node
    severity_suggested: str
    suggested_next_action: str
    initial_risk_assessment: str


# ---------------------------------------------------------------------------
# JSON schemas Groq must match exactly (strict structured output mode).
# ---------------------------------------------------------------------------
EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "product_type": {
            "type": "string",
            "enum": ["API", "FDF"],
            "description": "API = active pharmaceutical ingredient (bulk raw material). "
                            "FDF = finished dose form (the packaged end product, e.g. capsules/tablets).",
        },
        "complaint_source": {"type": "string", "description": "Who/what channel the complaint came from, e.g. 'Customer email', 'Pharmacy report'."},
        "customer_name": {"type": "string"},
        "product_name": {"type": "string"},
        "product_strength_grade": {"type": "string", "description": "e.g. '500 mg' for FDF, or a grade like 'USP Grade' for API."},
        "batch_lot_number": {"type": "string"},
        "affected_quantity": {"type": "string", "description": "Include units as stated, e.g. '3 of 20 drums', '50 kg'."},
        "manufacturing_date": {"type": "string", "description": "ISO format YYYY-MM-DD if a specific day is known, otherwise YYYY-MM if only month/year is given."},
        "expiry_date": {"type": "string", "description": "Same date format rules as manufacturing_date."},
        "originating_site_block": {"type": "string", "description": "Manufacturing site/block if mentioned, else 'Not specified'."},
        "impacted_npm": {"type": "string", "description": "Impacted non-product material if mentioned, else 'None reported'."},
        "complaint_category": {"type": "string", "description": "Short category label, e.g. 'Discoloration', 'Contamination', 'Packaging defect'."},
        "complaint_description": {
            "type": "string",
            "description": "A synthesized 1-3 sentence summary of the complaint IN YOUR OWN WORDS. "
                            "Do not copy-paste the raw input verbatim.",
        },
    },
    "required": [
        "product_type", "complaint_source", "customer_name", "product_name",
        "product_strength_grade", "batch_lot_number", "affected_quantity",
        "manufacturing_date", "expiry_date", "originating_site_block",
        "impacted_npm", "complaint_category", "complaint_description",
    ],
    "additionalProperties": False,
}

RISK_SCHEMA = {
    "type": "object",
    "properties": {
        "severity_suggested": {"type": "string", "enum": ["Minor", "Major", "Critical"]},
        "suggested_next_action": {"type": "string", "description": "A concrete next step for the QA team, one sentence."},
        "initial_risk_assessment": {
            "type": "string",
            "description": "1-3 complete sentences of substantive reasoning explaining WHY this severity "
                            "was chosen (quality/safety/compliance implications). Never a bare label or tag.",
        },
    },
    "required": ["severity_suggested", "suggested_next_action", "initial_risk_assessment"],
    "additionalProperties": False,
}


def call_groq_structured(model: str, system_prompt: str, user_prompt: str, schema_name: str, schema: dict) -> dict:
    """Call Groq's chat completion API in strict JSON-schema mode and parse the result."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": schema,
            },
        },
    )
    return json.loads(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# Node 1: extraction
# ---------------------------------------------------------------------------
def extract_complaint_details(state: ComplaintState) -> dict:
    extracted = call_groq_structured(
        model=EXTRACTION_MODEL,
        system_prompt=(
            "You are a pharmaceutical QMS intake assistant. Read a raw customer "
            "complaint and extract structured fields for logging into a complaint "
            "management system. If a field isn't mentioned in the text, use "
            "'Not specified' (or 'None reported' for impacted_npm)."
        ),
        user_prompt=f"Raw complaint text:\n\n{state['raw_text']}",
        schema_name="complaint_extraction",
        schema=EXTRACTION_SCHEMA,
    )
    return extracted


# ---------------------------------------------------------------------------
# Node 2: risk assessment (reads the fields node 1 just produced)
# ---------------------------------------------------------------------------
def assess_risk(state: ComplaintState) -> dict:
    context = {
        "product_type": state.get("product_type"),
        "product_name": state.get("product_name"),
        "affected_quantity": state.get("affected_quantity"),
        "complaint_category": state.get("complaint_category"),
        "complaint_description": state.get("complaint_description"),
    }
    risk = call_groq_structured(
        model=RISK_MODEL,
        system_prompt=(
            "You are a pharmaceutical QA risk assessor. Given structured details about a "
            "logged complaint, suggest a severity level, a concrete next action, and reasoning "
            "for that severity.\n\n"
            "initial_risk_assessment MUST be 1-3 complete sentences of substantive reasoning "
            "explaining WHY this severity was chosen (specific quality, safety, or compliance "
            "implications). It must NEVER be a bare label, tag, or short phrase like 'Medium' "
            "or 'High Priority Investigation' repeated as the whole value — that field always "
            "needs real reasoning, not a restated category name. It must also NEVER begin with "
            "a severity word or category label followed by a dash or colon (e.g. 'Medium - ', "
            "'High:') — start directly with the substantive reasoning itself; severity_suggested "
            "already captures the category separately."
        ),
        user_prompt=f"Complaint details:\n\n{json.dumps(context, indent=2)}",
        schema_name="risk_assessment",
        schema=RISK_SCHEMA,
    )
    return risk


# ---------------------------------------------------------------------------
# Build the main intake graph: extract -> assess_risk -> END
# ---------------------------------------------------------------------------
builder = StateGraph(ComplaintState)
builder.add_node("extract", extract_complaint_details)
builder.add_node("assess_risk", assess_risk)
builder.set_entry_point("extract")
builder.add_edge("extract", "assess_risk")
builder.add_edge("assess_risk", END)
graph = builder.compile()


def run_pipeline(raw_text: str) -> dict:
    final_state = graph.invoke({"raw_text": raw_text})
    final_state.pop("raw_text", None)
    return final_state


# ---------------------------------------------------------------------------
# Correction flow: its own small graph, since it runs on a different input
# shape (current state + a correction message) at a different time than
# the initial intake pipeline above.
# ---------------------------------------------------------------------------
class CorrectionState(TypedDict, total=False):
    current_complaint: dict
    correction_message: str
    changed_fields: dict


def _make_correction_property(spec: dict) -> dict:
    """Widen an extraction/risk field spec so '' can mean 'not changed'.

    Deliberately does NOT repeat the "empty string means unchanged" rule in
    every field's description - stuffing the same sentence into all 16
    descriptions made the schema so repetitive that the model started
    echoing the schema's own structure back instead of filling in values.
    The rule is stated once, in the system prompt, instead.
    """
    new_spec = dict(spec)
    if "enum" in spec:
        new_spec["enum"] = spec["enum"] + [""]
    return new_spec


_CORRECTION_PROPERTIES = {
    name: _make_correction_property(spec)
    for name, spec in {**EXTRACTION_SCHEMA["properties"], **RISK_SCHEMA["properties"]}.items()
}
CORRECTION_SCHEMA = {
    "type": "object",
    "properties": _CORRECTION_PROPERTIES,
    "required": list(_CORRECTION_PROPERTIES.keys()),
    "additionalProperties": False,
}


def apply_correction(state: CorrectionState) -> dict:
    raw_output = call_groq_structured(
        model=EXTRACTION_MODEL,
        system_prompt=(
            "You are given the CURRENT state of a logged pharmaceutical complaint and a short "
            "correction message from a user. Identify ONLY the field(s) the correction message "
            "explicitly changes, and output the complete NEW value for just those fields.\n\n"
            'For every field NOT explicitly mentioned in the correction message, output an empty '
            'string "". Do not guess, re-derive, rephrase, or "improve" any field that was not '
            "explicitly mentioned, even if you think you could infer a better value."
        ),
        user_prompt=(
            f"Current complaint state:\n{json.dumps(state['current_complaint'], indent=2)}\n\n"
            f"Correction message:\n{state['correction_message']}"
        ),
        schema_name="complaint_correction",
        schema=CORRECTION_SCHEMA,
    )
    changed_fields = {key: value for key, value in raw_output.items() if value != ""}
    return {"changed_fields": changed_fields}


correction_builder = StateGraph(CorrectionState)
correction_builder.add_node("apply_correction", apply_correction)
correction_builder.set_entry_point("apply_correction")
correction_builder.add_edge("apply_correction", END)
correction_graph = correction_builder.compile()


def run_correction(current_complaint: dict, correction_message: str) -> dict:
    result = correction_graph.invoke({
        "current_complaint": current_complaint,
        "correction_message": correction_message,
    })
    return result["changed_fields"]


# ---------------------------------------------------------------------------
# PDF ingestion: a plain text-extraction preprocessing step, not an LLM
# node. Its output (raw text) feeds into the exact same extraction node
# used above — nothing about the graph changes.
# ---------------------------------------------------------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _create_sample_complaint_pdf(path: str) -> None:
    """Test-only helper: writes a simple sample complaint PDF so PDF ingestion can be demoed."""
    from fpdf import FPDF

    text = (
        "Complaint Intake Form\n\n"
        "MedLife Retail Pharmacy submitted a complaint regarding Metoprolol "
        "Tartrate Tablets 50 mg, batch MTT250815. Several tablets in a sealed "
        "strip of 10 were found chipped and partially broken upon opening. "
        "Manufacturing date: 20 August 2025. Expiry date: 19 August 2027. "
        "Customer requests a replacement and an investigation into the "
        "packaging/handling process."
    )
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 8, text)
    pdf.output(path)


if __name__ == "__main__":
    # --- Test 1 & 2: fresh complaints through the main intake pipeline ---
    test_cases = {
        "FDF example (reference demo input)": (
            "Apollo Pharmacy reported discolored capsules in Amoxicillin Capsules "
            "500 mg. Batch number AMX240602. Manufacturing date March 2026. "
            "Expiry date February 2028. Please log this complaint."
        ),
        "API example (bulk drum contamination)": (
            "Received a complaint from Meridian Fine Chemicals regarding visible "
            "black particulate contamination found in 2 of 10 drums of "
            "Metformin Hydrochloride API, USP grade, from batch MHC250917. "
            "The drums were manufactured at our Site B - Block 4 facility. "
            "Manufacturing date was 15 September 2025 and expiry is 14 "
            "September 2027. Customer is requesting an urgent investigation "
            "before using the remaining stock."
        ),
    }

    results = {}
    for label, text in test_cases.items():
        print("=" * 80)
        print(label)
        print("=" * 80)
        result = run_pipeline(text)
        results[label] = result
        print(json.dumps(result, indent=2))
        print()

    # --- Test 3: correction on top of the FDF result ---
    print("=" * 80)
    print("Correction example (based on FDF result)")
    print("=" * 80)
    fdf_result = results["FDF example (reference demo input)"]
    correction_message = "ah sorry the batch number is BMX240602 and affected quantity is 48 capsules"
    print(f"Correction message: {correction_message!r}\n")

    changed = run_correction(fdf_result, correction_message)
    print("Fields the model reported as changed:")
    print(json.dumps(changed, indent=2))
    print(f"\nOnly expected batch_lot_number + affected_quantity to change: "
          f"{set(changed.keys()) == {'batch_lot_number', 'affected_quantity'}}")
    print(f"  batch_lot_number: {fdf_result['batch_lot_number']!r} -> {changed.get('batch_lot_number')!r}")
    print(f"  affected_quantity: {fdf_result['affected_quantity']!r} -> {changed.get('affected_quantity')!r}")
    print()

    # --- Test 4: PDF ingestion feeding into the same extraction pipeline ---
    print("=" * 80)
    print("PDF ingestion example")
    print("=" * 80)
    sample_pdf_path = "sample_complaint.pdf"
    _create_sample_complaint_pdf(sample_pdf_path)
    pdf_text = extract_text_from_pdf(sample_pdf_path)
    print("Text extracted from PDF:")
    print(pdf_text)
    print()

    pdf_result = run_pipeline(pdf_text)
    print("Structured output from PDF-sourced text:")
    print(json.dumps(pdf_result, indent=2))
