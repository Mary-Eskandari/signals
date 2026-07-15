"""LLM clinical-report generation, grounded in extracted pipeline features.

Design (per project architecture): the same Pydantic models that validate
pipeline output are serialized directly into the prompt as the grounding
payload — no hand-written re-description of numbers, which eliminates a
whole class of transcription-hallucination risk. The model is forced to
respond via tool use against a schema, so output is validated structured
data, not free text to regex-parse. The disclaimer is never something the
model is asked to write — it's the schema's fixed default, appended by code.
"""

import hashlib
import json

from anthropic import Anthropic
from pydantic import BaseModel

from backend.app.config import ALLOWED_REPORT_MODELS, ANTHROPIC_API_KEY, REPORT_MODEL
from pipeline import store
from pipeline.schemas import ClinicalReport, PatientTrendSummary, ProcedureSummary, ReportSection

SYSTEM_PROMPT = """\
You are drafting a structured, DEMONSTRATION-ONLY summary of hemodynamic \
signal-processing output for a technical portfolio project. It is built \
entirely from public, de-identified research data (PhysioNet SCG-RHC \
catheterization procedures; Chiron CHF telemonitoring dataset). It is NOT a \
real diagnosis, NOT a clinical tool, and NOT for use in any actual \
patient-care decision.

GROUNDING RULE (strict): only state numeric values that appear in the JSON \
payload provided in the user message. If a metric is null, missing, or \
flagged low-quality, say so explicitly ("not available in this window") \
rather than estimating, inferring, or filling in a plausible-sounding number. \
Do not invent measurements.

Clinical reference context (static, for framing only — do not treat as \
patient-specific facts):
- Normal resting pulmonary artery pressure: systolic ~15-30 mmHg, diastolic \
~4-12 mmHg, mean ~8-20 mmHg. Pulmonary hypertension is generally defined as \
mean PA pressure >20 mmHg (current ESC/ERS definition).
- A sustained rise in PA pressure trend, or rapid weight gain (e.g. >2kg over \
3 days), is a classic early-warning pattern for heart-failure decompensation \
— the clinical rationale behind implantable PA pressure monitoring systems \
(e.g. CardioMEMS, Cordella) validated in trials such as CHAMPION and \
PROACTIVE-HF.
- HRV metrics (SDNN, RMSSD) reflect autonomic/rhythm variability; unusually \
high values can indicate arrhythmia (e.g. atrial fibrillation) rather than \
"healthy" variability.
- The SCG AO-AC interval approximates left-ventricular ejection time; this is \
a heuristic, non-clinically-validated measurement in this project (no \
established SCG feature-extraction library exists; detection was custom-built).

Write in a clear, structured clinical-summary tone, but always keep in mind \
— and make clear in the summary — that this is a demonstration/portfolio \
artifact, not a real report.
"""


class _ReportDraft(BaseModel):
    """What the model fills in. Matches ClinicalReport minus `disclaimer`,
    which is never model-authored — see module docstring."""

    summary: str
    pa_pressure_findings: ReportSection
    rhythm_hrv_findings: ReportSection
    scg_findings: ReportSection | None = None
    trend_findings: ReportSection | None = None
    flags: list[str] = []


def build_grounding_payload(
    procedure_summary: ProcedureSummary | None,
    trend_summary: PatientTrendSummary | None,
) -> dict:
    return {
        "pa_pressure_ecg_scg_summary": procedure_summary.model_dump(mode="json") if procedure_summary else None,
        "telemonitoring_trend_summary": trend_summary.model_dump(mode="json") if trend_summary else None,
    }


def payload_hash(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def generate_report(
    procedure_summary: ProcedureSummary | None = None,
    trend_summary: PatientTrendSummary | None = None,
    model: str | None = None,
) -> ClinicalReport:
    if procedure_summary is None and trend_summary is None:
        raise ValueError("need at least one of procedure_summary/trend_summary to generate a report")
    model = model or REPORT_MODEL
    if model not in ALLOWED_REPORT_MODELS:
        raise ValueError(f"model must be one of {ALLOWED_REPORT_MODELS}, got {model!r}")
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is not set (check .env)")

    payload = build_grounding_payload(procedure_summary, trend_summary)
    cache_key = payload_hash(payload)
    cached = store.read_cached_report(cache_key, model)
    if cached is not None:
        return cached

    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    response = client.messages.create(
        model=model,
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(payload, indent=2)}],
        tools=[
            {
                "name": "submit_clinical_report",
                "description": "Submit the structured demonstration clinical report.",
                "input_schema": _ReportDraft.model_json_schema(),
            }
        ],
        tool_choice={"type": "tool", "name": "submit_clinical_report"},
    )

    if response.stop_reason == "max_tokens":
        raise RuntimeError("report generation was truncated (hit max_tokens) — retry or shorten the payload")

    tool_use = next(block for block in response.content if block.type == "tool_use")
    draft = _ReportDraft.model_validate(tool_use.input)
    report = ClinicalReport(**draft.model_dump())

    store.write_cached_report(cache_key, model, report)
    return report
