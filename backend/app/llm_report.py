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
from pydantic import BaseModel, ValidationError

from backend.app.config import ALLOWED_REPORT_MODELS, ANTHROPIC_API_KEY, REPORT_MODEL
from pipeline import store
from pipeline.schemas import ClinicalReport, PatientTrendSummary, ProcedureSummary, ReportSection

SYSTEM_PROMPT = """\
You are drafting a structured clinical-style summary of hemodynamic \
signal-processing output (PhysioNet SCG-RHC catheterization data; Chiron CHF \
telemonitoring data) for a technical portfolio project.

The demonstration/not-a-real-diagnosis framing is shown to the reader \
separately (a persistent banner and a fixed disclaimer field) — do NOT repeat \
it, reference it, or hedge with it in the summary or flags. Write the summary \
and flags as pure clinical-style findings only; assume the reader already \
knows this is a demo.

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

Write in a clear, structured clinical-summary tone. `flags` should be the \
notable clinical-style callouts from THIS data (e.g. "PA pressure markedly \
elevated", "HRV elevated, possible arrhythmia") — never the demo/disclaimer \
notice itself.
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


def _parse_draft(tool_input: dict) -> _ReportDraft:
    """Some models occasionally wrap the payload under an extra key (e.g.
    {"report": {...}}) despite the flat tool schema — observed in practice,
    not hypothetical. Unwrap that shape defensively before giving up."""
    try:
        return _ReportDraft.model_validate(tool_input)
    except ValidationError:
        if len(tool_input) == 1:
            (inner,) = tool_input.values()
            if isinstance(inner, dict):
                return _ReportDraft.model_validate(inner)
        raise


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
    # Include the prompt in the cache key so editing SYSTEM_PROMPT auto-invalidates
    # old cached reports instead of silently serving stale (e.g. pre-fix) output.
    cache_key = payload_hash({"payload": payload, "system_prompt": SYSTEM_PROMPT})
    cached = store.read_cached_report(cache_key, model)
    if cached is not None:
        return cached

    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    tools = [
        {
            "name": "submit_clinical_report",
            "description": "Submit the structured demonstration clinical report.",
            "input_schema": _ReportDraft.model_json_schema(),
        }
    ]
    tool_choice = {"type": "tool", "name": "submit_clinical_report"}
    user_content = json.dumps(payload, indent=2)

    last_error: Exception | None = None
    for attempt in range(2):  # one retry for occasional tool-schema drift, see _parse_draft
        response = client.messages.create(
            model=model,
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
            tools=tools,
            tool_choice=tool_choice,
        )
        if response.stop_reason == "max_tokens":
            raise RuntimeError("report generation was truncated (hit max_tokens) — retry or shorten the payload")

        tool_use = next(block for block in response.content if block.type == "tool_use")
        try:
            draft = _parse_draft(tool_use.input)
            break
        except ValidationError as e:
            last_error = e
    else:
        raise RuntimeError(f"model did not return a valid report after retry: {last_error}")

    report = ClinicalReport(**draft.model_dump())
    store.write_cached_report(cache_key, model, report)
    return report
