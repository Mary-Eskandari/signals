"""Automated slice of the grounding rubric: every numeric claim in a generated
report should trace to either the grounding payload or the system prompt's
static reference ranges (e.g. "normal PA systolic ~15-30 mmHg" is legitimate
framing, not a hallucinated patient value). This is a smoke-test complement
to manual review, not a replacement for it — text-based number extraction is
inherently approximate (units, rounding, ranges).
"""

import re

from backend.app.llm_report import SYSTEM_PROMPT
from pipeline.schemas import ClinicalReport

_NUMBER_RE = re.compile(r"-?\d+\.\d+|-?\d+")
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")  # ISO dates aren't "numeric claims" to ground
_ID_RE = re.compile(r"\b[A-Za-z]+\d+[A-Za-z0-9-]*\b")  # record/beat IDs, e.g. TRM278-RHC1


def _extract_numbers(text: str) -> set[float]:
    text = _DATE_RE.sub(" ", text)
    text = _ID_RE.sub(" ", text)
    return {float(m) for m in _NUMBER_RE.findall(text)}


def _payload_numbers(obj) -> set[float]:
    """Numeric values in the payload — including numeric-looking IDs (e.g. patient_id="9"),
    which are stored as strings but are still legitimately "in the payload", not invented."""
    numbers = set()
    if isinstance(obj, dict):
        for v in obj.values():
            numbers |= _payload_numbers(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            numbers |= _payload_numbers(v)
    elif isinstance(obj, (int, float)):
        numbers.add(round(float(obj), 2))
    elif isinstance(obj, str) and _NUMBER_RE.fullmatch(obj):
        numbers.add(round(float(obj), 2))
    return numbers


def check_grounding(report: ClinicalReport, payload: dict, tolerance: float = 0.5) -> dict:
    allowed = _payload_numbers(payload) | _extract_numbers(SYSTEM_PROMPT)

    sections = [report.pa_pressure_findings, report.rhythm_hrv_findings, report.scg_findings, report.trend_findings]
    text = report.summary + " " + " ".join(f for s in sections if s for f in s.findings)
    found = _extract_numbers(text)

    ungrounded = sorted(n for n in found if not any(abs(n - a) <= tolerance for a in allowed))
    return {
        "n_numbers_checked": len(found),
        "n_ungrounded": len(ungrounded),
        "ungrounded_numbers": ungrounded,
        "has_disclaimer": bool(report.disclaimer),
    }
