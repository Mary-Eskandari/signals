import pytest

from backend.app.config import ANTHROPIC_API_KEY
from backend.app.llm_report import build_grounding_payload, generate_report, payload_hash
from backend.app.report_eval import check_grounding
from pipeline.chiron_features import patient_trend_summary
from pipeline.run_pipeline import process_record
from pipeline import fetch_chiron


def test_build_grounding_payload_handles_missing_sides():
    payload = build_grounding_payload(None, None)
    assert payload == {"pa_pressure_ecg_scg_summary": None, "telemonitoring_trend_summary": None}


def test_payload_hash_is_deterministic_and_order_independent():
    a = {"x": 1, "y": 2}
    b = {"y": 2, "x": 1}
    assert payload_hash(a) == payload_hash(b)

    c = {"x": 1, "y": 3}
    assert payload_hash(a) != payload_hash(c)


def test_generate_report_requires_at_least_one_summary():
    with pytest.raises(ValueError):
        generate_report()


def test_generate_report_rejects_unknown_model():
    from pipeline.schemas import ProcedureSummary

    fake_summary = ProcedureSummary(
        record_id="R1", patient_id="P1", n_beats_total=10, n_beats_included=10,
        pap_systolic_median_mmhg=60, pap_systolic_iqr=(55, 65), pap_diastolic_median_mmhg=20,
        pap_mean_median_mmhg=35, hrv_sdnn_ms=50, hrv_rmssd_ms=40,
    )
    with pytest.raises(ValueError):
        generate_report(procedure_summary=fake_summary, model="gpt-4o")


@pytest.mark.skipif(not ANTHROPIC_API_KEY, reason="requires ANTHROPIC_API_KEY for a live call")
def test_real_report_is_grounded_and_cached():
    proc_summary = process_record("TRM278-RHC1")
    df = fetch_chiron.load()
    trend_summary = patient_trend_summary(df, 9)

    report = generate_report(procedure_summary=proc_summary, trend_summary=trend_summary)
    assert report.disclaimer
    assert "not fda-cleared" in report.disclaimer.lower() or "not a clinical tool" in report.disclaimer.lower()

    payload = build_grounding_payload(proc_summary, trend_summary)
    eval_result = check_grounding(report, payload)
    # allow up to 1 false positive from the documented arithmetic-derivation limitation
    # (e.g. "5 excluded" derived from n_beats_total=90 minus n_beats_included=85)
    assert eval_result["n_ungrounded"] <= 1, eval_result["ungrounded_numbers"]

    # second call should hit the cache — verified by not raising even if network were down
    cached_report = generate_report(procedure_summary=proc_summary, trend_summary=trend_summary)
    assert cached_report.summary == report.summary
