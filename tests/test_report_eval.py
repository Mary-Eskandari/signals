from backend.app.report_eval import _extract_numbers, check_grounding
from pipeline.schemas import ClinicalReport, ReportSection


def test_dates_and_record_ids_are_not_treated_as_numbers():
    text = "For record TRM278-RHC1 on 2000-06-07, beat00042 showed systolic 66.6 mmHg."
    numbers = _extract_numbers(text)
    assert numbers == {66.6}


def test_range_notation_is_not_misparsed_as_a_negative_number():
    """Regression test: 'IQR 64.20-68.56 mmHg' is a range (two positive endpoints), not
    64.20 and -68.56 — the hyphen here is a separator glued to the preceding number, not
    a minus sign. A real negative reading (e.g. '-45.0 mmHg' preceded by whitespace)
    must still parse as negative."""
    numbers = _extract_numbers("IQR 64.20-68.56 mmHg, flush artifact -45.0 mmHg")
    assert numbers == {64.20, 68.56, -45.0}


def test_check_grounding_flags_a_fabricated_value():
    report = ClinicalReport(
        summary="Systolic pressure was 999.9 mmHg.",
        pa_pressure_findings=ReportSection(title="PA", findings=["Systolic: 999.9 mmHg"]),
        rhythm_hrv_findings=ReportSection(title="HRV", findings=["SDNN: 50 ms"]),
    )
    payload = {"pa_pressure_ecg_scg_summary": {"pap_systolic_median_mmhg": 66.6, "hrv_sdnn_ms": 50}}

    result = check_grounding(report, payload)

    assert result["n_ungrounded"] == 1
    assert 999.9 in result["ungrounded_numbers"]


def test_check_grounding_accepts_values_within_tolerance():
    report = ClinicalReport(
        summary="Systolic pressure was approximately 66.5 mmHg.",
        pa_pressure_findings=ReportSection(title="PA", findings=["Systolic: 66.5 mmHg"]),
        rhythm_hrv_findings=ReportSection(title="HRV", findings=["no data"]),
    )
    payload = {"pa_pressure_ecg_scg_summary": {"pap_systolic_median_mmhg": 66.575996}}

    result = check_grounding(report, payload)

    assert result["n_ungrounded"] == 0
