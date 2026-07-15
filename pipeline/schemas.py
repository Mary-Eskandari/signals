"""Shared Pydantic contract: pipeline output schema, API response models, and LLM grounding payload."""

from datetime import date
from typing import Literal

from pydantic import BaseModel


class BeatFeatures(BaseModel):
    beat_id: str
    record_id: str
    patient_id: str
    onset_time_s: float
    pap_systolic_mmhg: float
    pap_diastolic_mmhg: float
    pap_mean_mmhg: float
    pulse_pressure_mmhg: float
    rr_interval_ms: float
    sqi_score: float
    quality_flag: Literal["good", "questionable", "excluded"]
    scg_ao_time_s: float | None = None
    scg_ao_amplitude: float | None = None
    scg_ac_time_s: float | None = None
    scg_ac_amplitude: float | None = None


class ProcedureSummary(BaseModel):
    record_id: str
    patient_id: str
    n_beats_total: int
    n_beats_included: int
    pap_systolic_median_mmhg: float
    pap_systolic_iqr: tuple[float, float]
    pap_diastolic_median_mmhg: float
    pap_mean_median_mmhg: float
    hrv_sdnn_ms: float
    hrv_rmssd_ms: float
    scg_ao_amplitude_mean: float | None = None
    scg_ac_amplitude_mean: float | None = None
    scg_ao_ac_interval_ms: float | None = None


class DailyTelemetry(BaseModel):
    patient_id: str
    date: date
    weight_kg: float
    systolic_bp_mmhg: float
    diastolic_bp_mmhg: float
    spo2_pct: float
    hr_bpm: float
    activity_score: float | None = None
    symptom_score: float | None = None
    flags: list[str] = []


class PatientTrendSummary(BaseModel):
    patient_id: str
    window_start: date
    window_end: date
    weight_slope_kg_per_day: float
    bp_trend_flags: list[str] = []
    flagged_events: list[str] = []


class ReportSection(BaseModel):
    title: str
    findings: list[str]
    unavailable: list[str] = []


class ClinicalReport(BaseModel):
    summary: str
    pa_pressure_findings: ReportSection
    rhythm_hrv_findings: ReportSection
    scg_findings: ReportSection | None = None
    trend_findings: ReportSection | None = None
    flags: list[str] = []
    disclaimer: str = (
        "Demonstration only, generated from public de-identified research data "
        "(PhysioNet SCG-RHC / Chiron CHF telemonitoring). Not a clinical tool, "
        "not FDA-cleared, not for real patient use."
    )
