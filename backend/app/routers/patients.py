"""Chiron telemonitoring patients: listing, daily telemetry, trend summaries."""

from fastapi import APIRouter, HTTPException

from pipeline import fetch_chiron, store
from pipeline.chiron_features import daily_telemetry as compute_daily_telemetry
from pipeline.run_chiron_pipeline import process_patient
from pipeline.schemas import DailyTelemetry, PatientTrendSummary

router = APIRouter(prefix="/patients", tags=["patients"])


@router.get("")
def list_patients() -> list[dict]:
    df = fetch_chiron.load()
    counts = df.groupby("Patient_ID")["Day"].agg(["count", "min", "max"])
    return [
        {"patient_id": str(pid), "n_days": int(row["count"]), "day_start": int(row["min"]), "day_end": int(row["max"])}
        for pid, row in counts.iterrows()
    ]


@router.get("/{patient_id}/trend", response_model=PatientTrendSummary)
def get_patient_trend(patient_id: int, force: bool = False) -> PatientTrendSummary:
    if not force:
        cached = store.read_patient_trend_summary(str(patient_id))
        if cached is not None:
            return cached

    df = fetch_chiron.load()
    summary = process_patient(df, patient_id)
    if summary is None:
        raise HTTPException(status_code=422, detail=f"insufficient data for patient {patient_id}")
    return summary


@router.get("/{patient_id}/daily", response_model=list[DailyTelemetry])
def get_patient_daily_telemetry(patient_id: int) -> list[DailyTelemetry]:
    records = store.read_daily_telemetry(str(patient_id))
    if not records:
        df = fetch_chiron.load()
        records = compute_daily_telemetry(df, patient_id)
        if not records:
            raise HTTPException(status_code=404, detail=f"no telemetry for patient {patient_id}")
    return records
