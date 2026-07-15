"""SCG-RHC catheterization records: listing, feature summaries, beats, waveform windows."""

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from pipeline import store
from pipeline.fetch_scg_rhc import fetch_pa_window, fetch_records_list, get_chamber_events
from pipeline.run_pipeline import process_record
from pipeline.schemas import BeatFeatures, ProcedureSummary

router = APIRouter(prefix="/records", tags=["records"])


@router.get("")
def list_records() -> list[dict]:
    records = fetch_records_list()
    return [{"record_id": r, "patient_id": r.split("-")[0]} for r in records]


@router.get("/{record_id}/summary", response_model=ProcedureSummary)
def get_record_summary(record_id: str, force: bool = False) -> ProcedureSummary:
    if not force:
        cached = store.read_procedure_summary(record_id)
        if cached is not None:
            return cached

    try:
        return process_record(record_id, force=force)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"could not process {record_id}: {e}") from e


@router.get("/{record_id}/beats", response_model=list[BeatFeatures])
def get_record_beats(record_id: str) -> list[BeatFeatures]:
    beats = store.read_beats(record_id)
    if not beats:
        raise HTTPException(status_code=404, detail=f"no beats found for {record_id}; call /summary first")
    return beats


@router.get("/{record_id}/waveform")
def get_record_waveform(
    record_id: str,
    channels: str = Query("RHC_pressure,ECG_lead_II,patch_ACC_dv", description="comma-separated channel names"),
    max_points: int = Query(2000, le=10000, description="downsample cap per channel"),
) -> dict:
    """Windowed waveform for charting — never ships the full raw record."""
    npz_path = fetch_pa_window(record_id)
    data = np.load(npz_path)
    signal = data["signal"]
    channel_names = list(data["channel_names"])
    fs = float(data["fs"])
    start_s = float(data["start_s"])

    requested = [c.strip() for c in channels.split(",")]
    unknown = [c for c in requested if c not in channel_names]
    if unknown:
        raise HTTPException(status_code=422, detail=f"unknown channels: {unknown}")

    stride = max(1, len(signal) // max_points)
    downsampled = signal[::stride]
    time_s = start_s + np.arange(0, len(signal), stride) / fs

    return {
        "record_id": record_id,
        "fs_effective_hz": fs / stride,
        "time_s": time_s.tolist(),
        "channels": {c: downsampled[:, channel_names.index(c)].tolist() for c in requested},
    }


@router.get("/{record_id}/chamber_events")
def get_chamber_event_timestamps(record_id: str) -> dict:
    return get_chamber_events(record_id)
