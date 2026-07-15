"""Ingest -> extract -> aggregate -> write, for one or more SCG-RHC records.

Usage: python -m pipeline.run_pipeline TRM278-RHC1 TRM178-RHC1
"""

import argparse

import numpy as np

from pipeline import store
from pipeline.beat_features import compute_hrv, detect_r_peaks, extract_beats, summarize_procedure
from pipeline.fetch_scg_rhc import PA_WINDOW_MAX_S, fetch_pa_window
from pipeline.schemas import ProcedureSummary
from pipeline.scg_features import SCG_CHANNEL


def process_record(record_name: str, duration_s: int = PA_WINDOW_MAX_S, force: bool = False) -> ProcedureSummary:
    npz_path = fetch_pa_window(record_name, duration_s=duration_s, force=force)
    data = np.load(npz_path)
    signal = data["signal"]
    channel_names = list(data["channel_names"])
    fs = float(data["fs"])
    patient_id = record_name.split("-")[0]

    ecg = signal[:, channel_names.index("ECG_lead_II")]
    r_peaks = detect_r_peaks(ecg, fs)
    hrv = compute_hrv(r_peaks, fs)
    beats = extract_beats(signal, channel_names, fs, record_name, patient_id, r_peaks, scg_channel=SCG_CHANNEL)
    summary = summarize_procedure(beats, hrv, record_name, patient_id)

    store.write_beats(beats)
    store.write_procedure_summary(summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", nargs="+", help="Record names, e.g. TRM278-RHC1")
    parser.add_argument("--force", action="store_true", help="Re-fetch and re-process even if cached")
    args = parser.parse_args()

    for record_name in args.records:
        summary = process_record(record_name, force=args.force)
        print(
            f"{record_name}: systolic={summary.pap_systolic_median_mmhg:.1f} "
            f"diastolic={summary.pap_diastolic_median_mmhg:.1f} "
            f"mean={summary.pap_mean_median_mmhg:.1f} "
            f"sdnn={summary.hrv_sdnn_ms:.1f}ms "
            f"beats={summary.n_beats_included}/{summary.n_beats_total} "
            f"scg_ao_ac_interval={summary.scg_ao_ac_interval_ms}"
        )
