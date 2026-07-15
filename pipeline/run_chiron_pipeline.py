"""Ingest -> extract -> aggregate -> write, for one or more Chiron patients.

Usage: python -m pipeline.run_chiron_pipeline 9 10 2
"""

import argparse

from pipeline import chiron_features, fetch_chiron, store
from pipeline.schemas import PatientTrendSummary


def process_patient(df, patient_id: int) -> PatientTrendSummary | None:
    records = chiron_features.daily_telemetry(df, patient_id)
    if not records:
        return None
    store.write_daily_telemetry(records)

    summary = chiron_features.patient_trend_summary(df, patient_id)
    if summary:
        store.write_patient_trend_summary(summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("patient_ids", nargs="+", type=int)
    args = parser.parse_args()

    df = fetch_chiron.load()
    for patient_id in args.patient_ids:
        summary = process_patient(df, patient_id)
        if summary is None:
            print(f"patient {patient_id}: insufficient data")
            continue
        print(
            f"patient {patient_id}: weight_slope={summary.weight_slope_kg_per_day:.3f}kg/day "
            f"bp_flags={summary.bp_trend_flags} n_flagged_events={len(summary.flagged_events)}"
        )
