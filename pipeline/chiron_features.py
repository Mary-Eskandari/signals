"""Daily telemetry rows + patient-level trend summaries from the Chiron dataset.

The dataset's "Day" column is a study-relative day index, not a calendar date
(no enrollment dates are published) — mapped onto a synthetic date anchored to
an arbitrary epoch purely so it fits the DailyTelemetry.date field; only
relative spacing between days is meaningful, not the actual calendar date.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd

from pipeline.schemas import DailyTelemetry, PatientTrendSummary

EPOCH = date(2000, 1, 1)
WEIGHT_GAIN_LOOKBACK_DAYS = 3
WEIGHT_GAIN_THRESHOLD_KG = 2.0
LOW_SPO2_THRESHOLD_PCT = 90.0
BP_RISING_SLOPE_THRESHOLD_MMHG_PER_DAY = 0.5


def _day_to_date(day: int) -> date:
    return EPOCH + timedelta(days=int(day))


def _row_flags(patient_df: pd.DataFrame, row: pd.Series) -> list[str]:
    flags = []

    if pd.notna(row["Weight"]):
        lookback = patient_df[
            (patient_df["Day"] >= row["Day"] - WEIGHT_GAIN_LOOKBACK_DAYS) & (patient_df["Day"] < row["Day"])
        ]["Weight"].dropna()
        if not lookback.empty and (row["Weight"] - lookback.min()) >= WEIGHT_GAIN_THRESHOLD_KG:
            flags.append("weight_gain_3d_gt_2kg")

    if pd.notna(row["SpO2"]) and row["SpO2"] < LOW_SPO2_THRESHOLD_PCT:
        flags.append("low_spo2")

    return flags


def daily_telemetry(df: pd.DataFrame, patient_id: int) -> list[DailyTelemetry]:
    patient_df = df[df["Patient_ID"] == patient_id].sort_values("Day")
    records = []
    for _, row in patient_df.iterrows():
        if pd.isna(row["Weight"]) or pd.isna(row["SystolicBP"]) or pd.isna(row["DiastolicBP"]):
            continue
        records.append(
            DailyTelemetry(
                patient_id=str(patient_id),
                date=_day_to_date(row["Day"]),
                weight_kg=float(row["Weight"]),
                systolic_bp_mmhg=float(row["SystolicBP"]),
                diastolic_bp_mmhg=float(row["DiastolicBP"]),
                spo2_pct=float(row["SpO2"]) if pd.notna(row["SpO2"]) else float("nan"),
                hr_bpm=float(row["HR_avg"]) if pd.notna(row["HR_avg"]) else float("nan"),
                activity_score=float(row["Energy"]) if pd.notna(row.get("Energy")) else None,
                symptom_score=None,  # not present as a single column in this CSV export
                flags=_row_flags(patient_df, row),
            )
        )
    return records


def _day_weighted_slope(days: np.ndarray, values: np.ndarray) -> float:
    """Linear-regression slope against actual day values (units per day).

    Not using tsfel.slope() here: it regresses against sample *index* (0,1,2,...),
    which silently gives the wrong "per day" rate whenever readings aren't on
    consecutive days — common in this dataset (patients skip days between visits).
    """
    if len(values) < 3:
        return 0.0
    return float(np.polyfit(days, values, 1)[0])


def patient_trend_summary(df: pd.DataFrame, patient_id: int) -> PatientTrendSummary | None:
    records = daily_telemetry(df, patient_id)
    if len(records) < 3:
        return None

    days = np.array([(r.date - EPOCH).days for r in records])
    weights = np.array([r.weight_kg for r in records])
    systolic = np.array([r.systolic_bp_mmhg for r in records])

    weight_slope = _day_weighted_slope(days, weights)
    bp_slope = _day_weighted_slope(days, systolic)

    bp_trend_flags = []
    if bp_slope > BP_RISING_SLOPE_THRESHOLD_MMHG_PER_DAY:
        bp_trend_flags.append("systolic_bp_rising_trend")

    flagged_events = [
        f"{r.date.isoformat()}: {flag}" for r in records for flag in r.flags
    ]

    return PatientTrendSummary(
        patient_id=str(patient_id),
        window_start=records[0].date,
        window_end=records[-1].date,
        weight_slope_kg_per_day=weight_slope,
        bp_trend_flags=bp_trend_flags,
        flagged_events=flagged_events,
    )
