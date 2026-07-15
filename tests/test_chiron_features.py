import pandas as pd

from pipeline import fetch_chiron
from pipeline.chiron_features import _day_weighted_slope, daily_telemetry, patient_trend_summary


def _synthetic_df(rows):
    """rows: list of (day, weight, sbp, dbp, spo2)"""
    return pd.DataFrame(
        [
            {
                "Patient_ID": 1,
                "Day": day,
                "Weight": weight,
                "SystolicBP": sbp,
                "DiastolicBP": dbp,
                "SpO2": spo2,
                "HR_avg": 70,
                "Energy": None,
            }
            for day, weight, sbp, dbp, spo2 in rows
        ]
    )


def test_weight_gain_flag_triggers_within_lookback():
    df = _synthetic_df(
        [
            (100, 70.0, 120, 80, 96),
            (101, 70.2, 120, 80, 96),
            (102, 72.5, 120, 80, 96),  # +2.5kg vs day 100, within 3-day lookback
        ]
    )
    records = daily_telemetry(df, 1)
    assert records[2].flags == ["weight_gain_3d_gt_2kg"]
    assert records[0].flags == []
    assert records[1].flags == []


def test_weight_gain_flag_does_not_trigger_outside_lookback():
    df = _synthetic_df(
        [
            (100, 70.0, 120, 80, 96),
            (105, 72.5, 120, 80, 96),  # +2.5kg but 5 days later, outside 3-day window
        ]
    )
    records = daily_telemetry(df, 1)
    assert records[1].flags == []


def test_low_spo2_flag():
    df = _synthetic_df([(100, 70.0, 120, 80, 85.0)])
    records = daily_telemetry(df, 1)
    assert "low_spo2" in records[0].flags


def test_day_weighted_slope_handles_gaps_correctly():
    import numpy as np

    # +10 units over 5 real days = slope 2.0/day, even though only 2 samples (irregular gap)
    days = np.array([0, 5])
    values = np.array([0.0, 10.0])
    slope = _day_weighted_slope(days, values)
    assert slope == 0.0  # fewer than 3 points -> defined as 0.0, not misleadingly extrapolated

    days = np.array([0, 5, 10])
    values = np.array([0.0, 10.0, 20.0])
    slope = _day_weighted_slope(days, values)
    assert abs(slope - 2.0) < 1e-6


def test_real_chiron_trend_summary_is_plausible():
    df = fetch_chiron.load()
    summary = patient_trend_summary(df, 9)

    assert summary is not None
    assert summary.window_end > summary.window_start
    # a stable outpatient's weight shouldn't drift more than ~1kg/day on average
    assert -1.0 < summary.weight_slope_kg_per_day < 1.0
    assert len(summary.flagged_events) > 0  # this patient has a known documented gain event
