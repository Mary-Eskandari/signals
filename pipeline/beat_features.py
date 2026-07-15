"""Per-beat PAP feature extraction, anchored to ECG R-peaks, plus HRV.

R-peaks (NeuroKit2) segment the record into cardiac cycles. Within each cycle,
the PAP pulse onset/systolic-peak/diastolic-point are located with simple,
well-scoped heuristics (a "thin adapter layer" over scipy, per the project's
architecture decision to drop the unmaintained vital_sqi dependency) rather
than a full published algorithm. Beats are quality-flagged before aggregation
so artifacts (catheter flush transients, lead noise) never reach the LLM.
"""

import neurokit2 as nk
import numpy as np

from pipeline.schemas import BeatFeatures, ProcedureSummary

ECG_CHANNEL = "ECG_lead_II"
PAP_CHANNEL = "RHC_pressure"

# Physiologically implausible even for severe pulmonary hypertension / artifact-prone
# invasive catheter recordings — used only to flag beats, not to clip/alter values.
PAP_PLAUSIBLE_RANGE_MMHG = (-10.0, 150.0)
MIN_PULSE_PRESSURE_MMHG = 2.0
RR_PLAUSIBLE_RANGE_MS = (300.0, 2000.0)  # 30-200 bpm


def detect_r_peaks(ecg: np.ndarray, fs: float) -> np.ndarray:
    # NeuroKit2's default "neurokit" method badly undercounts peaks on this dataset's
    # Mac-Lab-recorded ECG (verified: found 24 of ~91 true beats on a validation record,
    # cross-checked against autocorrelation-implied HR). Pan-Tompkins matches ground truth.
    cleaned = nk.ecg_clean(ecg, sampling_rate=fs, method="pantompkins1985")
    _, info = nk.ecg_peaks(cleaned, sampling_rate=fs, method="pantompkins1985")
    return np.asarray(info["ECG_R_Peaks"])


def compute_hrv(r_peaks: np.ndarray, fs: float) -> dict[str, float]:
    if len(r_peaks) < 3:
        return {"sdnn_ms": float("nan"), "rmssd_ms": float("nan")}
    hrv = nk.hrv_time(r_peaks, sampling_rate=fs, show=False)
    return {
        "sdnn_ms": float(hrv["HRV_SDNN"].iloc[0]),
        "rmssd_ms": float(hrv["HRV_RMSSD"].iloc[0]),
    }


def _score_beat_quality(systolic: float, diastolic: float, rr_ms: float) -> float:
    score = 1.0
    lo, hi = PAP_PLAUSIBLE_RANGE_MMHG
    if not (lo <= diastolic <= hi) or not (lo <= systolic <= hi):
        score -= 0.6
    if (systolic - diastolic) < MIN_PULSE_PRESSURE_MMHG:
        score -= 0.4
    rr_lo, rr_hi = RR_PLAUSIBLE_RANGE_MS
    if not (rr_lo <= rr_ms <= rr_hi):
        score -= 0.3
    return max(0.0, min(1.0, score))


def _quality_flag(score: float) -> str:
    if score >= 0.8:
        return "good"
    if score >= 0.5:
        return "questionable"
    return "excluded"


def extract_beats(
    signal: np.ndarray,
    channel_names: list[str],
    fs: float,
    record_id: str,
    patient_id: str,
    r_peaks: np.ndarray,
) -> list[BeatFeatures]:
    pap = signal[:, channel_names.index(PAP_CHANNEL)]

    beats = []
    for i in range(len(r_peaks) - 1):
        start, end = r_peaks[i], r_peaks[i + 1]
        beat_pap = pap[start:end]
        if len(beat_pap) < 5:
            continue

        systolic_idx = int(np.argmax(beat_pap))
        systolic = float(beat_pap[systolic_idx])
        # onset: local minimum before the systolic upstroke (isovolumic contraction dip)
        onset_idx = int(np.argmin(beat_pap[: max(systolic_idx, 1)])) if systolic_idx > 0 else 0
        # diastolic: lowest point after the systolic peak / dicrotic notch decay
        diastolic = float(np.min(beat_pap[systolic_idx:])) if systolic_idx < len(beat_pap) - 1 else systolic
        mean_pressure = float(np.mean(beat_pap))
        rr_ms = (end - start) / fs * 1000

        score = _score_beat_quality(systolic, diastolic, rr_ms)
        beats.append(
            BeatFeatures(
                beat_id=f"{record_id}-beat{i:05d}",
                record_id=record_id,
                patient_id=patient_id,
                onset_time_s=(start + onset_idx) / fs,
                pap_systolic_mmhg=systolic,
                pap_diastolic_mmhg=diastolic,
                pap_mean_mmhg=mean_pressure,
                pulse_pressure_mmhg=systolic - diastolic,
                rr_interval_ms=rr_ms,
                sqi_score=score,
                quality_flag=_quality_flag(score),
            )
        )
    return beats


def summarize_procedure(
    beats: list[BeatFeatures],
    hrv: dict[str, float],
    record_id: str,
    patient_id: str,
) -> ProcedureSummary:
    """Median/IQR rollup from 'good'-quality beats only — this is what the LLM narrates from."""
    good = [b for b in beats if b.quality_flag == "good"]
    if not good:
        raise ValueError(f"no good-quality beats for {record_id}; cannot summarize")

    systolic = np.array([b.pap_systolic_mmhg for b in good])
    diastolic = np.array([b.pap_diastolic_mmhg for b in good])
    mean_p = np.array([b.pap_mean_mmhg for b in good])

    return ProcedureSummary(
        record_id=record_id,
        patient_id=patient_id,
        n_beats_total=len(beats),
        n_beats_included=len(good),
        pap_systolic_median_mmhg=float(np.median(systolic)),
        pap_systolic_iqr=(float(np.percentile(systolic, 25)), float(np.percentile(systolic, 75))),
        pap_diastolic_median_mmhg=float(np.median(diastolic)),
        pap_mean_median_mmhg=float(np.median(mean_p)),
        hrv_sdnn_ms=hrv["sdnn_ms"],
        hrv_rmssd_ms=hrv["rmssd_ms"],
    )
