"""Per-beat PAP feature extraction, anchored to ECG R-peaks, plus HRV.

R-peaks (NeuroKit2) segment the record into cardiac cycles. Within each cycle,
the PAP pulse onset is located with the Zong et al. 2003 slope-sum-function
algorithm, the dicrotic notch with a second-derivative fiducial-point method,
and systolic/diastolic extrema with simple, well-scoped argmax/argmin (a "thin
adapter layer" over scipy, per the project's architecture decision to drop the
unmaintained vital_sqi dependency) — see the per-function docstrings below for
citations. Beats are quality-flagged before aggregation so artifacts (catheter
flush transients, lead noise) never reach the LLM.
"""

import neurokit2 as nk
import numpy as np
from scipy import stats as scipy_stats
from scipy.signal import find_peaks

from pipeline.schemas import BeatFeatures, ProcedureSummary
from pipeline.scg_features import extract_scg_beat_features

ECG_CHANNEL = "ECG_lead_II"
PAP_CHANNEL = "RHC_pressure"

# Physiologically implausible even for severe pulmonary hypertension / artifact-prone
# invasive catheter recordings — used only to flag beats, not to clip/alter values.
PAP_PLAUSIBLE_RANGE_MMHG = (-10.0, 150.0)
MIN_PULSE_PRESSURE_MMHG = 2.0
RR_PLAUSIBLE_RANGE_MS = (300.0, 2000.0)  # 30-200 bpm

# Zong, Heldt, Moody & Mark, "An open-source algorithm to detect onset of arterial
# blood pressure pulses", Computers in Cardiology 2003 — windowed slope-sum-function
# (SSF) onset detector. 128ms window and a threshold at a fraction of the beat's peak
# SSF value both follow the paper's reference parameters.
SSF_WINDOW_MS = 128
SSF_THRESHOLD_FRACTION = 0.5

# NeuroKit2's frequency-domain / nonlinear HRV metrics need a longer, steadier RR
# series than SDNN/RMSSD to avoid raising or silently returning NaN/inf (verified
# empirically: nk.hrv_frequency raises below ~5 beats and returns NaN below ~50 beats
# on this dataset's beat-to-beat variability; nk.hrv_nonlinear's SampEn is unstable
# below ~15-20 beats). These thresholds gate the *attempt*; a finite-value check below
# is the final safety net regardless of beat count.
MIN_BEATS_HRV_FREQUENCY = 50
MIN_BEATS_HRV_NONLINEAR = 15


def detect_r_peaks(ecg: np.ndarray, fs: float) -> np.ndarray:
    # NeuroKit2's default "neurokit" method badly undercounts peaks on this dataset's
    # Mac-Lab-recorded ECG (verified: found 24 of ~91 true beats on a validation record,
    # cross-checked against autocorrelation-implied HR). Pan-Tompkins matches ground truth.
    cleaned = nk.ecg_clean(ecg, sampling_rate=fs, method="pantompkins1985")
    _, info = nk.ecg_peaks(cleaned, sampling_rate=fs, method="pantompkins1985")
    return np.asarray(info["ECG_R_Peaks"])


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def compute_hrv(r_peaks: np.ndarray, fs: float) -> dict[str, float | None]:
    result: dict[str, float | None] = {
        "sdnn_ms": float("nan"),
        "rmssd_ms": float("nan"),
        "lf_hf_ratio": None,
        "sd1_ms": None,
        "sd2_ms": None,
        "sample_entropy": None,
    }
    if len(r_peaks) < 3:
        return result

    hrv_time = nk.hrv_time(r_peaks, sampling_rate=fs, show=False)
    result["sdnn_ms"] = float(hrv_time["HRV_SDNN"].iloc[0])
    result["rmssd_ms"] = float(hrv_time["HRV_RMSSD"].iloc[0])

    # LF/HF power ratio (nk.hrv_frequency, Welch PSD over the 0.04-0.15Hz / 0.15-0.4Hz
    # bands) — standard autonomic-balance metric per the Task Force of the European
    # Society of Cardiology / North American Society of Pacing and Electrophysiology
    # (1996) HRV standards. Guarded: too few/irregular beats make the spline
    # interpolation nk.hrv_frequency relies on raise or degenerate.
    if len(r_peaks) >= MIN_BEATS_HRV_FREQUENCY:
        try:
            hrv_freq = nk.hrv_frequency(r_peaks, sampling_rate=fs, show=False)
            result["lf_hf_ratio"] = _finite_or_none(hrv_freq["HRV_LFHF"].iloc[0])
        except Exception:
            pass

    # Poincaré SD1/SD2 (short-/long-term beat-to-beat variability) and sample entropy
    # (regularity/complexity) via nk.hrv_nonlinear — see Brennan et al. 2001 (Poincaré
    # geometry of HRV) and Richman & Moorman 2000 (sample entropy). Guarded the same way.
    if len(r_peaks) >= MIN_BEATS_HRV_NONLINEAR:
        try:
            hrv_nonlinear = nk.hrv_nonlinear(r_peaks, sampling_rate=fs, show=False)
            result["sd1_ms"] = _finite_or_none(hrv_nonlinear["HRV_SD1"].iloc[0])
            result["sd2_ms"] = _finite_or_none(hrv_nonlinear["HRV_SD2"].iloc[0])
            result["sample_entropy"] = _finite_or_none(hrv_nonlinear["HRV_SampEn"].iloc[0])
        except Exception:
            pass

    return result


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


def _slope_sum_function(x: np.ndarray, window_n: int) -> np.ndarray:
    """Zong et al. 2003 slope sum function: a trailing windowed sum of the
    positive-going first derivative, which sharpens the systolic upstroke and
    flattens noise/diastolic decay ahead of threshold-based onset detection."""
    dP = np.diff(x, prepend=x[0])
    dP_pos = np.clip(dP, 0, None)
    kernel = np.ones(window_n)
    return np.convolve(dP_pos, kernel, mode="full")[: len(x)]


def _ssf_onset_idx(beat_pap: np.ndarray, fs: float, systolic_idx: int) -> int:
    """Zong, Heldt, Moody & Mark (2003), "An open-source algorithm to detect onset of
    arterial blood pressure pulses", Computers in Cardiology — applied within each
    R-peak-segmented beat rather than as a free-running detector, since beat
    boundaries are already known here. Threshold-crosses the slope sum function on the
    systolic upstroke, then backtracks to the local pressure minimum immediately
    preceding the crossing (the paper's own refinement step, since the SSF threshold
    crossing itself lags true onset by roughly the window length).
    Falls back to the previous local-minimum heuristic when the upstroke is too flat/
    noisy for the SSF to cross threshold (e.g. a damped or clipped beat)."""
    if systolic_idx < 2:
        return 0
    window_n = max(1, int(SSF_WINDOW_MS / 1000 * fs))
    upstroke = beat_pap[: systolic_idx + 1]
    ssf = _slope_sum_function(upstroke, window_n)
    peak_ssf = float(np.max(ssf))
    if peak_ssf <= 0:
        return int(np.argmin(upstroke))
    threshold = SSF_THRESHOLD_FRACTION * peak_ssf
    crossings = np.flatnonzero(ssf >= threshold)
    if len(crossings) == 0:
        return int(np.argmin(upstroke))
    cross_idx = int(crossings[0])
    search_start = max(0, cross_idx - window_n)
    return search_start + int(np.argmin(upstroke[search_start : cross_idx + 1]))


def _dicrotic_notch_idx(beat_pap: np.ndarray, systolic_idx: int, diastolic_idx: int) -> int | None:
    """Dicrotic notch: the local-minimum-then-rise inflection on the diastolic runoff
    that marks pulmonic/aortic valve closure, equivalent to the first negative-to-
    positive second-derivative sign change after the systolic peak — a standard
    fiducial-point technique in pulse-contour-analysis / PPG waveform literature
    (Millasseau et al. 2002, Hypertension; Elgendi 2012 review of second-derivative
    waveform landmark detection). Implemented via scipy's local-minima finder (a
    numerically stabler equivalent of scanning raw second-derivative sign changes on
    noisy catheter data) restricted to the systole-to-diastole window, so the first
    sufficiently prominent minimum found is the notch, not the final diastolic trough.
    Returns None when no prominent notch is found (e.g. an overdamped catheter trace)."""
    if diastolic_idx <= systolic_idx + 2:
        return None
    tail = beat_pap[systolic_idx : diastolic_idx + 1]
    pulse_pressure = beat_pap[systolic_idx] - np.min(tail)
    if pulse_pressure <= 0:
        return None
    # mmHg floor rejects sub-noise-floor ripples; fraction scales with pulse pressure
    # so the same relative notch depth is detected across mild and severe hypertension.
    min_prominence = max(0.05 * pulse_pressure, 0.3)
    minima_idx, _ = find_peaks(-tail, prominence=min_prominence)
    if len(minima_idx) == 0:
        return None
    return systolic_idx + int(minima_idx[0])


def _upstroke_slope(beat_pap: np.ndarray, fs: float, onset_idx: int, systolic_idx: int) -> float | None:
    """Maximum dP/dt during the systolic upstroke — a standard pulse-wave-analysis
    morphological feature reflecting contractility / ventricular-arterial coupling
    (see e.g. Cannesson et al. 2008, Anesthesiology, on arterial pulse contour
    analysis; general pulse-wave-analysis reviews, e.g. Chowienczyk & Cockcroft)."""
    if systolic_idx <= onset_idx:
        return None
    segment = beat_pap[onset_idx : systolic_idx + 1]
    if len(segment) < 2:
        return None
    return float(np.max(np.diff(segment) * fs))


def _beat_shape_stats(beat_pap: np.ndarray, fs: float) -> dict[str, float]:
    """Beat-integral (area under the pressure curve, a proxy for stroke work/cardiac
    workload) and distribution skewness/kurtosis of the intra-beat pressure samples —
    standard morphological descriptors used alongside fiducial-point timing in
    pulse-wave-analysis / arterial-waveform-contour literature (see e.g. the PWA
    reviews cited in _upstroke_slope)."""
    return {
        "auc": float(np.trapezoid(beat_pap, dx=1.0 / fs)),
        "skewness": float(scipy_stats.skew(beat_pap)),
        "kurtosis": float(scipy_stats.kurtosis(beat_pap)),  # excess kurtosis, Fisher convention
    }


def extract_beats(
    signal: np.ndarray,
    channel_names: list[str],
    fs: float,
    record_id: str,
    patient_id: str,
    r_peaks: np.ndarray,
    scg_channel: str | None = None,
    start_s: float = 0.0,
) -> list[BeatFeatures]:
    """`start_s`: absolute procedure-time offset of this signal array's sample 0 —
    must match the /waveform endpoint's convention (fetch_window's start_s) so
    beat-time fields (onset_time_s, scg_ao/ac_time_s) align with waveform.time_s
    for frontend annotation. Without this, times default to array-relative (0-based),
    which silently never overlaps a waveform window that doesn't start at t=0."""
    pap = signal[:, channel_names.index(PAP_CHANNEL)]

    scg_per_beat = None
    if scg_channel and scg_channel in channel_names:
        scg = signal[:, channel_names.index(scg_channel)]
        scg_per_beat = extract_scg_beat_features(scg, fs, r_peaks, start_s=start_s)

    beats = []
    for i in range(len(r_peaks) - 1):
        start, end = r_peaks[i], r_peaks[i + 1]
        beat_pap = pap[start:end]
        if len(beat_pap) < 5:
            continue

        systolic_idx = int(np.argmax(beat_pap))
        systolic = float(beat_pap[systolic_idx])
        # onset: Zong et al. 2003 slope-sum-function threshold crossing, backtracked
        # to the preceding local pressure minimum (see _ssf_onset_idx)
        onset_idx = _ssf_onset_idx(beat_pap, fs, systolic_idx)
        # diastolic: lowest point after the systolic peak / dicrotic notch decay
        if systolic_idx < len(beat_pap) - 1:
            diastolic_idx = systolic_idx + int(np.argmin(beat_pap[systolic_idx:]))
            diastolic = float(beat_pap[diastolic_idx])
        else:
            diastolic_idx = systolic_idx
            diastolic = systolic
        mean_pressure = float(np.mean(beat_pap))
        rr_ms = (end - start) / fs * 1000

        notch_idx = _dicrotic_notch_idx(beat_pap, systolic_idx, diastolic_idx)
        upstroke_slope = _upstroke_slope(beat_pap, fs, onset_idx, systolic_idx)
        shape = _beat_shape_stats(beat_pap, fs)

        score = _score_beat_quality(systolic, diastolic, rr_ms)
        scg = scg_per_beat[i] if scg_per_beat else {}
        beats.append(
            BeatFeatures(
                beat_id=f"{record_id}-beat{i:05d}",
                record_id=record_id,
                patient_id=patient_id,
                onset_time_s=start_s + (start + onset_idx) / fs,
                pap_systolic_mmhg=systolic,
                pap_diastolic_mmhg=diastolic,
                pap_mean_mmhg=mean_pressure,
                pulse_pressure_mmhg=systolic - diastolic,
                rr_interval_ms=rr_ms,
                sqi_score=score,
                quality_flag=_quality_flag(score),
                scg_ao_time_s=scg.get("scg_ao_time_s"),
                scg_ao_amplitude=scg.get("scg_ao_amplitude"),
                scg_ac_time_s=scg.get("scg_ac_time_s"),
                scg_ac_amplitude=scg.get("scg_ac_amplitude"),
                scg_detection_confidence=scg.get("scg_detection_confidence"),
                dicrotic_notch_time_s=(start_s + (start + notch_idx) / fs) if notch_idx is not None else None,
                dicrotic_notch_pressure_mmhg=float(beat_pap[notch_idx]) if notch_idx is not None else None,
                upstroke_slope_mmhg_s=upstroke_slope,
                beat_auc_mmhg_s=shape["auc"],
                beat_skewness=shape["skewness"],
                beat_kurtosis=shape["kurtosis"],
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

    # SCG quality is independent of PAP quality (different signal, different artifacts) —
    # aggregate over all beats with a successful AO/AC detection, not just PAP-"good" ones.
    scg_beats = [b for b in beats if b.scg_ao_amplitude is not None and b.scg_ac_amplitude is not None]
    scg_ao_mean = float(np.mean([b.scg_ao_amplitude for b in scg_beats])) if scg_beats else None
    scg_ac_mean = float(np.mean([b.scg_ac_amplitude for b in scg_beats])) if scg_beats else None
    scg_interval_mean = (
        float(np.mean([(b.scg_ac_time_s - b.scg_ao_time_s) * 1000 for b in scg_beats])) if scg_beats else None
    )

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
        hrv_lf_hf_ratio=hrv.get("lf_hf_ratio"),
        hrv_sd1_ms=hrv.get("sd1_ms"),
        hrv_sd2_ms=hrv.get("sd2_ms"),
        hrv_sample_entropy=hrv.get("sample_entropy"),
        scg_ao_amplitude_mean=scg_ao_mean,
        scg_ac_amplitude_mean=scg_ac_mean,
        scg_ao_ac_interval_ms=scg_interval_mean,
    )
