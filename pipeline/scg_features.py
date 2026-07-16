"""Custom seismocardiogram (SCG) AO/AC extraction.

No mature Python library exists for this — verified during project research
(the closest tools, e.g. MSCardio/SeismoNet, are dataset-specific ML model
code, not reusable toolboxes; the general SCG toolbox PulsatioMech is
MATLAB-only). Built on NeuroKit2's filtering primitives + scipy peak-detection,
anchored to ECG R-peaks, within literature-informed timing windows — this is
the project's highest-risk, most differentiating module.

Channel choice (patch_ACC_dv, the dorsoventral/Z axis) was validated
empirically on this dataset by R-peak-aligned epoch averaging across all
three accelerometer axes: DV showed by far the largest averaged-epoch
amplitude, consistent with published SCG literature that the axis
perpendicular to the chest wall carries the strongest cardiac-mechanical
signal (Zanetti & Salerno 1991; Crow et al. 1994; Taebi et al. reviews).
"""

import neurokit2 as nk
import numpy as np

SCG_CHANNEL = "patch_ACC_dv"
BANDPASS_LOW_HZ = 1
BANDPASS_HIGH_HZ = 40

# Literature-informed post-R-peak search windows: AO ~ isovolumic contraction /
# early systole, AC ~ end-systole (coincides roughly with the second heart sound).
AO_WINDOW_S = (0.02, 0.16)
AC_WINDOW_S = (0.20, 0.42)


def filter_scg(signal: np.ndarray, fs: float) -> np.ndarray:
    return nk.signal_filter(
        signal, sampling_rate=fs, lowcut=BANDPASS_LOW_HZ, highcut=BANDPASS_HIGH_HZ,
        method="butterworth", order=4,
    )


def _largest_peak_in_window(beat_scg: np.ndarray, fs: float, window_s: tuple[float, float]):
    lo = int(window_s[0] * fs)
    hi = min(int(window_s[1] * fs), len(beat_scg))
    if hi <= lo:
        return None
    segment = beat_scg[lo:hi]
    idx = int(np.argmax(np.abs(segment)))
    return lo + idx, float(segment[idx])


# Individual-beat AO/AC timing within this many seconds of the ensemble-average
# template's timing scores 1.0 confidence; agreement decays linearly to 0.0 beyond it.
# ~40ms is a generous fraction of the AO/AC search windows above, wide enough to tolerate
# normal beat-to-beat physiological jitter while still down-weighting real outliers.
CONFIDENCE_TOLERANCE_S = 0.04


def _ensemble_average_template(filtered: np.ndarray, r_peaks: np.ndarray, fs: float) -> np.ndarray | None:
    """R-peak-aligned coherent/ensemble averaging: stack every beat's SCG segment on a
    common R-peak-relative time axis and average sample-wise. Averaging cancels
    beat-to-beat noise (respiration, motion, sensor coupling) while cardiac-mechanical
    events (AO/AC) that recur at a consistent latency reinforce — the classical SCG
    ensemble-averaging technique used to build a clean reference waveform (Zanetti &
    Salerno 1991; Crow et al. 1994; see also Di Rienzo et al. 2013 on SCG ensemble
    averaging for fiducial-point reliability)."""
    template_len = int(AC_WINDOW_S[1] * fs)
    if len(r_peaks) < 3 or template_len < 2:
        return None
    segments = []
    for i in range(len(r_peaks) - 1):
        start = r_peaks[i]
        end = min(start + template_len, len(filtered))
        if end - start < template_len:
            continue
        segments.append(filtered[start:end])
    if len(segments) < 2:
        return None
    return np.mean(np.stack(segments), axis=0)


def _timing_confidence(individual_time_s: float | None, template_time_s: float | None) -> float | None:
    """Score one beat's AO/AC timing against the ensemble-average template's timing —
    a real-vs-noisy-detection cross-check used in SCG literature to flag individual
    beats whose fiducial points drift from the coherent-average reference (see
    _ensemble_average_template citation) rather than discarding low-SNR beats outright."""
    if individual_time_s is None or template_time_s is None:
        return None
    delta = abs(individual_time_s - template_time_s)
    return float(max(0.0, 1.0 - delta / CONFIDENCE_TOLERANCE_S))


def extract_scg_beat_features(
    scg_signal: np.ndarray, fs: float, r_peaks: np.ndarray, start_s: float = 0.0
) -> list[dict]:
    """One dict of scg_ao_time_s/scg_ao_amplitude/scg_ac_time_s/scg_ac_amplitude/
    scg_detection_confidence per beat.

    Values are None where no prominent deflection was found in the search window
    (e.g. a noisy beat) — callers should treat that as "unavailable", not zero.
    `start_s`: absolute procedure-time offset of sample 0 — see extract_beats() docstring.
    """
    filtered = filter_scg(scg_signal, fs)

    template = _ensemble_average_template(filtered, r_peaks, fs)
    template_ao = _largest_peak_in_window(template, fs, AO_WINDOW_S) if template is not None else None
    template_ac = _largest_peak_in_window(template, fs, AC_WINDOW_S) if template is not None else None
    template_ao_time_s = template_ao[0] / fs if template_ao else None
    template_ac_time_s = template_ac[0] / fs if template_ac else None

    results = []
    for i in range(len(r_peaks) - 1):
        start, end = r_peaks[i], r_peaks[i + 1]
        beat = filtered[start:end]
        ao = _largest_peak_in_window(beat, fs, AO_WINDOW_S)
        ac = _largest_peak_in_window(beat, fs, AC_WINDOW_S)

        ao_confidence = _timing_confidence(ao[0] / fs if ao else None, template_ao_time_s)
        ac_confidence = _timing_confidence(ac[0] / fs if ac else None, template_ac_time_s)
        confidences = [c for c in (ao_confidence, ac_confidence) if c is not None]
        detection_confidence = float(np.mean(confidences)) if confidences else None

        results.append(
            {
                "scg_ao_time_s": start_s + (start + ao[0]) / fs if ao else None,
                "scg_ao_amplitude": ao[1] if ao else None,
                "scg_ac_time_s": start_s + (start + ac[0]) / fs if ac else None,
                "scg_ac_amplitude": ac[1] if ac else None,
                "scg_detection_confidence": detection_confidence,
            }
        )
    return results
