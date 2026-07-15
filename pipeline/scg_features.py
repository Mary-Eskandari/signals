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


def extract_scg_beat_features(
    scg_signal: np.ndarray, fs: float, r_peaks: np.ndarray, start_s: float = 0.0
) -> list[dict]:
    """One dict of scg_ao_time_s/scg_ao_amplitude/scg_ac_time_s/scg_ac_amplitude per beat.

    Values are None where no prominent deflection was found in the search window
    (e.g. a noisy beat) — callers should treat that as "unavailable", not zero.
    `start_s`: absolute procedure-time offset of sample 0 — see extract_beats() docstring.
    """
    filtered = filter_scg(scg_signal, fs)
    results = []
    for i in range(len(r_peaks) - 1):
        start, end = r_peaks[i], r_peaks[i + 1]
        beat = filtered[start:end]
        ao = _largest_peak_in_window(beat, fs, AO_WINDOW_S)
        ac = _largest_peak_in_window(beat, fs, AC_WINDOW_S)
        results.append(
            {
                "scg_ao_time_s": start_s + (start + ao[0]) / fs if ao else None,
                "scg_ao_amplitude": ao[1] if ao else None,
                "scg_ac_time_s": start_s + (start + ac[0]) / fs if ac else None,
                "scg_ac_amplitude": ac[1] if ac else None,
            }
        )
    return results
