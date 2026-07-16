import numpy as np

from pipeline.scg_features import AC_WINDOW_S, AO_WINDOW_S, extract_scg_beat_features


def test_recovers_known_ao_ac_peaks():
    fs = 500
    n_beats = 5
    beat_len = fs  # 1s beats
    r_peaks = np.arange(0, n_beats * beat_len, beat_len)

    signal = np.random.default_rng(0).normal(0, 0.1, n_beats * beat_len)
    ao_time_s, ao_amp = 0.08, -10.0
    ac_time_s, ac_amp = 0.30, 6.0
    for start in r_peaks:
        signal[start + int(ao_time_s * fs)] = ao_amp
        signal[start + int(ac_time_s * fs)] = ac_amp

    results = extract_scg_beat_features(signal, fs, r_peaks)

    assert len(results) == n_beats - 1
    for i, beat in enumerate(results):
        expected_ao_time = r_peaks[i] / fs + ao_time_s
        expected_ac_time = r_peaks[i] / fs + ac_time_s
        assert abs(beat["scg_ao_time_s"] - expected_ao_time) < 0.01
        assert abs(beat["scg_ac_time_s"] - expected_ac_time) < 0.01
        # bandpass filtering attenuates a single-sample impulse, so check sign + that it
        # stands out from the ~0.1-amplitude noise floor rather than exact equality
        assert beat["scg_ao_amplitude"] < -0.5
        assert beat["scg_ac_amplitude"] > 0.5


def test_search_windows_are_ordered_and_non_overlapping():
    assert AO_WINDOW_S[1] <= AC_WINDOW_S[0]
    assert AO_WINDOW_S[0] < AO_WINDOW_S[1]
    assert AC_WINDOW_S[0] < AC_WINDOW_S[1]


def test_ensemble_average_confidence_flags_the_outlier_beat():
    """Ensemble/coherent-averaging cross-check (Zanetti & Salerno 1991; Di Rienzo et al.
    2013): beats whose AO timing matches the R-peak-aligned ensemble-average template
    should score near-1.0 confidence; a beat whose AO fires well outside the tolerance
    window relative to every other beat should score visibly lower."""
    fs = 500
    n_beats = 6
    beat_len = fs
    r_peaks = np.arange(0, n_beats * beat_len, beat_len)

    signal = np.random.default_rng(0).normal(0, 0.1, n_beats * beat_len)
    ao_time_s, ao_amp = 0.08, -10.0
    ac_time_s, ac_amp = 0.30, 6.0
    outlier_beat_idx = n_beats - 2  # last beat with a full [start, end) window
    for i, start in enumerate(r_peaks):
        ao_offset = ao_time_s + (0.10 if i == outlier_beat_idx else 0.0)
        signal[start + int(ao_offset * fs)] = ao_amp
        signal[start + int(ac_time_s * fs)] = ac_amp

    results = extract_scg_beat_features(signal, fs, r_peaks)

    consistent_confidences = [
        r["scg_detection_confidence"] for i, r in enumerate(results) if i != outlier_beat_idx
    ]
    assert all(c is not None and c >= 0.9 for c in consistent_confidences)

    outlier_confidence = results[outlier_beat_idx]["scg_detection_confidence"]
    assert outlier_confidence is not None
    assert outlier_confidence < min(consistent_confidences)
