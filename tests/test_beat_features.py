import numpy as np

from pipeline.beat_features import compute_hrv, detect_r_peaks, extract_beats, summarize_procedure
from pipeline.fetch_scg_rhc import fetch_pa_window


def _synthetic_signal(fs: int, n_beats: int, systolic: float, diastolic: float):
    """A fake PAP waveform with known, exact systolic/diastolic per beat and evenly spaced R-peaks."""
    beat_len = fs  # 1s beats -> 60bpm
    r_peaks = np.arange(0, n_beats * beat_len, beat_len)
    pap = np.full(n_beats * beat_len, diastolic, dtype=float)
    for start in r_peaks:
        onset = start + 5
        systolic_idx = start + beat_len // 3
        pap[onset:systolic_idx] = np.linspace(diastolic, systolic, systolic_idx - onset)
        pap[systolic_idx : start + beat_len] = np.linspace(
            systolic, diastolic, start + beat_len - systolic_idx
        )
    n_samples = n_beats * beat_len
    signal = np.zeros((n_samples, 2))
    signal[:, 1] = pap
    channel_names = ["ECG_lead_II", "RHC_pressure"]
    return signal, channel_names, np.array(r_peaks)


def test_extract_beats_recovers_known_systolic_diastolic():
    fs = 500
    signal, channel_names, r_peaks = _synthetic_signal(fs, n_beats=10, systolic=70.0, diastolic=20.0)

    beats = extract_beats(signal, channel_names, fs, "REC1", "PAT1", r_peaks)

    assert len(beats) == 9  # n_beats - 1 intervals
    good = [b for b in beats if b.quality_flag == "good"]
    assert len(good) == len(beats)
    for beat in good:
        assert abs(beat.pap_systolic_mmhg - 70.0) < 1.0
        assert abs(beat.pap_diastolic_mmhg - 20.0) < 1.0
        assert beat.rr_interval_ms == 1000.0


def test_excludes_negative_pulse_pressure_beat():
    fs = 500
    signal, channel_names, r_peaks = _synthetic_signal(fs, n_beats=5, systolic=30.0, diastolic=28.5)

    beats = extract_beats(signal, channel_names, fs, "REC1", "PAT1", r_peaks)

    assert any(b.quality_flag != "good" for b in beats)


def test_summarize_procedure_uses_only_good_beats():
    fs = 500
    signal, channel_names, r_peaks = _synthetic_signal(fs, n_beats=10, systolic=70.0, diastolic=20.0)
    beats = extract_beats(signal, channel_names, fs, "REC1", "PAT1", r_peaks)
    hrv = {"sdnn_ms": 50.0, "rmssd_ms": 40.0}

    summary = summarize_procedure(beats, hrv, "REC1", "PAT1")

    assert summary.n_beats_included == len([b for b in beats if b.quality_flag == "good"])
    assert 65.0 < summary.pap_systolic_median_mmhg < 75.0
    assert 15.0 < summary.pap_diastolic_median_mmhg < 25.0


def test_real_pa_window_extraction_is_physiologically_plausible():
    """End-to-end against real PhysioNet data — verifies the PA-window fetch + full pipeline."""
    npz_path = fetch_pa_window("TRM278-RHC1", duration_s=90)
    data = np.load(npz_path)
    signal = data["signal"]
    channel_names = list(data["channel_names"])
    fs = float(data["fs"])

    ecg = signal[:, channel_names.index("ECG_lead_II")]
    r_peaks = detect_r_peaks(ecg, fs)
    hrv = compute_hrv(r_peaks, fs)
    beats = extract_beats(signal, channel_names, fs, "TRM278-RHC1", "TRM278", r_peaks)
    summary = summarize_procedure(beats, hrv, "TRM278-RHC1", "TRM278")

    # This dataset's implanted heart-rate range; this specific patient has a pacemaker (~60bpm).
    median_rr_ms = float(np.median([b.rr_interval_ms for b in beats]))
    assert 400 < median_rr_ms < 1500

    # This patient has diagnosed pulmonary hypertension — markedly elevated PAP is expected.
    assert 30 < summary.pap_systolic_median_mmhg < 150
    assert 5 < summary.pap_diastolic_median_mmhg < 100
    assert summary.pap_systolic_median_mmhg > summary.pap_diastolic_median_mmhg
    assert summary.n_beats_included > 0
    assert summary.hrv_sdnn_ms > 0
