import numpy as np

from pipeline.beat_features import compute_hrv, detect_r_peaks, extract_beats, summarize_procedure
from pipeline.fetch_scg_rhc import fetch_pa_window
from pipeline.scg_features import SCG_CHANNEL


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


def _synthetic_beat_with_isovolumic_dip(fs: int, n_beats: int, diastolic: float, dip: float, systolic: float):
    """A PAP waveform with a distinct pre-systolic dip (isovolumic contraction) at a
    known sample offset — the shape the Zong et al. 2003 slope-sum-function onset
    detector is meant to key on, as opposed to a flat/noisy baseline where a naive
    argmin has no real minimum to find."""
    beat_len = fs
    r_peaks = np.arange(0, n_beats * beat_len, beat_len)
    pap = np.full(n_beats * beat_len, diastolic, dtype=float)
    true_onset_offset = 200  # samples after R-peak
    dip_width = 15
    for start in r_peaks:
        dip_start = start + true_onset_offset - dip_width
        onset = start + true_onset_offset
        systolic_idx = onset + 75
        beat_end = start + beat_len
        pap[start:dip_start] = diastolic
        pap[dip_start:onset] = np.linspace(diastolic, dip, onset - dip_start)
        pap[onset:systolic_idx] = np.linspace(dip, systolic, systolic_idx - onset)
        pap[systolic_idx:beat_end] = np.linspace(systolic, diastolic, beat_end - systolic_idx)
    signal = np.zeros((n_beats * beat_len, 2))
    signal[:, 1] = pap
    channel_names = ["ECG_lead_II", "RHC_pressure"]
    return signal, channel_names, np.array(r_peaks), true_onset_offset


def test_ssf_onset_recovers_known_pre_systolic_dip():
    """Zong et al. 2003 slope-sum-function onset detector: given a beat with a
    physiologically realistic isovolumic-contraction dip immediately before the
    systolic upstroke, onset_time_s should land within a few samples of the true dip,
    not at an arbitrary point in the flat pre-dip baseline."""
    fs = 500
    signal, channel_names, r_peaks, true_onset_offset = _synthetic_beat_with_isovolumic_dip(
        fs, n_beats=6, diastolic=20.0, dip=17.0, systolic=70.0
    )

    beats = extract_beats(signal, channel_names, fs, "REC1", "PAT1", r_peaks)

    for i, beat in enumerate(beats):
        true_onset_sample = r_peaks[i] + true_onset_offset
        recovered_sample = beat.onset_time_s * fs
        assert abs(recovered_sample - true_onset_sample) <= 5  # within 10ms at 500Hz


def _synthetic_beat_with_notch(fs: int, n_beats: int, diastolic: float, systolic: float, notch_pressure: float):
    """A PAP waveform with a clear dicrotic notch (decline, brief dip, small dicrotic-
    wave rise, further decline) at a known sample offset and known pressure."""
    beat_len = fs
    r_peaks = np.arange(0, n_beats * beat_len, beat_len)
    pap = np.full(n_beats * beat_len, diastolic, dtype=float)
    onset_offset, upstroke_len = 40, 60
    notch_offset_after_peak, dicrotic_wave_len = 80, 20
    for start in r_peaks:
        onset = start + onset_offset
        systolic_idx = onset + upstroke_len
        notch_idx = systolic_idx + notch_offset_after_peak
        wave_peak_idx = notch_idx + dicrotic_wave_len
        beat_end = start + beat_len
        pap[start:onset] = diastolic
        pap[onset:systolic_idx] = np.linspace(diastolic, systolic, systolic_idx - onset)
        pap[systolic_idx:notch_idx] = np.linspace(systolic, notch_pressure, notch_idx - systolic_idx)
        pap[notch_idx:wave_peak_idx] = np.linspace(notch_pressure, notch_pressure + 3, wave_peak_idx - notch_idx)
        pap[wave_peak_idx:beat_end] = np.linspace(notch_pressure + 3, diastolic, beat_end - wave_peak_idx)
    signal = np.zeros((n_beats * beat_len, 2))
    signal[:, 1] = pap
    channel_names = ["ECG_lead_II", "RHC_pressure"]
    true_notch_offset = onset_offset + upstroke_len + notch_offset_after_peak
    return signal, channel_names, np.array(r_peaks), true_notch_offset


def test_dicrotic_notch_recovers_known_notch():
    fs = 500
    signal, channel_names, r_peaks, true_notch_offset = _synthetic_beat_with_notch(
        fs, n_beats=6, diastolic=20.0, systolic=70.0, notch_pressure=45.0
    )

    beats = extract_beats(signal, channel_names, fs, "REC1", "PAT1", r_peaks)

    for i, beat in enumerate(beats):
        assert beat.dicrotic_notch_time_s is not None
        true_notch_sample = r_peaks[i] + true_notch_offset
        recovered_sample = beat.dicrotic_notch_time_s * fs
        assert abs(recovered_sample - true_notch_sample) <= 3
        assert abs(beat.dicrotic_notch_pressure_mmhg - 45.0) < 0.5


def test_dicrotic_notch_absent_on_smooth_monotonic_decay():
    """An overdamped beat with no notch (pure exponential-like decay from systole to
    diastole) should not report a spurious notch."""
    fs = 500
    n_beats = 4
    beat_len = fs
    r_peaks = np.arange(0, n_beats * beat_len, beat_len)
    diastolic, systolic = 20.0, 70.0
    pap = np.full(n_beats * beat_len, diastolic, dtype=float)
    for start in r_peaks:
        onset = start + 40
        systolic_idx = onset + 60
        beat_end = start + beat_len
        pap[onset:systolic_idx] = np.linspace(diastolic, systolic, systolic_idx - onset)
        decay = np.linspace(0, 5, beat_end - systolic_idx)
        pap[systolic_idx:beat_end] = diastolic + (systolic - diastolic) * np.exp(-decay)
    signal = np.zeros((n_beats * beat_len, 2))
    signal[:, 1] = pap
    channel_names = ["ECG_lead_II", "RHC_pressure"]

    beats = extract_beats(signal, channel_names, fs, "REC1", "PAT1", np.array(r_peaks))

    assert all(b.dicrotic_notch_time_s is None for b in beats)


def test_upstroke_slope_and_beat_shape_stats_match_known_waveform():
    """Deterministic check of the PWA morphological features against a triangular
    beat with an analytically known upstroke slope and AUC."""
    fs = 500
    signal, channel_names, r_peaks = _synthetic_signal(fs, n_beats=6, systolic=70.0, diastolic=20.0)

    beats = extract_beats(signal, channel_names, fs, "REC1", "PAT1", r_peaks)

    beat_len = fs
    onset_offset, systolic_offset = 5, beat_len // 3
    expected_slope = (70.0 - 20.0) / ((systolic_offset - onset_offset) / fs)
    for beat in beats:
        assert beat.upstroke_slope_mmhg_s is not None
        assert abs(beat.upstroke_slope_mmhg_s - expected_slope) / expected_slope < 0.05
        assert beat.beat_auc_mmhg_s > 0
        assert np.isfinite(beat.beat_skewness)
        assert np.isfinite(beat.beat_kurtosis)


def test_compute_hrv_extended_metrics_guarded_for_short_windows():
    """nk.hrv_frequency/nk.hrv_nonlinear can raise or return non-finite values on short
    RR-interval series; compute_hrv must degrade to None rather than propagate that."""
    fs = 500
    rng = np.random.default_rng(0)

    r_peaks_tiny = np.cumsum(rng.normal(0.85, 0.05, 3) * fs).astype(int)
    hrv_tiny = compute_hrv(r_peaks_tiny, fs)
    assert hrv_tiny["lf_hf_ratio"] is None
    assert hrv_tiny["sd1_ms"] is None
    assert hrv_tiny["sd2_ms"] is None
    assert hrv_tiny["sample_entropy"] is None

    r_peaks_short = np.cumsum(rng.normal(0.85, 0.05, 10) * fs).astype(int)
    hrv_short = compute_hrv(r_peaks_short, fs)
    # below MIN_BEATS_HRV_FREQUENCY / MIN_BEATS_HRV_NONLINEAR — still no exception, still gated to None
    assert hrv_short["lf_hf_ratio"] is None
    assert hrv_short["sd1_ms"] is None

    r_peaks_long = np.cumsum(rng.normal(0.85, 0.05, 80) * fs).astype(int)
    hrv_long = compute_hrv(r_peaks_long, fs)
    assert hrv_long["lf_hf_ratio"] is None or np.isfinite(hrv_long["lf_hf_ratio"])
    assert hrv_long["sd1_ms"] is None or np.isfinite(hrv_long["sd1_ms"])
    # with enough beats and real beat-to-beat variability, SD1/SD2 should actually be available
    assert hrv_long["sd1_ms"] is not None
    assert hrv_long["sd2_ms"] is not None


def test_real_pa_window_extraction_is_physiologically_plausible():
    """End-to-end against real PhysioNet data — verifies the PA-window fetch + full pipeline."""
    npz_path = fetch_pa_window("TRM278-RHC1", duration_s=90)
    data = np.load(npz_path)
    signal = data["signal"]
    channel_names = list(data["channel_names"])
    fs = float(data["fs"])
    start_s = float(data["start_s"])

    ecg = signal[:, channel_names.index("ECG_lead_II")]
    r_peaks = detect_r_peaks(ecg, fs)
    hrv = compute_hrv(r_peaks, fs)
    beats = extract_beats(
        signal, channel_names, fs, "TRM278-RHC1", "TRM278", r_peaks, scg_channel=SCG_CHANNEL, start_s=start_s
    )
    summary = summarize_procedure(beats, hrv, "TRM278-RHC1", "TRM278")

    # Regression check: onset_time_s must be absolute (matching the /waveform endpoint's
    # time_s convention, which also adds start_s), not relative to the fetched array —
    # a real bug where beat times defaulted to 0-based and never overlapped the actual
    # waveform's absolute time range, silently breaking every frontend annotation.
    assert all(start_s <= b.onset_time_s <= start_s + 90 for b in beats)

    # This dataset's implanted heart-rate range; this specific patient has a pacemaker (~60bpm).
    median_rr_ms = float(np.median([b.rr_interval_ms for b in beats]))
    assert 400 < median_rr_ms < 1500

    # This patient has diagnosed pulmonary hypertension — markedly elevated PAP is expected.
    assert 30 < summary.pap_systolic_median_mmhg < 150
    assert 5 < summary.pap_diastolic_median_mmhg < 100
    assert summary.pap_systolic_median_mmhg > summary.pap_diastolic_median_mmhg
    assert summary.n_beats_included > 0
    assert summary.hrv_sdnn_ms > 0

    # AO/AC interval approximates left-ventricular ejection time; literature range roughly
    # 150-450ms depending on HR. Heuristic detection, not clinically validated — wide bounds.
    assert summary.scg_ao_ac_interval_ms is not None
    assert 100 < summary.scg_ao_ac_interval_ms < 450

    # Extended HRV: with 91 R-peaks over 90s this comfortably clears MIN_BEATS_HRV_FREQUENCY/
    # MIN_BEATS_HRV_NONLINEAR, so the frequency-domain and nonlinear metrics should be populated
    # (not silently skipped) and physiologically finite, not just "doesn't crash".
    assert summary.hrv_lf_hf_ratio is not None and summary.hrv_lf_hf_ratio > 0
    assert summary.hrv_sd1_ms is not None and summary.hrv_sd1_ms > 0
    assert summary.hrv_sd2_ms is not None and summary.hrv_sd2_ms > 0

    # Dicrotic notch: a real (if fluid-filled-catheter-damped) PAP trace should show a
    # detectable notch on at least some beats, but not on every beat (damping/noise are
    # expected on invasive catheter data) — a non-trivial, non-universal detection rate.
    good = [b for b in beats if b.quality_flag == "good"]
    n_with_notch = sum(1 for b in good if b.dicrotic_notch_time_s is not None)
    assert 0 < n_with_notch < len(good)
    for b in good:
        if b.dicrotic_notch_pressure_mmhg is not None:
            assert b.pap_diastolic_mmhg <= b.dicrotic_notch_pressure_mmhg <= b.pap_systolic_mmhg

    # Upstroke slope: positive, and in a plausible order of magnitude for a 500Hz PAP trace.
    slopes = [b.upstroke_slope_mmhg_s for b in good if b.upstroke_slope_mmhg_s is not None]
    assert slopes
    assert all(0 < s < 5000 for s in slopes)

    # Beat AUC (mean pressure * beat duration, roughly) should be positive and in the
    # right ballpark given the observed mean PAP and ~1s beats.
    aucs = [b.beat_auc_mmhg_s for b in good]
    assert all(0 < a < 200 for a in aucs)

    # SCG detection confidence, where computable, must be a valid [0, 1] agreement score.
    confidences = [b.scg_detection_confidence for b in beats if b.scg_detection_confidence is not None]
    assert confidences
    assert all(0.0 <= c <= 1.0 for c in confidences)
