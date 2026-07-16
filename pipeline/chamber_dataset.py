"""Offline dataset builder for the chamber-position classification task.

For each record x chamber (RA/RV/PA/PCW), fetches a short window, extracts the
same per-beat engineered features the main pipeline computes (pipeline.beat_features),
plus a fixed-length raw waveform snippet per beat for the CNN path, and labels
every beat with its chamber. Network-bound (332 record x chamber fetches) and
resumable — fetch_window()'s own caching skips already-fetched windows, and this
builder checkpoints its output periodically so an interrupted run doesn't lose
everything.

Usage: python -m pipeline.chamber_dataset [--records ID ...] [--force] [--checkpoint-every N]
"""

import argparse

import numpy as np
import pandas as pd

from pipeline.beat_features import detect_r_peaks, extract_beats
from pipeline.fetch_scg_rhc import CHAMBER_ORDER, fetch_chamber_window, fetch_records_list
from pipeline.paths import DATA_PROCESSED
from pipeline.scg_features import SCG_CHANNEL

RAW_CHANNELS = ["RHC_pressure", "ECG_lead_II", SCG_CHANNEL]
SNIPPET_PRE_S = 0.1
SNIPPET_POST_S = 0.9
DEFAULT_WINDOW_S = 20.0

BEATS_PATH = DATA_PROCESSED / "chamber_beats.parquet"
SNIPPETS_PATH = DATA_PROCESSED / "chamber_raw_snippets.npy"

ENGINEERED_COLUMNS = [
    "pap_systolic_mmhg",
    "pap_diastolic_mmhg",
    "pap_mean_mmhg",
    "pulse_pressure_mmhg",
    "rr_interval_ms",
    "sqi_score",
    "quality_flag",
    "scg_ao_amplitude",
    "scg_ac_amplitude",
    "scg_detection_confidence",
    "dicrotic_notch_time_s",
    "dicrotic_notch_pressure_mmhg",
    "upstroke_slope_mmhg_s",
    "beat_auc_mmhg_s",
    "beat_skewness",
    "beat_kurtosis",
]


def _raw_snippet(signal: np.ndarray, channel_indices: list[int], r_peak_idx: int, fs: float) -> np.ndarray:
    """Fixed-length (channels, samples) snippet around an R-peak, zero-padded at
    the edges of the fetched window rather than dropping edge beats entirely."""
    pre_n = int(SNIPPET_PRE_S * fs)
    post_n = int(SNIPPET_POST_S * fs)
    lo, hi = r_peak_idx - pre_n, r_peak_idx + post_n
    snippet = np.zeros((len(channel_indices), pre_n + post_n), dtype=np.float32)
    src_lo, src_hi = max(0, lo), min(len(signal), hi)
    dst_lo = src_lo - lo
    snippet[:, dst_lo : dst_lo + (src_hi - src_lo)] = signal[src_lo:src_hi, :][:, channel_indices].T
    return snippet


def process_record_chamber(record_name: str, chamber: str, duration_s: float = DEFAULT_WINDOW_S):
    """Returns (rows, snippets) for this record/chamber, or (None, None) on failure
    (e.g. a too-short window, missing channels, or wfdb reading past the record's end —
    the latter is retried with a shrinking duration before giving up)."""
    npz_path = None
    for attempt_duration in (duration_s, duration_s / 2, duration_s / 4):
        try:
            npz_path = fetch_chamber_window(record_name, chamber, duration_s=attempt_duration)
            break
        except Exception:
            continue
    if npz_path is None:
        return None, None

    data = np.load(npz_path)
    signal = data["signal"]
    channel_names = list(data["channel_names"])
    fs = float(data["fs"])
    start_s = float(data["start_s"])

    if not all(c in channel_names for c in RAW_CHANNELS):
        return None, None

    ecg = signal[:, channel_names.index("ECG_lead_II")]
    r_peaks = detect_r_peaks(ecg, fs)
    if len(r_peaks) < 2:
        return None, None

    patient_id = record_name.split("-")[0]
    beats = extract_beats(
        signal, channel_names, fs, record_name, patient_id, r_peaks,
        scg_channel=SCG_CHANNEL, start_s=start_s,
    )
    if not beats:
        return None, None

    channel_indices = [channel_names.index(c) for c in RAW_CHANNELS]
    snippets = np.stack([_raw_snippet(signal, channel_indices, int(r_peaks[i]), fs) for i in range(len(beats))])

    rows = [
        {"record_id": record_name, "chamber": chamber, "beat_id": b.beat_id, **{c: getattr(b, c) for c in ENGINEERED_COLUMNS}}
        for b in beats
    ]
    return rows, snippets


def _load_existing():
    if BEATS_PATH.exists() and SNIPPETS_PATH.exists():
        return pd.read_parquet(BEATS_PATH), np.load(SNIPPETS_PATH)
    return pd.DataFrame(columns=["record_id", "chamber", "beat_id", *ENGINEERED_COLUMNS]), np.zeros((0, 3, 0), dtype=np.float32)


def _checkpoint(existing_df, existing_snippets, new_rows, new_snippets):
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(new_rows)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True) if not existing_df.empty else new_df
    combined_snippets = (
        np.concatenate([existing_snippets, *new_snippets], axis=0) if existing_snippets.size else np.concatenate(new_snippets, axis=0)
    )
    combined_df.to_parquet(BEATS_PATH, index=False)
    np.save(SNIPPETS_PATH, combined_snippets)
    return combined_df, combined_snippets


def build_dataset(record_names: list[str], force: bool = False, checkpoint_every: int = 10) -> dict:
    existing_df, existing_snippets = (
        (pd.DataFrame(columns=["record_id", "chamber", "beat_id", *ENGINEERED_COLUMNS]), np.zeros((0, 3, 0), dtype=np.float32))
        if force
        else _load_existing()
    )
    done_pairs = set(zip(existing_df["record_id"], existing_df["chamber"])) if not existing_df.empty else set()

    new_rows, new_snippets = [], []
    n_pairs_since_checkpoint = 0
    failures = []
    pairs = [(r, c) for r in record_names for c in CHAMBER_ORDER]

    for i, (record_name, chamber) in enumerate(pairs, 1):
        if (record_name, chamber) in done_pairs:
            continue
        rows, snippets = process_record_chamber(record_name, chamber)
        if rows is None:
            failures.append((record_name, chamber))
            print(f"[{i}/{len(pairs)}] {record_name} {chamber}: FAILED")
        else:
            new_rows.extend(rows)
            new_snippets.append(snippets)
            print(f"[{i}/{len(pairs)}] {record_name} {chamber}: {len(rows)} beats")
        n_pairs_since_checkpoint += 1

        if n_pairs_since_checkpoint >= checkpoint_every and new_rows:
            existing_df, existing_snippets = _checkpoint(existing_df, existing_snippets, new_rows, new_snippets)
            done_pairs |= {(r["record_id"], r["chamber"]) for r in new_rows}
            new_rows, new_snippets = [], []
            n_pairs_since_checkpoint = 0
            print(f"  checkpointed: {len(existing_df)} beats total")

    if new_rows:
        existing_df, existing_snippets = _checkpoint(existing_df, existing_snippets, new_rows, new_snippets)

    return {"n_beats": len(existing_df), "n_failures": len(failures), "failures": failures}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", nargs="*", help="Specific record names; defaults to all 83")
    parser.add_argument("--force", action="store_true", help="Rebuild from scratch, ignoring cached progress")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    args = parser.parse_args()

    records = args.records or fetch_records_list()
    result = build_dataset(records, force=args.force, checkpoint_every=args.checkpoint_every)
    print(f"\nDone: {result['n_beats']} beats, {result['n_failures']} failed record/chamber pairs")
    if result["failures"]:
        print("Failures:", result["failures"])
