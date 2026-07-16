"""Cache the PhysioNet SCG-RHC Wearable + Right Heart Catheter DB locally.

Dataset: https://physionet.org/content/scg-rhc-wearable-database/1.0.0/
83 RHC procedures, each a 500Hz WFDB record with ECG, seismocardiogram-proxy
accelerometer channels, and the invasive RHC_pressure channel, spanning up to
an hour+ per procedure (a full record's .dat file can run 100-200MB).

Rather than downloading full-length raw files, this fetches a bounded time
window per record via wfdb's remote range-read (fast: a 30s/17-channel window
takes ~10s vs. minutes+ for a full file) and caches it as a compact .npz.
Later pipeline stages should re-fetch a different/longer window on demand
using the same fetch_record() call if more context is needed.
"""

import argparse
import json

import numpy as np
import requests
import wfdb

from pipeline.paths import SCG_RHC_RAW_DIR

PN_DIR = "scg-rhc-wearable-database/1.0.0"
SAMPLING_RATE_HZ = 500  # confirmed from record headers; consistent across the dataset
DEFAULT_WINDOW_S = 120
PA_WINDOW_MAX_S = 90  # cap fetch time; the true PA-placement segment can run several minutes

META_FILES = (
    "meta_information/acronyms_RHC.csv",
    "meta_information/HO_names_RHC.csv",
    "meta_information/clinic_names_RHC.csv",
    "meta_information/PAM_PCWP_timestamp_in_TBME.json",
    "meta_information/RHC_values.csv",
)


def _download(url: str, dest_path, force: bool) -> None:
    if dest_path.exists() and not force:
        return
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    dest_path.write_bytes(response.content)


def fetch_records_list(force: bool = False) -> list[str]:
    dest = SCG_RHC_RAW_DIR / "RECORDS"
    _download(f"https://physionet.org/files/{PN_DIR}/RECORDS", dest, force)
    return [line.split("/")[-1] for line in dest.read_text().splitlines() if line.strip()]


def fetch_meta(force: bool = False) -> None:
    for filename in META_FILES:
        _download(f"https://physionet.org/files/{PN_DIR}/{filename}", SCG_RHC_RAW_DIR / filename, force)


def fetch_record_json(record_name: str, force: bool = False) -> None:
    filename = f"processed_data/{record_name}.json"
    _download(f"https://physionet.org/files/{PN_DIR}/{filename}", SCG_RHC_RAW_DIR / filename, force)


def get_chamber_events(record_name: str, force: bool = False) -> dict[str, float]:
    """Catheter chamber-entry timestamps (seconds from procedure start): RA, RV, PA, PCW."""
    fetch_record_json(record_name, force)
    meta = json.loads((SCG_RHC_RAW_DIR / "processed_data" / f"{record_name}.json").read_text())
    return meta["ChamEvents_in_s"]


def fetch_window(record_name: str, start_s: float, duration_s: int, tag: str, force: bool = False):
    # Cache key must encode the actual window bounds, not just a caller-chosen tag —
    # otherwise a request for a shorter/different window silently returns a stale
    # cached file from an earlier, differently-sized request under the same tag.
    dest = SCG_RHC_RAW_DIR / "processed_data" / f"{record_name}__{tag}_{int(start_s)}_{int(duration_s)}.npz"
    if dest.exists() and not force:
        return dest

    record = wfdb.rdrecord(
        record_name,
        pn_dir=f"{PN_DIR}/processed_data",
        sampfrom=int(start_s * SAMPLING_RATE_HZ),
        sampto=int((start_s + duration_s) * SAMPLING_RATE_HZ),
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dest,
        signal=record.p_signal.astype(np.float32),
        channel_names=np.array(record.sig_name),
        fs=record.fs,
        record_name=record_name,
        start_s=start_s,
    )
    return dest


def fetch_record(record_name: str, window_s: int = DEFAULT_WINDOW_S, force: bool = False):
    """Fetch the first `window_s` seconds of a record (tag='default')."""
    dest = fetch_window(record_name, start_s=0, duration_s=window_s, tag="default", force=force)
    fetch_record_json(record_name, force)
    return dest


CHAMBER_ORDER = ["RA", "RV", "PA", "PCW"]


def fetch_chamber_window(
    record_name: str,
    chamber: str,
    duration_s: float = 20.0,
    pad_before_s: float = 2.0,
    force: bool = False,
):
    """Fetch the segment where the catheter is in the given chamber (RA/RV/PA/PCW).

    Uses each record's ChamEvents_in_s metadata (catheter chamber-entry timestamps)
    rather than the start of the file. Bounded chambers (RA/RV/PA) are clipped to
    the next chamber's timestamp if that arrives sooner than `duration_s`; PCW has
    no next event so it keeps the requested fixed duration — callers should retry
    with a shorter duration if wfdb errors reading past the record's actual end.
    """
    events = get_chamber_events(record_name, force)
    if chamber not in events:
        raise ValueError(f"{record_name} has no {chamber!r} timestamp in ChamEvents_in_s: {events}")
    start_s = max(0.0, events[chamber] - pad_before_s)

    idx = CHAMBER_ORDER.index(chamber)
    if idx + 1 < len(CHAMBER_ORDER):
        next_chamber = CHAMBER_ORDER[idx + 1]
        if next_chamber in events:
            duration_s = max(5.0, min(duration_s, events[next_chamber] - start_s))

    return fetch_window(record_name, start_s, duration_s, tag=f"chamber_{chamber}", force=force)


def fetch_pa_window(record_name: str, duration_s: int = PA_WINDOW_MAX_S, pad_before_s: int = 5, force: bool = False):
    """Fetch the segment where the catheter is actually in the pulmonary artery.

    Thin wrapper over fetch_chamber_window with the historical "pa" cache tag and
    defaults, kept separate since the rest of the pipeline (run_pipeline.py etc.)
    depends on this exact tag/signature.
    """
    events = get_chamber_events(record_name, force)
    start_s = max(0.0, events["PA"] - pad_before_s)
    end_s = events.get("PCW", start_s + duration_s)
    duration_s = max(10.0, min(duration_s, end_s - start_s))
    return fetch_window(record_name, start_s, duration_s, tag="pa", force=force)


def fetch_all(window_s: int = DEFAULT_WINDOW_S, force: bool = False) -> list[str]:
    records = fetch_records_list(force)
    fetch_meta(force)
    for i, record_name in enumerate(records, 1):
        fetch_record(record_name, window_s, force)
        print(f"[{i}/{len(records)}] cached {record_name}")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", help="Fetch a single record (e.g. TRM278-RHC1)")
    parser.add_argument("--window-s", type=int, default=DEFAULT_WINDOW_S, help="Seconds to cache per record")
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    args = parser.parse_args()

    if args.record:
        dest = fetch_record(args.record, args.window_s, args.force)
        print(f"cached {args.record} ({args.window_s}s window) -> {dest}")
    else:
        records = fetch_all(args.window_s, args.force)
        print(f"cached {len(records)} records -> {SCG_RHC_RAW_DIR}")
