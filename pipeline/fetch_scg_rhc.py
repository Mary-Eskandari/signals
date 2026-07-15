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

import numpy as np
import requests
import wfdb

from pipeline.paths import SCG_RHC_RAW_DIR

PN_DIR = "scg-rhc-wearable-database/1.0.0"
SAMPLING_RATE_HZ = 500  # confirmed from record headers; consistent across the dataset
DEFAULT_WINDOW_S = 120

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


def fetch_record(record_name: str, window_s: int = DEFAULT_WINDOW_S, force: bool = False):
    dest = SCG_RHC_RAW_DIR / "processed_data" / f"{record_name}.npz"
    if dest.exists() and not force:
        return dest

    record = wfdb.rdrecord(
        record_name,
        pn_dir=f"{PN_DIR}/processed_data",
        sampto=window_s * SAMPLING_RATE_HZ,
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dest,
        signal=record.p_signal.astype(np.float32),
        channel_names=np.array(record.sig_name),
        fs=record.fs,
        record_name=record_name,
    )
    fetch_record_json(record_name, force)
    return dest


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
