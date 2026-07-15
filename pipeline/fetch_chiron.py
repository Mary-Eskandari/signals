"""Cache the Chiron CHF telemonitoring dataset (Mlakar et al., PLoS ONE 2018) locally.

S1 Dataset: daily weight, BP, SpO2, wearable-derived HR, activity, and symptom
score for 24 CHF patients over 1,086 patient-days.
https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0190323
"""

import argparse

import pandas as pd
import requests

from pipeline.paths import CHIRON_RAW_DIR

DATASET_URL = "https://doi.org/10.1371/journal.pone.0190323.s017"
DEST_FILENAME = "chiron_telemonitoring.csv"


def fetch(force: bool = False):
    dest_path = CHIRON_RAW_DIR / DEST_FILENAME
    if dest_path.exists() and not force:
        return dest_path
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(DATASET_URL, timeout=60)
    response.raise_for_status()
    dest_path.write_bytes(response.content)
    return dest_path


def load(force: bool = False) -> pd.DataFrame:
    """Parse the cached CSV. Semicolon-delimited with '?' as the missing-value marker."""
    path = fetch(force)
    return pd.read_csv(path, sep=";", na_values=["?"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    args = parser.parse_args()

    path = fetch(args.force)
    print(f"cached Chiron telemonitoring dataset -> {path}")
