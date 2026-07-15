"""Parquet feature store + DuckDB query layer.

One parquet file per record under data/processed/beats/, plus a single
procedure_summaries.parquet. DuckDB queries these directly (read_parquet)
rather than maintaining a separately-loaded database.
"""

import duckdb
import pandas as pd

from pipeline.paths import DATA_PROCESSED
from pipeline.schemas import BeatFeatures, ProcedureSummary

BEATS_DIR = DATA_PROCESSED / "beats"
SUMMARIES_PATH = DATA_PROCESSED / "procedure_summaries.parquet"


def write_beats(beats: list[BeatFeatures]) -> None:
    if not beats:
        return
    record_id = beats[0].record_id
    BEATS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([b.model_dump() for b in beats])
    df.to_parquet(BEATS_DIR / f"{record_id}.parquet", index=False)


def write_procedure_summary(summary: ProcedureSummary) -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    row = summary.model_dump()
    row["pap_systolic_iqr_low"], row["pap_systolic_iqr_high"] = row.pop("pap_systolic_iqr")
    new_row = pd.DataFrame([row])

    if SUMMARIES_PATH.exists():
        existing = pd.read_parquet(SUMMARIES_PATH)
        existing = existing[existing["record_id"] != summary.record_id]
        if not existing.empty:
            new_row = pd.concat([existing, new_row], ignore_index=True)
    new_row.to_parquet(SUMMARIES_PATH, index=False)


def query(sql: str) -> pd.DataFrame:
    con = duckdb.connect()
    beats_glob = str(BEATS_DIR / "*.parquet")
    con.execute(f"CREATE VIEW beats AS SELECT * FROM read_parquet('{beats_glob}')")
    if SUMMARIES_PATH.exists():
        con.execute(f"CREATE VIEW procedure_summaries AS SELECT * FROM read_parquet('{SUMMARIES_PATH}')")
    return con.execute(sql).df()
