"""Parquet feature store + DuckDB query layer.

One parquet file per record/patient under data/processed/{beats,daily_telemetry}/,
plus single upserted files for the two summary tables. DuckDB queries these
directly (read_parquet) rather than maintaining a separately-loaded database.
"""

import duckdb
import pandas as pd

from pipeline.paths import DATA_PROCESSED
from pipeline.schemas import BeatFeatures, DailyTelemetry, PatientTrendSummary, ProcedureSummary

BEATS_DIR = DATA_PROCESSED / "beats"
DAILY_TELEMETRY_DIR = DATA_PROCESSED / "daily_telemetry"
PROCEDURE_SUMMARIES_PATH = DATA_PROCESSED / "procedure_summaries.parquet"
PATIENT_TREND_SUMMARIES_PATH = DATA_PROCESSED / "patient_trend_summaries.parquet"


def _upsert(path, key_col: str, key_value, new_row: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_parquet(path)
        existing = existing[existing[key_col] != key_value]
        if not existing.empty:
            new_row = pd.concat([existing, new_row], ignore_index=True)
    new_row.to_parquet(path, index=False)


def write_beats(beats: list[BeatFeatures]) -> None:
    if not beats:
        return
    record_id = beats[0].record_id
    BEATS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([b.model_dump() for b in beats])
    df.to_parquet(BEATS_DIR / f"{record_id}.parquet", index=False)


def write_procedure_summary(summary: ProcedureSummary) -> None:
    row = summary.model_dump()
    row["pap_systolic_iqr_low"], row["pap_systolic_iqr_high"] = row.pop("pap_systolic_iqr")
    _upsert(PROCEDURE_SUMMARIES_PATH, "record_id", summary.record_id, pd.DataFrame([row]))


def write_daily_telemetry(records: list[DailyTelemetry]) -> None:
    if not records:
        return
    patient_id = records[0].patient_id
    DAILY_TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([r.model_dump() for r in records])
    df.to_parquet(DAILY_TELEMETRY_DIR / f"{patient_id}.parquet", index=False)


def write_patient_trend_summary(summary: PatientTrendSummary) -> None:
    _upsert(PATIENT_TREND_SUMMARIES_PATH, "patient_id", summary.patient_id, pd.DataFrame([summary.model_dump()]))


def _has_files(directory) -> bool:
    return directory.exists() and any(directory.glob("*.parquet"))


def query(sql: str) -> pd.DataFrame:
    con = duckdb.connect()
    if _has_files(BEATS_DIR):
        con.execute(f"CREATE VIEW beats AS SELECT * FROM read_parquet('{BEATS_DIR / '*.parquet'}')")
    if PROCEDURE_SUMMARIES_PATH.exists():
        con.execute(f"CREATE VIEW procedure_summaries AS SELECT * FROM read_parquet('{PROCEDURE_SUMMARIES_PATH}')")
    if _has_files(DAILY_TELEMETRY_DIR):
        con.execute(f"CREATE VIEW daily_telemetry AS SELECT * FROM read_parquet('{DAILY_TELEMETRY_DIR / '*.parquet'}')")
    if PATIENT_TREND_SUMMARIES_PATH.exists():
        con.execute(
            f"CREATE VIEW patient_trend_summaries AS SELECT * FROM read_parquet('{PATIENT_TREND_SUMMARIES_PATH}')"
        )
    return con.execute(sql).df()
