"""Parquet feature store + DuckDB query layer.

One parquet file per record/patient under data/processed/{beats,daily_telemetry}/,
plus single upserted files for the two summary tables. DuckDB queries these
directly (read_parquet) rather than maintaining a separately-loaded database.
"""

import duckdb
import pandas as pd

from pipeline.paths import DATA_PROCESSED
from pipeline.schemas import BeatFeatures, ClinicalReport, DailyTelemetry, PatientTrendSummary, ProcedureSummary

BEATS_DIR = DATA_PROCESSED / "beats"
DAILY_TELEMETRY_DIR = DATA_PROCESSED / "daily_telemetry"
PROCEDURE_SUMMARIES_PATH = DATA_PROCESSED / "procedure_summaries.parquet"
PATIENT_TREND_SUMMARIES_PATH = DATA_PROCESSED / "patient_trend_summaries.parquet"
LLM_REPORTS_PATH = DATA_PROCESSED / "llm_reports.parquet"


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


def read_procedure_summary(record_id: str) -> ProcedureSummary | None:
    if not PROCEDURE_SUMMARIES_PATH.exists():
        return None
    df = pd.read_parquet(PROCEDURE_SUMMARIES_PATH)
    match = df[df["record_id"] == record_id]
    if match.empty:
        return None
    row = match.iloc[0].to_dict()
    row["pap_systolic_iqr"] = (row.pop("pap_systolic_iqr_low"), row.pop("pap_systolic_iqr_high"))
    return ProcedureSummary(**row)


def read_beats(record_id: str) -> list[BeatFeatures]:
    path = BEATS_DIR / f"{record_id}.parquet"
    if not path.exists():
        return []
    df = pd.read_parquet(path).sort_values("beat_id")
    return [BeatFeatures(**row) for row in df.to_dict(orient="records")]


def read_patient_trend_summary(patient_id: str) -> PatientTrendSummary | None:
    if not PATIENT_TREND_SUMMARIES_PATH.exists():
        return None
    df = pd.read_parquet(PATIENT_TREND_SUMMARIES_PATH)
    match = df[df["patient_id"] == patient_id]
    if match.empty:
        return None
    return PatientTrendSummary(**match.iloc[0].to_dict())


def read_daily_telemetry(patient_id: str) -> list[DailyTelemetry]:
    path = DAILY_TELEMETRY_DIR / f"{patient_id}.parquet"
    if not path.exists():
        return []
    df = pd.read_parquet(path).sort_values("date")
    return [DailyTelemetry(**row) for row in df.to_dict(orient="records")]


def write_cached_report(payload_hash: str, model: str, report: ClinicalReport) -> None:
    cache_key = f"{payload_hash}:{model}"
    row = {"cache_key": cache_key, "payload_hash": payload_hash, "model": model, "report_json": report.model_dump_json()}
    _upsert(LLM_REPORTS_PATH, "cache_key", cache_key, pd.DataFrame([row]))


def read_cached_report(payload_hash: str, model: str) -> ClinicalReport | None:
    if not LLM_REPORTS_PATH.exists():
        return None
    df = pd.read_parquet(LLM_REPORTS_PATH)
    match = df[(df["payload_hash"] == payload_hash) & (df["model"] == model)]
    if match.empty:
        return None
    return ClinicalReport.model_validate_json(match.iloc[0]["report_json"])


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
