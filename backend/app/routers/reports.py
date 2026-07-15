"""LLM clinical report generation, tying together a record and/or patient."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.app.config import ALLOWED_REPORT_MODELS, REPORT_MODEL
from backend.app.llm_report import generate_report
from backend.app.routers.patients import ensure_patient_trend
from backend.app.routers.records import ensure_record_processed
from pipeline.schemas import ClinicalReport

router = APIRouter(prefix="/reports", tags=["reports"])


class GenerateReportRequest(BaseModel):
    record_id: str | None = None
    patient_id: str | None = None
    model: str = REPORT_MODEL


@router.get("/models")
def list_models() -> dict:
    return {"allowed_models": ALLOWED_REPORT_MODELS, "default": REPORT_MODEL}


@router.post("/generate", response_model=ClinicalReport)
def generate(req: GenerateReportRequest) -> ClinicalReport:
    if not req.record_id and not req.patient_id:
        raise HTTPException(status_code=422, detail="need at least one of record_id/patient_id")

    procedure_summary = ensure_record_processed(req.record_id) if req.record_id else None
    trend_summary = ensure_patient_trend(int(req.patient_id)) if req.patient_id else None

    try:
        return generate_report(procedure_summary=procedure_summary, trend_summary=trend_summary, model=req.model)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        # Never let a report-generation failure (LLM API error, schema drift the retry
        # didn't resolve, etc.) escape as an unhandled 500 — that can arrive at the
        # browser without CORS headers and surface as an opaque "Failed to fetch".
        raise HTTPException(status_code=502, detail=f"report generation failed: {e}") from e
