"""LLM clinical report generation, tying together a record and/or patient."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.app.config import ALLOWED_REPORT_MODELS, REPORT_MODEL
from backend.app.llm_report import generate_report
from pipeline import store
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

    procedure_summary = store.read_procedure_summary(req.record_id) if req.record_id else None
    if req.record_id and procedure_summary is None:
        raise HTTPException(status_code=404, detail=f"no processed summary for record {req.record_id}; call /records/{{id}}/summary first")

    trend_summary = store.read_patient_trend_summary(req.patient_id) if req.patient_id else None
    if req.patient_id and trend_summary is None:
        raise HTTPException(status_code=404, detail=f"no trend summary for patient {req.patient_id}; call /patients/{{id}}/trend first")

    try:
        return generate_report(procedure_summary=procedure_summary, trend_summary=trend_summary, model=req.model)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
