from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Cordella HF Signals API",
    description="Signal processing + LLM clinical reporting demo. Not a clinical tool.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# Core flow (implemented in later phases):
#   POST /signals/upload   -> accept a raw signal file, run it through pipeline/, return a record_id
#   GET  /signals/{id}/features -> ProcedureSummary / DailyTelemetry features for the uploaded signal
#   POST /signals/{id}/report   -> generate a ClinicalReport via the Claude API from extracted features
