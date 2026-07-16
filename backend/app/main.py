from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.routers import classification, patients, records, reports

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

app.include_router(records.router)
app.include_router(patients.router)
app.include_router(reports.router)
app.include_router(classification.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
