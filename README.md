# Cordella-Adjacent HF Signal Insights

A demonstration web app that processes hemodynamic and physiological signals — pulmonary artery (PA) pressure waveforms, ECG, seismocardiogram (SCG), and longitudinal heart-failure telemonitoring data — into clinical features, and uses an LLM to generate grounded, clinician-style narrative reports.

Built as a technical portfolio project exploring the signal-processing and remote patient-monitoring problem space behind implantable PA pressure sensor systems (e.g. Cordella, CardioMEMS) used in heart failure management.

> **Disclaimer:** This project uses exclusively public, de-identified research datasets (PhysioNet SCG-RHC, Chiron CHF telemonitoring). It is **not** a clinical tool, is **not FDA-cleared**, contains **no real patient health information**, and must not be used for actual clinical decision-making. It exists solely to demonstrate signal-processing and applied-ML engineering.

## Datasets

- **SCG-RHC Wearable + Right Heart Catheter DB** (PhysioNet, ODC-Attribution license) — real PA pressure waveform + wearable ECG + seismocardiogram.
  https://physionet.org/content/scg-rhc-wearable-database/1.0.0/
- **Chiron CHF telemonitoring dataset** (Mlakar et al., PLoS ONE 2018, CC BY) — daily weight, BP, SpO2, HR, activity, and symptom score for CHF patients.
  https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0190323

## Architecture

- `pipeline/` — offline signal-processing pipeline (wfdb → NeuroKit2/pyPPG/vital_sqi/tsfel → Parquet + DuckDB feature store)
- `backend/` — FastAPI service serving precomputed features/waveform windows and generating LLM narrative reports (Claude API)
- `frontend/` — React + Vite + TypeScript app with an annotated waveform viewer and telemonitoring trend dashboard

See `docs/` for the full implementation plan and methods writeup as the project progresses.

## Status

Early scaffold — see project plan for the phased roadmap.
