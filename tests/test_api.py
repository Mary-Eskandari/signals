import pytest
from fastapi.testclient import TestClient

from backend.app.config import ANTHROPIC_API_KEY
from backend.app.main import app

client = TestClient(app)


def test_list_records():
    response = client.get("/records")
    assert response.status_code == 200
    records = response.json()
    assert len(records) == 83
    assert {"record_id": "TRM278-RHC1", "patient_id": "TRM278"} in records


def test_record_summary_and_beats():
    response = client.get("/records/TRM278-RHC1/summary")
    assert response.status_code == 200
    summary = response.json()
    assert summary["record_id"] == "TRM278-RHC1"
    assert summary["n_beats_included"] > 0

    response = client.get("/records/TRM278-RHC1/beats")
    assert response.status_code == 200
    beats = response.json()
    assert len(beats) == summary["n_beats_total"]


def test_record_waveform():
    response = client.get("/records/TRM278-RHC1/waveform", params={"max_points": 100})
    assert response.status_code == 200
    data = response.json()
    assert len(data["time_s"]) <= 100
    assert set(data["channels"].keys()) == {"RHC_pressure", "ECG_lead_II", "patch_ACC_dv"}


def test_record_waveform_rejects_unknown_channel():
    response = client.get("/records/TRM278-RHC1/waveform", params={"channels": "not_a_real_channel"})
    assert response.status_code == 422


def test_record_chamber_events():
    response = client.get("/records/TRM278-RHC1/chamber_events")
    assert response.status_code == 200
    events = response.json()
    assert events["PA"] < events["PCW"]


def test_list_patients():
    response = client.get("/patients")
    assert response.status_code == 200
    patients = response.json()
    assert any(p["patient_id"] == "9" for p in patients)


def test_patient_trend_and_daily():
    response = client.get("/patients/9/trend")
    assert response.status_code == 200
    trend = response.json()
    assert trend["patient_id"] == "9"

    response = client.get("/patients/9/daily")
    assert response.status_code == 200
    assert len(response.json()) > 0


def test_reports_models():
    response = client.get("/reports/models")
    assert response.status_code == 200
    assert "claude-sonnet-5" in response.json()["allowed_models"]


def test_reports_generate_requires_record_or_patient():
    response = client.post("/reports/generate", json={})
    assert response.status_code == 422


def test_reports_generate_404s_on_unknown_record():
    response = client.post("/reports/generate", json={"record_id": "NOT-A-REAL-RECORD"})
    assert response.status_code == 422


@pytest.mark.skipif(not ANTHROPIC_API_KEY, reason="requires ANTHROPIC_API_KEY for a live call")
def test_reports_generate_auto_processes_without_prior_calls():
    """/reports/generate must not depend on the client having called /summary or
    /trend first — regression test for the race condition where the frontend's
    waveform/beats panel could fetch beats before the summary panel had processed
    the record, and reports could 404 if generated before /trend had ever run."""
    response = client.post("/reports/generate", json={"record_id": "TRM278-RHC1", "patient_id": "9"})
    assert response.status_code == 200
    report = response.json()
    assert report["disclaimer"]
    assert report["summary"]
    # regression test: the model must not repeat the demo/disclaimer notice in the
    # summary or flags — that's the header/footer's job, not the model's
    assert "demonstration" not in report["summary"].lower()
    assert "not a clinical tool" not in report["summary"].lower()
    assert not any("demonstration" in f.lower() for f in report["flags"])
