"""Phase 1 verification: both raw data sources stream/parse end-to-end."""

from pipeline import fetch_chiron, fetch_scg_rhc


def test_scg_rhc_record_streams():
    dest = fetch_scg_rhc.fetch_record("TRM278-RHC1", window_s=10)
    assert dest.exists()

    import numpy as np

    data = np.load(dest)
    assert data["signal"].shape == (10 * fetch_scg_rhc.SAMPLING_RATE_HZ, 17)
    assert "RHC_pressure" in list(data["channel_names"])
    assert "patch_ECG" in list(data["channel_names"])


def test_chiron_parses():
    df = fetch_chiron.load()
    assert df.shape[0] > 1000
    assert {"Patient_ID", "Day", "SystolicBP", "DiastolicBP", "Weight", "SpO2"}.issubset(df.columns)
    assert df["Patient_ID"].nunique() > 1
