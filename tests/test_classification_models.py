import numpy as np
import pandas as pd
import pytest

from pipeline.chamber_dataset import BEATS_PATH
from pipeline.classification_models import (
    MODEL_REGISTRY,
    NUMERIC_FEATURE_COLUMNS,
    _fit_predict,
    group_train_test_split,
    load_dataset,
    train_and_evaluate,
)
from pipeline.fetch_scg_rhc import CHAMBER_ORDER


def _synthetic_engineered_df(n_per_record=20, n_records_per_chamber=3):
    """Trivially-separable synthetic data: each chamber gets a distinct, tight
    systolic-pressure range with no overlap, so any reasonable classifier should
    get near-perfect accuracy — this validates the pipeline plumbing, not the
    models' handling of hard real-world separability (that's what the real-data
    test below checks)."""
    rng = np.random.default_rng(0)
    rows = []
    record_counter = 0
    for chamber_idx, chamber in enumerate(CHAMBER_ORDER):
        base = chamber_idx * 100.0
        for _ in range(n_records_per_chamber):
            record_id = f"REC{record_counter:03d}"
            record_counter += 1
            for i in range(n_per_record):
                rows.append(
                    {
                        "record_id": record_id,
                        "chamber": chamber,
                        "beat_id": f"{record_id}-beat{i:03d}",
                        "pap_systolic_mmhg": base + rng.normal(0, 1),
                        "pap_diastolic_mmhg": base / 2 + rng.normal(0, 1),
                        "pap_mean_mmhg": base * 0.75 + rng.normal(0, 1),
                        "pulse_pressure_mmhg": base / 2 + rng.normal(0, 1),
                        "rr_interval_ms": 800 + rng.normal(0, 20),
                        "sqi_score": 1.0,
                        "scg_ao_amplitude": rng.normal(0, 1),
                        "scg_ac_amplitude": rng.normal(0, 1),
                    }
                )
    df = pd.DataFrame(rows)
    df["label"] = df["chamber"].map(CHAMBER_ORDER.index)
    return df


def test_group_split_never_leaks_records():
    df = _synthetic_engineered_df()
    train_idx, test_idx = group_train_test_split(df, test_size=0.3)
    train_records = set(df.loc[train_idx, "record_id"])
    test_records = set(df.loc[test_idx, "record_id"])
    assert train_records.isdisjoint(test_records)
    assert len(train_idx) > 0 and len(test_idx) > 0


def test_manual_split_respects_explicit_ids():
    df = _synthetic_engineered_df()
    all_records = df["record_id"].unique().tolist()
    train_ids, test_ids = all_records[:8], all_records[8:]
    train_idx, test_idx = group_train_test_split(df, manual_train_ids=train_ids, manual_test_ids=test_ids)
    assert set(df.loc[train_idx, "record_id"]) == set(train_ids)
    assert set(df.loc[test_idx, "record_id"]) == set(test_ids)


def test_classic_model_learns_trivially_separable_synthetic_data():
    df = _synthetic_engineered_df()
    train_idx, test_idx = group_train_test_split(df, test_size=0.3)
    dummy_snippets = np.zeros((len(df), 3, 10), dtype=np.float32)
    y_test, y_pred = _fit_predict("random_forest", df, dummy_snippets, train_idx, test_idx)
    accuracy = (y_test == y_pred).mean()
    assert accuracy > 0.9, f"expected near-perfect accuracy on trivially separable data, got {accuracy}"


def test_cnn_runs_on_synthetic_raw_snippets():
    n = 80
    rng = np.random.default_rng(1)
    labels = np.tile(np.arange(4), n // 4)
    snippets = rng.normal(0, 1, size=(n, 3, 500)).astype(np.float32)
    # bias each class's mean so the CNN has *something* learnable, without
    # requiring it to actually converge in this fast smoke test
    for c in range(4):
        snippets[labels == c] += c * 2.0
    df = pd.DataFrame(
        {
            "record_id": [f"REC{i:03d}" for i in range(n)],
            "chamber": [CHAMBER_ORDER[label] for label in labels],
            "beat_id": [f"b{i}" for i in range(n)],
            "label": labels,
            **{c: 0.0 for c in NUMERIC_FEATURE_COLUMNS},
        }
    )
    train_idx, test_idx = group_train_test_split(df, test_size=0.3)
    y_test, y_pred = _fit_predict("cnn", df, snippets, train_idx, test_idx)
    assert y_pred.shape == y_test.shape
    assert set(y_pred.tolist()).issubset(set(range(4)))


def test_model_registry_covers_all_tiers():
    tiers = {meta["tier"] for meta in MODEL_REGISTRY.values()}
    assert tiers == {"classic", "ensemble", "neural"}
    feature_sets = {meta["feature_set"] for meta in MODEL_REGISTRY.values()}
    assert feature_sets == {"engineered", "raw"}
    assert MODEL_REGISTRY["cnn"]["feature_set"] == "raw"


@pytest.mark.skipif(not BEATS_PATH.exists(), reason="chamber dataset not built yet")
def test_real_dataset_trains_above_random_baseline():
    """Smoke test against whatever real chamber data has been built so far —
    tolerant of the dataset build still being in progress in the background."""
    df, snippets = load_dataset()
    if df["record_id"].nunique() < 4:
        pytest.skip("not enough records built yet for a meaningful group split")

    result = train_and_evaluate("random_forest", test_size=0.3)
    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["accuracy"] > 0.25  # better than random 4-class baseline
    assert result["n_train_beats"] > 0 and result["n_test_beats"] > 0
