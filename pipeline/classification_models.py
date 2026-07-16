"""Chamber-position classification: model registry, grouped train/test splitting,
training, and evaluation — for both engineered-feature classic/ensemble/MLP models
and the raw-waveform PyTorch CNN.

Reads the offline dataset built by pipeline.chamber_dataset (chamber_beats.parquet +
chamber_raw_snippets.npy) — this module never touches the network; training should
be fast (seconds for classic models, well under a minute for the CNN on a handful
of epochs with a small dataset).
"""

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from pipeline.chamber_dataset import BEATS_PATH, SNIPPETS_PATH
from pipeline.fetch_scg_rhc import CHAMBER_ORDER

NUMERIC_FEATURE_COLUMNS = [
    "pap_systolic_mmhg",
    "pap_diastolic_mmhg",
    "pap_mean_mmhg",
    "pulse_pressure_mmhg",
    "rr_interval_ms",
    "sqi_score",
    "scg_ao_amplitude",
    "scg_ac_amplitude",
]

MODEL_REGISTRY = {
    "logistic_regression": {"tier": "classic", "feature_set": "engineered", "display_name": "Logistic Regression"},
    "knn": {"tier": "classic", "feature_set": "engineered", "display_name": "k-Nearest Neighbors"},
    "decision_tree": {"tier": "classic", "feature_set": "engineered", "display_name": "Decision Tree"},
    "random_forest": {"tier": "ensemble", "feature_set": "engineered", "display_name": "Random Forest"},
    "gradient_boosting": {"tier": "ensemble", "feature_set": "engineered", "display_name": "Gradient Boosting"},
    "mlp": {"tier": "neural", "feature_set": "engineered", "display_name": "Neural Net (MLP)"},
    "cnn": {"tier": "neural", "feature_set": "raw", "display_name": "1D-CNN (raw waveform, GPU)"},
}


def _make_classic_model(model_name: str):
    if model_name == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    if model_name == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    if model_name == "decision_tree":
        return DecisionTreeClassifier(max_depth=8, random_state=42)
    if model_name == "random_forest":
        return RandomForestClassifier(n_estimators=200, random_state=42)
    if model_name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=42)
    if model_name == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42))
    raise ValueError(f"unknown classic model {model_name!r}")


class ChamberCNN(nn.Module):
    """3-block 1D-CNN over (channels=3, samples=500) raw PAP/ECG/SCG snippets."""

    def __init__(self, n_channels: int = 3, n_classes: int = 4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 16, kernel_size=7, padding=3), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.conv(x).squeeze(-1)
        return self.fc(x)


def get_device() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def load_dataset() -> tuple[pd.DataFrame, np.ndarray]:
    if not BEATS_PATH.exists() or not SNIPPETS_PATH.exists():
        raise FileNotFoundError("chamber dataset not built yet — run `python -m pipeline.chamber_dataset` first")
    df = pd.read_parquet(BEATS_PATH).reset_index(drop=True)
    snippets = np.load(SNIPPETS_PATH)
    df["label"] = df["chamber"].map(CHAMBER_ORDER.index)
    return df, snippets


def group_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.3,
    manual_train_ids: list[str] | None = None,
    manual_test_ids: list[str] | None = None,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Always grouped by record_id — beats from one procedure never span train/test."""
    if manual_train_ids is not None and manual_test_ids is not None:
        train_idx = df.index[df["record_id"].isin(manual_train_ids)].to_numpy()
        test_idx = df.index[df["record_id"].isin(manual_test_ids)].to_numpy()
        return train_idx, test_idx
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=df["record_id"]))
    return df.index.to_numpy()[train_idx], df.index.to_numpy()[test_idx]


def _prepare_engineered(df: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
    return df.loc[idx, NUMERIC_FEATURE_COLUMNS].fillna(0.0).to_numpy(dtype=np.float32)


def _normalize_raw(X: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None):
    if mean is None:
        mean = X.mean(axis=(0, 2), keepdims=True)
        std = X.std(axis=(0, 2), keepdims=True) + 1e-6
    return (X - mean) / std, mean, std


def _train_cnn(X_train: np.ndarray, y_train: np.ndarray, epochs: int = 25, batch_size: int = 32, lr: float = 1e-3):
    device = get_device()
    model = ChamberCNN(n_channels=X_train.shape[1], n_classes=len(CHAMBER_ORDER)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
    return model


def _predict_cnn(model: "ChamberCNN", X: np.ndarray) -> np.ndarray:
    device = get_device()
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(X, dtype=torch.float32).to(device))
        return out.argmax(dim=1).cpu().numpy()


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = list(range(len(CHAMBER_ORDER)))
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": CHAMBER_ORDER,
        "per_class": [
            {"label": CHAMBER_ORDER[i], "precision": float(precision[i]), "recall": float(recall[i]),
             "f1": float(f1[i]), "support": int(support[i])}
            for i in labels
        ],
    }


def _fit_predict(model_name: str, df: pd.DataFrame, snippets: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray):
    feature_set = MODEL_REGISTRY[model_name]["feature_set"]
    y_train = df.loc[train_idx, "label"].to_numpy()
    y_test = df.loc[test_idx, "label"].to_numpy()

    if feature_set == "raw":
        X_train_raw, X_test_raw = snippets[train_idx], snippets[test_idx]
        X_train, mean, std = _normalize_raw(X_train_raw)
        X_test, _, _ = _normalize_raw(X_test_raw, mean, std)
        model = _train_cnn(X_train, y_train)
        y_pred = _predict_cnn(model, X_test)
    else:
        X_train = _prepare_engineered(df, train_idx)
        X_test = _prepare_engineered(df, test_idx)
        model = _make_classic_model(model_name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    return y_test, y_pred


def train_and_evaluate(
    model_name: str,
    test_size: float = 0.3,
    manual_train_ids: list[str] | None = None,
    manual_test_ids: list[str] | None = None,
    cv_folds: int | None = None,
) -> dict:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"model must be one of {list(MODEL_REGISTRY)}, got {model_name!r}")

    df, snippets = load_dataset()
    start = time.time()

    if cv_folds:
        gkf = GroupKFold(n_splits=cv_folds)
        fold_results = []
        for train_idx, test_idx in gkf.split(df, groups=df["record_id"]):
            train_idx, test_idx = df.index.to_numpy()[train_idx], df.index.to_numpy()[test_idx]
            y_test, y_pred = _fit_predict(model_name, df, snippets, train_idx, test_idx)
            fold_results.append(_evaluate(y_test, y_pred))
        accuracies = [f["accuracy"] for f in fold_results]
        macro_f1s = [f["macro_f1"] for f in fold_results]
        return {
            "model": model_name,
            "feature_set": MODEL_REGISTRY[model_name]["feature_set"],
            "cv_folds": cv_folds,
            "cv_results": fold_results,
            "mean_accuracy": float(np.mean(accuracies)),
            "std_accuracy": float(np.std(accuracies)),
            "mean_macro_f1": float(np.mean(macro_f1s)),
            "std_macro_f1": float(np.std(macro_f1s)),
            "train_seconds": time.time() - start,
        }

    train_idx, test_idx = group_train_test_split(df, test_size, manual_train_ids, manual_test_ids)
    y_test, y_pred = _fit_predict(model_name, df, snippets, train_idx, test_idx)
    result = _evaluate(y_test, y_pred)
    result.update(
        {
            "model": model_name,
            "feature_set": MODEL_REGISTRY[model_name]["feature_set"],
            "n_train_beats": len(train_idx),
            "n_test_beats": len(test_idx),
            "train_seconds": time.time() - start,
        }
    )
    return result
