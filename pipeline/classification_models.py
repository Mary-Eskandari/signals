"""Chamber-position classification: model registry, grouped train/test splitting,
training, and evaluation — for both engineered-feature classic/ensemble/MLP models
and the raw-waveform PyTorch CNN.

Reads the offline dataset built by pipeline.chamber_dataset (chamber_beats.parquet +
chamber_raw_snippets.npy) — this module never touches the network; training should
be fast (seconds for classic models, well under a minute for the CNN on a handful
of epochs with a small dataset).
"""

import os
import time

# PyTorch and XGBoost each bundle their own OpenMP runtime; loading both in the
# same process reliably segfaults on macOS during XGBoost's .fit() (reproduced:
# crash-free if either library is imported alone, or if only OMP_NUM_THREADS=1
# without KMP_DUPLICATE_LIB_OK, or vice versa — needs both together). Must be set
# before either library is imported, since OpenMP reads these at first load.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

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
from xgboost import XGBClassifier

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
    # richer PWA/SCG features (see pipeline/beat_features.py, scg_features.py citations) —
    # absolute-time fields (dicrotic_notch_time_s) intentionally excluded, same as
    # onset_time_s/scg_ao_time_s/scg_ac_time_s above, since raw timestamps aren't
    # meaningfully comparable across beats without normalization.
    "scg_detection_confidence",
    "dicrotic_notch_pressure_mmhg",
    "upstroke_slope_mmhg_s",
    "beat_auc_mmhg_s",
    "beat_skewness",
    "beat_kurtosis",
]

MODEL_REGISTRY = {
    "logistic_regression": {"tier": "classic", "feature_set": "engineered", "display_name": "Logistic Regression"},
    "knn": {"tier": "classic", "feature_set": "engineered", "display_name": "k-Nearest Neighbors"},
    "decision_tree": {"tier": "classic", "feature_set": "engineered", "display_name": "Decision Tree"},
    "random_forest": {"tier": "ensemble", "feature_set": "engineered", "display_name": "Random Forest"},
    "gradient_boosting": {"tier": "ensemble", "feature_set": "engineered", "display_name": "Gradient Boosting (sklearn)"},
    "xgboost": {"tier": "ensemble", "feature_set": "engineered", "display_name": "XGBoost"},
    "mlp": {"tier": "neural", "feature_set": "engineered", "display_name": "Neural Net (MLP)"},
    "cnn": {"tier": "neural", "feature_set": "raw", "display_name": "1D-ResNet (raw waveform, GPU)"},
    "lstm": {"tier": "neural", "feature_set": "raw", "display_name": "BiLSTM + Attention (raw waveform, GPU)"},
}


# Per-model tunable hyperparameters, exposed to the frontend so users can design
# their own model rather than only picking from fixed presets. `default` is what's
# used when the caller doesn't override a given key.
HYPERPARAMETER_SCHEMAS: dict[str, list[dict]] = {
    "logistic_regression": [
        {"name": "C", "type": "float", "default": 1.0, "min": 0.001, "max": 100.0},
        {"name": "max_iter", "type": "int", "default": 2000, "min": 100, "max": 5000},
    ],
    "knn": [
        {"name": "n_neighbors", "type": "int", "default": 5, "min": 1, "max": 50},
    ],
    "decision_tree": [
        {"name": "max_depth", "type": "int", "default": 8, "min": 1, "max": 30},
    ],
    "random_forest": [
        {"name": "n_estimators", "type": "int", "default": 200, "min": 10, "max": 1000},
        {"name": "max_depth", "type": "int", "default": 20, "min": 1, "max": 50},
    ],
    "gradient_boosting": [
        {"name": "n_estimators", "type": "int", "default": 100, "min": 10, "max": 1000},
        {"name": "learning_rate", "type": "float", "default": 0.1, "min": 0.001, "max": 1.0},
        {"name": "max_depth", "type": "int", "default": 3, "min": 1, "max": 15},
    ],
    "xgboost": [
        {"name": "n_estimators", "type": "int", "default": 300, "min": 10, "max": 1000},
        {"name": "max_depth", "type": "int", "default": 6, "min": 1, "max": 15},
        {"name": "learning_rate", "type": "float", "default": 0.1, "min": 0.001, "max": 1.0},
    ],
    "mlp": [
        {"name": "hidden_layer_1", "type": "int", "default": 32, "min": 4, "max": 256},
        {"name": "hidden_layer_2", "type": "int", "default": 16, "min": 0, "max": 256, "description": "0 = single hidden layer"},
        {"name": "max_iter", "type": "int", "default": 500, "min": 50, "max": 2000},
    ],
    "cnn": [
        {"name": "epochs", "type": "int", "default": 25, "min": 1, "max": 200},
        {"name": "batch_size", "type": "int", "default": 32, "min": 4, "max": 256},
        {"name": "learning_rate", "type": "float", "default": 0.001, "min": 0.00001, "max": 0.1},
    ],
    "lstm": [
        {"name": "hidden_size", "type": "int", "default": 32, "min": 4, "max": 128},
        {"name": "epochs", "type": "int", "default": 25, "min": 1, "max": 200},
        {"name": "batch_size", "type": "int", "default": 32, "min": 4, "max": 256},
        {"name": "learning_rate", "type": "float", "default": 0.001, "min": 0.00001, "max": 0.1},
    ],
}


def _resolve_hyperparams(model_name: str, overrides: dict | None) -> dict:
    resolved = {h["name"]: h["default"] for h in HYPERPARAMETER_SCHEMAS.get(model_name, [])}
    resolved.update(overrides or {})
    return resolved


def _make_classic_model(model_name: str, hyperparams: dict | None = None):
    p = _resolve_hyperparams(model_name, hyperparams)
    if model_name == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(C=p["C"], max_iter=p["max_iter"]))
    if model_name == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=p["n_neighbors"]))
    if model_name == "decision_tree":
        return DecisionTreeClassifier(max_depth=p["max_depth"], random_state=42)
    if model_name == "random_forest":
        return RandomForestClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"], random_state=42)
    if model_name == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=p["n_estimators"], learning_rate=p["learning_rate"], max_depth=p["max_depth"], random_state=42
        )
    if model_name == "xgboost":
        return XGBClassifier(
            n_estimators=p["n_estimators"], max_depth=p["max_depth"], learning_rate=p["learning_rate"],
            random_state=42, eval_metric="mlogloss",
        )
    if model_name == "mlp":
        hidden_layers = tuple(n for n in (p["hidden_layer_1"], p["hidden_layer_2"]) if n > 0)
        return make_pipeline(
            StandardScaler(), MLPClassifier(hidden_layer_sizes=hidden_layers, max_iter=p["max_iter"], random_state=42)
        )
    raise ValueError(f"unknown classic model {model_name!r}")


class ResidualBlock1D(nn.Module):
    """Two conv1d layers + a skip connection — residual learning (He et al. 2015,
    "Deep Residual Learning for Image Recognition"), adapted to 1D signals, in the
    same spirit as 1D-ResNet architectures used in the ECG/PPG deep-learning
    literature. The skip connection lets gradients flow directly through, so the
    network can go deeper without the vanishing-gradient degradation plain stacked
    convnets hit."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.shortcut = (
            nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride), nn.BatchNorm1d(out_channels))
            if (in_channels != out_channels or stride != 1)
            else nn.Identity()
        )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ChamberCNN(nn.Module):
    """1D-ResNet over (channels=3, samples=500) raw PAP/ECG/SCG snippets: a conv
    stem followed by 3 residual blocks, global average pooling, and a linear head."""

    def __init__(self, n_channels: int = 3, n_classes: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_channels, 16, kernel_size=7, padding=3), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.layer1 = ResidualBlock1D(16, 32, stride=2)
        self.layer2 = ResidualBlock1D(32, 64, stride=2)
        self.layer3 = ResidualBlock1D(64, 64, stride=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class BiLSTMAttention(nn.Module):
    """Bidirectional LSTM + additive attention pooling over raw PAP/ECG/SCG
    snippets. A plain LSTM classifier (using only the final hidden state) is a
    known weak point — it forces the whole sequence into one fixed vector and
    struggles to say *which* part of the cycle mattered. Attention pooling
    (Bahdanau-style additive attention over every timestep's LSTM output, e.g.
    Bahdanau et al. 2015) lets the model learn a weighted combination instead,
    which is the standard fix used in sequence-classification literature. Full
    Transformer-based time-series models exist and can do better still, but
    need far more data than this project's few-thousand-beat dataset to earn
    that extra complexity — BiLSTM+attention is the pragmatic ceiling here."""

    def __init__(self, n_channels: int = 3, hidden_size: int = 32, n_classes: int = 4):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_channels, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Linear(hidden_size * 2, n_classes)

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, channels, samples) -> (batch, samples, channels)
        outputs, _ = self.lstm(x)  # (batch, samples, hidden*2)
        attn_scores = self.attention(outputs).squeeze(-1)  # (batch, samples)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        context = (outputs * attn_weights).sum(dim=1)  # (batch, hidden*2)
        return self.fc(context)


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


def resolve_feature_columns(feature_columns: list[str] | None) -> list[str]:
    """Validates a user-chosen feature subset against the known engineered columns;
    None/empty falls back to the full default set."""
    if not feature_columns:
        return list(NUMERIC_FEATURE_COLUMNS)
    unknown = set(feature_columns) - set(NUMERIC_FEATURE_COLUMNS)
    if unknown:
        raise ValueError(f"unknown feature column(s): {sorted(unknown)}")
    return list(feature_columns)


def _prepare_engineered(df: pd.DataFrame, idx: np.ndarray, feature_columns: list[str]) -> np.ndarray:
    return df.loc[idx, feature_columns].fillna(0.0).to_numpy(dtype=np.float32)


def _normalize_raw(X: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None):
    if mean is None:
        mean = X.mean(axis=(0, 2), keepdims=True)
        std = X.std(axis=(0, 2), keepdims=True) + 1e-6
    return (X - mean) / std, mean, std


def _train_torch_model(model: nn.Module, X_train: np.ndarray, y_train: np.ndarray, p: dict, on_epoch_end=None):
    device = get_device()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=p["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=min(p["batch_size"], len(dataset)), shuffle=True)

    for epoch in range(p["epochs"]):
        model.train()
        total_loss, n_batches = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        if on_epoch_end is not None:
            train_acc = float((_predict_torch_model(model, X_train) == y_train).mean())
            on_epoch_end(
                {
                    "type": "epoch",
                    "epoch": epoch + 1,
                    "total_epochs": p["epochs"],
                    "loss": total_loss / max(n_batches, 1),
                    "train_accuracy": train_acc,
                }
            )
    return model


def _train_cnn(X_train: np.ndarray, y_train: np.ndarray, hyperparams: dict | None = None, on_epoch_end=None):
    p = _resolve_hyperparams("cnn", hyperparams)
    model = ChamberCNN(n_channels=X_train.shape[1], n_classes=len(CHAMBER_ORDER))
    return _train_torch_model(model, X_train, y_train, p, on_epoch_end)


def _train_lstm(X_train: np.ndarray, y_train: np.ndarray, hyperparams: dict | None = None, on_epoch_end=None):
    p = _resolve_hyperparams("lstm", hyperparams)
    model = BiLSTMAttention(n_channels=X_train.shape[1], hidden_size=p["hidden_size"], n_classes=len(CHAMBER_ORDER))
    return _train_torch_model(model, X_train, y_train, p, on_epoch_end)


def _predict_torch_model(model: nn.Module, X: np.ndarray) -> np.ndarray:
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


def _fit_predict(
    model_name: str,
    df: pd.DataFrame,
    snippets: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    hyperparams: dict | None = None,
    on_progress=None,
    feature_columns: list[str] | None = None,
):
    feature_set = MODEL_REGISTRY[model_name]["feature_set"]
    y_train = df.loc[train_idx, "label"].to_numpy()
    y_test = df.loc[test_idx, "label"].to_numpy()

    if feature_set == "raw":
        X_train_raw, X_test_raw = snippets[train_idx], snippets[test_idx]
        X_train, mean, std = _normalize_raw(X_train_raw)
        X_test, _, _ = _normalize_raw(X_test_raw, mean, std)
        train_fn = _train_lstm if model_name == "lstm" else _train_cnn
        model = train_fn(X_train, y_train, hyperparams, on_epoch_end=on_progress)
        y_pred = _predict_torch_model(model, X_test)
    else:
        if on_progress is not None:
            on_progress({"type": "fitting", "model": model_name})
        resolved_columns = resolve_feature_columns(feature_columns)
        X_train = _prepare_engineered(df, train_idx, resolved_columns)
        X_test = _prepare_engineered(df, test_idx, resolved_columns)
        model = _make_classic_model(model_name, hyperparams)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    return y_test, y_pred


def train_and_evaluate(
    model_name: str,
    test_size: float = 0.3,
    manual_train_ids: list[str] | None = None,
    manual_test_ids: list[str] | None = None,
    cv_folds: int | None = None,
    hyperparameters: dict | None = None,
    on_progress=None,
    feature_columns: list[str] | None = None,
) -> dict:
    """`on_progress`, if given, is called with dicts describing training progress:
    {"type": "epoch", ...} per CNN epoch, {"type": "fold", ...} per completed CV
    fold, {"type": "fitting", ...} once for classic models (which train too fast
    for meaningful sub-progress). Callers (e.g. the streaming API endpoint) use
    this to surface tqdm-like live progress instead of a plain spinner.

    `feature_columns`, if given, restricts engineered-feature models (classic/
    ensemble/mlp) to that subset of NUMERIC_FEATURE_COLUMNS — lets a user compare
    which features actually drive accuracy. Ignored for raw-waveform models
    (cnn/lstm), which always use the fixed raw channels."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"model must be one of {list(MODEL_REGISTRY)}, got {model_name!r}")
    resolved_feature_columns = resolve_feature_columns(feature_columns)

    df, snippets = load_dataset()
    start = time.time()
    resolved_hyperparams = _resolve_hyperparams(model_name, hyperparameters)

    if cv_folds:
        gkf = GroupKFold(n_splits=cv_folds)
        fold_results = []
        for fold_i, (train_idx, test_idx) in enumerate(gkf.split(df, groups=df["record_id"]), 1):
            train_idx, test_idx = df.index.to_numpy()[train_idx], df.index.to_numpy()[test_idx]
            y_test, y_pred = _fit_predict(
                model_name, df, snippets, train_idx, test_idx, resolved_hyperparams, on_progress,
                feature_columns=resolved_feature_columns,
            )
            fold_metrics = _evaluate(y_test, y_pred)
            fold_results.append(fold_metrics)
            if on_progress is not None:
                on_progress(
                    {
                        "type": "fold",
                        "fold": fold_i,
                        "total_folds": cv_folds,
                        "accuracy": fold_metrics["accuracy"],
                        "macro_f1": fold_metrics["macro_f1"],
                    }
                )
        accuracies = [f["accuracy"] for f in fold_results]
        macro_f1s = [f["macro_f1"] for f in fold_results]
        return {
            "model": model_name,
            "feature_set": MODEL_REGISTRY[model_name]["feature_set"],
            "feature_columns": resolved_feature_columns,
            "hyperparameters": resolved_hyperparams,
            "cv_folds": cv_folds,
            "cv_results": fold_results,
            "mean_accuracy": float(np.mean(accuracies)),
            "std_accuracy": float(np.std(accuracies)),
            "mean_macro_f1": float(np.mean(macro_f1s)),
            "std_macro_f1": float(np.std(macro_f1s)),
            "train_seconds": time.time() - start,
        }

    train_idx, test_idx = group_train_test_split(df, test_size, manual_train_ids, manual_test_ids)
    y_test, y_pred = _fit_predict(
        model_name, df, snippets, train_idx, test_idx, resolved_hyperparams, on_progress,
        feature_columns=resolved_feature_columns,
    )
    result = _evaluate(y_test, y_pred)
    result.update(
        {
            "model": model_name,
            "feature_set": MODEL_REGISTRY[model_name]["feature_set"],
            "feature_columns": resolved_feature_columns,
            "hyperparameters": resolved_hyperparams,
            "n_train_beats": len(train_idx),
            "n_test_beats": len(test_idx),
            "train_seconds": time.time() - start,
        }
    )
    return result
