"""Chamber-position classification: model menu, dataset status, train & evaluate.

/train returns a single JSON result. /train/stream returns newline-delimited JSON
progress events (epoch-by-epoch for the CNN, fold-by-fold for CV) followed by a
final {"type": "result", "data": ...} line — training runs on a background thread
so progress events can be yielded as they happen rather than blocking until done.
"""

import json
import queue
import threading
from typing import Literal

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from pipeline.chamber_dataset import BEATS_PATH
from pipeline.classification_models import (
    HYPERPARAMETER_SCHEMAS,
    MODEL_REGISTRY,
    NUMERIC_FEATURE_COLUMNS,
    load_dataset,
    train_and_evaluate,
)
from pipeline.fetch_scg_rhc import CHAMBER_ORDER, fetch_records_list

router = APIRouter(prefix="/classification", tags=["classification"])

# PyTorch's MPS (Apple GPU) backend doesn't handle concurrent training loops from
# multiple threads well — reproduced: overlapping CNN/LSTM training requests (e.g.
# a client that disconnects mid-stream, leaving the server-side thread running,
# followed by a new request) caused severe contention that looked like a hang
# (29 accumulated threads, backend stuck in uninterruptible sleep). A single global
# lock is the right fix for a single-user demo app: only one training job runs at
# a time; anything else gets a clear 409 instead of silently degrading.
_TRAINING_LOCK = threading.Lock()


@router.get("/models")
def list_models() -> dict:
    return {"models": MODEL_REGISTRY, "hyperparameters": HYPERPARAMETER_SCHEMAS}


@router.get("/status")
def dataset_status() -> dict:
    total_records = len(fetch_records_list())
    if not BEATS_PATH.exists():
        return {"n_records_covered": 0, "total_records": total_records, "n_beats": 0, "per_chamber": {}}

    df = pd.read_parquet(BEATS_PATH)
    return {
        "n_records_covered": int(df["record_id"].nunique()),
        "total_records": total_records,
        "n_beats": len(df),
        "per_chamber": df["chamber"].value_counts().to_dict(),
    }


@router.get("/records")
def list_dataset_records() -> list[str]:
    if not BEATS_PATH.exists():
        return []
    df = pd.read_parquet(BEATS_PATH)
    return sorted(df["record_id"].unique().tolist())


@router.get("/labels")
def labels() -> list[str]:
    return CHAMBER_ORDER


@router.get("/features")
def features() -> list[str]:
    """Engineered feature columns available for classic/ensemble/mlp models — a
    user can train on any non-empty subset (via TrainRequest.feature_columns) to
    see which features actually drive accuracy. Raw-waveform models (cnn/lstm)
    ignore this; they always use the fixed raw signal channels."""
    return NUMERIC_FEATURE_COLUMNS


class SplitConfig(BaseModel):
    mode: Literal["auto", "manual"] = "auto"
    test_size: float = 0.3
    train_record_ids: list[str] | None = None
    test_record_ids: list[str] | None = None


class TrainRequest(BaseModel):
    model: str
    split: SplitConfig = SplitConfig()
    cv_folds: int | None = None
    hyperparameters: dict | None = None
    feature_columns: list[str] | None = None


def _validate_train_request(req: TrainRequest) -> None:
    if req.model not in MODEL_REGISTRY:
        raise HTTPException(status_code=422, detail=f"model must be one of {list(MODEL_REGISTRY)}")
    if req.cv_folds is not None and not (2 <= req.cv_folds <= 10):
        raise HTTPException(status_code=422, detail="cv_folds must be between 2 and 10")
    if req.split.mode == "manual" and not (req.split.train_record_ids and req.split.test_record_ids):
        raise HTTPException(status_code=422, detail="manual split needs both train_record_ids and test_record_ids")
    if req.feature_columns is not None:
        unknown = set(req.feature_columns) - set(NUMERIC_FEATURE_COLUMNS)
        if unknown:
            raise HTTPException(status_code=422, detail=f"unknown feature column(s): {sorted(unknown)}")
        if not req.feature_columns:
            raise HTTPException(status_code=422, detail="feature_columns, if given, must be non-empty")


def _run(req: TrainRequest, on_progress=None) -> dict:
    if req.cv_folds:
        return train_and_evaluate(
            req.model, cv_folds=req.cv_folds, hyperparameters=req.hyperparameters, on_progress=on_progress,
            feature_columns=req.feature_columns,
        )
    if req.split.mode == "manual":
        return train_and_evaluate(
            req.model,
            manual_train_ids=req.split.train_record_ids,
            manual_test_ids=req.split.test_record_ids,
            hyperparameters=req.hyperparameters,
            on_progress=on_progress,
            feature_columns=req.feature_columns,
        )
    return train_and_evaluate(
        req.model, test_size=req.split.test_size, hyperparameters=req.hyperparameters, on_progress=on_progress,
        feature_columns=req.feature_columns,
    )


@router.post("/train")
def train(req: TrainRequest) -> dict:
    _validate_train_request(req)
    if not _TRAINING_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="a training job is already in progress — wait for it to finish")
    try:
        load_dataset()  # raises FileNotFoundError with a clear message if not built yet
        return _run(req)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"training failed: {e}") from e
    finally:
        _TRAINING_LOCK.release()


def _stream_events(req: TrainRequest):
    progress_q: queue.Queue = queue.Queue()

    def worker():
        try:
            result = _run(req, on_progress=progress_q.put)
            progress_q.put({"type": "result", "data": result})
        except Exception as e:
            progress_q.put({"type": "error", "message": str(e)})
        finally:
            progress_q.put(None)
            _TRAINING_LOCK.release()

    threading.Thread(target=worker, daemon=True).start()

    while True:
        event = progress_q.get()
        if event is None:
            return
        yield json.dumps(event) + "\n"


@router.post("/train/stream")
def train_stream(req: TrainRequest) -> StreamingResponse:
    _validate_train_request(req)
    if not _TRAINING_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="a training job is already in progress — wait for it to finish")
    try:
        load_dataset()
    except FileNotFoundError as e:
        _TRAINING_LOCK.release()
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception:
        _TRAINING_LOCK.release()
        raise
    return StreamingResponse(_stream_events(req), media_type="application/x-ndjson")
