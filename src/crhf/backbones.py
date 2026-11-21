"""Fit weighted CatBoost increment models for available or actual power."""

import numpy as np
from catboost import CatBoostRegressor
from .protocol import SEED, train_transform, transform


def fit_cat(x, y, scale, task):
    t = train_transform(x)
    m = CatBoostRegressor(
        iterations=320 if task == "available" else 240,
        depth=6 if task == "available" else 7,
        learning_rate=0.04,
        loss_function="MAE",
        thread_count=4,
        random_seed=SEED,
        verbose=False,
        allow_writing_files=False,
    )
    m.fit(transform(x, t), y, sample_weight=scale / scale.mean())
    # Prediction uses the fitted trees. Omit local training timestamps and
    # build identifiers from exported checkpoints.
    metadata = m.get_metadata()
    for field in ("train_finish_time", "catboost_version_info", "model_guid"):
        if field in metadata:
            del metadata[field]
    return dict(model=m, transform=t, task=task)


def predict_cat(m, x, current, scale):
    raw = current + scale * m["model"].predict(transform(x, m["transform"]))
    if m["task"] == "available":
        raw = current + 0.75 * (np.maximum(raw, 0) - current)
    return np.maximum(raw, 0)
