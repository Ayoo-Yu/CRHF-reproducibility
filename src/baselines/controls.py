"""Seed-matched CatBoost comparisons for the direct static predictor."""
import numpy as np
from catboost import CatBoostRegressor

from .preprocessing import center_scale, scale_values


def paired_controls(data, tabular, fit_mask, issue_mask):
    from crhf.spatial import spatial_features

    frame = data["frame"]
    n = len(frame)
    per_station = spatial_features(data["source"], frame).to_numpy(np.float32)
    seq = data["sequence"].reshape(n, -1)
    design = np.column_stack((seq, tabular, per_station))
    base = frame.actual_operational.to_numpy(float)
    scale = frame.capacity_proxy.to_numpy(float)
    y = (frame.target_actual_operational.to_numpy(float) - base) / scale
    weight = scale[fit_mask] / scale[fit_mask].mean()
    for label, width in (("CatBoost_common", 765), ("CatBoost_spatial", 855)):
        train = design[fit_mask, :width]
        stats = center_scale(train, float32_statistics=True)
        values, missing = scale_values(train, stats)
        x = np.column_stack((values, missing.astype(np.float32)))
        model = CatBoostRegressor(iterations=240, depth=7, learning_rate=.04,
                                  loss_function="MAE", random_seed=20260914,
                                  thread_count=4, verbose=False, allow_writing_files=False)
        model.fit(x, y[fit_mask], sample_weight=weight)
        del x, train, values, missing
        values, missing = scale_values(design[issue_mask, :width], stats)
        increments = model.predict(np.column_stack((values, missing.astype(np.float32))))
        yield label, np.maximum(base[issue_mask] + scale[issue_mask] * increments, 0)
