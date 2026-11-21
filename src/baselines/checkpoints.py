"""Predict from portable, data-free fit parameters. Numeric archives disallow pickle."""
import numpy as np
from threadpoolctl import threadpool_limits

from .models import ForecastNet
from .training import predict_network


def network(path, name, cfg, seed, testing):
    import torch
    net = ForecastNet(name, testing.tab.shape[1], cfg)
    with np.load(path / f"{name}_{seed}.npz", allow_pickle=False) as weights:
        net.load_state_dict({key: torch.from_numpy(weights[key]) for key in weights.files})
    return np.maximum(testing.base + testing.scale * predict_network(net, testing), 0)


def ensemble_trees(matrix, arrays, name):
    # Each tree's child indices are relative to its position in the concatenated arrays.
    offsets = arrays["offsets"]
    left, right = arrays["left"], arrays["right"]
    threshold, feature, value = arrays["threshold"], arrays["feature"], arrays["value"]
    total = np.full(len(matrix), arrays["base"][0] if name == "HGB" else 0., dtype=float)
    rows = np.arange(len(matrix))
    for offset in offsets[:-1]:
        node = np.full(len(matrix), offset, dtype=np.int64)
        active = rows
        while active.size:
            index = node[active]
            active = active[left[index] >= 0]
            index = node[active]
            go_left = matrix[active, feature[index]] <= threshold[index]
            node[active] = offset + np.where(go_left, left[index], right[index])
        total += value[node]
    if name == "ExtraTrees":
        total /= len(offsets) - 1
    return total


def conventional(path, name, testing):
    matrix = testing.flat()
    if name == "CatBoost":
        from catboost import CatBoostRegressor
        model = CatBoostRegressor()
        model.load_model(str(path / "CatBoost.json"), format="json")
        result = model.predict(matrix, thread_count=6)
    elif name == "LightGBM":
        from lightgbm import Booster
        model = Booster(model_str=(path / "LightGBM.txt").read_text(encoding="utf-8"))
        result = model.predict(matrix, num_threads=6)
    else:
        with np.load(path / f"{name}.npz", allow_pickle=False) as arrays:
            if name == "Ridge":
                with threadpool_limits(6):
                    result = matrix @ arrays["coef"] + arrays["intercept"]
            else:
                result = ensemble_trees(matrix, arrays, name)
    return np.maximum(testing.base + testing.scale * result, 0)
