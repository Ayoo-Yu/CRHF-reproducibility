"""Registered model families, configurations and validation choices."""
from types import SimpleNamespace

import torch
from torch import nn

from .linear_cores import DecompositionCore, MixingCore

SEEDS = (20260913, 20260914, 20260915)
TREE_MODELS = ("Ridge", "ExtraTrees", "HGB", "CatBoost", "LightGBM")
ADAMW_MODELS = ("MLP", "GRU", "PatchTST", "MixLinear", "MDLinear")

# Two configurations were compared on the same validation rows with SEEDS[0].
# Each winning epoch count is reused for all three final, independently seeded fits.
SELECTION = {
    "Ridge": (1, None), "ExtraTrees": (1, None), "HGB": (1, None),
    "CatBoost": (1, None), "LightGBM": (0, None), "MLP": (1, 3),
    "GRU": (0, 13), "PatchTST": (1, 14), "MixLinear": (0, 20), "MDLinear": (0, 13),
}


def tree_model(name, cfg, seed=SEEDS[0]):
    if name == "Ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=(1., 100.)[cfg])
    if name == "ExtraTrees":
        from sklearn.ensemble import ExtraTreesRegressor
        return ExtraTreesRegressor(n_estimators=240, min_samples_leaf=(8, 24)[cfg],
                                   max_features=.8, n_jobs=6, random_state=seed)
    if name == "HGB":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(loss="absolute_error", max_iter=240,
            max_leaf_nodes=(15, 31)[cfg], learning_rate=.04, l2_regularization=1.,
            early_stopping=False, random_state=seed)
    if name == "LightGBM":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=240, num_leaves=(15, 31)[cfg],
            objective="regression_l1", learning_rate=.04, min_child_samples=24,
            reg_lambda=1., n_jobs=6, random_state=seed, verbosity=-1)
    if name == "CatBoost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=240, depth=(5, 7)[cfg],
            loss_function="MAE", learning_rate=.04, thread_count=6, random_seed=seed,
            verbose=False, allow_writing_files=False)
    raise ValueError(name)


class ForecastNet(nn.Module):
    """The sequence prediction and tabular linear prediction are added in increment space."""
    def __init__(self, name, tab_dim, cfg):
        super().__init__()
        self.name = name
        width = (32, 64)[cfg]
        self.context = nn.Linear(tab_dim, 1)
        nn.init.zeros_(self.context.weight)
        nn.init.zeros_(self.context.bias)
        if name == "MLP":
            self.core = nn.Sequential(nn.Flatten(), nn.Linear(96 * 6, width * 2), nn.ReLU(),
                                      nn.Dropout(.1), nn.Linear(width * 2, width), nn.ReLU(),
                                      nn.Linear(width, 1))
        elif name == "GRU":
            self.core = nn.GRU(6, width, num_layers=1, batch_first=True)
            self.readout = nn.Linear(width, 1)
        elif name == "PatchTST":
            from .patchtst.PatchTST import Model
            conf = SimpleNamespace(enc_in=6, seq_len=96, pred_len=1, e_layers=2, n_heads=4,
                d_model=width, d_ff=2 * width, dropout=.1, fc_dropout=.1, head_dropout=0.,
                individual=False, patch_len=16, stride=8, padding_patch="end", revin=True,
                affine=True, subtract_last=False, decomposition=False, kernel_size=25)
            self.core = Model(conf)
            self.readout = nn.Linear(6, 1)
        elif name == "MixLinear":
            self.core = MixingCore((6, 12)[cfg])
            self.readout = nn.Linear(6, 1)
        elif name == "MDLinear":
            self.core = DecompositionCore()
        else:
            raise ValueError(name)

    def forward(self, sequence, tabular):
        if self.name == "GRU":
            value = self.readout(self.core(sequence)[1][-1])
        elif self.name in ("PatchTST", "MixLinear"):
            value = self.readout(self.core(sequence)[:, -1, :])
        elif self.name == "MDLinear":
            value = self.core(sequence)[:, -1, :]
        else:
            value = self.core(sequence)
        return (value + self.context(tabular)).squeeze(-1)
