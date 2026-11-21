"""Align 96 six-channel historical slots and define the target-quality scoring mask."""

import json
import numpy as np
import pandas as pd
from pathlib import Path


CORE = [
    "x_actual_operational",
    "x_potential",
    "x_dispatch_history",
    "x_physics_score",
    "c_x_actual_operational",
    "c_x_physics_score",
]


LENGTH = 96


def write_json(path, data):
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def windows(grid, keys, columns=CORE, length=LENGTH):
    """Exact clock offsets, including absent/invalid observations as NaN."""
    result = np.full((len(keys), length, len(columns)), np.nan, np.float32)
    for farm, ids in keys.groupby("farm", sort=True).indices.items():
        g = grid.loc[grid.farm.eq(farm)].set_index("Timestamp").sort_index()
        if g.index.has_duplicates:
            raise ValueError("Duplicate grid timestamp")
        times = keys.iloc[ids].Timestamp.to_numpy(dtype="datetime64[ns]")
        wanted = times[:, None] - np.arange(length - 1, -1, -1)[None, :] * np.timedelta64(15, "m")
        values = g.reindex(wanted.ravel())[columns].to_numpy(np.float32)
        result[ids] = values.reshape(len(ids), length, len(columns))
    return result


def eligible(f, cutoff=None, consistent=True):
    good = f.evaluation_eligible.to_numpy(bool) & np.isfinite(
        f.target_actual_operational.to_numpy(float)
    )
    if consistent:
        good &= ~f.target_available_inconsistent.to_numpy(bool)
    if cutoff is not None:
        good &= f.target_available_at.lt(pd.Timestamp(cutoff)).to_numpy()
    return good
