"""Historical-input transformation, chronological splits and weighting."""
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd


def center_scale(matrix, *, float32_statistics=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        center = np.nanmedian(matrix, axis=0)
        # Scalar calls retain the benchmark's float32 interpolation/rounding.
        # The matched CatBoost preprocessing instead computes a joint pair.
        if float32_statistics:
            lower, upper = np.nanquantile(matrix, [.25, .75], axis=0)
        else:
            lower = np.nanquantile(matrix, .25, axis=0)
            upper = np.nanquantile(matrix, .75, axis=0)
    center = np.nan_to_num(center)
    spread = upper - lower
    spread = np.where(spread > 1e-5, spread, 1.)
    if float32_statistics:
        return center.astype(np.float32), spread.astype(np.float32)
    return center, spread


def scale_values(values, stats):
    missing = ~np.isfinite(values)
    center, spread = stats
    scaled = np.clip((np.where(missing, center, values) - center) / spread, -10, 10)
    return scaled.astype(np.float32), missing


def masks(frame):
    valid = (frame.evaluation_eligible.to_numpy(bool)
             & ~frame.target_available_inconsistent.to_numpy(bool)
             & np.isfinite(frame.target_actual_operational.to_numpy(float)))
    future = frame.Timestamp
    start_test = future[frame.phase.eq("test")].min().normalize()
    start_validation = start_test - pd.DateOffset(months=1)
    label_time = frame.target_available_at
    fit = valid & future.lt(start_validation) & label_time.lt(start_validation)
    val = valid & future.ge(start_validation) & future.lt(start_test) & label_time.lt(start_test)
    final = valid & future.lt(start_test) & label_time.lt(start_test)
    test = valid & future.ge(start_test)
    result = tuple(np.asarray(mask, dtype=bool) for mask in (fit, val, final, test))
    counts = tuple(int(mask.sum()) for mask in (result[0], result[2], result[3]))
    if counts != (60045, 72861, 43721):
        raise ValueError(f"Baseline split counts differ from the release: {counts}")
    return result


@dataclass
class Batch:
    seq: np.ndarray
    tab: np.ndarray
    y: np.ndarray
    base: np.ndarray
    scale: np.ndarray

    def flat(self):
        return np.column_stack((self.seq.reshape(len(self.y), -1), self.tab))


def transform_for_fit(sequence, tabular, fit_mask):
    return (center_scale(sequence[fit_mask].reshape(-1, 6)),
            center_scale(tabular[fit_mask]))


def batch(frame, sequence, tabular, mask, stats):
    part = frame.loc[mask]
    seq, missing = scale_values(sequence[mask], stats[0])
    features, _ = scale_values(tabular[mask], stats[1])
    tab = np.column_stack((features, missing.reshape(len(seq), -1).astype(np.float32)))
    base = part.actual_operational.to_numpy(np.float32)
    scale = part.capacity_proxy.to_numpy(np.float32)
    y = (part.target_actual_operational.to_numpy(np.float32) - base) / scale
    return Batch(seq, tab, y, base, scale)
