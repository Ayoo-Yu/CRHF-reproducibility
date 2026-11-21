"""Define forward fits, sample selection and training-only robust feature transformations."""

import numpy as np
import pandas as pd
from .sequences import eligible


SEED = 20260914


FOLDS = ("1997-09-01", "1997-10-01", "1997-11-01", "1997-12-01", "1998-01-01", "1998-02-01")


ROLES = ("full", "no_available", "II_current", "observed")


KEYS = ["farm", "Timestamp", "target_timestamp"]


def training_mask(f, cutoff):
    return (
        eligible(f, cutoff)
        & f.Timestamp.lt(cutoff).to_numpy()
        & np.isfinite(f.target_potential.to_numpy(float))
    )


def month_mask(f, start):
    start = pd.Timestamp(start)
    return f.Timestamp.ge(start).to_numpy() & (
        f.Timestamp.lt(start + pd.offsets.MonthBegin()).to_numpy()
        if start < pd.Timestamp("1998-02-01")
        else True
    )


def static_frame(f):
    return f.drop(
        columns=[c for c in f if c.startswith("target_")]
        + ["evaluation_eligible", "sample_valid"],
        errors="ignore",
    )


def train_transform(x):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = np.nanmedian(x, axis=0)
        q = np.nanquantile(x, [0.25, 0.75], axis=0)
    m = np.where(np.isfinite(m), m, 0).astype(np.float32)
    s = q[1] - q[0]
    s = np.where(np.isfinite(s) & (s > 1e-5), s, 1).astype(np.float32)
    return dict(median=m, scale=s)


def transform(x, t):
    x = np.asarray(x, np.float32)
    bad = ~np.isfinite(x)
    z = np.clip((np.where(bad, t["median"], x) - t["median"]) / t["scale"], -10, 10)
    return np.concatenate([z, bad.astype(np.float32)], axis=1)
