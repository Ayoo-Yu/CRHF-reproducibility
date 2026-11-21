"""Identify forecast variants, model-fitting periods, and observable feedback fields."""

import numpy as np
from .protocol import FOLDS, KEYS, ROLES, static_frame, train_transform, transform


STARTS = FOLDS[2:]


CAP = 0.025


SEEDS = (20260914, 20260915, 20260916)


POLICIES = ("A_versioned", "B_carry", "C_robust")


def feedback(f):
    return f[
        KEYS
        + [
            "target_available_at",
            "target_actual_operational",
            "target_potential",
            "target_available_inconsistent",
        ]
    ].copy()


def versions(f):
    # Model fitting actually stops at February; calendar changes are not new versions.
    return np.array([min(v, "1998-02-01") for v in f.Timestamp.dt.strftime("%Y-%m-01")])
