"""Assemble current or forecast auxiliary inputs and local-regression query features."""

import numpy as np
from .protocol import ROLES


def role_values(f, available, role, bias=0.0):
    if role not in ROLES:
        raise ValueError(role)
    q0 = f.teacher_probability.to_numpy(float)
    a0 = f.potential.to_numpy(float)
    q = q0 if role == "observed" else f.risk_q.to_numpy(float)
    a = (
        a0
        if role in ("no_available", "observed")
        else np.asarray(
            available["spatial_current" if role == "II_current" else "spatial_full"], float
        )
    )
    if role not in ("no_available", "observed"):
        a = np.maximum(a + bias * f.capacity_proxy.to_numpy(float), 0)
    return a, q


def query(f, available, anchor, role, bias=0.0):
    a, q = role_values(f, available, role, bias)
    s = f.capacity_proxy.to_numpy(float)
    a0 = f.potential.to_numpy(float)
    p = f.actual_operational.to_numpy(float)
    v = np.c_[
        p / s,
        a0 / s,
        f.dispatch_history.to_numpy(float) / s,
        f.teacher_probability.to_numpy(float),
        (a - a0) / s,
        q - f.teacher_probability.to_numpy(float),
        f.x_gap.to_numpy(float),
        f.a_delta_4.to_numpy(float),
        f.x_tracking.to_numpy(float),
        f.x_hour_sin.to_numpy(float),
        f.x_hour_cos.to_numpy(float),
        (anchor - p) / s,
    ]
    return np.clip(np.where(np.isfinite(v), v, 0), -4, 4)
