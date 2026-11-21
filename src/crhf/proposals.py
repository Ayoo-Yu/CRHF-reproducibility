"""Compute bounded fixed candidates and sensitivity variants from the residual models."""

import numpy as np
from .residual_models import features, predict_deltas, role_values


def proposals(model, f, available, online, static, role, width, bias=0.0):
    scale = f.capacity_proxy.to_numpy(float)
    center, q = role_values(f, available, role)
    center = np.maximum(center + bias * scale, 0) if role in ("full", "II_current") else center
    nominal, _ = features(
        f,
        available,
        online,
        static,
        role,
        model["history_columns"],
        model["farms"],
        override=center,
    )
    delta = predict_deltas(model, nominal)
    if role in ("full", "II_current"):
        radius = np.asarray(width, float) * scale
        low, _ = features(
            f,
            available,
            online,
            static,
            role,
            model["history_columns"],
            model["farms"],
            override=np.maximum(center - radius, 0),
        )
        high, _ = features(
            f,
            available,
            online,
            static,
            role,
            model["history_columns"],
            model["farms"],
            override=center + radius,
        )
        ds = np.stack([predict_deltas(model, low), delta, predict_deltas(model, high)], axis=1)
        agree = (ds > 0).all(axis=1) | (ds < 0).all(axis=1)
        robust = np.where(agree, np.sign(delta) * np.min(abs(ds), axis=1), 0.0)
    else:
        agree = np.ones_like(delta, bool)
        robust = delta.copy()
    plans = np.repeat(np.asarray(online)[:, None, None], 6, axis=1).reshape(len(f), 2, 3)
    plans[:, 0, 1:] = np.maximum(online[:, None] + scale[:, None] * delta, 0)
    plans[:, 1, 1:] = np.maximum(online[:, None] + scale[:, None] * robust, 0)
    executed = np.ones_like(plans, bool)
    executed[:, 1, 1:] = agree
    return dict(
        plans=plans,
        executed=executed,
        nominal_delta=delta,
        robust_delta=robust,
        sign_agreement=agree,
    )
