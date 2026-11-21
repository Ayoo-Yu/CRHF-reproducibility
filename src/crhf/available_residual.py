"""Fit and apply robust risk-conditioned corrections to available power."""

import numpy as np
import pandas as pd


VERSION = "risk_available_oof_calibration_v1"


FAMILIES = ("A_innovation", "B_transition", "C_augmentation")


ROLES = ("current", "cal")


LAMBDAS = (0.001, 0.01, 0.1)


STRENGTHS = (0.25, 0.5)


CAP = 0.1


SMOOTHING = 0.005


ITERATIONS = 18


AUGMENT = (-0.2, 0.0, 0.2)


BASE_COLUMNS = (
    "x_potential",
    "x_actual_operational",
    "x_dispatch_history",
    "x_gap",
    "x_tracking",
    "teacher_probability",
    "a_delta_1",
    "a_delta_4",
    "a_delta_8",
    "a_std_4",
    "a_std_16",
    "a_deviation_4",
    "a_deviation_16",
    "a_input_inconsistent",
    "x_hour_sin",
    "x_hour_cos",
)


def gate(frame, role, override=None):
    if role not in ROLES:
        raise ValueError("Unknown risk role")
    values = np.asarray(
        override
        if override is not None
        else frame["teacher_probability" if role == "current" else "risk_probability"],
        float,
    )
    if (
        values.shape != (len(frame),)
        or not np.isfinite(values).all()
        or np.any((values < 0) | (values > 1))
    ):
        raise ValueError("Risk must be finite probabilities")
    return values


def transition_weights(current, future):
    current, future = np.asarray(current, float), np.asarray(future, float)
    if (
        current.shape != future.shape
        or not np.isfinite(current).all()
        or not np.isfinite(future).all()
    ):
        raise ValueError("Invalid transition probabilities")
    if np.any((current < 0) | (current > 1) | (future < 0) | (future > 1)):
        raise ValueError("Transition probabilities out of range")
    # Current state is observed; soft q describes the state at the forecast lead.
    state = (current >= 0.5).astype(float)
    return np.column_stack(
        ((1 - state) * (1 - future), (1 - state) * future, state * (1 - future), state * future)
    )


def design(frame, family, role, farms, override=None):
    if family not in FAMILIES:
        raise ValueError("Unknown family")
    q = gate(frame, role, override)
    q0 = gate(frame, "current")
    scale = frame.capacity_proxy.to_numpy(float)
    a0, ah = frame.potential.to_numpy(float), frame.available_pred.to_numpy(float)
    if (
        not np.isfinite(scale).all()
        or np.any(scale <= 0)
        or not np.isfinite(a0).all()
        or not np.isfinite(ah).all()
    ):
        raise ValueError("Invalid available power or frozen scale")
    if np.any(a0 < 0) or np.any(ah < 0):
        raise ValueError("Available power must be nonnegative")
    base = frame[list(BASE_COLUMNS)].to_numpy(float)
    delta = (ah - a0) / scale
    level = a0 / scale
    ramp = frame.a_delta_4.to_numpy(float)
    gap = frame.x_gap.to_numpy(float)
    onehot = (
        np.column_stack([frame.farm.eq(f).to_numpy(float) for f in farms[1:]])
        if len(farms) > 1
        else np.empty((len(frame), 0))
    )
    common = np.column_stack((base, delta, onehot))
    if family == "B_transition":
        weights = transition_weights(q0, q)
        basis = np.column_stack((np.ones(len(frame)), level, delta, ramp))
        extra = (weights[:, :, None] * basis[:, None, :]).reshape(len(frame), -1)
    else:
        innovation = q - q0
        extra = innovation[:, None] * np.column_stack(
            (np.ones(len(frame)), level, delta, ramp, gap)
        )
    result = np.column_stack((common, extra))
    if np.isinf(result).any():
        raise ValueError("Infinite feature")
    return result


def fitted_transform(x):
    median = np.array([np.nanmedian(col) if np.isfinite(col).any() else 0.0 for col in x.T])
    filled = np.where(np.isnan(x), median, x)
    spread = np.quantile(filled, 0.75, axis=0) - np.quantile(filled, 0.25, axis=0)
    spread = np.where(spread > 1e-6, spread, 1.0)
    return dict(median=median, spread=spread)


def transformed(x, transform):
    filled = np.where(np.isnan(x), transform["median"], x)
    z = np.clip((filled - transform["median"]) / transform["spread"], -8.0, 8.0)
    return np.column_stack((np.ones(len(x)), z))


def validate_training(frame, cutoff):
    end = pd.Timestamp(cutoff)
    if len(frame) < 32 or frame[["farm", "Timestamp"]].duplicated().any():
        raise ValueError("Insufficient or duplicate training rows")
    times = {
        c: pd.to_datetime(frame[c])
        for c in (
            "Timestamp",
            "target_available_at",
            "risk_fold_start",
            "available_fold_start",
            "available_training_latest_label",
        )
    }
    if any(x.isna().any() for x in times.values()):
        raise ValueError("Missing training chronology")
    if times["Timestamp"].ge(end).any() or times["target_available_at"].gt(end).any():
        raise ValueError("Training label or issue beyond cutoff")
    if (
        times["risk_fold_start"].gt(times["Timestamp"]).any()
        or times["available_fold_start"].gt(times["Timestamp"]).any()
    ):
        raise ValueError("Out-of-sample model fitted after issue")
    if times["available_training_latest_label"].gt(times["available_fold_start"]).any():
        raise ValueError("OOF training label after fold cutoff")
    if (
        not frame.evaluation_eligible.eq(True).all()
        or not frame.target_available_inconsistent.eq(False).all()
    ):
        raise ValueError("Training targets must be observed and consistent")
    if not np.isfinite(frame.target_potential.to_numpy(float)).all():
        raise ValueError("Non-finite available target")


def fit(frame, family, role, penalty, cutoff):
    validate_training(frame, cutoff)
    if penalty not in LAMBDAS:
        raise ValueError("Unplanned penalty")
    farms = sorted(frame.farm.unique())
    x = design(frame, family, role, farms)
    transform = fitted_transform(x)
    scale = frame.capacity_proxy.to_numpy(float)
    target = (
        frame.target_potential.to_numpy(float) - frame.available_pred.to_numpy(float)
    ) / scale
    q = gate(frame, role)
    if family == "C_augmentation":
        x = np.vstack(
            [design(frame, family, role, farms, np.clip(q + d, 0.0, 1.0)) for d in AUGMENT]
        )
        target = np.tile(target, len(AUGMENT))
        scale = np.tile(scale, len(AUGMENT))
    z = transformed(x, transform)
    weight = scale / scale.sum()
    ridge = np.diag(np.r_[0.0, np.full(z.shape[1] - 1, penalty)])
    beta = np.zeros(z.shape[1])
    # Fixed-iteration IRLS for smooth absolute loss plus an L2 coefficient penalty.
    # Each observation is weighted by its frozen scale to approximate pooled MW MAE.
    for _ in range(ITERATIONS):
        robust = weight / np.sqrt((target - z @ beta) ** 2 + SMOOTHING**2)
        lhs = z.T @ (robust[:, None] * z) + ridge
        beta = np.linalg.solve(lhs, z.T @ (robust * target))
    if not np.isfinite(beta).all():
        raise ValueError("Non-finite residual fit")
    return dict(
        version=VERSION,
        family=family,
        role=role,
        penalty=penalty,
        farms=farms,
        transform=transform,
        beta=beta,
        cutoff=str(pd.Timestamp(cutoff)),
        latest_label=str(frame.target_available_at.max()),
        rows=len(frame),
        target="(future total available - chronological available forecast)/frozen scale",
        augmentation=tuple(AUGMENT) if family == "C_augmentation" else (0.0,),
    )


def predict(model, frame, strength, override=None):
    if strength not in STRENGTHS:
        raise ValueError("A positive preplanned residual strength is required")
    x = design(frame, model["family"], model["role"], model["farms"], override)
    residual = np.clip(transformed(x, model["transform"]) @ model["beta"], -CAP, CAP)
    out = np.maximum(
        frame.available_pred.to_numpy(float)
        + strength * frame.capacity_proxy.to_numpy(float) * residual,
        0.0,
    )
    if not np.isfinite(out).all():
        raise ValueError("Non-finite available prediction")
    return out
