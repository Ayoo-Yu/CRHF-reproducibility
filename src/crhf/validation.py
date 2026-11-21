"""Validate chronological joins and compare per-issue forecasts and decisions."""

import numpy as np


KEYS = ["farm", "Timestamp", "target_timestamp"]


def validate_keys(frame, name):
    if not len(frame) or not set(KEYS).issubset(frame):
        raise ValueError(f"{name}: empty frame or missing forecast keys")
    if frame[KEYS].isna().any().any() or frame[KEYS].duplicated().any():
        raise ValueError(f"{name}: missing or duplicate forecast keys")


def align_reference(frame, reference):
    """Allow extra historical reference rows, but require every requested key."""
    validate_keys(frame, "issued frame")
    validate_keys(reference, "reference")
    merged = frame[KEYS].merge(
        reference, on=KEYS, how="left", validate="one_to_one", indicator=True, sort=False
    )
    if not merged["_merge"].eq("both").all():
        missing = merged.loc[merged["_merge"].ne("both"), KEYS].head().to_dict("records")
        raise ValueError(f"Reference is missing issued keys: {missing}")
    return merged.drop(columns="_merge").reset_index(drop=True)


def validate_predictions(frame, predictions):
    validate_keys(frame, "issued frame")
    validate_keys(predictions, "predictions")
    if not frame[KEYS].reset_index(drop=True).equals(predictions[KEYS].reset_index(drop=True)):
        raise ValueError("Prediction keys/order do not match issued rows")
    if predictions.columns.duplicated().any():
        raise ValueError("Duplicate prediction columns")
    for name in predictions:
        if name in KEYS:
            continue
        values = predictions[name].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{name}: non-finite predictions or actions")
        if name.endswith("_action"):
            if not np.isin(values, [0, 1, 2]).all():
                raise ValueError(f"{name}: invalid action")
        elif name == "risk_q":
            if ((values < 0) | (values > 1)).any():
                raise ValueError("risk_q: values outside [0, 1]")
        elif (values < 0).any():
            raise ValueError(f"{name}: negative power")


def compare_series(new, old, name, atol=3e-5, action=False):
    left, right = np.asarray(new, dtype=float), np.asarray(old, dtype=float)
    if left.shape != right.shape or left.ndim != 1 or not left.size:
        raise ValueError(f"{name}: incompatible or empty comparison arrays")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError(f"{name}: non-finite comparison values")
    error = float(np.max(np.abs(left - right)))
    if action:
        if not np.isin(left, [0, 1, 2]).all() or not np.isin(right, [0, 1, 2]).all():
            raise ValueError(f"{name}: invalid action")
        differences = int(np.count_nonzero(left != right))
        if differences:
            raise ValueError(f"{name}: {differences} changed actions")
        return dict(series=name, action_differences=0)
    if not np.isfinite(atol) or atol <= 0 or error >= atol:
        raise ValueError(f"{name}: error {error} exceeds tolerance {atol}")
    return dict(
        series=name,
        max_abs_error=error,
        unit="probability" if name == "risk_q" else "MW",
        tolerance=atol,
    )


def assign_shared(predictions, name, values):
    """Do not silently overwrite a supposedly common backbone across roles."""
    value = np.asarray(values, dtype=float)
    if value.shape != (len(predictions),) or not np.isfinite(value).all():
        raise ValueError(f"{name}: invalid shared forecast")
    if name in predictions and not np.array_equal(predictions[name].to_numpy(), value):
        raise ValueError(f"{name}: shared upstream changed between ablation roles")
    predictions[name] = value
