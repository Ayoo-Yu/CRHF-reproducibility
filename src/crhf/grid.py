"""Attach one-hour targets by slot and enforce the chronological split."""

import math
import pandas as pd


FARMS = ["F2", "F1", "F3", "F5", "F4"]


SIGNALS = ["actual_operational", "potential", "dispatch_history"]


LABELS = ["curtailed", "teacher_probability", "severity", "curtailment_mw"]


def attach_targets(frame, horizon, config):
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("Horizon must be a positive integer")
    out = frame.copy()
    delay = pd.Timedelta(minutes=int(config.get("observation_delay_minutes", 0)))
    if delay < pd.Timedelta(0):
        raise ValueError("Observation delay must be non-negative")
    out["observation_end"] = out.Timestamp
    out["observation_available_at"] = out.Timestamp + delay
    # Stay on the issue grid when reporting delay is not a multiple of 15 min.
    # Waiting to the next grid point is conservative and leaves fixed leads exact.
    rounded_delay = pd.Timedelta(
        minutes=math.ceil(delay.total_seconds() / 60 / int(config["step_minutes"]))
        * int(config["step_minutes"])
    )
    out["Timestamp"] += rounded_delay
    out["target_timestamp"] = out.Timestamp + pd.Timedelta(
        minutes=horizon * int(config["step_minutes"])
    )
    target_cols = SIGNALS + LABELS + ["input_valid"]
    lookup = frame[["farm", "Timestamp", *target_cols]].rename(
        columns={"Timestamp": "target_timestamp", **{c: f"target_{c}" for c in target_cols}}
    )
    out = out.merge(lookup, on=["farm", "target_timestamp"], how="left", validate="one_to_one")
    out["target_available_at"] = out.target_timestamp + delay
    out["horizon"] = horizon
    out["sample_valid"] = out.input_valid & out.target_input_valid.fillna(False).astype(bool)
    return out


def assign_phases(frame, config):
    out = frame.copy()
    calibration, train, validation, end = [
        pd.Timestamp(config[c])
        for c in ("calibration_end", "train_end", "validation_end", "test_end")
    ]
    if not calibration < train < validation < end:
        raise ValueError("Time boundaries must be strictly increasing")
    out["phase"] = "excluded"
    out["evaluation_eligible"] = False
    for name, start, finish in [
        ("train", calibration, train),
        ("validation", train, validation),
        ("test", validation, end),
    ]:
        issue_mask = out.Timestamp.ge(start) & out.Timestamp.lt(finish) & out.input_valid
        eligible = issue_mask & out.target_available_at.le(finish) & out.sample_valid
        # Training needs an observed target. In validation/test, ISSUE predictions
        # even if their future outcome is missing: target validity cannot decide
        # whether an online action occurs. Only scoring is masked retrospectively.
        mask = eligible if name == "train" else issue_mask
        out.loc[mask, "phase"] = name
        out.loc[eligible, "evaluation_eligible"] = True
    return out
