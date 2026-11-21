"""Additional Stage 2 factors and Stage 3 comparisons reported in the paper."""

import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from .available_residual import fit as fit_center, predict as predict_center
from .backbones import fit_cat, predict_cat
from .inference import saved_predict
from .pipeline import (
    KEYS,
    EPOCH,
    STEP,
    ROLES,
    add_role_predictions,
    compare,
    feedback,
    public_keys,
    reconstruct,
    static_frame,
)
from .protocol import FOLDS, month_mask, training_mask
from .risk_bridge import build_risk_stream, predict_risk
from .spatial import spatial_features


def forward_risk(data, bridges_path=None):
    frame = data["frame"].copy()
    if bridges_path is None:
        history = data["all_frame"]
        phases = [history.loc[history.phase.eq(p)].copy() for p in ("train", "validation", "test")]
        with threadpool_limits(limits=8):
            *_, bridges, _ = build_risk_stream(
                *phases, data["risk_spec"], data["config"], 4, print
            )
    else:
        bridges = joblib.load(bridges_path)
    frame["risk_q"] = np.nan
    frame["risk_fold_start"] = pd.NaT
    for start in FOLDS:
        selected = month_mask(frame, start)
        bundle = bridges[str(pd.Timestamp(start))] if start < "1997-12-01" else bridges["final"]
        _, calibrated = predict_risk(bundle, static_frame(frame.loc[selected]))
        frame.loc[selected, "risk_q"] = calibrated
        frame.loc[selected, "risk_fold_start"] = pd.Timestamp(bundle["cutoff"])
    return frame


def available_experiment(out, bridges_path=None):
    """Fit the nine combinations of spatial input and available-power correction."""
    data = reconstruct(sequence=False)
    frame = forward_risk(data, bridges_path)
    schema = data["schema"]["columns"]
    spatial = spatial_features(data["source"], frame).astype(np.float32)
    all_features = pd.concat([frame[schema["aggregate"]], spatial], axis=1)
    current = frame.potential.to_numpy(float)
    scale = frame.capacity_proxy.to_numpy(float)
    target = (frame.target_potential.to_numpy(float) - current) / scale
    out_frame = frame[KEYS].copy()
    for setting in ("own", "aggregate", "spatial"):
        features = all_features[schema[setting]].to_numpy(np.float32)
        correction = frame.copy()
        correction["available_pred"] = np.nan
        correction["risk_probability"] = frame.risk_q
        correction["available_fold_start"] = pd.NaT
        correction["available_training_latest_label"] = pd.NaT
        for variant in ("raw", "full", "current"):
            out_frame[f"{setting}_{variant}"] = np.nan
        for start in FOLDS:
            train = training_mask(frame, start)
            predict = month_mask(frame, start)
            model = fit_cat(features[train], target[train], scale[train], "available")
            raw = predict_cat(model, features[predict], current[predict], scale[predict])
            out_frame.loc[predict, f"{setting}_raw"] = raw
            correction.loc[predict, "available_pred"] = raw
            correction.loc[predict, "available_fold_start"] = pd.Timestamp(start)
            correction.loc[predict, "available_training_latest_label"] = frame.loc[
                train, "target_available_at"
            ].max()
            for variant, risk in (("full", "cal"), ("current", "current")):
                if start == FOLDS[0]:
                    values = raw
                else:
                    select = train & frame.Timestamp.ge(FOLDS[0]).to_numpy()
                    select &= (
                        correction.available_pred.notna().to_numpy()
                        & correction.risk_q.notna().to_numpy()
                    )
                    residual_model = fit_center(
                        correction.loc[select].reset_index(drop=True),
                        "A_innovation",
                        risk,
                        0.1,
                        start,
                    )
                    values = predict_center(residual_model, correction.loc[predict], 0.5)
                out_frame.loc[predict, f"{setting}_{variant}"] = values
            print(
                f"Available power, {setting}, model cutoff slot {(pd.Timestamp(start) - EPOCH) // STEP}",
                flush=True,
            )
    out_frame["observed_available_mw"] = frame.target_potential
    test = out_frame.loc[frame.Timestamp.ge(FOLDS[-1])]
    public_keys(test).to_csv(out / "available_predictions.csv.gz", index=False)


def diagnostic_experiment(out, bundle_path, quick=False):
    """Replay fixed candidates, substitutions and the two prescribed availability shifts."""
    torch.set_num_threads(2)
    data = reconstruct()
    all_rows = data["frame"]
    take = all_rows.Timestamp.ge(FOLDS[-1])
    if quick:
        take &= all_rows.Timestamp.lt("1998-02-03")
    test = all_rows.loc[take].reset_index(drop=True)
    pending = feedback(
        all_rows.loc[all_rows.Timestamp.ge("1998-01-01") & all_rows.Timestamp.lt(FOLDS[-1])]
    )
    facts = feedback(test)
    sequence = data["sequence"][take]
    bundle = joblib.load(bundle_path)
    result = test[KEYS].copy()
    role_forecasts = test[KEYS].copy()
    result["current_score"] = test.teacher_probability
    observed = static_frame(test)
    result["pooled_score"], result["future_probability"] = predict_risk(
        bundle["risk_model"], observed
    )
    plans = [
        ("full", 0.0, "CRHF"),
        ("no_available", 0.0, "final_current_A"),
        ("II_current", 0.0, "final_current_II"),
        ("observed", 0.0, "final_observed"),
        ("full", -0.05, "bias_low"),
        ("full", 0.05, "bias_high"),
    ]
    for role, shift, column in plans:
        print(f"Computing {column} on {len(test):,} issue rows.", flush=True)
        replay = saved_predict(
            bundle,
            observed,
            sequence,
            data["source"],
            facts,
            role=role,
            pending_feedback=pending,
            bias=shift,
        )
        result[column] = replay["predictions"][:, 0]
        if shift == 0:
            add_role_predictions(role_forecasts, role, replay)
        if shift == 0 and role in ("full", "no_available"):
            suffix = "forecast_A" if role == "full" else "current_A"
            result["linear_" + suffix] = replay["tools"][:, 0, 1]
            result["neural_" + suffix] = replay["tools"][:, 0, 2]
        if shift == 0 and role == "full":
            result["DirectSpatial"] = replay["static"]
            result["B"] = replay["online"]["predictions"]
            result["action"] = np.array(["base", "linear", "neural"])[replay["actions"][:, 0]]
            for i, name in enumerate(("linear", "neural"), start=1):
                result[name + "_record_count"] = replay["counts"][:, 0, i]
                result[name + "_valid_days"] = replay["days"][:, 0, i]
                result[name + "_saved_score"] = replay["scores"][:, 0, i]
                result[name + "_executed"] = replay["executed"][:, 0, i]
    public_keys(result).to_csv(out / "diagnostics.csv.gz", index=False)
    public_keys(role_forecasts).to_csv(out / "predictions.csv.gz", index=False)
    compare(out, test, role_forecasts, ROLES)
