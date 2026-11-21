"""Train time-separated classifiers, pooling weights, and monotone Platt calibration."""

import numpy as np
import pandas as pd
from .risk_features import (
    apply_fusion,
    fit_fusion,
    make_classifier,
    predict_learner,
    rule_predictions,
)
from .transform import FeatureTransform
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression


def logit(probability):
    p = np.clip(np.asarray(probability, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def raw_risk(bundle, frame):
    predictions = rule_predictions(frame, bundle["prior"])
    predictions.update({n: predict_learner(m, frame) for n, m in bundle["models"].items()})
    return np.clip(apply_fusion(predictions, bundle["weights"]), 0, 1)


def predict_risk(bundle, frame):
    raw = raw_risk(bundle, frame)
    calibrated = (
        bundle["calibrator"].predict_proba(logit(raw))[:, 1]
        if bundle["calibrator"] is not None
        else raw.copy()
    )
    return raw, calibrated


def fit_risk_before(train, cutoff, spec, config, horizon, progress=print):
    cutoff = pd.Timestamp(cutoff)
    fit_end, weight_end = cutoff - pd.Timedelta(days=14), cutoff - pd.Timedelta(days=7)
    fit = train.loc[train.Timestamp.lt(fit_end) & train.target_available_at.le(fit_end)]
    weight = train.loc[
        train.Timestamp.ge(fit_end)
        & train.Timestamp.lt(weight_end)
        & train.target_available_at.le(weight_end)
    ]
    calibrate = train.loc[
        train.Timestamp.ge(weight_end)
        & train.Timestamp.lt(cutoff)
        & train.target_available_at.le(cutoff)
    ]
    if min(len(fit), len(weight), len(calibrate)) < 16:
        raise ValueError(
            f"Insufficient past-only risk data at {cutoff}: {len(fit)}/{len(weight)}/{len(calibrate)}"
        )
    assert fit.target_available_at.max() <= fit_end
    assert weight.target_available_at.max() <= weight_end
    assert calibrate.target_available_at.max() <= cutoff
    if horizon <= 4:
        members = ["hgb_no_cluster", "logistic_full", "hgb_full"]
        if horizon > 1:
            members += ["history16", "history96", "state_persistence", "physics_current"]
        architecture = "restored_horizon_fusion"
    else:
        members = ["history16", "history96", "physics_current", "hgb_full"]
        architecture = "restored_residual_fusion"
    y = fit.target_curtailed.to_numpy(int)
    bundle = {
        "models": {},
        "prior": float(y.mean()),
        "calibrator": None,
        "cutoff": str(cutoff),
        "architecture": architecture,
        "horizon": horizon,
    }
    needed = [n for n in members if n.startswith(("hgb_", "logistic_"))]
    for name in needed:
        family = "HGB" if name.startswith("hgb") else "Logistic"
        variant = "no_cluster" if name.endswith("no_cluster") else "full"
        transform = FeatureTransform.fit(fit, spec["variants"][variant])
        model = (
            make_classifier(family, config, int(config["seed"]) + horizon)
            if len(np.unique(y)) > 1
            else DummyClassifier(strategy="prior")
        )
        model.fit(transform.apply(fit), y)
        bundle["models"][name] = {"model": model, "transform": transform}
    weight_scores = rule_predictions(weight, bundle["prior"])
    weight_scores.update({n: predict_learner(m, weight) for n, m in bundle["models"].items()})
    bundle["weights"] = fit_fusion(weight.target_curtailed.to_numpy(int), weight_scores, members)
    raw = raw_risk(bundle, calibrate)
    y_cal = calibrate.target_curtailed.to_numpy(int)
    calibration = "identity_degenerate_window"
    if len(np.unique(y_cal)) == 2 and np.std(raw) > 1e-8:
        # No class balancing: the predicted value must retain probability meaning.
        platt = LogisticRegression(C=1.0, max_iter=500).fit(logit(raw), y_cal)
        if float(platt.coef_[0, 0]) >= 0:
            bundle["calibrator"] = platt
            calibration = "nonnegative_slope_platt"
        else:
            calibration = "identity_negative_slope"
    meta = {
        "issue_block_start": str(cutoff),
        "architecture": architecture,
        "fit_end": str(fit_end),
        "weight_end": str(weight_end),
        "calibration_end": str(cutoff),
        "rows": {"fit": len(fit), "weight": len(weight), "calibration": len(calibrate)},
        "latest_label": {
            "fit": str(fit.target_available_at.max()),
            "weight": str(weight.target_available_at.max()),
            "calibration": str(calibrate.target_available_at.max()),
        },
        "weights": bundle["weights"],
        "calibration": calibration,
        "calibration_positive_rows": int(y_cal.sum()),
        "calibration_prevalence": float(y_cal.mean()),
        "calibration_slope": float(bundle["calibrator"].coef_[0, 0])
        if bundle["calibrator"] is not None
        else None,
        "past_only_parameters": True,
    }
    progress(
        f"h={horizon}: risk cutoff={cutoff.date()}, fit/weight/cal={len(fit)}/{len(weight)}/{len(calibrate)}, {calibration}"
    )
    return bundle, meta


def build_risk_stream(train, validation, test, spec, config, horizon, progress=print):
    starts = [pd.Timestamp(s) for s in config["risk_oof_starts"]]
    end = pd.Timestamp(config["train_end"])
    out_parts, bundles, metadata = [], {}, []

    def apply(frame, bundle):
        out = frame.copy()
        out["risk_raw"], out["risk_probability"] = predict_risk(bundle, frame)
        out["risk_fold_start"] = pd.Timestamp(bundle["cutoff"])
        assert out.Timestamp.ge(out.risk_fold_start).all()
        return out

    for start, finish in zip(starts, [*starts[1:], end]):
        future = train.loc[train.Timestamp.ge(start) & train.Timestamp.lt(finish)]
        if future.empty:
            continue
        bundle, meta = fit_risk_before(train, start, spec, config, horizon, progress)
        out_parts.append(apply(future, bundle))
        bundles[str(start)] = bundle
        metadata.append(meta)
    if not out_parts:
        raise ValueError("No chronological OOF rows")
    final, meta = fit_risk_before(train, end, spec, config, horizon, progress)
    metadata.append(meta)
    bundles["final"] = final
    # One common frozen upstream mapping for OUTER validation AND test.
    return (
        pd.concat(out_parts, ignore_index=True),
        apply(validation, final),
        apply(test, final),
        bundles,
        metadata,
    )
