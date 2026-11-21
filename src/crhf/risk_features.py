"""Construct causal classifier inputs, physical scores, and nonnegative opinion-pool weights."""

import numpy as np
import pandas as pd
from .grid import FARMS, SIGNALS
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


VARIANTS = {
    "full": set(),
    "no_cluster": {"cluster"},
    "no_memory": {"memory"},
    "no_physics": {"physics"},
}


BASE_COLUMNS = [
    "farm",
    "Timestamp",
    *SIGNALS,
    "teacher_probability",
    "curtailed",
    "severity",
    "curtailment_mw",
    "capacity_proxy",
    "input_valid",
    "sample_coverage",
    "suspected_outage",
]


RULE_COLUMNS = {
    "state_persistence": "x_state",
    "history16": "m_x_state_mean_16",
    "history96": "m_x_state_mean_96",
    "physics_current": "x_physics_score",
    "residual_current": "r_soft",
    "residual_rate16": "r_flag_mean_16",
    "residual_rate96": "r_flag_mean_96",
    "residual_soft16": "r_soft_mean_16",
    "residual_soft96": "r_soft_mean_96",
}


def build_feature_bank(frame, config):
    """Rebuild on the FULL grid, before any train/score row filtering.

    no_physics removes engineered dispatch/label features, while retaining the
    same raw A/P/D observations. It tests encoding, not absence of dispatch data.
    """
    out = frame[BASE_COLUMNS].sort_values(["farm", "Timestamp"]).reset_index(drop=True).copy()
    tags, generated = {}, {}

    def add(name, values, dependencies=()):
        generated[name] = values
        tags[name] = sorted(set(dependencies))

    step = pd.Timedelta(minutes=int(config["step_minutes"]))
    for farm, g in out.groupby("farm", sort=False):
        if g.Timestamp.duplicated().any() or not g.Timestamp.diff().dropna().eq(step).all():
            raise ValueError(f"{farm}: features require an uncompressed regular time grid")
    cap = out.capacity_proxy
    raw_names = []
    for col in SIGNALS:
        name = "x_" + col
        add(name, out[col] / cap)
        raw_names.append(name)
    minute = out.Timestamp.dt.hour * 60 + out.Timestamp.dt.minute
    day = out.Timestamp.dt.dayofyear
    for name, values in {
        "x_hour_sin": np.sin(2 * np.pi * minute / 1440),
        "x_hour_cos": np.cos(2 * np.pi * minute / 1440),
        "x_year_sin": np.sin(2 * np.pi * day / 365.25),
        "x_year_cos": np.cos(2 * np.pi * day / 365.25),
        "x_farm_code": out.farm.map({f: i for i, f in enumerate(FARMS)}),
    }.items():
        add(name, values)
    tracking = (
        (out.actual_operational - out.potential).abs()
        - (out.actual_operational - out.dispatch_history).abs()
    ) / cap
    physical = {
        "x_physics_score": out.teacher_probability,
        "x_gap": (out.potential - out.dispatch_history).clip(lower=0) / cap,
        "x_tracking": 1
        / (1 + np.exp(-np.clip(tracking / float(config["sigmoid_fraction"]), -40, 40))),
        "x_state": out.curtailed,
        "x_severity": out.severity,
        "x_loss": out.curtailment_mw / cap,
    }
    for name, values in physical.items():
        add(name, values, ["physics"])
    good = out.input_valid.astype(bool)
    total_count = good.groupby(out.Timestamp).transform("sum")
    neighbors = total_count - good.astype(int)
    add("c_count", neighbors, ["cluster"])
    cluster_names = []
    for name in [*raw_names, "x_physics_score", "x_gap", "x_tracking"]:
        values = generated[name].where(good)
        other_mean = pd.Series(np.nan, index=out.index)
        # Sum the other farms directly: total-minus-self can leave roundoff
        # dependence on the excluded farm near a downstream tree boundary.
        for farm in out.farm.unique():
            own = out.farm.eq(farm)
            total = values.loc[~own].groupby(out.loc[~own, "Timestamp"]).sum(min_count=1)
            other_mean.loc[own] = out.loc[own, "Timestamp"].map(total) / neighbors.loc[
                own
            ].replace(0, np.nan)
        cluster = "c_" + name
        add(cluster, other_mean, [*tags[name], "cluster"])
        cluster_names.append(cluster)
    sync = out.teacher_probability.ge(0.5).where(good, False).astype(int)
    add(
        "c_sync",
        (sync.groupby(out.Timestamp).transform("sum") - sync) / neighbors.replace(0, np.nan),
        ["cluster", "physics"],
    )
    cluster_names.append("c_sync")
    # All windows are trailing, inclusive of the completed interval at issue.
    for name in [*raw_names, *physical, *cluster_names]:
        values = pd.Series(generated[name], index=out.index)
        grouped = values.groupby(out.farm, sort=False)
        for window in (4, 8, 16, 96):
            for stat in ("mean", "max"):
                rolled = grouped.transform(
                    lambda s, w=window, op=stat: getattr(s.rolling(w, min_periods=1), op)()
                )
                add(f"m_{name}_{stat}_{window}", rolled, [*tags[name], "memory"])
    for name in raw_names:
        grouped = generated[name].groupby(out.farm, sort=False)
        for lag in (1, 2, 4, 8, 16, 24, 96):
            add(f"m_{name}_lag_{lag}", grouped.shift(lag), ["memory"])
    for name in ("x_physics_score", "x_gap", "c_sync"):
        add(
            f"m_{name}_delta_4",
            generated[name].groupby(out.farm, sort=False).diff(4),
            [*tags[name], "memory"],
        )
    streak = pd.Series(0.0, index=out.index)
    for _, g in out.groupby("farm", sort=False):
        flag = g.curtailed.fillna(0).eq(1)
        length = flag.groupby(flag.ne(flag.shift()).cumsum()).cumcount() + 1
        streak.loc[g.index] = np.where(flag, length, 0)
    add("m_streak", streak, ["physics", "memory"])
    features = pd.DataFrame(generated, index=out.index)
    out = pd.concat([out, features], axis=1)
    # Residual-only comparators retain their original restricted information set.
    residual = ((out.potential - out.actual_operational).clip(lower=0) / cap).clip(0, 1)
    residuals = {
        "r_ratio": residual,
        "r_flag": residual.ge(float(config["gap_fraction"])).astype(float).where(good),
        "r_soft": 1
        / (
            1
            + np.exp(
                -np.clip(
                    (residual - float(config["gap_fraction"])) / float(config["sigmoid_fraction"]),
                    -40,
                    40,
                )
            )
        ),
    }
    for col in ("r_flag", "r_soft"):
        grouped = residuals[col].groupby(out.farm, sort=False)
        for window in (4, 8, 16, 96):
            residuals[f"{col}_mean_{window}"] = grouped.transform(
                lambda s, w=window: s.rolling(w, min_periods=1).mean()
            )
            if col == "r_soft":
                residuals[f"{col}_max_{window}"] = grouped.transform(
                    lambda s, w=window: s.rolling(w, min_periods=1).max()
                )
    out = pd.concat([out, pd.DataFrame(residuals, index=out.index)], axis=1)
    spec = {
        "tags": tags,
        "variants": {
            v: [n for n, dep in tags.items() if not set(dep) & removed]
            for v, removed in VARIANTS.items()
        },
        "residual": [*residuals, "x_hour_sin", "x_hour_cos", "x_year_sin", "x_year_cos"],
        "no_physics_meaning": "Remove engineered physics and label descendants; retain raw A/P/D and common validity mask.",
    }
    return out, spec


def probability_metrics(y, probability):
    y = np.asarray(y, int)
    p = np.asarray(probability, float)
    two = len(np.unique(y)) == 2
    return {
        "AP": float(average_precision_score(y, p)) if two else None,
        "Brier": float(np.mean((p - y) ** 2)),
    }


def make_classifier(family, config, seed):
    iterations = int(config["risk_max_iter"])
    if family == "HGB":
        return HistGradientBoostingClassifier(
            max_iter=iterations,
            learning_rate=0.035,
            max_leaf_nodes=16,
            min_samples_leaf=35,
            l2_regularization=0.04,
            early_stopping=False,
            random_state=seed,
        )
    if family == "Logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
        )
    if family == "LightGBM":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=iterations,
            learning_rate=0.035,
            num_leaves=16,
            min_child_samples=35,
            reg_lambda=0.04,
            random_state=seed,
            n_jobs=int(config["threads"]),
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        )
    if family == "CatBoost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=iterations,
            learning_rate=0.035,
            depth=4,
            loss_function="Logloss",
            random_seed=seed,
            thread_count=int(config["threads"]),
            verbose=False,
            allow_writing_files=False,
        )
    raise ValueError(family)


def predict_learner(learner, frame):
    x = learner["transform"].apply(frame)
    model = learner["model"]
    classes = list(model.classes_)
    return model.predict_proba(x)[:, classes.index(1)] if 1 in classes else np.zeros(len(frame))


def rule_predictions(frame, prior):
    return {
        name: frame[col].fillna(prior).clip(0, 1).to_numpy(float)
        for name, col in RULE_COLUMNS.items()
    }


def fit_fusion(y, predictions, members):
    weight = {}
    for name in members:
        ap = probability_metrics(y, predictions[name])["AP"]
        weight[name] = max(ap if ap is not None else 0.0, 1e-6)
    total = sum(weight.values())
    return {name: value / total for name, value in weight.items()}


def apply_fusion(predictions, weights):
    return sum(weight * predictions[name] for name, weight in weights.items())
