"""Recompute the plotted statistics from one row per farm and forecast origin."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

FARMS = [f"F{i}" for i in range(1, 6)]
MONTHS = [2, 3, 4, 5]
MODELS = [
    "Persistence",
    "Ridge",
    "MixLinear",
    "MDLinear",
    "ExtraTrees",
    "HGB",
    "CatBoost",
    "LightGBM",
    "MLP",
    "GRU",
    "PatchTST",
]
SPACES = ["own", "aggregate", "spatial"]
CORRECTIONS = ["raw", "current", "full"]
SCORE = {
    "current_score": "current_score",
    "raw_score": "pooled_score",
    "calibrated_probability": "future_probability",
}
FIXED = {
    "CatBoost": "CatBoost",
    "Linear": "linear_forecast_A",
    "Neural": "neural_forecast_A",
    "CRHF": "CRHF",
}


def errors(rows, field, target="actual_mw"):
    error = rows[field].to_numpy() - rows[target].to_numpy()
    return np.array([np.mean(abs(error)), np.sqrt(np.mean(error**2)), np.mean(error)])


def group_sets(rows):
    for i in FARMS:
        yield "farm", i, rows.loc[rows.farm_id == i]
    for m in MONTHS:
        yield "month", str(m), rows.loc[rows.month == m]


def frame(rows, columns):
    return pd.DataFrame(rows, columns=columns.split())


def paired_intervals(d, baseline, proposed, family):
    """Synchronous target-time block resampling across farms (10,000 replicates)."""
    delta = (d[baseline] - d.actual_mw).abs() - (d[proposed] - d.actual_mw).abs()
    # Blocks start at the first test-origin day, on the shared 15-minute grid.
    start = 96 * (d.issue_slot.min() // 96)
    blocks = (d.target_slot - start) // (7 * 96)
    grouped = delta.groupby(blocks).agg(["sum", "count"])
    draws = np.random.default_rng(20260914).integers(0, len(grouped), size=(10000, len(grouped)))
    sampled = grouped["sum"].to_numpy()[draws].sum(axis=1) / grouped["count"].to_numpy()[draws].sum(
        axis=1
    )
    return np.quantile(sampled, [0.025, 0.975, 0.025 / family, 1 - 0.025 / family]), len(grouped)


def paper_statistics(d):
    result = {}
    group_errors, fixed_errors, adoption = [], [], []
    for scope, key, g in [("overall", "Overall", d), *group_sets(d)]:
        for state, subset in g.groupby("target_state"):
            benchmarks = np.stack([errors(subset, m)[:2] for m in MODELS])
            best_idx = benchmarks.argmin(axis=0)
            best = benchmarks.min(axis=0)
            err = errors(subset, "CRHF")[:2]
            gain = 100 * (best - err) / best
            group_errors.append(
                [
                    scope,
                    key,
                    "curtailed" if state else "uncurtailed",
                    len(subset),
                    err[0],
                    MODELS[best_idx[0]],
                    best[0],
                    gain[0],
                    err[1],
                    MODELS[best_idx[1]],
                    best[1],
                    gain[1],
                ]
            )
        if scope != "overall":
            ref = errors(g, "CatBoost")[:2]
            for label, model in FIXED.items():
                err = errors(g, model)[:2]
                fixed_errors.append(
                    [
                        "site" if scope == "farm" else scope,
                        key,
                        len(g),
                        label,
                        *err,
                        *(100 * (ref - err) / ref),
                    ]
                )
    result["05_group_errors"] = frame(
        group_errors,
        "scope group state n crhf_mae best_mae_method best_mae mae_gain_pct crhf_rmse best_rmse_method best_rmse rmse_gain_pct",
    )
    result["06_group_errors"] = frame(
        fixed_errors,
        "group_type group_label n model MAE_MW RMSE_MW MAE_improvement_pct RMSE_improvement_pct",
    )

    scores, bins, calibration, distribution = [], [], [], []
    for name, g in [("Overall", d), *d.groupby("farm_id")]:
        y = g.target_state.to_numpy()
        for label, col in SCORE.items():
            p = g[col].to_numpy()
            scores.append(
                [name, label, len(g), y.sum(), average_precision_score(y, p), ((p - y) ** 2).mean()]
            )
    result["07_scores"] = frame(scores, "group score n positive_count AP Brier")
    for name, g in [
        ("All records", d),
        ("Currently non-curtailed", d[d.current_state == 0]),
        ("Currently curtailed", d[d.current_state == 1]),
    ]:
        y = g.target_state.to_numpy()
        for label, col in SCORE.items():
            p = g[col].to_numpy()
            ids = np.minimum(np.floor(10 * p).astype(int), 9)
            ece = 0.0
            for i in range(10):
                mask = ids == i
                cnt = mask.sum()
                a = p[mask].mean() if cnt else np.nan
                b = y[mask].mean() if cnt else np.nan
                if cnt:
                    ece += abs(a - b) * cnt / len(g)
                bins.append(
                    [
                        name,
                        label,
                        i,
                        i / 10,
                        (i + 1) / 10,
                        cnt,
                        cnt / len(g),
                        a,
                        b,
                        label == "calibrated_probability" and 0 < cnt <= 0.01 * len(g),
                    ]
                )
            calibration.append([name, label, len(g), ((p - y) ** 2).mean(), ece, y.sum()])
    result["08_bins"] = frame(
        bins,
        "group score bin lower upper n fraction_of_condition_records mean_prediction observed_frequency is_nonempty_low_count_q_bin",
    )
    result["08_calibration"] = frame(
        calibration, "group score n BS ECE_10_equal_width_bins target_positive_count"
    )

    transition = d.current_state.astype(str) + d.target_state.astype(str)
    counts, transition_error = [], []
    for s, g in d.groupby(transition):
        for label, col in {
            "current_score": "current_score",
            "calibrated_probability": "future_probability",
            "delta_q": "delta_q",
        }.items():
            p = g[col].to_numpy()
            distribution.append(
                [
                    s[0] + "->" + s[1],
                    label,
                    len(g),
                    *np.quantile(p, [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]),
                    p.mean(),
                ]
            )
        counts.append([s, len(g), 100 * len(g) / len(d)])
        ref = errors(g, "neural_forecast_A")[:2]
        for label, col in [
            ("linear", "linear_forecast_A"),
            ("neural", "neural_forecast_A"),
            ("CRHF", "CRHF"),
        ]:
            e = errors(g, col)
            transition_error.append([s, label, len(g), *e, *(ref - e[:2])])
    result["09_scores"] = frame(
        distribution, "transition score n min p05 q1 median q3 p95 max mean"
    )
    result["13_counts"] = frame(counts, "state n share_percent")
    result["13_transitions"] = frame(
        transition_error,
        "state model n MAE RMSE Bias MAE_improvement_over_neural_MW RMSE_improvement_over_neural_MW",
    )

    spatial, relative = [], []
    for space in SPACES:
        for correction in CORRECTIONS:
            spatial.append(
                [
                    space,
                    correction,
                    len(d),
                    *errors(d, f"{space}_{correction}", "observed_available_mw"),
                ]
            )
        for key, g in [*d.groupby("farm_id"), ("Pooled", d)]:
            current = errors(g, space + "_current", "observed_available_mw")
            future = errors(g, space + "_full", "observed_available_mw")
            for k, metric in enumerate(["MAE", "RMSE"]):
                diff = current[k] - future[k]
                relative.append(
                    [
                        metric,
                        key,
                        len(g),
                        space,
                        current[k],
                        future[k],
                        diff,
                        100 * diff / current[k],
                    ]
                )
    result["10_available"] = frame(spatial, "space correction n MAE_MW RMSE_MW Bias_MW")
    result["S2_available"] = frame(
        relative, "metric group n space current_MW future_MW improvement_MW improvement_percent"
    )

    candidates = []
    for name, a, b in [
        ("Linear correction", "linear_current_A", "linear_forecast_A"),
        ("Neural correction", "neural_current_A", "neural_forecast_A"),
        ("CRHF", "final_current_A", "CRHF"),
    ]:
        ea, eb = errors(d, a), errors(d, b)
        candidates.append([name, len(d), a, b, ea[0], eb[0], ea[1], eb[1], *(ea - eb)[:2]])
    result["11_candidates"] = frame(
        candidates,
        "method n current_column forecast_column current_MAE_MW forecast_MAE_MW current_RMSE_MW forecast_RMSE_MW MAE_improvement_MW RMSE_improvement_MW",
    )
    grouping = [*group_sets(d), *(("context", k, g) for k, g in d.groupby("risk_regime"))]
    for scope, key, g in grouping:
        cnt = (
            g["action"]
            .value_counts()
            .reindex(["base", "linear", "neural"], fill_value=0)
            .to_numpy()
        )
        adoption.append(
            ["site" if scope == "farm" else scope, key, len(g), *cnt, *(100 * cnt / len(g))]
        )
    result["12_adoption"] = frame(
        adoption,
        "group label n action_0_count action_1_count action_2_count action_0_percent action_1_percent action_2_percent",
    )

    full = errors(d, "CRHF")
    variants, bias, bias_group = [], [], []
    for role, field in [
        ("full", "CRHF"),
        ("no_available", "final_current_A"),
        ("II_current", "final_current_II"),
        ("observed", "final_observed"),
    ]:
        e = errors(d, field)
        variants.append([role, field, len(d), *e[:2], *(e - full)[:2]])
    result["S3_auxiliary"] = frame(
        variants,
        "saved_role field n MAE_MW RMSE_MW MAE_full_improvement_MW RMSE_full_improvement_MW",
    )
    for field in ["CRHF", "bias_low", "bias_high"]:
        e = errors(d, field)
        diff = e - full
        bias.append(
            [field, len(d), *e, diff[0], 100 * diff[0] / full[0], diff[1], 100 * diff[1] / full[1]]
        )
        if field != "CRHF":
            for group, key, g in [
                *(("farm", k, g) for k, g in d.groupby("farm_id")),
                *(("state", k, g) for k, g in d.groupby(transition)),
            ]:
                bias_group.append(
                    [group, key, len(g), field, errors(g, field)[0] - errors(g, "CRHF")[0]]
                )
    result["14_bias"] = frame(
        bias,
        "scenario n MAE_MW RMSE_MW Bias_MW MAE_increase_MW MAE_relative_increase_pct RMSE_increase_MW RMSE_relative_increase_pct",
    )
    result["14_groups"] = frame(bias_group, "group label n scenario MAE_increase_MW")

    overall = []
    ref = errors(d, "Persistence")[:2]
    for model in [*MODELS, "CRHF"]:
        e = errors(d, model)[:2]
        normalized = np.mean(
            [100 * errors(g, model)[:2] / g.farm_scale_mw.iloc[0] for _, g in d.groupby("farm_id")],
            axis=0,
        )
        overall.append([model, *e, len(d), *normalized, *(100 * (ref - e) / ref)])
    result["S1_overall"] = frame(
        overall,
        "model MAE_MW RMSE_MW n NMAE_pct NRMSE_pct MAE_reduction_vs_Persistence_pct RMSE_reduction_vs_Persistence_pct",
    )

    edges = np.arange(-98, 163, 2)
    grid = np.arange(-98, 162.01, 0.25)
    hist = pd.DataFrame(
        {
            "bin_lower_percent": edges[:-1],
            "bin_upper_percent": edges[1:],
            "bin_center_percent": (edges[1:] + edges[:-1]) / 2,
        }
    )
    curves = pd.DataFrame({"error_percent": grid})
    series = {
        name: (100 * (d[name] - d.actual_mw) / d.farm_scale_mw).to_numpy()
        for name in ("CatBoost", "CRHF")
    }
    joined = np.concatenate(list(series.values()))
    bandwidth = (
        0.9
        * min(joined.std(ddof=1), np.subtract(*np.quantile(joined, [0.75, 0.25])) / 1.34)
        * len(d) ** (-0.2)
    )
    for name, values in series.items():
        counts = np.histogram(values, bins=edges)[0]
        hist[name + "_count"] = counts
        hist[name + "_hist_density"] = counts / (len(d) * np.diff(edges))
        # Evaluate Gaussian kernels in chunks to bound memory independently of the grid length.
        density = []
        for points in np.array_split(grid, 66):
            z = (points[:, None] - values) / bandwidth
            density.extend(np.exp(-0.5 * z * z).mean(axis=1) / (np.sqrt(2 * np.pi) * bandwidth))
        curves[name + "_kde_density"] = density
        hist[name + "_kde_density"] = np.interp(hist.bin_center_percent, grid, density)
    result["S1_histogram"] = hist
    result["S1_density"] = curves

    qualified = np.logical_and.reduce(
        [
            (d[f"{m}_valid_days"] >= 3) & (d[f"{m}_record_count"] >= 32) & d[f"{m}_executed"]
            for m in ("linear", "neural")
        ]
    )
    if not np.array_equal(qualified, d.history_qualified):
        raise ValueError("Saved and recomputed history qualification differ")
    adopted = d.action.ne("base")
    cnts = [(qualified & ~adopted).sum(), adopted.sum(), (~qualified).sum()]
    result["S4_adoption"] = frame(
        [
            [k, n, 100 * n / len(d)]
            for k, n in zip(
                ["Qualified history; not adopted", "Correction adopted", "Insufficient history"],
                cnts,
            )
        ],
        "category n percent",
    )
    edges = np.linspace(-0.021, 0.011, 65)
    a = d.loc[qualified, "linear_saved_score"]
    b = d.loc[qualified, "neural_saved_score"]
    mat, x, y = np.histogram2d(a, b, bins=[edges, edges])
    if int(mat.sum()) != int(qualified.sum()):
        raise ValueError("The plotting grid omits valid selection scores")
    result["S4_density"] = frame(
        [[x[i], x[i + 1], y[j], y[j + 1], int(mat[i, j])] for i, j in zip(*np.nonzero(mat))],
        "linear_bin_left linear_bin_right neural_bin_left neural_bin_right n",
    )
    # Preserve the experiment-specific multiplicity families, including the secondary one-pair test.
    for i, row in result["11_candidates"].iterrows():
        family = 3 if row.method == "CRHF" else 2
        ci, blocks = paired_intervals(d, row.current_column, row.forecast_column, family)
        for name, value in zip(
            [
                "MAE_saved_unadjusted_lower_MW",
                "MAE_saved_unadjusted_upper_MW",
                "MAE_saved_adjusted_lower_MW",
                "MAE_saved_adjusted_upper_MW",
                "MAE_original_family_size",
                "MAE_block_count",
                "MAE_saved_gain_MW",
            ],
            [*ci, family, blocks, row.MAE_improvement_MW],
        ):
            result["11_candidates"].loc[i, name] = value
    for i, row in result["S3_auxiliary"].iterrows():
        if row.field == "CRHF":
            continue
        family = 1 if row.field == "final_current_II" else 3
        ci, _ = paired_intervals(d, row.field, "CRHF", family)
        result["S3_auxiliary"].loc[
            i, ["MAE_95_lower_MW", "MAE_95_upper_MW", "original_family_size"]
        ] = [*ci[2:], family]
    for i, row in result["14_bias"].iterrows():
        if row.scenario == "CRHF":
            continue
        ci, _ = paired_intervals(d, row.scenario, "CRHF", 1)
        result["14_bias"].loc[i, ["MAE_95_lower_MW", "MAE_95_upper_MW", "original_family_size"]] = [
            *ci[:2],
            1,
        ]
    return result


KEYS = {
    "05_group_errors": ["scope", "group", "state"],
    "06_group_errors": ["group_type", "group_label", "model"],
    "07_scores": ["group", "score"],
    "08_bins": ["group", "score", "bin"],
    "08_calibration": ["group", "score"],
    "09_scores": ["transition", "score"],
    "10_available": ["space", "correction"],
    "11_candidates": ["method"],
    "12_adoption": ["group", "label"],
    "13_counts": ["state"],
    "13_transitions": ["state", "model"],
    "14_bias": ["scenario"],
    "14_groups": ["group", "label", "scenario"],
    "S1_overall": ["model"],
    "S1_histogram": ["bin_lower_percent"],
    "S1_density": ["error_percent"],
    "S2_available": ["space", "group", "metric"],
    "S3_auxiliary": ["saved_role"],
    "S4_adoption": ["category"],
    "S4_density": ["linear_bin_left", "neural_bin_left"],
}


def compare_statistics(computed, reference_dir):
    """Return computed rows in publication order, plus archived interval columns for rendering."""
    verified = {}
    for name, current in computed.items():
        reference = pd.read_csv(
            Path(reference_dir) / (name + ".csv"),
            float_precision="round_trip",
            dtype={"group": str, "label": str, "state": str, "group_label": str},
        )
        keys = KEYS[name]
        old = reference.set_index(keys)
        new = current.set_index(keys)
        if not old.index.is_unique or not new.index.is_unique or set(old.index) != set(new.index):
            raise ValueError(f"Row identifiers differ for {name}")
        new = new.reindex(old.index)
        for col in new:
            if pd.api.types.is_numeric_dtype(new[col].dtype):
                np.testing.assert_allclose(
                    new[col],
                    old[col],
                    rtol=1e-9,
                    atol=1e-10,
                    equal_nan=True,
                    err_msg=f"{name}: {col}",
                )
            elif not np.array_equal(new[col].values, old[col].values):
                raise ValueError(f"{name}: string field {col} differs")
            old[col] = new[col]
        verified[name] = old.reset_index()
    return verified
