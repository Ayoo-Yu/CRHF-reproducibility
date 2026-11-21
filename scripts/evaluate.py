"""Recalculate the paper's stage-specific metrics and paired uncertainty intervals."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd



ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "results" / "reference"
BASELINES = [
    "Persistence", "Ridge", "MixLinear", "MDLinear", "ExtraTrees", "HGB",
    "CatBoost", "LightGBM", "MLP", "GRU", "PatchTST",
]
SEEDED = ("MLP", "GRU", "PatchTST", "MixLinear", "MDLinear")
SEEDS = (20260913, 20260914, 20260915)
BOOTSTRAP_SEED = 20260914
SLOTS_PER_DAY = 96
TEST_START_SLOT = 20640


def metrics(group, actual, forecast):
    error = (group[forecast] - group[actual]).to_numpy()
    return {
        "n": len(group),
        "MAE_MW": np.abs(error).mean(),
        "RMSE_MW": np.sqrt(np.mean(error ** 2)),
        "bias_MW": error.mean(),
    }


def groups(frame):
    yield "overall", "all", frame
    for label in ["farm_id", "month", "current_state", "target_state", "risk_regime", "action"]:
        for value, group in frame.groupby(label, sort=True):
            yield label, str(value), group
    for (origin, target), group in frame.groupby(["current_state", "target_state"]):
        yield "observed_transition", f"{origin}{target}", group


def power_metrics(frame):
    models = BASELINES + [
        "DirectSpatial", "CatBoostMatchedLocal", "CatBoostFixedSeed", "B", "CRHF",
        "linear_forecast_A", "neural_forecast_A", "linear_current_A", "neural_current_A",
        "final_current_A", "final_current_II", "final_observed", "bias_low", "bias_high",
    ]
    available = [f"{spatial}_{correction}" for spatial in ["own", "aggregate", "spatial"]
                 for correction in ["raw", "current", "full"]]
    rows = []
    for key, value, group in groups(frame):
        for target, methods in [("actual_mw", models), ("observed_available_mw", available)]:
            for method in methods:
                rows.append({"group_type": key, "group": value, "outcome": target,
                             "method": method, **metrics(group, target, method)})
    return pd.DataFrame(rows)


def average_precision_score(labels, scores):
    # Evaluate precision at the end of each group of ties, weighting by recall increments.
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    if not ranked.sum():
        return 0.0
    ordered_scores = scores[order]
    ends = np.flatnonzero(np.r_[ordered_scores[1:] != ordered_scores[:-1], True])
    cumulative_positive = np.cumsum(ranked)[ends]
    precision = cumulative_positive / (ends + 1)
    recall_increment = np.diff(np.r_[0, cumulative_positive]) / ranked.sum()
    return float(np.sum(precision * recall_increment))


def risk_metrics(frame):
    rows, bins = [], []
    for key, value, group in groups(frame):
        if key not in ("overall", "farm_id", "current_state"):
            continue
        y = group.target_state.to_numpy()
        for method in ["current_score", "pooled_score", "future_probability"]:
            p = group[method].to_numpy()
            indices = np.minimum(np.floor(p * 10).astype(int), 9)
            ece = 0.0
            for i in range(10):
                mask = indices == i
                count = mask.sum()
                mean_p = p[mask].mean() if count else np.nan
                mean_y = y[mask].mean() if count else np.nan
                if count:
                    ece += count * abs(mean_p - mean_y) / len(group)
                bins.append({"group_type": key, "group": value, "method": method,
                             "bin": i, "n": count, "mean_probability": mean_p, "frequency": mean_y})
            rows.append({"group_type": key, "group": value, "method": method, "n": len(group),
                         "positives": y.sum(), "AP": average_precision_score(y, p),
                         "BS": np.mean((p - y) ** 2), "ECE": ece})
    return pd.DataFrame(rows), pd.DataFrame(bins)


def paired_intervals(frame):
    # Contrasts and multiplicity families are specified in Supplementary Section D.
    # A positive gain means the second member has smaller absolute errors.
    pairs = [
        ("base_plus_gate", "B", "CRHF", 3),
        ("substitute_current_available", "final_current_A", "CRHF", 3),
        ("substitute_both", "final_observed", "CRHF", 3),
        ("substitute_stage2_risk", "final_current_II", "CRHF", 1),
        ("linear_available_forecast", "linear_current_A", "linear_forecast_A", 2),
        ("neural_available_forecast", "neural_current_A", "neural_forecast_A", 2),
        ("negative_available_shift", "CRHF", "bias_low", 1),
        ("positive_available_shift", "CRHF", "bias_high", 1),
        ("matched_catboost_spatial_input", "CatBoostMatchedLocal", "DirectSpatial", 1),
    ]
    block_id = (frame.target_slot.to_numpy() - TEST_START_SLOT) // (7 * SLOTS_PER_DAY)
    ids, index = np.unique(block_id, return_inverse=True)
    if len(ids) != 15 or not np.array_equal(ids, np.arange(15)):
        raise ValueError("The paired analysis expects the 15 common test blocks.")
    counts = np.bincount(index)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(ids), size=(10_000, len(ids)))
    denominators = counts[draws].sum(axis=1)
    rows = []
    for name, reference, proposed, family_size in pairs:
        delta = (frame[reference] - frame.actual_mw).abs() - (frame[proposed] - frame.actual_mw).abs()
        sums = np.bincount(index, weights=delta.to_numpy(), minlength=len(ids))
        samples = sums[draws].sum(axis=1) / denominators
        q = np.quantile(samples, [0.025, 0.975, 0.025 / family_size, 1 - 0.025 / family_size])
        rows.append({"contrast": name, "reference": reference, "proposed": proposed,
                     "n": len(frame), "blocks": len(ids), "replicates": len(samples),
                     "family_size": family_size, "gain_MW": delta.mean(), "lower_MW": q[0],
                     "upper_MW": q[1], "adjusted_lower_MW": q[2], "adjusted_upper_MW": q[3]})
    return pd.DataFrame(rows)


def subset_metrics(frame):
    active = frame.action != "base"
    hh = frame.risk_regime == "HH"
    sets = {"all": np.ones(len(frame), dtype=bool), "all_corrected": active, "all_HH": hh,
            "corrected_HH": active & hh, "corrected_non_HH": active & ~hh}
    rows = []
    for label, mask in sets.items():
        group = frame.loc[mask]
        base, full = [metrics(group, "actual_mw", method) for method in ("B", "CRHF")]
        row = {"subset": label, "n": len(group)}
        for measure in ("MAE", "RMSE"):
            field = f"{measure}_MW"
            gain = base[field] - full[field]
            row.update({f"base_{field}": base[field], f"CRHF_{field}": full[field],
                        f"gain_{field}": gain, f"gain_{measure}_pct": 100 * gain / base[field]})
        rows.append(row)
    return pd.DataFrame(rows)


def verify(frame, seed_file, intervals, subsets, risks):
    keys = ["farm_id", "issue_slot", "target_slot"]
    if frame.duplicated(keys).any() or not (frame.target_slot - frame.issue_slot).eq(4).all():
        raise ValueError("Forecast keys are duplicated or have the wrong horizon.")
    np.testing.assert_array_equal(frame.groupby("farm_id", sort=True).size(), [8226, 8293, 9228, 9233, 8741])
    np.testing.assert_array_equal(frame.loc[frame.action == "base", "B"], frame.loc[frame.action == "base", "CRHF"])
    seeds = pd.read_csv(seed_file, float_precision="round_trip").set_index(keys)
    final = frame.set_index(keys)
    if not seeds.index.is_unique or not seeds.index.sort_values().equals(final.index.sort_values()):
        raise ValueError("Per-seed forecast keys do not match the common test records.")
    seeds = seeds.reindex(final.index)
    for model in SEEDED:
        average = seeds[[f"{model}_{seed}" for seed in SEEDS]].mean(axis=1)
        np.testing.assert_allclose(average, final[model], rtol=0, atol=1e-12)
    summary = subsets.set_index("subset")
    np.testing.assert_array_equal(summary.loc[["all","all_corrected","all_HH","corrected_HH"], "n"],
                                  [43721, 6771, 9470, 3168])
    np.testing.assert_allclose(summary.loc["all", ["CRHF_MAE_MW","CRHF_RMSE_MW"]],
                               [8.9588,17.0049], rtol=0, atol=0.00005)
    np.testing.assert_allclose(summary.loc["corrected_HH", ["gain_MAE_MW","gain_RMSE_MW"]],
                               [0.1701,0.4803], rtol=0, atol=0.00005)
    expected_intervals = {
        "base_plus_gate": [0.0105, -0.0036, 0.0289],
        "matched_catboost_spatial_input": [0.0147, -0.0064, 0.0347],
        "linear_available_forecast": [0.0306, 0.0175, 0.0440],
        "neural_available_forecast": [0.0125, -0.0088, 0.0365],
    }
    for name, expected in expected_intervals.items():
        actual = intervals.set_index("contrast").loc[name, ["gain_MW", "adjusted_lower_MW", "adjusted_upper_MW"]]
        np.testing.assert_allclose(actual.to_numpy(float), expected, rtol=0, atol=0.00005)
    initial = risks[(risks.group_type == "current_state") & (risks.group == "1")].set_index("method")
    ece_reduction = 100 * (1 - initial.loc["future_probability", "ECE"] / initial.loc["pooled_score", "ECE"])
    np.testing.assert_allclose(ece_reduction, 73.48, rtol=0, atol=0.005)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=REFERENCE / "test_predictions.csv.gz")
    parser.add_argument("--seeds", type=Path, default=REFERENCE / "three_seed_predictions.csv.gz")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "evaluation")
    parser.add_argument("--check", action="store_true", help="Check counts, three-seed averages and published numbers.")
    args = parser.parse_args()
    frame = pd.read_csv(args.input, float_precision="round_trip")
    risks, bins = risk_metrics(frame)
    intervals, subsets = paired_intervals(frame), subset_metrics(frame)
    if args.check:
        verify(frame, args.seeds, intervals, subsets, risks)
        print("Confirmed the common test set, seed means, core paper estimates and paired intervals.")
    outputs = {"power_metrics": power_metrics(frame), "risk_metrics": risks, "calibration_bins": bins,
               "paired_MAE_intervals": intervals, "correction_subsets": subsets}
    args.output.mkdir(parents=True, exist_ok=True)
    for name, table in outputs.items():
        table.to_csv(args.output / f"{name}.csv", index=False)
    print(f"Wrote {len(outputs)} tables to {args.output}")


if __name__ == "__main__":
    main()
