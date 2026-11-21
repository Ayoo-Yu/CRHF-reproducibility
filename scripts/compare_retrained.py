"""Compare forecasts produced by training or replay with the published common test set."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
KEYS = ["farm_id", "issue_slot", "target_slot"]
DIRECT = ["Persistence", "Ridge", "ExtraTrees", "HGB", "CatBoost", "LightGBM", "MLP",
          "GRU", "PatchTST", "MixLinear", "MDLinear"]
DIAGNOSTICS = ["current_score", "pooled_score", "future_probability", "linear_current_A",
               "linear_forecast_A", "neural_current_A", "neural_forecast_A", "bias_low",
               "bias_high", "final_current_A", "final_current_II", "final_observed",
               "CRHF", "B", "DirectSpatial"]
AVAILABLE = [f"{space}_{correction}" for space in ["own", "aggregate", "spatial"]
             for correction in ["raw", "current", "full"]] + ["observed_available_mw"]
MAPPINGS = {
    "crhf": {
        "risk_q": "future_probability", "available_full": "spatial_full",
        "available_current": "spatial_current", "available_raw": "spatial_raw",
        "direct_static": "DirectSpatial", "strong_online": "B", "A_versioned/full": "CRHF",
        "A_versioned/full_action": "action", "A_versioned/no_available": "final_current_A",
        "A_versioned/II_current": "final_current_II", "A_versioned/observed": "final_observed",
    },
    "diagnostics": {field: field for field in DIAGNOSTICS},
    "stage2": {field: field for field in AVAILABLE},
    "baselines": {
        **{field: field for field in DIRECT},
        "CatBoost_seed14": "CatBoostFixedSeed", "CatBoost_common": "CatBoostMatchedLocal",
        "CatBoost_spatial": "DirectSpatial",
    },
}


def keyed(path):
    frame = pd.read_csv(path, float_precision="round_trip")
    frame = frame.rename(columns={"farm": "farm_id", "slot": "issue_slot"})
    if frame.duplicated(KEYS).any() or frame[KEYS].isna().any(axis=None):
        raise ValueError(f"Duplicate or missing forecast keys in {path}.")
    return frame.set_index(KEYS)


def compare(kind, path, expected):
    test = keyed(path)
    missing = expected.index.difference(test.index)
    if len(missing):
        raise ValueError(f"{path} misses {len(missing)} of {len(expected)} common test keys.")
    result = test.reindex(expected.index)
    mappings = MAPPINGS[kind]
    if kind != "baselines":
        missing_fields = set(mappings) - set(result.columns)
        if missing_fields:
            raise ValueError(f"Missing {kind} fields in {path}: {sorted(missing_fields)}")
    rows = []
    for column, published in mappings.items():
        if column not in result:
            # Direct model runs may contain only the models chosen with --models.
            continue
        observed = result[column]
        reference = expected[published]
        if published == "action":
            observed = observed.map({0: "base", 1: "linear", 2: "neural",
                                     "base": "base", "linear": "linear", "neural": "neural"})
            if observed.isna().any():
                raise ValueError("Unrecognized correction action.")
            errors = observed.ne(reference).astype(float).to_numpy()
        else:
            errors = np.abs(observed.to_numpy(float) - reference.to_numpy(float))
            if not np.isfinite(errors).all():
                raise ValueError(f"Missing or infinite prediction in {kind}/{column}.")
        rows.append({"input_type": kind, "generated_field": column, "reference_field": published,
                     "n": len(result), "mean_abs_difference": float(errors.mean()),
                     "max_abs_difference": float(errors.max())})
    if not rows:
        raise ValueError(f"No recognized model output found in {path}.")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for kind in MAPPINGS:
        parser.add_argument("--" + kind, type=Path, help="Path to the generated forecast CSV or CSV.gz.")
    parser.add_argument("--reference", type=Path, default=ROOT / "results/reference/test_predictions.csv.gz")
    parser.add_argument("--atol", type=float, default=1e-8, help="Maximum permitted absolute difference.")
    parser.add_argument("--output", type=Path, help="Optional output CSV of differences by variable.")
    args = parser.parse_args()
    jobs = [(k, getattr(args, k)) for k in MAPPINGS if getattr(args, k) is not None]
    if not jobs or not np.isfinite(args.atol) or args.atol < 0:
        parser.error("Supply at least one forecast file and a finite nonnegative tolerance.")
    expected = keyed(args.reference)
    if len(expected) != 43721:
        raise ValueError("The published reference must contain all 43,721 common records.")
    summary = pd.DataFrame([row for kind, path in jobs for row in compare(kind, path, expected)])
    print(summary.to_string(index=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.output, index=False)
    if (summary.max_abs_difference > args.atol).any():
        raise SystemExit("At least one output differs from the reference by more than --atol.")


if __name__ == "__main__":
    main()
