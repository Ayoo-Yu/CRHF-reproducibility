"""Rebuild the four quantitative manuscript tables and compare their displayed cells."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "tables" / "reference"
METHODS = (
    "Persistence", "Ridge", "MixLinear", "MDLinear", "ExtraTrees", "HGB", "CatBoost",
    "LightGBM", "MLP", "GRU", "PatchTST", "CRHF",
)
OBSERVATION_FIELDS = [
    "farm", "slot", "actual_operational", "potential", "input_valid", "curtailed", "capacity_proxy",
]


def read_reference(number):
    with (REFERENCE / f"table_{number}.csv").open(encoding="utf-8", newline="") as stream:
        return list(csv.reader(stream))


def errors(records, column):
    error = (records[column] - records["actual_mw"]).to_numpy(dtype=float)
    if not len(error) or not np.isfinite(error).all():
        raise ValueError(f"Missing or non-finite forecast error for {column}")
    return np.abs(error).mean(), np.sqrt(np.square(error).mean())


def relative_reduction(reference, value):
    return 100 * (reference - value) / reference


def sample_summary(observations, predictions):
    # Interval ends are 15 minutes apart. A target must exist at t + 4 on the original grid.
    target = observations[["farm", "slot", "input_valid", "curtailed", "actual_operational", "potential"]].copy()
    target["slot"] -= 4
    joined = observations.merge(target, on=["farm", "slot"], suffixes=("", "_target"), validate="one_to_one")
    eligible = joined[joined["input_valid"] & joined["input_valid_target"]]
    training = eligible[(eligible["slot"] >= 2976) & (eligible["slot"] < 20640)]
    test = eligible[(eligible["slot"] >= 20640) & (
        eligible["actual_operational_target"] <= eligible["potential_target"] + 0.03 * eligible["capacity_proxy"]
    )]
    if predictions.duplicated(["farm_id", "issue_slot"]).any():
        raise ValueError("Duplicated prediction key")
    paired = test.merge(
        predictions, left_on=["farm", "slot"], right_on=["farm_id", "issue_slot"],
        how="outer", validate="one_to_one", indicator=True,
    )
    if not paired["_merge"].eq("both").all():
        raise ValueError("The prediction keys differ from the test cohort derived from the observations")
    if not np.array_equal(paired["target_slot"], paired["slot"] + 4):
        raise ValueError("The prediction horizon differs from four 15-minute intervals")
    if not np.array_equal(paired["target_state"], paired["curtailed_target"]):
        raise ValueError("The recorded target states disagree with the observation grid")
    if not np.allclose(paired["actual_mw"], paired["actual_operational_target"], rtol=0, atol=1e-5):
        raise ValueError("The target power in the record predictions differs from the observation grid")
    scales = observations.groupby("farm")["capacity_proxy"]
    if not scales.nunique().eq(1).all():
        raise ValueError("Farm scales vary across records")
    farm_scales = scales.first()
    rows = []
    for farm in [*farm_scales.index, "All farms"]:
        tv = training if farm == "All farms" else training[training["farm"].eq(farm)]
        ts = test if farm == "All farms" else test[test["farm"].eq(farm)]
        rows.append([
            farm, f"{100 * tv['curtailed_target'].mean():.2f}", f"{100 * ts['curtailed_target'].mean():.2f}",
            "—" if farm == "All farms" else f"{farm_scales[farm]:.2f}", str(len(ts)),
        ])
    return rows


def benchmark_table(predictions):
    reference_mae, reference_rmse = errors(predictions, "Persistence")
    rows = []
    for method in METHODS:
        mae, rmse = errors(predictions, method)
        farm_metrics = []
        for _, group in predictions.groupby("farm_id"):
            scales = group["farm_scale_mw"].unique()
            if len(scales) != 1 or not scales[0] > 0:
                raise ValueError("Each farm must have one positive normalization scale")
            farm_metrics.append(np.array(errors(group, method)) * (100 / scales[0]))
        nmae, nrmse = np.mean(farm_metrics, axis=0)
        rows.append([
            method, f"{mae:.3f}", f"{rmse:.3f}", f"{relative_reduction(reference_mae, mae):.2f}",
            f"{relative_reduction(reference_rmse, rmse):.2f}", f"{nmae:.2f}", f"{nrmse:.2f}",
        ])
    return rows


def component_table(predictions):
    rows = []
    methods = ("CatBoost", "DirectSpatial", "B", "CRHF")
    for labels, method in zip(read_reference(4)[1:], methods, strict=True):
        mae, rmse = errors(predictions, method)
        rows.append([*labels[:2], f"{mae:.4f}", f"{rmse:.4f}"])
    return rows


def correction_table(predictions):
    if not set(predictions["action"].unique()) <= {"base", "linear", "neural"}:
        raise ValueError("Unrecognized correction strategy")
    corrected = predictions["action"].isin(["linear", "neural"])
    hh = predictions["risk_regime"].eq("HH")
    groups = (("All corrected records", corrected), ("HH regime", hh), ("Corrected HH records", corrected & hh))
    rows = []
    for name, mask in groups:
        group = predictions[mask]
        b_mae, b_rmse = errors(group, "B")
        mae, rmse = errors(group, "CRHF")
        rows.append([
            name, str(len(group)), f"{b_mae:.4f}", f"{mae:.4f}", f"{relative_reduction(b_mae, mae):.2f}",
            f"{b_rmse:.4f}", f"{rmse:.4f}", f"{relative_reduction(b_rmse, rmse):.2f}",
        ])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, default=ROOT / "data" / "observations_15min.csv.gz")
    parser.add_argument("--predictions", type=Path, default=ROOT / "results" / "reference" / "test_predictions.csv.gz")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "tables")
    parser.add_argument("--no-check", action="store_true", help="Write estimates without checking paper rounding.")
    args = parser.parse_args()
    observations = pd.read_csv(args.observations, usecols=OBSERVATION_FIELDS)
    predictions = pd.read_csv(args.predictions)
    tables = {
        1: sample_summary(observations, predictions), 3: benchmark_table(predictions),
        4: component_table(predictions), 5: correction_table(predictions),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    for number, rows in tables.items():
        reference = read_reference(number)
        if not args.no_check and rows != reference[1:]:
            for actual, expected in zip(rows, reference[1:], strict=True):
                if actual != expected:
                    raise ValueError(f"Table {number}: computed {actual}, paper {expected}")
            raise ValueError(f"Table {number} row count differs from the paper")
        dest = args.out / f"table_{number}.csv"
        with dest.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream, lineterminator="\n").writerows([reference[0], *rows])
        print(f"Table {number}: {len(rows)} rows written to {dest}")


if __name__ == "__main__":
    main()
