"""Rebuild the 15-minute analysis grid from the public native-cadence observations."""
import argparse
import gzip
import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NATIVE_CADENCE_MINUTES = {"F1": 5, "F2": 5, "F3": 5, "F4": 1, "F5": 5}
SIGNALS = ["actual_operational", "potential", "dispatch_history"]
LABELS = ["teacher_probability", "curtailed", "severity", "curtailment_mw"]
COLUMNS = ["farm", "slot", *SIGNALS, *LABELS, "capacity_proxy", "input_valid",
           "sample_coverage", "suspected_outage"]
STEP_MINUTES = 15
CALIBRATION_END_SLOT = 2976


def aggregate(native, cadence):
    minutes = native["minute_offset"]
    if minutes.isna().any() or not minutes.is_monotonic_increasing or not minutes.is_unique:
        raise ValueError("Native observation times must be complete, unique and sorted.")
    if not minutes.eq(minutes.astype("int64")).all():
        raise ValueError("Minute offsets must be whole numbers.")
    start = int(minutes.min()) // STEP_MINUTES * STEP_MINUTES
    expected = pd.RangeIndex(start, int(minutes.max()) + 1, cadence)
    values = native.set_index("minute_offset")[SIGNALS].reindex(expected)
    values = values.where(np.isfinite(values) & values.ge(0))
    # The observation for slot k covers [(k-1)*15, k*15) minutes.
    groups = values.groupby(expected // STEP_MINUTES + 1, sort=True)
    bins = groups.mean()
    bins["sample_coverage"] = groups.count().min(axis=1) / (STEP_MINUTES // cadence)
    bins["input_valid"] = bins.sample_coverage.ge(1.0) & bins[SIGNALS].notna().all(axis=1)
    return bins.rename_axis("slot").reset_index()


def label(bins):
    prefix = bins.slot.le(CALIBRATION_END_SLOT) & bins.input_valid & bins.potential.gt(0)
    if prefix.sum() < 96:
        raise ValueError("Insufficient positive calibration observations.")
    scale = max(float(bins.loc[prefix, "potential"].quantile(0.995)), 1.0)
    gap = (bins.potential - bins.dispatch_history).clip(lower=0)
    tracking = ((bins.actual_operational - bins.potential).abs()
                - (bins.actual_operational - bins.dispatch_history).abs())
    threshold = 0.03 * scale
    smooth = 0.02 * scale
    bins["suspected_outage"] = (
        bins.actual_operational.le(0.05) & bins.potential.gt(threshold)
        & bins.dispatch_history.gt(threshold)
    )
    bins["input_valid"] &= ~bins.suspected_outage
    event = bins.input_valid & gap.gt(threshold) & tracking.gt(0)
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -40, 40)))
    bins["teacher_probability"] = sigmoid((gap - threshold) / smooth) * sigmoid(tracking / smooth)
    bins["curtailed"] = event.astype(float)
    bins["curtailment_mw"] = np.where(event, (bins.potential - bins.actual_operational).clip(lower=0), 0.0)
    bins["severity"] = (bins.curtailment_mw / bins.potential.clip(lower=0.1)).clip(0, 1)
    bins["capacity_proxy"] = scale
    bins.loc[~bins.input_valid, SIGNALS + LABELS] = np.nan
    return bins


def build(directory):
    grids = []
    for farm, cadence in NATIVE_CADENCE_MINUTES.items():
        raw = pd.read_csv(directory / f"{farm}.csv.gz", float_precision="round_trip")
        if raw.columns.tolist() != ["minute_offset", *SIGNALS]:
            raise ValueError(f"{farm} has unexpected columns.")
        bins = label(aggregate(raw, cadence))
        bins["farm"] = farm
        grids.append(bins[COLUMNS])
    return pd.concat(grids, ignore_index=True)


def verify(result, reference):
    ref = pd.read_csv(reference, float_precision="round_trip")
    pd.testing.assert_frame_equal(result, ref, check_exact=False, atol=1e-12, rtol=1e-13)
    print(f"Verified {len(result):,} rows across {result.farm.nunique()} farms.")


def save(result, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use a fresh file so an interrupted write leaves the old grid intact.
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        temp = Path(tmp.name)
    try:
        with temp.open("wb") as dst, gzip.GzipFile(filename="", fileobj=dst, mode="wb", mtime=0) as gz:
            with io.TextIOWrapper(gz, encoding="utf-8", newline="") as txt:
                result.to_csv(txt, index=False, lineterminator="\n")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dir", type=Path, default=ROOT / "data/native")
    parser.add_argument("--output", type=Path, help="Write the rebuilt CSV.GZ to this path.")
    parser.add_argument("--reference", type=Path, default=ROOT / "data/observations_15min.csv.gz")
    parser.add_argument("--no-verify", action="store_true", help="Skip comparison with the frozen grid.")
    args = parser.parse_args()
    result = build(args.native_dir)
    if not args.no_verify:
        verify(result, args.reference)
    if args.output:
        save(result, args.output)


if __name__ == "__main__":
    main()