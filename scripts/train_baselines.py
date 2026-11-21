"""Refit or replay the direct benchmark from the released 15-minute grid."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from baselines import checkpoints
from baselines.controls import paired_controls
from baselines.models import ADAMW_MODELS, SEEDS, SELECTION, TREE_MODELS
from baselines.preprocessing import batch, masks, transform_for_fit
from baselines.training import fit_predict, select_configuration
from crhf.pipeline import new_output, public_keys, reconstruct


def score(actual, prediction):
    error = np.asarray(prediction, dtype=float) - actual
    return {"MAE_MW": float(np.abs(error).mean()), "RMSE_MW": float(np.sqrt((error ** 2).mean()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choices = ["Persistence", *TREE_MODELS, *ADAMW_MODELS]
    parser.add_argument("--models", nargs="+", choices=choices, default=choices)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "direct")
    parser.add_argument("--reselect", action="store_true", help="Repeat the two January configuration trials per model.")
    parser.add_argument("--checkpoints", action="store_true",
                        help="Use the published fitted parameters for all selected baseline models.")
    parser.add_argument("--neural-checkpoints", action="store_true",
                        help="Use published CPU weights for neural models; requested tree models are still refitted.")
    parser.add_argument("--seed-control", action="store_true",
                        help="Also fit the fixed-seed standard, common-column and spatial-column CatBoost controls.")
    args = parser.parse_args()
    if args.reselect and (args.neural_checkpoints or args.checkpoints):
        parser.error("Choose a fresh selection run or the archived neural checkpoints.")
    out = new_output(args.out)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)

    data = reconstruct(sequence=True)
    frame = data["frame"]
    sequence = data["sequence"]
    tab = frame[data["schema"]["observed"]].to_numpy(np.float32)
    fit, val, final, test = masks(frame)
    issued = frame.phase.eq("test").to_numpy()
    valid = test[issued]
    results = public_keys(frame.loc[test, ["farm", "Timestamp", "target_timestamp"]]).reset_index(drop=True)
    actual = frame.loc[test, "target_actual_operational"].to_numpy(float)
    metrics, receipt = {}, {"rows": dict(zip(("selection", "validation", "final", "test"), (
        int(mask.sum()) for mask in (fit, val, final, test)))), "configuration": {}}
    selected = dict(SELECTION)
    if args.reselect:
        stats = transform_for_fit(sequence, tab, fit)
        train = batch(frame, sequence, tab, fit, stats)
        validation = batch(frame, sequence, tab, val, stats)
        for name in args.models:
            if name != "Persistence":
                selected[name] = select_configuration(name, train, validation)
        del train, validation
    stats = transform_for_fit(sequence, tab, final)
    testing = batch(frame, sequence, tab, issued, stats)
    fresh = args.seed_control or (not args.checkpoints and any(n in TREE_MODELS or
                (n in ADAMW_MODELS and not args.neural_checkpoints) for n in args.models))
    training = batch(frame, sequence, tab, final, stats) if fresh else None

    def record(name, value):
        prediction = np.asarray(value)[valid]
        if not np.isfinite(prediction).all():
            raise ValueError(f"Missing or invalid forecast for {name}")
        results[name] = prediction
        metrics[name] = score(actual, prediction)
        print(name, metrics[name], flush=True)

    for name in dict.fromkeys(args.models):
        if name == "Persistence":
            record(name, frame.loc[issued, "actual_operational"].to_numpy(float))
            continue
        cfg, epochs = selected[name]
        receipt["configuration"][name] = {"config": cfg, "epochs": epochs}
        if name in ADAMW_MODELS:
            seed_predictions = []
            for seed in SEEDS:
                if args.neural_checkpoints or args.checkpoints:
                    p = checkpoints.network(ROOT / "models/direct", name, cfg, seed, testing)
                else:
                    p, _ = fit_predict(name, cfg, training, testing, seed=seed, epochs=epochs)
                # Published constituent forecasts were saved as float32 decimal CSVs and
                # subsequently read as float64 before averaging. Reproduce that boundary.
                seed_predictions.append(np.asarray(p, np.float32).astype(str).astype(float))
            record(name, np.mean(seed_predictions, axis=0))
        else:
            pred = (checkpoints.conventional(ROOT / "models/direct", name, testing)
                    if args.checkpoints else fit_predict(name, cfg, training, testing)[0])
            record(name, pred)
    if args.seed_control:
        record("CatBoost_seed14", fit_predict("CatBoost", 1, training, testing, seed=SEEDS[1])[0])
        for name, p in paired_controls(data, tab, final, issued):
            record(name, p)
    results.to_csv(out / "predictions.csv.gz", index=False)
    pd.DataFrame.from_dict(metrics, orient="index").rename_axis("model").to_csv(out / "metrics.csv")
    receipt.update(all_checkpoints=args.checkpoints, neural_checkpoints=args.neural_checkpoints, reselected=args.reselect,
                   neural_final_seeds=list(SEEDS), seed_control=args.seed_control)
    (out / "run.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
