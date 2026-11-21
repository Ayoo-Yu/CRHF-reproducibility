# Direct actual-power forecasts

The 11 benchmarks in Table 3 use the common scoring set of 43,721 records. Persistence carries forward the current observation. The ten learned methods share the 96-step, six-channel history and 189 tabular features. The model-specific inputs, missingness indicators, and configuration choices are documented in supplementary Tables C1 and G1 and in the [source](../src/baselines/).

Run these commands from the root of the repository using the pinned training dependencies.

```sh
python scripts/train_baselines.py --checkpoints --out outputs/direct-replay
python scripts/compare_retrained.py --baselines outputs/direct-replay/predictions.csv.gz
python scripts/train_baselines.py --seed-control --out outputs/direct-fit
python scripts/compare_retrained.py --baselines outputs/direct-fit/predictions.csv.gz
```

* `--checkpoints` reconstructs inputs from the public observation grid and runs all 11 methods with the released fitted parameters.
* The default fits the 10 learned methods using the selected configurations. `--seed-control` adds the CatBoost comparisons for the supplementary fixed-seed analysis.
* `--reselect` repeats chronological selection. The selection fit uses 60,045 origins with 12,801 origins for validation. The final fit uses 72,861 origins.
* `--models` selects a subset, for example `--models Persistence Ridge MLP`. `--neural-checkpoints` uses the saved neural fits and refits any selected tabular models.
* Each neural model has a separate fit for seeds 20260913, 20260914, and 20260915. Average the three predictions for each record before calculating errors.

Each command needs a fresh output folder and writes `predictions.csv.gz`, `metrics.csv` and `run.json`. The models' [attributions and third-party licenses](../src/baselines/THIRD_PARTY.md) accompany their implementations.
