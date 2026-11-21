# Direct forecasting benchmark

The default run fits the ten learning-based models on the released observations and reports persistence on the same evaluation set. The networks predict a normalized one-hour increment, and each model’s output is converted to nonnegative MW before the three-seed average. The forecast error is evaluated in MW.

From the repository root:

```sh
python scripts/train_baselines.py --out outputs/direct
python scripts/compare_retrained.py --baselines outputs/direct/predictions.csv.gz
```

## Focused runs

| Goal | Command |
| --- | --- |
| Refit one conventional baseline | `python scripts/train_baselines.py --models Ridge --out outputs/ridge` |
| Repeat both validation trials and refit one network | `python scripts/train_baselines.py --models MLP --reselect --out outputs/mlp-selection` |
| Replay all fifteen saved network fits | `python scripts/train_baselines.py --models MLP GRU PatchTST MixLinear MDLinear --neural-checkpoints --out outputs/direct-networks` |
| Refit ordinary CatBoost and the seed-matched comparisons | `python scripts/train_baselines.py --models CatBoost --seed-control --out outputs/paired-direct` |

Use a fresh output directory for each run. Each contains `run.json` (selected configurations, seeds and row counts), `predictions.csv.gz` (one row per shared test case indexed by farm, issue slot and target slot), and `metrics.csv` (MAE and RMSE in MW). The frozen numeric neural weights are in `models/direct/`. Input standardization statistics are recomputed using training records, including during checkpoint replay. The only model weights read by that option are the fifteen .npz files. Model configuration and third-party source attribution are in [models.py](models.py) and [THIRD_PARTY.md](THIRD_PARTY.md).

## Training and inputs

The first training set has 60,045 eligible observations. For selection, two configurations per model are compared using MAE on the subsequent 12,801 eligible validation records. Only labels available before the next cutoff enter each fit. The final refit uses 72,861 records. There are 44,777 issued forecasts in the test period and 43,721 meet the shared evaluation criteria. The default command uses the hyperparameters chosen in this chronological selection. `--reselect` repeats that step.

The six channels in the 96-slot sequence and the 189 tabular variables are reconstructed by `crhf.pipeline.reconstruct`. Per-channel sequence and per-column tabular statistics are the training medians and interquartile ranges. Missing values are imputed by those medians. Values are clipped to ±10 after scaling. Flattening the sequence gives 576 values. The tree and Ridge inputs have 1,341 columns, consisting of the 576 sequence values, 189 tabular variables, and 576 sequence missingness indicators. For AdamW, the sequence goes to the main model and the 189 tabular values and 576 indicators go to a concurrent linear prediction.

Configuration selection for AdamW uses seed 20260913 and up to 40 epochs (patience 6). Each selected model is then fit to the final training set from random initialization with seeds 20260913, 20260914 and 20260915, for the winning number of epochs. Final results are the arithmetic mean of the three MW forecasts. Training uses a batch size of 512, weight decay 0.0001, gradient-norm limit 1 and power-scale-weighted absolute increment error. The CPU benchmark used four PyTorch threads and six threads for the conventional models.

The optional seed comparison outputs `CatBoost_seed14` (ordinary preprocessing, changed seed), `CatBoost_common` (fixed seed and common predictors under the alternative preprocessing) and `CatBoost_spatial` (same fixed seed, preprocessing and estimator, with 90 station-specific predictors added). The latter two fits apply per-column scaling and append an indicator per column, giving 1,530 and 1,710 total inputs, respectively.

Exact per-record predictions from the study are also released for independent metric calculation. Reduced-precision models can differ slightly across CPU and BLAS implementations after refitting. Examine both the maximum individual forecast difference and the pooled MAE/RMSE when comparing a new fit.
