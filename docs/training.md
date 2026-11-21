# Reproducing CRHF

Run these commands from the repository root after installing the pinned Python dependencies. Each output directory must be new. The default parameters reproduce the five-farm, 1-hour-ahead experiment.

| Command | Computation | Main output |
| --- | --- | --- |
| `python scripts/train_crhf.py prepare` | Build the feature grid and 96 × 6 histories from `data/observations_15min.csv.gz` | Prints the dimensions for 120,831 forecast origins across model fitting, warm-up, and testing |
| `python scripts/train_crhf.py train --output outputs/retrained` | Refit the four forward Stage 1 bridges, six available-power and six direct CatBoost fits, ten available-power residual fits, and four folds of Stage 3 models for all four roles. Rebuild the arrived-feedback histories | `models_h4.joblib`, `predictions.csv.gz`, `metrics.csv`, per-fold models and training receipts |
| `python scripts/train_crhf.py replay --output outputs/replay` | Generate fresh predictions using the release checkpoint, full test sequence, and arriving observations | `predictions.csv.gz`, per-role decision evidence and final states, `metrics.csv` |
| `python scripts/train_crhf.py stage2 --output outputs/stage2` | Independently train all nine available-power comparisons from three spatial inputs and three residual treatments | `available_predictions.csv.gz` for Fig. 10 and Fig. S2 |
| `python scripts/train_crhf.py diagnostics --output outputs/diagnostics` | Replay the four auxiliary-input roles and the two prescribed available-power biases. Export uncalibrated and calibrated risks, fixed candidates, gate evidence and selected forecasts | `diagnostics.csv.gz` for Figs. 6–9, 11–14, S3 and S4, and Table 5 |

For a short model check, add `--quick` to `replay` or `diagnostics`. This uses the first two test days. To replay a model you trained locally, pass `--bundle outputs/retrained/models_h4.joblib`. The Stage 2 experiment can reuse forward Stage 1 fits via `--bridges outputs/retrained/stage1_bridges.joblib`. Model serialization uses Joblib and depends on the pinned scikit-learn, CatBoost, NumPy, pandas, and Python versions. Only load model files from a trusted source.

## Data flow and time boundaries

`src/crhf/pipeline.py` reads the regular grid of 15-minute observations. All joins in the distributed CSVs use the public farm ID and integer slots. Internally a synthetic calendar retains the day-of-year and intraday terms needed by the models. Four slots separate each forecast origin (`slot`) and target (`target_slot`). Invalid observations remain missing in a sequence at their original offsets.

| Forecast component | Main implementation | Inputs and history |
| --- | --- | --- |
| Calibrated future curtailment probability | `risk_features.py`, `risk_bridge.py` | Use data preceding each fitting cutoff, separate model, weight and calibration windows, and cross-farm observations available by issue time |
| Available-power forecast | `spatial.py`, `backbones.py`, `available_residual.py` | Fit the available-power increment, then learn a robust correction from forward residuals. Disaggregated spatial information contributes 90 features |
| Direct forecast and base forecast B | `backbones.py`, `local_feedback.py`, `neighborhood.py` | Fit the direct CatBoost increment. Update the local weighted ridge correction using outcomes that have already arrived |
| Candidate corrections and selection | `residual_models.py`, `proposals.py`, `correction_selection.py` | Predict corrections using one robust linear fit and the mean of three neural seeds. Compare recent gains by score group and model fitting period |
| Alternative Stage 2 designs and sensitivity | `experiments.py` | Exercise the nine Stage 2 settings, the fixed candidates, observation substitutions and the ±5% available-power shifts |

The common test period starts at issue slot 20,640. There are 44,777 valid forecast origins in this period. Forecasting and feedback run on every valid origin. Evaluation requires an eligible observed target and finite available power, and applies the target-power consistency rule. This leaves the common 43,721 records used in the paper. Future target availability never determines whether a forecast is issued.

## Comparing outputs

`reference/crhf_replay.csv.gz` contains 44,777 previously saved forecasts with public keys. The train and replay commands read this file **after** generating predictions, and compare every issued key and model output for the requested roles. `reference_comparison.json` records the largest absolute discrepancy per series and the number of differing actions. The tests use a strict tolerance of 0.00003 MW for power, 1e-10 for calibrated probabilities, and exact equality for actions.

| Forecast column | Meaning |
| --- | --- |
| `risk_q` | Calibrated probability of target-time curtailment |
| `available_raw`, `available_current`, `available_full` | Spatial available-power model without residual correction, with the current score, and with forecast risk |
| `direct_static`, `strong_online` | Direct CatBoost with disaggregated spatial inputs and its locally adjusted base forecast B |
| `A_versioned/full` | Full CRHF with gains from the same model fitting period. The corresponding `_action` column uses 0 for B, 1 for the linear correction, and 2 for the neural correction |
| `no_available`, `II_current`, `observed` (role suffixes) | Substitute current available power in Stage 3, use current risk in Stage 2, or use both current observations in Stage 3, respectively |
| `B_carry`, `C_robust` (policy prefixes) | Archived additional policies using gains across fits or a sign-consistency rule. The paper uses `A_versioned` |

To compare the three files to the manuscript's 43,721-row analysis set:

```bash
python scripts/compare_retrained.py --crhf outputs/retrained/predictions.csv.gz --stage2 outputs/stage2/available_predictions.csv.gz --diagnostics outputs/diagnostics/diagnostics.csv.gz
```
