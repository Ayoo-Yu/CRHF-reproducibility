# Predictions on the common test set

The two gzip-compressed CSV files have 43,721 rows each. Join them by `farm_id`, `issue_slot`, and `target_slot`. The target is four 15-minute intervals after issue. Join observations in `data/observations_15min.csv.gz` by matching `farm_id` with `farm` and either public slot with `slot`. The month is the **issue** month used for the paper's monthly figures. Magnitudes retain the original MW scale and decimal precision. All five farm identifiers are pseudonyms.

## Test predictions

| Fields in `test_predictions.csv.gz` | Meaning |
| --- | --- |
| `actual_mw`, `observed_available_mw`, `farm_scale_mw` | Target actual wind power, target available power and the farm's fixed normalization scale, in MW. The first two are observations used for scoring. |
| `current_state`, `target_state` | Observed curtailment at issue and target respectively, with 1 indicating curtailment. |
| `current_score`, `pooled_score`, `future_probability`, `delta_q` | Current-state score, uncalibrated pooled score, calibrated target probability and calibrated probability minus current score. |
| `risk_regime`, `action` | The first letter of LL/LH/HL/HH is the current-score category, the second the future-probability category (high at 0.5 and above). The stored decision is `base`, `linear` or `neural`. Both variables are known at issue. |
| `Persistence`, `Ridge`, `MixLinear`, `MDLinear`, `ExtraTrees`, `HGB`, `CatBoost`, `LightGBM`, `MLP`, `GRU`, `PatchTST` | The 11 direct benchmark forecasts in Table 3, in MW. |
| `CatBoostFixedSeed`, `CatBoostMatchedLocal`, `DirectSpatial` | The three fixed-seed configurations in Supplementary Section D.1. In order these use conventional inputs and preprocessing, conventional inputs and matched preprocessing, and individual-farm inputs and matched preprocessing. |
| `B`, `CRHF` | Locally adapted base forecast and selected final actual-power forecast, in MW. |
| `own_*`, `aggregate_*`, `spatial_*` | Nine available-power forecasts, in MW. Spatial inputs are own farm, cross-farm aggregates, and aggregates plus individual source farms, respectively. The suffixes `raw`, `current`, and `full` denote no residual correction, current-state correction and future-risk correction. |
| `final_current_A`, `final_current_II`, `final_observed` | Final forecasts for three substitutions. The first substitutes current available power for forecast available power. The second uses the current risk score only in Stage 2. The third replaces both auxiliary forecasts by current observations. |
| `linear_current_A`, `linear_forecast_A`, `neural_current_A`, `neural_forecast_A` | Actual-power candidate forecasts using current or forecast available power, with the risk forecast held fixed. The two forecast-available-power candidates also give the fixed correction strategies. |
| `bias_low`, `bias_high` | Final actual-power forecasts after shifting the available-power input by −5% or +5% of the farm power scale. |
| `history_qualified` | Whether enough same-regime history was available for comparing the candidates. |
| `linear_valid_days`, `neural_valid_days`, `linear_record_count`, `neural_record_count` | Days with eligible observations and the corresponding record counts for each candidate. |
| `linear_executed`, `neural_executed` | Whether the candidate was computed at issue. The separate `action` column records the candidate actually applied. |
| `linear_saved_score`, `neural_saved_score` | Conservative, scale-normalized historical gain estimates. The stored value −1 marks insufficient history. Exclude these rows when summarizing the distribution of qualified scores. |

## Individual neural predictions

`three_seed_predictions.csv.gz` contains the same keys and month, followed by the forecast for each combination of five models and three seeds. For example, `GRU_20260914` is the GRU prediction from that individual fit. Average predictions across seeds **within each record**, then compute errors against the observation.

## Paired inference

Run `python scripts/evaluate.py --check` from the repository root. The script verifies the individual-seed means, reported sample counts and selected numerical results. It writes five CSV files under `results/evaluation/`, including grouped power errors, AP, Brier score, calibration bins, ECE, conditional improvements and confidence intervals. The target-slot blocks are anchored at the beginning of the common test period (slot 20,640). Each of the 10,000 replicates samples 15 blocks with replacement and keeps the farms together. Multiplicity families are specified in Supplementary Section D.

The neural candidate's adjusted upper confidence limit in Fig. 11 is 0.036548068 MW in the saved analysis and in the public recomputation. It rounds to 0.0365 MW at four decimal places. The manuscript text gives 0.0366 MW; both intervals include zero.


## Comparing a new fit with the published predictions

After running a new fit or replay, compare the row-level values on the same common test set. Supply any available outputs by filename. A direct baseline run may contain just the selected models.

```bash
python scripts/compare_retrained.py --crhf outputs/crhf/predictions.csv.gz
python scripts/compare_retrained.py --baselines outputs/direct/predictions.csv.gz
```

For the optional Stage 2 and diagnostics outputs, use `--stage2 <file>` and `--diagnostics <file>`. The script lists the largest absolute difference and the mean absolute difference for each forecast, checks that all common keys are present, and returns a failing exit status when a value exceeds the specified tolerance. Categorical correction decisions must match for a run to pass the default check.
