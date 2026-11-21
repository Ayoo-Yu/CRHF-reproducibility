# What was verified

Validation used the five public farm IDs and relative time keys on CPU. The exact package versions used for the release checks are in [reference-environment.txt](reference-environment.txt).

| Check | Result |
| --- | --- |
| Native measurements → observation grid | Rebuilt all 151,106 interval rows and checked the 13 fields, including invalid intervals, missing values, scales and labels. There were 132,324 valid intervals. |
| Feature reconstruction | Matched the 120,831 archived forecast-origin feature rows and their missingness in the 96 × 6 histories, 189 tabular features and 90 additional individual-farm columns used in the spatial control. |
| All 11 direct benchmarks | Reconstructed input features, replayed the fitted models and checked every forecast for all 43,721 test records. The maximum absolute difference was below 1 × 10⁻¹² MW. Fresh fits of Ridge, the three-seed MLP and the CatBoost configurations used for the same-seed control also matched the reference forecasts. |
| Complete CRHF fit → sequential forecasts | Refit from the public observations and generated forecasts at 44,777 eligible issue origins. All 30 series in the test replay comparison matched. The maximum absolute difference in full CRHF was 1.17 × 10⁻¹² MW. Candidate-selection decisions agreed record for record. The local base B matched exactly. The refit took approximately 11 minutes in the validation environment. |
| Nine Stage 2 configurations | Individually refit the 3 × 3 available-power comparison. Across 43,721 common test keys, the largest absolute difference from the archived predictions was below 8 × 10⁻¹³ MW. |
| Candidate and correction diagnostics | Replayed the three auxiliary-input substitutions, fixed candidates, two available-power shifts and the final selection. Compared 15 diagnostic fields on all 43,721 test keys. The maximum absolute discrepancy across these fields was 3.42 × 10⁻¹² or less. |
| Risk, calibration and comparisons | Recomputed probability scores, calibration bins, subgroup estimates and paired confidence intervals at the released aggregation levels. The common test set contains 6,771 records with an enabled correction, including 3,168 of the 9,470 HH records. |
| Paper tables | Calculated main Tables 1, 3, 4 and 5 and verified all displayed results against the reference CSVs. Table 2 and the five supplementary tables document designs, settings or comparison families. |
| Paper figures | Compared 20 plot-data tables and rendered 14 statistical figures. The four fixed method diagrams and the published layouts are included. The public captions for the partial final month omit its original calendar year. |

Individual model results, refit and checkpoint commands are described in [baselines.md](baselines.md). Run `python scripts/compare_retrained.py` with a generated predictions file to see maximum and mean discrepancies by series. Power errors are in MW. A new CPU, software version or implementation of a numerical routine may introduce floating-point differences; compare the mode, keys, thresholds and fitted settings before interpreting a mismatch.

## Precision note for Fig. 11

For the neural candidate, the upper end of the adjusted 95% interval for the paired MAE improvement is **0.0365480681713 MW**. The manuscript prints **0.0366 MW**. Rounding the full-precision result directly to four decimal places gives **0.0365 MW**. The reference series and evaluation code retain the full-precision estimate. The interval includes zero with either display rounding.
