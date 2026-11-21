# Manuscript tables

All five main-text tables and all five supplementary tables are in [reference](reference). The CSV files use UTF-8. Column headers retain the paper's units, mathematical expressions use inline LaTeX, and record counts are integers without thousands separators.

From the repository root, run `python scripts/tables.py`. The script writes Tables 1, 3, 4 and 5 to `outputs/tables/` and checks every displayed result against the reference CSVs. To evaluate a new model run that has the same prediction schema and test keys, supply `--predictions PATH --no-check`. Choose a different output directory with `--out PATH`.

| Table | Contents | Source |
| --- | --- | --- |
| [1](reference/table_1.csv) | Farm-specific curtailment rates, fixed scales and test-set counts | Observation grid and verified common-test keys |
| [2](reference/table_2.csv) | Stage 3 comparison designs | Published model definitions |
| [3](reference/table_3.csv) | Eleven direct benchmarks and CRHF, including equal-farm-weight metrics | Record-level actual power and forecasts |
| [4](reference/table_4.csv) | Direct CatBoost, individual-farm-input CatBoost, base forecast and full CRHF | Record-level actual power and forecasts |
| [5](reference/table_5.csv) | Corrected records, all HH records and corrected HH records | Forecasts, risk regime and candidate decision on each record |
| [C1](reference/table_C1.csv) | Features and windows for the six model components | Published feature specification |
| [C2](reference/table_C2.csv) | Farm scales, model settings, loss, clipping and seeds | Published parameter specification |
| [C3](reference/table_C3.csv) | Input transformations, local fitting and selection | Published estimation rules |
| [C4](reference/table_C4.csv) | Fit, refit and feedback schedule | Published schedule expressed in common 15-minute slots |
| [G1](reference/table_G1.csv) | Direct benchmark architectures and hyperparameters | Published benchmark settings |

For Table 1, forecast origins and their targets four slots later must both have valid observations. The training/validation curtailment rate uses target labels on these pairs. In the test period the target-consistency check additionally excludes records whose actual power exceeds available power by more than 3% of the farm scale. Table 1 takes the fixed scale from the calibration step in the data preparation pipeline.

For Table 3, each reduction uses the unrounded persistence error. The normalized columns average each farm's own MAE or RMSE divided by its fixed scale, giving each farm equal weight. For Table 5, a record is corrected when the gain rule selects the linear or neural candidate. Each percentage uses the unrounded base error on that same subset. These subsets overlap, and their counts should not be summed.
