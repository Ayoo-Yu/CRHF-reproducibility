# CRHF wind power forecasting

Code and de-identified data for *Curtailment-risk-aware hybrid forecasting of actual wind power under unknown dispatch: risk stratification, available-power forecasting and feedback-gated online correction*, by Zhongze Yu and Zhenqing Liu.

Given 15-minute observations from five wind farms, CRHF forecasts actual power four intervals ahead. The paper's common test set has **43,721** records. Full CRHF has **8.9588 MW MAE** and **17.0049 MW RMSE**. The local base forecast is used on 36,950 records. On the 6,771 records where a correction is applied, the observed MAE and RMSE improvements over the base are **0.0679 MW** and **0.2523 MW**, respectively. The all-record paired MAE improvement is 0.0105 MW and its adjusted 95% interval includes zero.

## Reproduce the paper's results

Start in a fresh clone of this repository with Python 3.12. The original CPU environment used CPython 3.12.2. A virtual environment is recommended.

```sh
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-analysis.txt
python scripts/verify_release.py
python scripts/reproduce.py
```

The last command rebuilds the 15-minute grid from the native measurements, verifies it against the public observation grid, evaluates the archived record-level forecasts, regenerates the four numerical manuscript tables, and exports all 18 figures. Outputs are written to `outputs/reproduction/`. The 14 empirical figures are drawn from the public predictions. Four method diagrams use their source artwork. The reference directory for each figure also provides its publication layout. Visual details in a regenerated chart can depend on local fonts and plotting software.

## Refit and replay the models

```sh
python -m pip install -r requirements.txt
python scripts/train_crhf.py prepare
python scripts/train_crhf.py replay --quick --output outputs/quick
python scripts/train_crhf.py train --output outputs/retrained
python scripts/train_crhf.py stage2 --bridges outputs/retrained/stage1_bridges.joblib --output outputs/stage2
python scripts/train_crhf.py diagnostics --bundle outputs/retrained/models_h4.joblib --output outputs/diagnostics
python scripts/train_baselines.py --checkpoints --out outputs/direct-replay
python scripts/train_baselines.py --seed-control --out outputs/direct
```

Use a fresh output directory for each run. `train` fits from observations and replays the four auxiliary-input configurations with causal feedback. The full baseline command fits the 10 learned benchmarks, computes persistence and runs the fixed-seed CatBoost control. `--neural-checkpoints` replays the 15 individual neural fits; `--reselect` repeats hyperparameter selection. The saved individual predictions allow verification of the within-record three-seed averages.

See [CRHF training and chronological replay](docs/training.md), [direct baselines](docs/baselines.md) and [verification results](docs/verification.md) for outputs, holdout boundaries, comparison commands and measured tolerances.

## Contents

| Directory | What it contains |
| --- | --- |
| [data](docs/data_dictionary.md) | 814,354 native observations from five farms, three signals in MW, 151,106 aligned 15-minute rows, missingness, quality filters, curtailment labels and historical power scales |
| [results/reference](results/reference/README.md) | Anonymized forecasts on the 43,721 common records, all individual neural predictions and a field dictionary for paired comparisons |
| [reference](reference/) | Issued forecasts at 44,777 origins, including origins subsequently excluded from the paper's scoring set |
| [models](models/) | Checkpoints for replay |
| [src](src/) and [scripts](scripts/) | Data preparation, model fitting, chronological replay, error metrics, synchronized bootstrap, figure rendering and table generation |
| [figures](figures/README.md) | Figures 1–14 and S1–S4, published PNG and SVG artwork, plot data and regeneration instructions |
| [tables](tables/README.md) | Main Tables 1–5 and Supplementary Tables C1–C4 and G1, exported to CSV with mathematical notation and units |

## De-identification

Farm identifiers are F1–F5. All farms use a common relative clock: native records carry minute offsets and model-ready records carry 15-minute slot numbers. An issue at slot `k` predicts the interval ending at `k+4`. The calendar used internally reproduces the seasonal features. Forecasting and feature construction preserve the order in which measurements became available.

The release omits operator and station names, the identifier lookup, coordinates, equipment tags, and original vendor files. Power, dispatch, gaps and simultaneous network-wide events retain their measured values. Published study months, farm scales and distinctive traces may allow someone with independent operating data to associate records with an approximate date or farm. This is a pseudonymized research dataset.

## Citation and licenses

[Citation metadata](CITATION.cff) lists the authors and study title. The project's own code is licensed under [MIT](LICENSE). The measurements, predictions, result data and original figure artwork are licensed under [CC BY 4.0](LICENSE-DATA.md). The [baseline notices](src/baselines/THIRD_PARTY.md) identify separately licensed components.
