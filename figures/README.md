# Figures

There are 14 figures in the main manuscript and four in the supplement. The PNG files in [published](published/) preserve the manuscript artwork at its original pixel dimensions on a white background. The public versions of Figs. 5, 6 and 12 replace the dated footer with “May* denotes a partial month.” The numerical panels and other labels retain their original pixels. The paired SVGs are the source artwork. Diagrams 2–4 contain embedded raster layers with editable vector annotations. The SVGs have no external file dependencies.

## Reproduce

From the repository root run the following command.

~~~bash
python scripts/figures.py
~~~

The command reads the 43,721 rows of [test predictions](../results/reference/test_predictions.csv.gz), rebuilds the plot series and checks 20 series tables in [source_data](source_data/). It then writes the plots in PNG and SVG under `outputs/figures/`. Options include `--verify-only` to check the numbers, `--output DIR` for another output directory, and `--predictions FILE` to check an alternate table with the same column schema. For exact reproduction of the paper figures this table must contain the published predictions. The four method diagrams have fixed source artwork and are copied on export. The 14 regenerated numerical plots use the same samples, groupings, methods and estimates with portable fonts and a simplified layout. The files under `published/` retain the typeset layout.

The saved plot tables record bootstrap interval endpoints for Figs. 11, 14 and S3. The computation repeats the synchronized 10,000-replicate target-time block bootstrap and checks the endpoints using the original comparison family. It also recomputes the plotted point estimates, counts, calibration bins, quantiles, histograms and Gaussian kernel densities from the public forecast series. The month labels refer to the forecast origin. May* covers the partial final month. Risk-score groups and observed curtailment transitions have separate columns in the common table.

## Figure index

| Figure | Subject | Publication artwork | Plotted series |
|---|---|---|---|
| 1 | Overall forecasting framework | [PNG](published/figure_01.png) · [SVG](published/figure_01.svg) | Method diagram |
| 2 | Pooling and calibration of curtailment scores | [PNG](published/figure_02.png) · [SVG](published/figure_02.svg) | Method diagram |
| 3 | Available-power increment and residual models | [PNG](published/figure_03.png) · [SVG](published/figure_03.svg) | Method diagram |
| 4 | Selection of the actual-power correction | [PNG](published/figure_04.png) · [SVG](published/figure_04.svg) | Method diagram |
| 5 | Errors by target curtailment status | [PNG](published/figure_05.png) · [SVG](published/figure_05.svg) | [group errors](source_data/05_group_errors.csv) |
| 6 | Fixed corrections and CRHF by farm and month | [PNG](published/figure_06.png) · [SVG](published/figure_06.svg) | [group errors](source_data/06_group_errors.csv) |
| 7 | Average precision and Brier score | [PNG](published/figure_07.png) · [SVG](published/figure_07.svg) | [scores](source_data/07_scores.csv) |
| 8 | Reliability diagrams, bin counts and calibration error | [PNG](published/figure_08.png) · [SVG](published/figure_08.svg) | [bins](source_data/08_bins.csv) · [calibration](source_data/08_calibration.csv) |
| 9 | Score and risk-shift distributions by observed transition | [PNG](published/figure_09.png) · [SVG](published/figure_09.svg) | [scores](source_data/09_scores.csv) |
| 10 | Available-power forecasts by input and correction | [PNG](published/figure_10.png) · [SVG](published/figure_10.svg) | [available](source_data/10_available.csv) |
| 11 | Current versus forecast available power in candidate corrections | [PNG](published/figure_11.png) · [SVG](published/figure_11.svg) | [candidates](source_data/11_candidates.csv) |
| 12 | Correction decisions by farm, month and risk regime | [PNG](published/figure_12.png) · [SVG](published/figure_12.svg) | [adoption](source_data/12_adoption.csv) |
| 13 | Errors at onset, release and persistence of curtailment | [PNG](published/figure_13.png) · [SVG](published/figure_13.svg) | [counts](source_data/13_counts.csv) · [transitions](source_data/13_transitions.csv) |
| 14 | Sensitivity to a systematic available-power input shift | [PNG](published/figure_14.png) · [SVG](published/figure_14.svg) | [bias](source_data/14_bias.csv) · [groups](source_data/14_groups.csv) |
| S1 | Overall forecasts, model comparisons and normalized signed errors | [PNG](published/figure_S1.png) · [SVG](published/figure_S1.svg) | [density](source_data/S1_density.csv) · [histogram](source_data/S1_histogram.csv) · [overall](source_data/S1_overall.csv) |
| S2 | Available-power improvement by farm and spatial input | [PNG](published/figure_S2.png) · [SVG](published/figure_S2.svg) | [available](source_data/S2_available.csv) |
| S3 | Forecast changes under auxiliary-input replacement | [PNG](published/figure_S3.png) · [SVG](published/figure_S3.svg) | [auxiliary](source_data/S3_auxiliary.csv) |
| S4 | Historical observation requirements and candidate gain scores | [PNG](published/figure_S4.png) · [SVG](published/figure_S4.svg) | [adoption](source_data/S4_adoption.csv) · [density](source_data/S4_density.csv) |
