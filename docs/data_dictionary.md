# Released observations

The public inputs are the three power measurements used by the paper, at their native sampling cadences. Each file identifies a farm only by its paper code, F1–F5. The files contain relative time, retain the measured MW values and preserve the alignment of simultaneous observations across sites. Empty cells are missing measurements. The native files contain one record per observed minute offset, after parsing, merging channels on their time keys and retaining the last reading at duplicate times as in the experiment. They preserve gaps in the original recordings. Vendor exports, sensor tag names, geographic metadata and the private site-code lookup are excluded.

## Native measurements

`data/native/F1.csv.gz` through `F5.csv.gz` each contain four columns.

| Column | Unit | Meaning |
| --- | --- | --- |
| `minute_offset` | minute | Whole minutes from a common, undisclosed recording origin. Zero is the start of the first 15-minute interval. |
| `actual_operational` | MW | Measured active power delivered by the wind farm (P). |
| `potential` | MW | Site-reported available active power (A). |
| `dispatch_history` | MW | Observed historical dispatch active-power setpoint (D). |

The public clock is relative. The software can use an equivalent surrogate calendar to evaluate seasonal features. Its years refer to a computational convention. The test-month labels in the figures follow the manuscript. The exact power series and coincident changes are retained so that predictions and error metrics can be reproduced. The release removes direct identifiers; parties with external observations could still link time series by their patterns.

## Analysis grid

`data/observations_15min.csv.gz` contains one row per farm and completed 15-minute interval, including unusable intervals. Its unique key is (`farm`, `slot`). The first slot is 1; slot k aggregates native offsets in [15(k−1), 15k) and becomes available at offset 15k. An issue at slot k predicts slot k+4.

| Column | Unit or type | Meaning |
| --- | --- | --- |
| `farm` | F1–F5 | Farm identifier used in the paper. |
| `slot` | 15-minute intervals | Common interval-end index. |
| `actual_operational`, `potential`, `dispatch_history` | MW | Mean of the corresponding native signal, masked if the interval fails input quality control. |
| `sample_coverage` | fraction | Minimum of the three valid reading counts divided by the expected count per channel. Non-finite and negative values are excluded. |
| `input_valid` | Boolean | All three signals have full coverage and the suspected-outage rule does not apply. |
| `suspected_outage` | Boolean | Actual output ≤ 0.05 MW, with both available power and dispatch above 3% of the farm scale. Computed separately from coverage. |
| `capacity_proxy` | MW | Fixed historical power scale: 99.5th percentile of strictly positive available power in complete calibration-prefix intervals through slot 2,976 inclusive, with a lower bound of 1 MW. The scale is fitted before suspected outages are removed. |
| `teacher_probability` | unit interval | Current-state score calculated from the observed positive dispatch gap and relative tracking proximity. |
| `curtailed` | 0 or 1 | Current-interval curtailed label. Requires a valid interval, a positive dispatch gap strictly above 3% of the farm scale, and positive tracking proximity. One qualifying interval suffices. |
| `curtailment_mw` | MW | max(A − P, 0) in a flagged interval and zero in other valid intervals. |
| `severity` | unit interval | Curtailment MW divided by max(A, 0.1 MW), clipped to [0, 1]. |

The direct observed score uses g = max(A − D, 0), u = |P − A| − |P − D| and s = `capacity_proxy`. Then `teacher_probability` = sigmoid((g − 0.03s)/(0.02s)) × sigmoid(u/(0.02s)), with each sigmoid argument clipped to [−40, 40]. This current-state score differs from the forecast probability for slot k+4. Invalid intervals retain their place in the grid, coverage, outage flag and scale; the three binned signals and four label fields are missing. A separate target-consistency filter is applied when evaluating the paper’s power forecasts.

| Farm | Native cadence (min) | Native rows | 15-minute rows | Incomplete coverage | Additional suspected outages | Valid rows | Scale (MW, rounded) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F1 | 5 | 90,495 | 30,221 | 5,424 | 287 | 24,510 | 26.74 |
| F2 | 5 | 90,498 | 30,222 | 4,717 | 1 | 25,504 | 22.33 |
| F3 | 5 | 90,526 | 30,231 | 2,229 | 7 | 27,995 | 110.55 |
| F4 | 1 | 452,344 | 30,213 | 2,126 | 0 | 28,087 | 308.64 |
| F5 | 5 | 90,491 | 30,219 | 3,948 | 43 | 26,228 | 40.81 |
| All | | 814,354 | 151,106 | 18,444 | 338 | 132,324 | |

The figures and tables use 43,721 common eligible test targets after applying the 1 h target horizon, observation validity, the evaluation-window boundary and the target power consistency filter. The larger grid also includes calibration, training and validation records and invalid intervals.

## Rebuild from native observations

From the repository root, run:

```sh
python scripts/prepare_observations.py
```

The script rebuilds the grid from all five native files and compares every column with the released grid, including missing values. Numeric checks allow absolute tolerance 1e-12 and relative tolerance 1e-13 for platform rounding. To save a recomputed copy after verification:

```sh
python scripts/prepare_observations.py --output outputs/observations_15min.csv.gz
```
