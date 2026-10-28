# FP Cluster Investigation Report

**Threshold**: 2.7 mm  |  **Total FP (OOF)**: 185

---

## Cluster 1 — Sugarcane × Vegetative / Sowing

**FP count**: 34  |  **Comparable TN count**: 387

### FP count by stage (within Sugarcane)

| growth_stage | FP_count |
|---|---|
| Vegetative | 18 |
| Sowing | 16 |

### FP count by soil type (within Sugarcane Veg/Sow)

| soil_type | FP_count |
|---|---|
| clay | 9 |
| loamy | 9 |
| sandy | 9 |
| silt | 7 |

### Feature comparison: FP vs TN rows (top differentiators)

Columns: feature | FP median | TN median | delta (FP−TN)

| feature | FP_n | FP_median | TN_median | delta_median |
|---|---|---|---|---|
| rainfall_mm | 34 | 1186.45 | 1317.94 | -131.49 |
| soil_moisture | 34 | 35.925 | 49.61 | -13.685 |
| dr_mm | 34 | 39.1696 | 29.5542 | 9.6154 |
| wind_speed_kmh | 34 | 12.77 | 9.49 | 3.28 |
| eff_rain_window_mm | 34 | 23.729 | 26.3588 | -2.6298 |
| rain_per_day_mm | 34 | 9.8871 | 10.9828 | -1.0958 |
| temperature_c | 34 | 27.51 | 26.44 | 1.07 |
| demand_minus_rain_mm | 34 | -4.2508 | -5.2352 | 0.9843 |

> **Rainfall_mm**: FP median = 1186.45, TN median = 1317.94 (delta = -131.49)
> **demand_minus_rain_mm**: FP median = -4.2508, TN median = -5.2352

### What this tells us

- FP and TN rows have similar rainfall (FP 1186 mm, TN 1318 mm). The cause is likely in other features (VPD, Kc, soil).
- `rain_covers_etc` (flag: rain ≥ ETc): FP mean = 0.68, TN mean = 0.83. **Rain covers ETc for many FPs** — the model is ignoring the rain-covers flag.

---

## Cluster 2 — Heavy Rain FPs

**FP count**: 16  |  **Heavy-rain TN count**: 401

### FP breakdown by crop

| crop | FP_count |
|---|---|
| potato | 7 |
| maize | 4 |
| cotton | 2 |
| wheat | 2 |
| rice | 1 |

### Feature comparison: heavy-rain FP vs heavy-rain TN (top differentiators)

| feature | FP_n | FP_median | TN_median | delta_median |
|---|---|---|---|---|
| dr_mm | 16 | 47.4044 | 28.397 | 19.0074 |
| humidity | 16 | 43.725 | 62.31 | -18.585 |
| rainfall_mm | 16 | 2386.27 | 2402.61 | -16.34 |
| soil_moisture | 16 | 36.59 | 44.35 | -7.76 |
| wind_speed_kmh | 16 | 6.505 | 9.81 | -3.305 |
| et0_mm | 16 | 5.6559 | 5.0381 | 0.6178 |
| vpd_kpa | 16 | 1.7397 | 1.2762 | 0.4636 |
| etc_mm | 16 | 4.3096 | 3.894 | 0.4156 |

> **rainfall_mm in heavy-rain FPs**: min=2300.6, median=2386.27, max=2481.94
> **rainfall_mm in heavy-rain TNs**: min=2300.78, median=2402.61, max=2499.58

### What this tells us

- `rain_covers_etc` mean for heavy-rain FPs = **1.00** → Rain does cover ETc for most heavy-rain FPs. The model is **not learning the rain-covers-etc signal** for these rows.
- Check `demand_minus_rain_mm` median: if negative for FPs, model ignores surplus rain.
  - Median demand_minus_rain_mm for heavy-rain FPs = **-11.70 mm** (negative → rain surplus; model fired anyway)

---

## Cluster 3 — Label Conflict FPs

**FP count**: 21  |  **Label-conflict TN count**: 38

### What is a label conflict?

A label conflict is flagged when the FAO-56 derived `water_amount_mm` target is > 0 but the binary `irrigate` label is 0 (or vice-versa). In such rows the regression target and binary label disagree — the model may produce predictions that reflect the continuous FAO signal while the binary ground truth says 'no irrigation'.

### Distribution of actual_water_amount_mm for label-conflict FPs

*(These rows are classed as FP because predicted ≥ 2.7 mm and actual < 2.7 mm, but the actual value is non-zero — borderline ground truth.)*

- count: 21  min: 2.0052  median: 2.4044  mean: 2.3607  max: 2.6432
- Rows where actual > 0 mm: **21 / 21** (100%) — genuine borderline cases.

### FP breakdown by crop

| crop | FP_count |
|---|---|
| potato | 8 |
| rice | 5 |
| maize | 3 |
| sugarcane | 2 |
| wheat | 2 |
| cotton | 1 |

### Feature comparison: label-conflict FP vs label-conflict TN (top differentiators)

| feature | FP_n | FP_median | TN_median | delta_median |
|---|---|---|---|---|
| rainfall_mm | 21 | 1321.91 | 539.885 | 782.025 |
| eff_rain_window_mm | 21 | 26.4382 | 10.7977 | 15.6405 |
| soil_moisture | 21 | 32.87 | 43.485 | -10.615 |
| temperature_c | 21 | 23.51 | 32.56 | -9.05 |
| humidity | 21 | 53.96 | 62.12 | -8.16 |
| rain_per_day_mm | 21 | 11.0159 | 4.499 | 6.5169 |
| wind_speed_kmh | 21 | 9.19 | 14.795 | -5.605 |
| demand_minus_rain_mm | 21 | -5.0207 | -1.0801 | -3.9406 |

### What this tells us

Label conflicts fall into two sub-types:

| Sub-type | Description | Model behaviour |
|---|---|---|
| **Ambiguous zero** | actual = 0 but FAO physics suggest partial need | Model picks up the physics signal; binary ground truth may be wrong |
| **Soft borderline** | actual > 0 but < 2.7 mm threshold | Model and FAO agree there's some need; only the threshold matters |

These are the **least actionable** FPs — they stem from inherent label noise, not model error. Removing or down-weighting label-conflict rows during training is the most principled fix (not post-hoc rule-based).

---

## Summary & Recommended Actions

| Cluster | FP Count | Root Cause | Recommended Fix |
|---|---|---|---|
| Sugarcane Veg/Sow | 34 | Model predicts high demand for sugarcane Veg/Sow regardless of rain | Add `rain_covers_etc` or `demand_minus_rain_mm` as explicit suppressor features; or post-hoc rule |
| Heavy Rain | 16 | demand_minus_rain_mm positive (ETc > rain) even in heavy-rain rows | Review ETc calculation for these crops; consider `eff_rain_window_mm` feature weight |
| Label Conflict | 21 | Ground truth label noise — FAO physics and binary label disagree | Remove/down-weight label-conflict rows in training, or set a 'conflict confidence' flag |

**Unique FPs covered by these 3 clusters**: 66 / 185 (36%)
**Remaining unexplained FPs**: 119 — worth checking with feature SHAP values on a per-row basis to find any other systematic drivers.

## Output Files

All in `results/fp_cluster_investigation/`:

- `cluster1_sugarcane_veg_sow.csv` — Full FP rows for cluster 1
- `cluster1_feature_comparison.csv` — FP vs TN feature stats for cluster 1
- `cluster2_heavy_rain.csv` — Full FP rows for cluster 2
- `cluster2_feature_comparison.csv` — FP vs TN feature stats for cluster 2
- `cluster3_label_conflict.csv` — Full FP rows for cluster 3
- `cluster3_feature_comparison.csv` — FP vs TN feature stats for cluster 3
- `cluster_investigation_report.md` — This report