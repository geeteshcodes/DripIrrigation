# False Positive Analysis — ExtraTreesReg @ 2.7 mm

**Dataset**: OOF predictions on 8,000 training rows (5-fold CV)
**Threshold**: 2.7 mm  |  **Total FP**: 185  |  **Total FN**: 33

## 1. Executive Summary

- **25 / 185 FPs (14%)** have predictions between 2.7 and 3.0 mm — they sit within 0.3 mm of the threshold.
- **77 / 185 FPs (42%)** occur in edge-case flagged rows.
- FP prediction range: 2.70 – 42.12 mm  (median 4.04 mm, mean 6.69 mm).

## 2. By Crop

| crop | n_total | FP_count | FN_count | FP_rate_% | FN_rate_% |
|---|---|---|---|---|---|
| sugarcane | 1343 | 45 | 1 | 3.35 | 0.07 |
| potato | 1330 | 37 | 4 | 2.78 | 0.3 |
| cotton | 1276 | 34 | 6 | 2.66 | 0.47 |
| maize | 1355 | 26 | 7 | 1.92 | 0.52 |
| wheat | 1327 | 24 | 1 | 1.81 | 0.08 |
| rice | 1369 | 19 | 14 | 1.39 | 1.02 |

## 3. By Soil Type

| soil_type | n_total | FP_count | FN_count | FP_rate_% | FN_rate_% |
|---|---|---|---|---|---|
| sandy | 2055 | 57 | 9 | 2.77 | 0.44 |
| loamy | 1970 | 47 | 2 | 2.39 | 0.1 |
| silt | 2018 | 44 | 17 | 2.18 | 0.84 |
| clay | 1957 | 37 | 5 | 1.89 | 0.26 |

## 4. By Growth Stage

| growth_stage | n_total | FP_count | FN_count | FP_rate_% | FN_rate_% |
|---|---|---|---|---|---|
| Vegetative | 2016 | 64 | 8 | 3.17 | 0.4 |
| Flowering | 1978 | 44 | 4 | 2.22 | 0.2 |
| Sowing | 1939 | 40 | 15 | 2.06 | 0.77 |
| Harvest | 2067 | 37 | 6 | 1.79 | 0.29 |

## 5. Prediction Proximity to Threshold

| pred_range_mm | FP_count | FP_pct | cumulative_pct |
|---|---|---|---|
| 2.7-3.0 | 25 | 13.5 | 13.5 |
| 3.0-3.5 | 45 | 24.3 | 37.8 |
| 3.5-4.0 | 20 | 10.8 | 48.6 |
| 4.0-5.0 | 28 | 15.1 | 63.7 |
| 5.0-10.0 | 35 | 18.9 | 82.6 |
| >10.0 | 32 | 17.3 | 99.9 |

> 25 FPs (14%) are in the 2.7–3.0 mm band.

## 6. Edge-Case Flag Distribution in FPs

| edge_reason | FP_count | FP_pct |
|---|---|---|
| no_edge_flag | 108 | 58.4 |
| label_conflict | 21 | 11.4 |
| high_wind | 20 | 10.8 |
| low_temp | 19 | 10.3 |
| heavy_rain | 16 | 8.6 |
| extreme_vpd | 10 | 5.4 |
| saturated_soil | 7 | 3.8 |
| very_dry_soil | 7 | 3.8 |
| near_zero_rain | 3 | 1.6 |

## 7. Crop x Soil Cross-Tab (top 15 by FP count)

| crop | soil_type | n_total | FP_count | FP_rate_% |
|---|---|---|---|---|
| sugarcane | sandy | 348 | 15 | 4.31 |
| cotton | sandy | 334 | 13 | 3.89 |
| potato | loamy | 334 | 13 | 3.89 |
| sugarcane | clay | 321 | 11 | 3.43 |
| maize | sandy | 320 | 11 | 3.44 |
| sugarcane | loamy | 337 | 10 | 2.97 |
| cotton | silt | 328 | 9 | 2.74 |
| sugarcane | silt | 337 | 9 | 2.67 |
| cotton | clay | 320 | 8 | 2.5 |
| maize | loamy | 342 | 8 | 2.34 |
| potato | clay | 306 | 8 | 2.61 |
| potato | sandy | 351 | 8 | 2.28 |
| potato | silt | 339 | 8 | 2.36 |
| wheat | silt | 328 | 7 | 2.13 |
| rice | silt | 328 | 6 | 1.83 |

## 8. Crop x Stage Cross-Tab (top 15 by FP count)

| crop | growth_stage | n_total | FP_count | FP_rate_% |
|---|---|---|---|---|
| sugarcane | Vegetative | 329 | 18 | 5.47 |
| sugarcane | Sowing | 319 | 16 | 5.02 |
| potato | Vegetative | 328 | 14 | 4.27 |
| cotton | Vegetative | 327 | 12 | 3.67 |
| potato | Flowering | 338 | 11 | 3.25 |
| wheat | Vegetative | 368 | 10 | 2.72 |
| potato | Harvest | 349 | 9 | 2.58 |
| cotton | Sowing | 316 | 8 | 2.53 |
| maize | Flowering | 355 | 8 | 2.25 |
| wheat | Flowering | 316 | 8 | 2.53 |
| cotton | Harvest | 322 | 7 | 2.17 |
| cotton | Flowering | 311 | 7 | 2.25 |
| maize | Harvest | 334 | 7 | 2.1 |
| sugarcane | Harvest | 368 | 6 | 1.63 |
| maize | Sowing | 333 | 6 | 1.8 |

## 9. Recommended Next Steps

1. **Identify the dominant (crop, stage) combo** in `fp_crop_x_stage.csv` with the highest FP rate — check if the FAO-56 target is structurally near-zero for that combination (e.g. cotton at Harvest after heavy rain).
2. **Examine `fp_full.csv`** sorted by `pred_distance_above_threshold` — the lowest values are borderline cases; look for shared feature patterns (high rainfall_mm, saturated soil_moisture) that could justify a rule-based override.
3. **Check rainfall_mm among FPs** — if heavy rain rows dominate, the model may not fully suppress irrigation when rainfall alone should suffice.
4. If a high-FP subgroup is found, consider a targeted post-hoc rule (e.g. 'suppress irrigation if rainfall_mm > X') rather than full retraining.

## 10. Output Files

All in `results/false_alarm_analysis/`:

- `fp_by_crop.csv` — FP/FN count and rate by crop
- `fp_by_soil.csv` — FP/FN count and rate by soil type
- `fp_by_stage.csv` — FP/FN count and rate by growth stage
- `fp_by_edge_reason.csv` — FP count by edge-case flag
- `fp_prediction_bins.csv` — FP distribution by predicted mm band
- `fp_crop_x_soil.csv` — Crop x soil FP cross-tab
- `fp_crop_x_stage.csv` — Crop x growth stage FP cross-tab
- `fp_full.csv` — All FP rows sorted by distance to threshold