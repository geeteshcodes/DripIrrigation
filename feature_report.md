# Feature engineering report

## 1. Run log
- primary: 10000 rows loaded, 0 duplicates dropped, 0 missing cells.
- tomato json: 393 rows; irrigation_needed = ETc - moisture% when moisture% < min threshold else 0 -> rule reproduced exactly: True; ETc = ETo x Kc holds: True.
- tomato csv: 3000 rows, 28 exact duplicates dropped.
- tomato csv: ETc/ETo disagrees with Kc in 90% of rows; pH outside 3.5-9.5 in 28%; ET values reach 1099 (implausible as mm/day) -> flagged low quality, no derived target created.
- iot: 4688 rows, 7 glitch rows (temp 178.7, pressure -2120, negative ADC, unparsable altitude) masked -> Kalman predicts through them; mean |step| of dryness 0.0121 raw -> 0.0058 filtered; max step 0.767 -> 0.274. Timestamp columns are constant (single date/time for all rows), so row order (id) is treated as time order.
- hydraulic: Raw 1329 rows (pressure 1/2, flow rate); Filtered keeps 911 rows with identical values (it only trims start-up/shut-down transients, same values: True) and has 9 summary rows (MEAN/MEDIAN/STD DEV) appended. These describe drip-line hydraulics, not crop water need -> not used as features.

## 2. What the primary dataset can and cannot support

Rows per crop: rice=1711, maize=1694, sugarcane=1678, potato=1663, wheat=1659, cotton=1595

Tomato is NOT in the primary dataset; see the separate tomato table.

Native irrigation-need class share by crop (flat = crop carries no signal):

| crop_type   |   High |   Low |   Medium |
|:------------|-------:|------:|---------:|
| cotton      |  0.034 | 0.582 |    0.384 |
| maize       |  0.04  | 0.576 |    0.384 |
| potato      |  0.031 | 0.574 |    0.396 |
| rice        |  0.026 | 0.602 |    0.372 |
| sugarcane   |  0.039 | 0.588 |    0.373 |
| wheat       |  0.032 | 0.596 |    0.372 |

By soil type:

| soil_type   |   High |   Low |   Medium |
|:------------|-------:|------:|---------:|
| clay        |  0.034 | 0.582 |    0.384 |
| loamy       |  0.03  | 0.599 |    0.372 |
| sandy       |  0.038 | 0.583 |    0.379 |
| silt        |  0.032 | 0.582 |    0.386 |

By growth stage (large differences = real signal):

| growth_stage   |   High |   Low |   Medium |
|:---------------|-------:|------:|---------:|
| Flowering      |  0.062 | 0.349 |    0.589 |
| Harvest        |  0.004 | 0.814 |    0.182 |
| Sowing         |  0.004 | 0.826 |    0.171 |
| Vegetative     |  0.065 | 0.355 |    0.58  |

### Ablation: native Low/Medium/High class (RandomForest, 3-fold CV, macro-F1)

| feature set | macro-F1 |
|---|---|
| baseline4 (moisture, temp, rain, humidity) | 0.5783 |
| baseline4 + growth_stage | 0.7276 |
| baseline4 + crop_type | 0.5721 |
| baseline4 + soil_type | 0.5657 |
| proposed7 (+crop, soil, stage) | 0.7276 |
| extended_raw (+wind, sunlight, mulching, ...) | 0.9338 |
| proposed7 + physics features | 0.732 |

### Ablation: derived water amount (RandomForest, held-out test split)

CAUTION: the derived target is built from crop, soil and stage parameters, so any gain from those columns here is partly by construction (circular).

| feature set | MAE (mm) | R2 |
|---|---|---|
| baseline4 | 15.833 | 0.3229 |
| proposed7 | 1.181 | 0.9926 |
| proposed7 + physics | 0.791 | 0.9962 |

## 3. Target design

- `need_class` / `need_ord`: the dataset's own Low/Medium/High label (0/1/2). The dataset has NO water-amount column.
- `water_amount_mm` (>= 0) and `irrigate` (1 if > 0): derived FAO-56 style. Irrigate when root-zone depletion exceeds the readily-available water (MAD); amount = depletion - effective rain over 3 days, capped at 75 mm; amounts under 2 mm count as no irrigation. Soil_Moisture is read as % of plant-available water; Rainfall_mm is read as a season total (120 days).
- Agreement with the native label (row = native class, column = derived irrigate flag):

| need_class   |     0 |     1 |
|:-------------|------:|------:|
| High         | 0.089 | 0.911 |
| Low          | 0.646 | 0.354 |
| Medium       | 0.398 | 0.602 |

- Derived irrigate share overall: 46.7%; mean positive amount 39.4 mm; min amount 0.00.

## 4. Edge-case rows (for 'edge cases / low temperature testing')

`is_edge_case` and `edge_reasons` mark: low_temp (<=15 C), very_dry_soil, saturated_soil, heavy_rain, near_zero_rain, high_wind, extreme_vpd (top 5%), label_conflict (derived flag disagrees strongly with the native label). Counts:

| flag | rows |
|---|---|
| low_temp | 1000 |
| very_dry_soil | 698 |
| saturated_soil | 916 |
| heavy_rain | 740 |
| near_zero_rain | 178 |
| high_wind | 1022 |
| extreme_vpd | 500 |
| label_conflict | 2108 |

Total edge-case rows: 5576 (1127 in the test split). Negative outputs: the target is floored at 0; clip model predictions with max(0, pred).

## 5. Tomato table

| source           |   rows |
|:-----------------|-------:|
| csv_augmented    |   2972 |
| json_real_sensor |    393 |

Only `json_real_sensor` has a source-provided irrigation amount and it is extremely imbalanced (14 positive rows). The csv rows are flagged where ET/pH values are implausible.

## 6. IoT sensor stream (Kalman)

Q=0.01, R=0.05. Dryness = ADC/480 (higher ADC = drier). Use this file to demonstrate noise handling; it has no crop/soil/stage columns so it is not a model-training set.

## 7. Assumptions to state in the paper

- FAO-56 Kc, root depth, MAD, field capacity and wilting point are textbook midpoints.
- ET0 uses Penman-Monteith with assumed 12 h day length, sea-level pressure, 2 m wind and season-based Ra.
- The primary dataset behaves like a rule-generated (synthetic) dataset: its label follows thresholds on soil moisture, rainfall, temperature, wind, growth stage and mulching. State this in the limitations.
