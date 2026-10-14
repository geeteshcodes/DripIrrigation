# Drip Irrigation Prediction — Machine Learning Pipeline

> **Regression-only architecture**: A single model predicts the irrigation water amount (`water_amount_mm`). The prediction itself encodes the binary irrigation decision — no separate classifier needed.
>
> `pred ≥ 2.0 mm` → **Irrigate** | `pred < 2.0 mm` → **No Irrigation**

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Dataset](#dataset)
4. [Feature Sets](#feature-sets)
5. [Models Trained](#models-trained)
6. [Key Results](#key-results)
7. [How to Run](#how-to-run)
8. [How to Use a Saved Model](#how-to-use-a-saved-model)
9. [Important Rules](#important-rules)

---

## Project Overview

This project builds a machine learning pipeline to predict **how much water (in mm) to apply** for drip irrigation of field crops. The pipeline is trained on a synthetic-but-realistic dataset of 10,000 crop-season records covering 6 crops (rice, maize, potato, wheat, cotton, sugarcane) across multiple soil types, growth stages, and seasons.

**Design decision**: Instead of training a separate binary classifier (irrigate / don't irrigate) on top of the regression model, the regression output alone drives the decision. If the model predicts ≥ 2.0 mm of irrigation is needed, the field is irrigated. If it predicts less, no irrigation is applied. This eliminates model redundancy and guarantees the decision is always consistent with the predicted amount.

---

## Repository Structure

```
work/
│
├── primary/
│   ├── primary_model_ready.csv        # 10,000 rows, numeric/one-hot, with split column
│   ├── primary_features_full.csv      # Human-readable version (strings, all columns)
│   ├── edge_cases.csv                 # Rows flagged as edge cases
│   └── per_crop/<crop>_model_ready.csv  # One file per crop
│
├── sensor/
│   └── iot_sensor_kalman.csv          # Noisy IoT stream with Kalman-filtered soil moisture
│
├── tomato/
│   └── tomato_features.csv            # Tomato data (not in primary dataset)
│
├── trained_models/
│   ├── sklearn_regressors/            # 44 joblib files (11 models × 4 feature sets)
│   ├── tensorflow/                    # 8 .keras files (2 models × 4 feature sets)
│   └── scaler_<featureset>.joblib     # StandardScaler per feature set (fit on train only)
│
├── results/
│   ├── regression_results.csv         # MAE, RMSE, R², + binary irrigation metrics (52 rows)
│   ├── edge_case_results.csv          # Same metrics on edge-case test rows only
│   ├── best_hyperparams.json          # Best hyperparameters per model
│   └── training_report.md            # Human-readable summary report
│
├── train_all_models.py                # Main training pipeline (run this to retrain)
├── generate_report.py                 # Standalone report generator
├── feature_engineering.py            # Feature engineering pipeline
├── feature_sets.json                 # Column lists for each feature set
├── data_dictionary.md                # Description of every column
├── feature_report.md                 # Feature engineering notes
└── HOW_TO_USE.md                     # Quick-start guide for the modelling team
```

---

## Dataset

| Property | Value |
|---|---|
| Total rows | 10,000 |
| Train rows | ~8,000 (80%) |
| Test rows | ~2,000 (20%) |
| Edge-case rows (test) | ~1,130 |
| Crops | Rice, Maize, Potato, Wheat, Cotton, Sugarcane |
| Split strategy | Stratified by crop × irrigation class, seed 42 |

**Targets**

| Column | Type | Description |
|---|---|---|
| `water_amount_mm` | Continuous ≥ 0 | Irrigation depth in mm (FAO-56 derived). **Primary regression target.** |
| `irrigate` | Binary 0/1 | 1 if `water_amount_mm > 0`. Used only to evaluate the binary decision from regression. |
| `need_ord` | Ordinal 0/1/2 | Dataset's own Low/Medium/High label. Honest native benchmark. |

**Edge cases** are rows with at least one of: low temperature, very dry soil, saturated soil, heavy rain, near-zero rain, high wind, extreme VPD, or conflicting labels. Performance on edge cases is reported separately.

---

## Feature Sets

Defined in [`feature_sets.json`](feature_sets.json). Four sets are used for training:

| Set | Features | Description |
|---|---|---|
| `baseline4` | 4 | Soil moisture, temperature, rainfall, humidity only |
| `proposed7` | 18 | baseline4 + crop type, soil type, growth stage (one-hot) |
| `extended_raw` | 26 | proposed7 + wind, sunlight, previous irrigation, season, mulching |
| `proposed7_plus_physics` | 31 | proposed7 + FAO-56 physics (ET₀, ETc, Kc, VPD, TAW, stress indices) |

> ⚠️ **Never use** columns in `leaky_for_amount_do_not_use_as_inputs` as model inputs. These columns (e.g. `dr_mm`, `raw_mm`, `below_mad`) are derived from the target and will cause data leakage.

---

## Models Trained

**52 total runs** (13 models × 4 feature sets).

### Sklearn Regressors (11 models)
LinearRegression, Ridge, Lasso, ElasticNet, DecisionTreeReg, RandomForestReg, HistGradientBoostingReg, AdaBoostReg, LinearSVR, KNNReg, ExtraTreesReg

### TensorFlow Neural Networks (2 models)
| Model | Architecture |
|---|---|
| `DNN_Regressor` | Dense(128) → BN → Dropout → Dense(64) → BN → Dropout → Dense(32) → Dense(1) |
| `WideDeep_Regressor` | Wide (linear) path + Deep path, concatenated → Dense(1) |

All TF models are trained with early stopping (patience=8 on val_MAE), with an 80/20 validation split from the training set only (no test-set leakage).

---

## Key Results

### Best Models — Full Test Set

| Rank | Model | Feature Set | MAE | R² | Irrigate F1 |
|---|---|---|---|---|---|
| 1 | ExtraTreesReg | proposed7_plus_physics | **0.71** | **0.9964** | 0.9699 |
| 2 | RandomForestReg | proposed7_plus_physics | 0.78 | 0.9963 | **0.9784** |
| 3 | HistGradientBoostingReg | proposed7_plus_physics | 0.90 | 0.9964 | 0.9638 |
| 4 | ExtraTreesReg | proposed7 | 0.75 | 0.9953 | 0.9699 |
| 5 | WideDeep_Regressor (TF) | extended_raw | 1.11 | 0.9860 | 0.9623 |

### Best Models — Edge Cases Only

| Rank | Model | Feature Set | MAE | Irrigate F1 |
|---|---|---|---|---|
| 1 | ExtraTreesReg | proposed7 | **0.77** | **0.9884** |
| 2 | WideDeep_Regressor (TF) | proposed7 | 1.17 | 0.9868 |
| 3 | RandomForestReg | proposed7 | 1.29 | 0.9790 |

> **Irrigate F1** = F1-score for the binary decision derived purely from the regression output using the 2.0 mm threshold. No classifier is trained or used.

See [`results/regression_results.csv`](results/regression_results.csv) and [`results/training_report.md`](results/training_report.md) for full tables.

---

## How to Run

### Requirements
```bash
pip install numpy pandas scikit-learn tensorflow joblib
```

### Retrain all models
```bash
python train_all_models.py
```

This trains all 52 models, saves them to `trained_models/`, and writes all result files to `results/`.

---

## How to Use a Saved Model

```python
import joblib
import numpy as np

# 1. Load the scaler and model for your chosen feature set
FEATURE_SET = "proposed7_plus_physics"
scaler = joblib.load(f"trained_models/scaler_{FEATURE_SET}.joblib")
model  = joblib.load(f"trained_models/sklearn_regressors/ExtraTreesReg__{FEATURE_SET}.joblib")

# 2. Prepare your input features (same columns as the feature set)
import json, pandas as pd
feature_cols = json.load(open("feature_sets.json"))[FEATURE_SET]
X_new = your_dataframe[feature_cols].values   # shape: (n_samples, n_features)

# 3. Scale and predict
X_scaled = scaler.transform(X_new)
water_mm  = model.predict(X_scaled).clip(min=0)   # predicted water amount in mm

# 4. Binary irrigation decision — no classifier needed
irrigate  = water_mm >= 2.0   # True = Irrigate, False = No Irrigation

print(f"Water amount : {water_mm} mm")
print(f"Irrigate?    : {irrigate}")
```

### For TensorFlow models

```python
import tensorflow as tf

model = tf.keras.models.load_model(
    f"trained_models/tensorflow/WideDeep_Regressor__{FEATURE_SET}.keras"
)
water_mm = model.predict(X_scaled, verbose=0).flatten().clip(min=0)
irrigate  = water_mm >= 2.0
```

---

## Important Rules

- **Do not re-split the data randomly.** Use the `split` column so all experiments are comparable.
- **Never use leaky columns as inputs** — `dr_mm`, `raw_mm`, `below_mad`, `eff_rain_window_mm`, `rain_per_day_mm`, `field_capacity`, `wilting_point` all derive from the target.
- **Never use** `meta_crop`, `split`, `irrigate`, `need_ord`, or `is_edge_case` as model features.
- **Always report edge-case performance separately** — use `test_df[test_df["is_edge_case"] == 1]`.
- **Clip predictions to 0** — irrigation amounts cannot be negative.
- **R² near 1.0 on physics feature sets is partly by construction** — `water_amount_mm` is FAO-56 derived using crop/soil/stage. For an honest independent benchmark, also check performance on the native `need_ord` label or on `baseline4` / `proposed7` feature sets.

---

## Reference

- FAO-56: Allen et al. (1998), *Crop Evapotranspiration — Guidelines for Computing Crop Water Requirements*.
- Data dictionary: [`data_dictionary.md`](data_dictionary.md)
- Full training report: [`results/training_report.md`](results/training_report.md)
