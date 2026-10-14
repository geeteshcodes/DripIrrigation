"""
=============================================================================
Multi-Model Training Pipeline — Irrigation Prediction (Regression-Only)
=============================================================================
Trains 11 sklearn regressors and 2 TensorFlow neural networks across
4 feature sets (52 total runs).

Design principle
----------------
A single regression model predicts `water_amount_mm`.  The regression
output itself encodes the irrigation decision — no separate classifier needed:

    pred >= IRRIGATE_THRESHOLD (2.0 mm)  →  Irrigate  (binary label = 1)
    pred <  IRRIGATE_THRESHOLD           →  No Irrigation (binary label = 0)

Binary irrigation metrics (accuracy, precision, recall, F1) are computed
directly from the clipped regression predictions.

Outputs
-------
  trained_models/sklearn_regressors/   — joblib-saved sklearn regressors
  trained_models/tensorflow/           — Keras .keras saved models
  results/regression_results.csv       — regression + binary irrigation metrics
  results/edge_case_results.csv        — edge-case-only metrics
  results/best_hyperparams.json        — best hyperparams per model
  results/training_report.md           — human-readable summary report
=============================================================================
"""

import os, sys, json, time, warnings, datetime
print("Loading numpy, pandas ...", flush=True)
import numpy as np
import pandas as pd
import joblib

# ── Sklearn ──────────────────────────────────────────────────────────────────
print("Loading sklearn ...", flush=True)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, precision_recall_fscore_support
)

# Regressors
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, HistGradientBoostingRegressor,
    AdaBoostRegressor, ExtraTreesRegressor
)
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor

# ── TensorFlow ───────────────────────────────────────────────────────────────
print("Loading TensorFlow (this may take a moment) ...", flush=True)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
print(f"  TensorFlow {tf.__version__} loaded. GPU: {tf.config.list_physical_devices('GPU') or 'None (CPU only)'}", flush=True)

warnings.filterwarnings("ignore")
np.random.seed(42)
tf.random.set_seed(42)

# Force unbuffered output so progress prints appear immediately
import functools
print = functools.partial(print, flush=True)

# =============================================================================
# 0. DIRECTORIES
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIRS = [
    os.path.join(BASE_DIR, "trained_models", "sklearn_regressors"),
    os.path.join(BASE_DIR, "trained_models", "tensorflow"),
    os.path.join(BASE_DIR, "results"),
]
for d in DIRS:
    os.makedirs(d, exist_ok=True)

print("=" * 80)
print("  IRRIGATION PREDICTION — REGRESSION-ONLY TRAINING PIPELINE")
print("=" * 80)
print(f"  Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# =============================================================================
# 1. LOAD DATA & FEATURE SETS
# =============================================================================
print("[1/6] Loading data ...")
df = pd.read_csv(os.path.join(BASE_DIR, "primary", "primary_model_ready.csv"))
with open(os.path.join(BASE_DIR, "feature_sets.json")) as f:
    feature_sets = json.load(f)

# Remove the leaky list — it's not a feature set to train on
LEAKY_COLS = feature_sets.pop("leaky_for_amount_do_not_use_as_inputs", [])

train_df = df[df["split"] == "train"].copy()
test_df  = df[df["split"] == "test"].copy()

# Edge-case subsets
test_edge  = test_df[test_df["is_edge_case"] == 1]
test_normal = test_df[test_df["is_edge_case"] == 0]

print(f"  Total rows  : {len(df):,}")
print(f"  Train rows  : {len(train_df):,}")
print(f"  Test rows   : {len(test_df):,}")
print(f"  Edge (test) : {len(test_edge):,}")
print(f"  Feature sets: {list(feature_sets.keys())}")
print()

# Target & threshold
REG_TARGET         = "water_amount_mm"
IRRIGATE_THRESHOLD = 2.0   # mm — pred >= threshold -> irrigate

# =============================================================================
# 2. MODEL DEFINITIONS  (regressors only)
# =============================================================================

# ── Sklearn Regressors + Hyperparameter Grids ─────────────────────────────────
SKLEARN_REGRESSORS = {
    "LinearRegression": {
        "model": LinearRegression(),
        "params": {},  # no tuning — baseline
    },
    "Ridge": {
        "model": Ridge(),
        "params": {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    },
    "Lasso": {
        "model": Lasso(max_iter=5000),
        "params": {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0]},
    },
    "ElasticNet": {
        "model": ElasticNet(max_iter=5000),
        "params": {
            "alpha": [0.01, 0.1, 1.0],
            "l1_ratio": [0.2, 0.5, 0.8],
        },
    },
    "DecisionTreeReg": {
        "model": DecisionTreeRegressor(random_state=42),
        "params": {
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
        },
    },
    "RandomForestReg": {
        # n_jobs=1 here; parallelism handled by GridSearchCV n_jobs=-1
        "model": RandomForestRegressor(random_state=42, n_jobs=1),
        "params": {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
        },
    },
    "HistGradientBoostingReg": {
        # 10-50x faster than GradientBoostingRegressor, native histogram binning
        "model": HistGradientBoostingRegressor(random_state=42),
        "params": {
            "max_iter": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7],
        },
    },
    "AdaBoostReg": {
        "model": AdaBoostRegressor(random_state=42),
        "params": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 1.0],
        },
    },
    "LinearSVR": {
        # LinearSVR is O(n) vs SVR's O(n^2-n^3) — massive speedup
        "model": LinearSVR(max_iter=5000, random_state=42),
        "params": {
            "C": [0.01, 0.1, 1.0, 10.0],
            "epsilon": [0.01, 0.1, 0.5],
        },
    },
    "KNNReg": {
        "model": KNeighborsRegressor(n_jobs=1),
        "params": {
            "n_neighbors": [3, 5, 7, 11],
            "weights": ["uniform", "distance"],
        },
    },
    "ExtraTreesReg": {
        # n_jobs=1 here; parallelism handled by GridSearchCV n_jobs=-1
        "model": ExtraTreesRegressor(random_state=42, n_jobs=1),
        "params": {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
        },
    },
}

# =============================================================================
# 3. HELPER FUNCTIONS
# =============================================================================

def regression_metrics(y_true, y_pred, irrigate_true=None):
    """Return dict of regression + binary irrigation metrics.

    Binary irrigation decision derived from regression:
        pred >= IRRIGATE_THRESHOLD  ->  irrigate (1)
        pred <  IRRIGATE_THRESHOLD  ->  no irrigation (0)

    When `irrigate_true` is supplied, also computes accuracy, precision,
    recall, and F1 for the binary irrigation decision — no classifier needed.
    """
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    result = {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}

    if irrigate_true is not None and len(irrigate_true) > 0:
        irrigate_pred = (y_pred >= IRRIGATE_THRESHOLD).astype(int)
        irr_acc  = accuracy_score(irrigate_true, irrigate_pred)
        irr_f1   = f1_score(irrigate_true, irrigate_pred, zero_division=0)
        prec, rec, _, _ = precision_recall_fscore_support(
            irrigate_true, irrigate_pred, average="binary", zero_division=0
        )
        result["Irrigate_Accuracy"]  = round(irr_acc,       4)
        result["Irrigate_Precision"] = round(float(prec),   4)
        result["Irrigate_Recall"]    = round(float(rec),    4)
        result["Irrigate_F1"]        = round(irr_f1,        4)
    return result


def train_sklearn_model(name, model, params, X_train, y_train, scoring, cv=3):
    """Train a single sklearn model with optional GridSearch. Returns best model, best params, time."""
    t0 = time.time()
    if params:
        # GridSearch for small grids (<=10 combos), RandomizedSearch otherwise
        n_combos = 1
        for v in params.values():
            n_combos *= len(v)
        if n_combos <= 10:
            print(f"GridSearchCV ({n_combos} combos x {cv} folds = {n_combos * cv} fits) ", end="", flush=True)
            search = GridSearchCV(
                model, params, cv=cv, scoring=scoring,
                n_jobs=-1, refit=True, error_score="raise"
            )
        else:
            n_iter = 10
            print(f"RandomizedSearchCV ({n_iter}/{n_combos} combos x {cv} folds = {n_iter * cv} fits) ", end="", flush=True)
            search = RandomizedSearchCV(
                model, params, n_iter=n_iter, cv=cv, scoring=scoring,
                n_jobs=-1, refit=True, random_state=42, error_score="raise"
            )
        search.fit(X_train, y_train)
        best_model  = search.best_estimator_
        best_params = search.best_params_
    else:
        print(f"fit (no tuning) ", end="", flush=True)
        model.fit(X_train, y_train)
        best_model  = model
        best_params = {}
    elapsed = round(time.time() - t0, 2)
    return best_model, best_params, elapsed


# =============================================================================
# 4. TF MODEL BUILDERS  (regression only)
# =============================================================================

def build_dnn_regressor(input_dim, lr=0.001):
    """Deep Neural Network for regression."""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="linear"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse", metrics=["mae"]
    )
    return model


def build_wide_and_deep_regressor(input_dim, lr=0.001):
    """Wide & Deep model for regression."""
    inputs = layers.Input(shape=(input_dim,))

    # Wide path (linear)
    wide = layers.Dense(1, activation="linear", name="wide")(inputs)

    # Deep path
    deep = layers.Dense(128, activation="relu")(inputs)
    deep = layers.BatchNormalization()(deep)
    deep = layers.Dropout(0.3)(deep)
    deep = layers.Dense(64, activation="relu")(deep)
    deep = layers.BatchNormalization()(deep)
    deep = layers.Dropout(0.2)(deep)
    deep = layers.Dense(32, activation="relu")(deep)

    # Merge
    merged = layers.Concatenate()([wide, deep])
    output = layers.Dense(1, activation="linear", name="output")(merged)

    model = keras.Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse", metrics=["mae"]
    )
    return model


def train_tf_model(build_fn, X_train, y_train,
                   model_name="dnn", fset_name="baseline4"):
    """Train a TF regression model with hyperparameter sweep.

    FIX: Uses an 80/20 split from TRAINING data for validation (not the test
    set), preventing data leakage into HP selection and early stopping.

    Returns best model, best hps, time.
    """
    # ── Hold out 20% of training data for validation (no test-set leakage) ──
    n = len(X_train)
    idx = np.random.permutation(n)
    val_size = int(0.2 * n)
    val_idx, trn_idx = idx[:val_size], idx[val_size:]
    X_trn, y_trn = X_train[trn_idx], y_train[trn_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    learning_rates = [0.001, 0.0005]
    batch_sizes    = [128, 256]        # larger batches -> fewer steps -> faster

    best_score = np.inf    # minimize val_mae
    best_model = None
    best_hp    = {}
    total_combos = len(learning_rates) * len(batch_sizes)
    combo_num = 0
    t0 = time.time()

    for lr in learning_rates:
        for bs in batch_sizes:
            combo_num += 1
            print(f"\n        HP {combo_num}/{total_combos}: lr={lr}, batch={bs} ... ", end="", flush=True)
            tf.keras.backend.clear_session()

            model    = build_fn(X_trn.shape[1], lr=lr)
            monitor  = "val_mae"

            es = callbacks.EarlyStopping(
                monitor=monitor, patience=8, mode="min",
                restore_best_weights=True, verbose=0
            )

            history = model.fit(
                X_trn, y_trn,
                validation_data=(X_val, y_val),
                epochs=150, batch_size=bs,
                callbacks=[es], verbose=0
            )

            epochs_ran = len(history.history.get(monitor, []))
            score  = min(history.history["val_mae"])
            marker = "star" if score < best_score else " "
            print(f"{epochs_ran} epochs, val_mae={score:.4f} {marker}", end="", flush=True)
            if score < best_score:
                best_score = score
                best_model = model
                best_hp = {"learning_rate": lr, "batch_size": bs,
                           "epochs_run": epochs_ran}

    elapsed = round(time.time() - t0, 2)
    return best_model, best_hp, elapsed


# =============================================================================
# 5. MAIN TRAINING LOOP
# =============================================================================
reg_results     = []   # regression + binary irrigation results rows
edge_results    = []   # edge-case results rows
all_best_params = {}   # model -> best params

# 11 sklearn regressors + 2 TF regressors, per feature set
total_models  = (len(SKLEARN_REGRESSORS) + 2) * len(feature_sets)
model_counter = 0

for fset_name, fset_cols in feature_sets.items():
    print(f"\n{'━' * 80}")
    print(f"  FEATURE SET: {fset_name}  ({len(fset_cols)} features)")
    print(f"{'━' * 80}")

    # Verify columns exist
    missing = [c for c in fset_cols if c not in train_df.columns]
    if missing:
        print(f"  Warning: Missing columns {missing}, skipping feature set.")
        continue

    X_train = train_df[fset_cols].values
    X_test  = test_df[fset_cols].values

    y_train_reg = train_df[REG_TARGET].values
    y_test_reg  = test_df[REG_TARGET].values

    irrigate_test      = test_df["irrigate"].values
    irrigate_test_edge = test_edge["irrigate"].values

    X_test_edge     = test_edge[fset_cols].values
    y_test_edge_reg = test_edge[REG_TARGET].values

    # Scale features
    scaler = StandardScaler()
    X_train_sc     = scaler.fit_transform(X_train)
    X_test_sc      = scaler.transform(X_test)
    X_test_edge_sc = scaler.transform(X_test_edge) if len(X_test_edge) > 0 else np.array([])

    # Save scaler
    scaler_path = os.path.join(BASE_DIR, "trained_models", f"scaler_{fset_name}.joblib")
    joblib.dump(scaler, scaler_path)

    # ── 5A. SKLEARN REGRESSORS ───────────────────────────────────────────────
    print(f"\n  ── Sklearn Regressors ({len(SKLEARN_REGRESSORS)} models) ──")
    for name, cfg in SKLEARN_REGRESSORS.items():
        model_counter += 1
        tag = f"{name}__{fset_name}"
        print(f"    [{model_counter}/{total_models}] {tag} ... ", end="", flush=True)

        try:
            best_model, best_params, elapsed = train_sklearn_model(
                name, cfg["model"], cfg["params"],
                X_train_sc, y_train_reg,
                scoring="neg_mean_absolute_error", cv=3
            )

            # Predict & clip (irrigation amounts are never negative)
            preds   = best_model.predict(X_test_sc).clip(min=0)
            metrics = regression_metrics(y_test_reg, preds, irrigate_test)
            metrics.update({"Model": name, "FeatureSet": fset_name,
                            "Time_s": elapsed, "Status": "OK"})
            reg_results.append(metrics)

            # Edge-case metrics
            if len(X_test_edge_sc) > 0:
                preds_edge = best_model.predict(X_test_edge_sc).clip(min=0)
                e_metrics  = regression_metrics(y_test_edge_reg, preds_edge, irrigate_test_edge)
                e_metrics.update({"Model": name, "FeatureSet": fset_name, "Subset": "edge_case"})
                edge_results.append(e_metrics)

            # Save model
            model_path = os.path.join(
                BASE_DIR, "trained_models", "sklearn_regressors", f"{tag}.joblib"
            )
            joblib.dump(best_model, model_path)
            all_best_params[tag] = best_params

            irr_f1 = metrics.get("Irrigate_F1", "—")
            irr_f1_str = f"{irr_f1:.4f}" if isinstance(irr_f1, float) else str(irr_f1)
            print(f"MAE={metrics['MAE']:.4f}  R2={metrics['R2']:.4f}  IrrigateF1={irr_f1_str}  ({elapsed}s)")

        except Exception as e:
            print(f"FAILED: {e}")
            reg_results.append({
                "Model": name, "FeatureSet": fset_name,
                "MAE": None, "RMSE": None, "R2": None,
                "Status": f"ERROR: {e}", "Time_s": 0
            })

    # ── 5B. TENSORFLOW REGRESSION MODELS ────────────────────────────────────
    print(f"\n  ── TensorFlow Regression Models (2 models) ──")
    tf_models_to_train = [
        ("DNN_Regressor",      build_dnn_regressor),
        ("WideDeep_Regressor", build_wide_and_deep_regressor),
    ]

    for tf_name, build_fn in tf_models_to_train:
        model_counter += 1
        tag = f"{tf_name}__{fset_name}"
        print(f"    [{model_counter}/{total_models}] {tag} ... ", end="", flush=True)

        try:
            best_model, best_hp, elapsed = train_tf_model(
                build_fn, X_train_sc, y_train_reg,
                model_name=tf_name, fset_name=fset_name
            )

            preds   = best_model.predict(X_test_sc, verbose=0).flatten().clip(min=0)
            metrics = regression_metrics(y_test_reg, preds, irrigate_test)
            metrics.update({"Model": tf_name, "FeatureSet": fset_name,
                            "Time_s": elapsed, "Status": "OK"})
            reg_results.append(metrics)

            # Edge-case
            if len(X_test_edge_sc) > 0:
                preds_edge = best_model.predict(X_test_edge_sc, verbose=0).flatten().clip(min=0)
                e_metrics  = regression_metrics(y_test_edge_reg, preds_edge, irrigate_test_edge)
                e_metrics.update({"Model": tf_name, "FeatureSet": fset_name, "Subset": "edge_case"})
                edge_results.append(e_metrics)

            # Save TF model
            model_save_path = os.path.join(
                BASE_DIR, "trained_models", "tensorflow", f"{tag}.keras"
            )
            best_model.save(model_save_path)
            all_best_params[tag] = {k: str(v) for k, v in best_hp.items()}

            irr_f1 = metrics.get("Irrigate_F1", "—")
            irr_f1_str = f"{irr_f1:.4f}" if isinstance(irr_f1, float) else str(irr_f1)
            print(f"MAE={metrics['MAE']:.4f}  R2={metrics['R2']:.4f}  IrrigateF1={irr_f1_str}  ({elapsed}s)")

        except Exception as e:
            print(f"FAILED: {e}")
            reg_results.append({
                "Model": tf_name, "FeatureSet": fset_name,
                "MAE": None, "RMSE": None, "R2": None,
                "Status": f"ERROR: {e}", "Time_s": 0
            })

# =============================================================================
# 6. SAVE RESULTS
# =============================================================================
print(f"\n{'=' * 80}")
print("  SAVING RESULTS")
print(f"{'=' * 80}")

results_dir = os.path.join(BASE_DIR, "results")

reg_df  = pd.DataFrame(reg_results)
edge_df = pd.DataFrame(edge_results)

reg_df.to_csv(os.path.join(results_dir, "regression_results.csv"),  index=False)
edge_df.to_csv(os.path.join(results_dir, "edge_case_results.csv"),  index=False)

with open(os.path.join(results_dir, "best_hyperparams.json"), "w") as f:
    json.dump(all_best_params, f, indent=2, default=str)

print(f"  OK regression_results.csv    ({len(reg_df)} rows)")
print(f"  OK edge_case_results.csv     ({len(edge_df)} rows)")
print(f"  OK best_hyperparams.json")

# =============================================================================
# 7. GENERATE REPORT
# =============================================================================
print("\n  Generating training_report.md ...")

report_lines = []
report_lines.append("# Irrigation Prediction — Regression-Only Training Report\n")
report_lines.append(f"**Generated**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
report_lines.append(f"**Dataset**: primary_model_ready.csv ({len(df):,} rows, "
                     f"train={len(train_df):,}, test={len(test_df):,})\n")
report_lines.append(
    f"> **Architecture**: A single regression model predicts `water_amount_mm`. "
    f"Predictions >= {IRRIGATE_THRESHOLD} mm are classified as **Irrigate**; "
    f"predictions < {IRRIGATE_THRESHOLD} mm are **No Irrigation**. "
    f"No separate classification model is used.\n"
)
report_lines.append("---\n")

# ── REGRESSION + BINARY IRRIGATION RESULTS ────────────────────────────────────
report_lines.append("## 1. Regression Results with Binary Irrigation Decision\n")
report_lines.append(f"> **Threshold**: pred >= {IRRIGATE_THRESHOLD} mm -> Irrigate | pred < {IRRIGATE_THRESHOLD} mm -> No Irrigation\n")

reg_ok = reg_df[reg_df["Status"] == "OK"].copy()
if len(reg_ok) > 0:
    # Best by MAE
    best_mae_row = reg_ok.loc[reg_ok["MAE"].idxmin()]
    report_lines.append(
        f"> **Best Regression (lowest MAE)**: {best_mae_row['Model']} on "
        f"`{best_mae_row['FeatureSet']}` — "
        f"MAE={best_mae_row['MAE']:.4f}, RMSE={best_mae_row['RMSE']:.4f}, "
        f"R2={best_mae_row['R2']:.4f}\n"
    )
    # Best by Irrigate_F1
    if "Irrigate_F1" in reg_ok.columns and reg_ok["Irrigate_F1"].notna().any():
        best_f1_row = reg_ok.loc[reg_ok["Irrigate_F1"].idxmax()]
        report_lines.append(
            f"> **Best Irrigation Decision (highest F1)**: {best_f1_row['Model']} on "
            f"`{best_f1_row['FeatureSet']}` — "
            f"F1={best_f1_row['Irrigate_F1']:.4f}, "
            f"Precision={best_f1_row.get('Irrigate_Precision', 0):.4f}, "
            f"Recall={best_f1_row.get('Irrigate_Recall', 0):.4f}\n"
        )

    # Per-feature-set tables
    for fs in feature_sets.keys():
        subset = reg_ok[reg_ok["FeatureSet"] == fs].sort_values("MAE")
        if len(subset) == 0:
            continue
        report_lines.append(f"\n### Feature Set: `{fs}`\n")
        report_lines.append(
            "| Rank | Model | MAE | RMSE | R2 | "
            "Irrigate Acc | Irrigate Prec | Irrigate Rec | Irrigate F1 | Time (s) |"
        )
        report_lines.append(
            "|------|-------|-----|------|----|"
            "--------------|---------------|--------------|-------------|----------|"
        )
        for rank, (_, row) in enumerate(subset.iterrows(), 1):
            def _fmt(col, r=row):
                v = r.get(col, "—")
                return f"{v:.4f}" if isinstance(v, float) else str(v)
            report_lines.append(
                f"| {rank} | {row['Model']} | {row['MAE']:.4f} | {row['RMSE']:.4f} | "
                f"{row['R2']:.4f} | {_fmt('Irrigate_Accuracy')} | "
                f"{_fmt('Irrigate_Precision')} | {_fmt('Irrigate_Recall')} | "
                f"{_fmt('Irrigate_F1')} | {row['Time_s']} |"
            )
        report_lines.append("")

# ── EDGE-CASE SUMMARY ─────────────────────────────────────────────────────────
report_lines.append("\n## 2. Edge-Case Performance\n")
report_lines.append(f"Edge-case test rows: {len(test_edge):,}\n")

if len(edge_df) > 0 and "MAE" in edge_df.columns:
    edge_reg_valid = edge_df.dropna(subset=["MAE"])
    if len(edge_reg_valid) > 0:
        report_lines.append("\n### Regression + Binary Irrigation (edge cases only)\n")
        report_lines.append("| Model | Feature Set | MAE | RMSE | R2 | Irrigate F1 |")
        report_lines.append("|-------|-------------|-----|------|----|-------------|")
        for _, row in edge_reg_valid.sort_values("MAE").head(20).iterrows():
            irr_f1_str = (
                f"{row['Irrigate_F1']:.4f}"
                if isinstance(row.get("Irrigate_F1"), float)
                else "—"
            )
            report_lines.append(
                f"| {row['Model']} | {row['FeatureSet']} | {row['MAE']:.4f} | "
                f"{row['RMSE']:.4f} | {row['R2']:.4f} | {irr_f1_str} |"
            )
        report_lines.append("")

# ── BEST HYPERPARAMETERS ──────────────────────────────────────────────────────
report_lines.append("\n## 3. Best Hyperparameters\n")
report_lines.append("| Model | Best Parameters |")
report_lines.append("|-------|-----------------|")
for tag, params in all_best_params.items():
    params_str = (
        ", ".join(f"{k}={v}" for k, v in params.items())
        if params else "default (no tuning)"
    )
    report_lines.append(f"| {tag} | {params_str} |")
report_lines.append("")

# ── SAVED MODELS ──────────────────────────────────────────────────────────────
report_lines.append("\n## 4. Saved Models\n")
report_lines.append("| Directory | Contents |")
report_lines.append("|-----------|----------|")
for subdir in ["sklearn_regressors", "tensorflow"]:
    full = os.path.join(BASE_DIR, "trained_models", subdir)
    if os.path.exists(full):
        files = os.listdir(full)
        report_lines.append(f"| `trained_models/{subdir}/` | {len(files)} models |")
report_lines.append("")

# ── NOTES ─────────────────────────────────────────────────────────────────────
report_lines.append("\n## 5. Notes\n")
report_lines.append("- **Single-model architecture**: the regression output drives both the water amount and the irrigation on/off decision.")
report_lines.append(f"- Binary rule: pred >= {IRRIGATE_THRESHOLD} mm -> Irrigate; pred < {IRRIGATE_THRESHOLD} mm -> No Irrigation.")
report_lines.append("- All predictions clipped to max(0, pred) — negative irrigation amounts are impossible.")
report_lines.append("- water_amount_mm is FAO-56 derived (partly by construction with crop/soil/stage).")
report_lines.append("- Train/test split uses the pre-defined `split` column (no random re-splitting).")
report_lines.append("- StandardScaler fitted on train set only; scalers saved alongside models.")
report_lines.append("- No separate classifier is saved or needed: zero/near-zero regression output = no irrigation.")
report_lines.append("")

report_text = "\n".join(report_lines)
report_path = os.path.join(results_dir, "training_report.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_text)

print(f"  OK training_report.md")

# =============================================================================
# 8. FINAL SUMMARY
# =============================================================================
end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"\n{'=' * 80}")
print("  TRAINING COMPLETE  (Regression-Only Pipeline)")
print(f"{'=' * 80}")
print(f"  Finished : {end_time}")
print(f"  Models   : {model_counter} regressors trained")

if len(reg_ok) > 0:
    best = reg_ok.loc[reg_ok["MAE"].idxmin()]
    print(f"  Best MAE : {best['Model']} ({best['FeatureSet']}) — MAE={best['MAE']:.4f}, R2={best['R2']:.4f}")
    if "Irrigate_F1" in reg_ok.columns and reg_ok["Irrigate_F1"].notna().any():
        best_f1 = reg_ok.loc[reg_ok["Irrigate_F1"].idxmax()]
        print(f"  Best F1  : {best_f1['Model']} ({best_f1['FeatureSet']}) — IrrigateF1={best_f1['Irrigate_F1']:.4f}")

print(f"\n  Irrigation decision: pred >= {IRRIGATE_THRESHOLD} mm -> Irrigate | pred < {IRRIGATE_THRESHOLD} mm -> No Irrigation")
print(f"  Results  : results/")
print(f"  Models   : trained_models/")
print(f"{'=' * 80}")
