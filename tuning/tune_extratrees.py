"""Tune ExtraTreesReg on proposed7 using the predefined train/test split.

Run from the DripIrrigation project folder with:
    python tune_extratrees.py
"""

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score

# Some scikit-learn/joblib combinations repeat this non-fatal warning for each
# parallel CV job. It does not affect model fitting or the reported scores.
warnings.filterwarnings(
    "ignore",
    message=r"`sklearn\.utils\.parallel\.delayed` should be used with `sklearn\.utils\.parallel\.Parallel`.*",
    category=UserWarning,
)


def report_metrics(label, y_true, predictions, irrigation_true):
    irrigation_pred = (predictions >= 2.0).astype(int)
    print(f"\n{label} (n={len(y_true)})")
    print(f"MAE: {mean_absolute_error(y_true, predictions):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, predictions)):.4f}")
    print(f"R²: {r2_score(y_true, predictions):.4f}")
    print(f"Irrigation accuracy: {accuracy_score(irrigation_true, irrigation_pred):.4f}")
    print(f"Irrigation precision: {precision_score(irrigation_true, irrigation_pred, zero_division=0):.4f}")
    print(f"Irrigation recall: {recall_score(irrigation_true, irrigation_pred, zero_division=0):.4f}")
    print(f"Irrigation F1: {f1_score(irrigation_true, irrigation_pred, zero_division=0):.4f}")


def main():
    df = pd.read_csv("primary/primary_model_ready.csv")
    with open("feature_sets.json", encoding="utf-8") as f:
        features = json.load(f)["proposed7"]

    train = df[df["split"] == "train"].copy()
    test = df[df["split"] == "test"].copy()
    X_train = train[features]
    y_train = train["water_amount_mm"]
    X_test = test[features]
    y_test = test["water_amount_mm"].to_numpy()

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    baseline = ExtraTreesRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1,
    )
    baseline_scores = -cross_val_score(
        baseline, X_train, y_train, cv=cv,
        scoring="neg_mean_absolute_error", n_jobs=-1
    )
    print(f"Baseline CV MAE: {baseline_scores.mean():.4f} ± {baseline_scores.std():.4f}")

    search = RandomizedSearchCV(
        estimator=ExtraTreesRegressor(random_state=42, n_jobs=-1),
        param_distributions={
            "n_estimators": [300, 500, 800],
            "max_depth": [10, 20, 30, None],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": [0.6, 0.8, 1.0],
        },
        n_iter=60,
        scoring="neg_mean_absolute_error",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    search.fit(X_train, y_train)

    print(f"\nBest CV MAE: {-search.best_score_:.4f}")
    print(f"Best parameters: {search.best_params_}")

    predictions = np.maximum(search.best_estimator_.predict(X_test), 0)
    irrigation_true = test["irrigate"].to_numpy().astype(int)
    edge = test["is_edge_case"].to_numpy() == 1
    report_metrics("Full test set", y_test, predictions, irrigation_true)
    report_metrics("Edge-case test rows", y_test[edge], predictions[edge], irrigation_true[edge])
    report_metrics("Non-edge-case test rows", y_test[~edge], predictions[~edge], irrigation_true[~edge])


if __name__ == "__main__":
    main()
