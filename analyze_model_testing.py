"""Review test-set false alarms and validate irrigation thresholds on train CV.

Run: python .\analyze_model_testing.py

The held-out test rows are used only for descriptive false-alarm review.
Threshold comparisons use out-of-fold predictions from training rows only.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
THRESHOLDS_MM = np.round(np.arange(0.5, 10.01, 0.1), 1).tolist()


def main():
    primary_path = BASE_DIR / "primary" / "primary_model_ready.csv"
    readable_path = BASE_DIR / "primary" / "primary_features_full.csv"
    data = pd.read_csv(primary_path)
    readable = pd.read_csv(readable_path)
    out_dir = BASE_DIR / "results"

    # Descriptive audit of false alarms from the existing test run.
    bad_path = out_dir / "model_test_bad_cases.csv"
    bad = pd.read_csv(bad_path)
    false_alarms = bad[
        (bad["actual_irrigate"] == 0) & (bad["predicted_irrigate"] == 1)
    ].copy()
    review_cols = [
        "crop_type", "soil_type", "growth_stage", "soil_moisture",
        "temperature_c", "rainfall_mm", "humidity", "water_amount_mm",
        "irrigate", "need_class", "need_ord", "edge_reasons", "is_edge_case",
    ]
    source_details = readable[review_cols].copy()
    source_details.index.name = "source_row_index"
    review = false_alarms.merge(
        source_details.reset_index(), on="source_row_index", how="left"
    )
    review_path = out_dir / "false_alarm_review.csv"
    review.sort_values("absolute_error_mm", ascending=False).to_csv(review_path, index=False)

    print(f"Test false alarms reviewed: {len(review)}")
    print("False alarms by native need class:")
    print(review["need_class"].value_counts(dropna=False).to_string())
    print("\nFalse alarms by edge-case reason:")
    print(review["edge_reasons"].value_counts(dropna=False).head(12).to_string())
    print(f"\nDetailed review: {review_path}")

    # Generate honest training-only out-of-fold amount predictions.
    with (BASE_DIR / "feature_sets.json").open(encoding="utf-8") as f:
        features = json.load(f)["proposed7"]
    train = data[data["split"] == "train"].copy().reset_index(drop=True)
    X = train[features].to_numpy()
    y_amount = train["water_amount_mm"].to_numpy()
    y_irrigate = train["irrigate"].to_numpy().astype(int)
    strata = train["meta_crop"].astype(str) + "_" + train["irrigate"].astype(str)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_predictions = np.zeros(len(train), dtype=float)
    for fold, (fit_idx, valid_idx) in enumerate(cv.split(X, strata), start=1):
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X[fit_idx])
        X_valid = scaler.transform(X[valid_idx])
        model = ExtraTreesRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=2,
            random_state=42,
            n_jobs=1,
        )
        model.fit(X_fit, y_amount[fit_idx])
        oof_predictions[valid_idx] = np.maximum(model.predict(X_valid), 0.0)
        print(f"Completed threshold validation fold {fold}/5")

    threshold_rows = []
    for threshold in THRESHOLDS_MM:
        predicted = (oof_predictions >= threshold).astype(int)
        tp = int(((predicted == 1) & (y_irrigate == 1)).sum())
        fp = int(((predicted == 1) & (y_irrigate == 0)).sum())
        fn = int(((predicted == 0) & (y_irrigate == 1)).sum())
        tn = int(((predicted == 0) & (y_irrigate == 0)).sum())
        threshold_rows.append({
            "threshold_mm": threshold,
            "accuracy": accuracy_score(y_irrigate, predicted),
            "precision": precision_score(y_irrigate, predicted, zero_division=0),
            "recall": recall_score(y_irrigate, predicted, zero_division=0),
            "f1": f1_score(y_irrigate, predicted, zero_division=0),
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
        })

    threshold_results = pd.DataFrame(threshold_rows)
    threshold_path = out_dir / "threshold_validation.csv"
    threshold_results.to_csv(threshold_path, index=False, float_format="%.6f")
    print("\nTraining-only out-of-fold threshold comparison (dense 0.1 mm grid saved to CSV).")
    both_under_100 = threshold_results[
        (threshold_results["false_positive"] < 100)
        & (threshold_results["false_negative"] < 100)
    ]
    if len(both_under_100):
        print("Thresholds with both false alarms and missed irrigations below 100:")
        print(both_under_100[[
            "threshold_mm", "false_positive", "false_negative", "precision", "recall", "f1"
        ]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    else:
        threshold_results["worst_error_count"] = threshold_results[
            ["false_positive", "false_negative"]
        ].max(axis=1)
        closest = threshold_results.sort_values(
            ["worst_error_count", "false_positive", "false_negative"]
        ).iloc[0]
        print("No threshold reached both targets (<100 each). Closest threshold by minimizing the larger error count:")
        print(closest[[
            "threshold_mm", "false_positive", "false_negative", "precision", "recall", "f1"
        ]].to_string(float_format=lambda x: f"{x:.4f}"))

    print("\nSelected threshold checkpoints:")
    checkpoints = threshold_results[
        np.isclose(threshold_results["threshold_mm"] % 0.5, 0.0, atol=1e-8)
    ]
    print(checkpoints[[
        "threshold_mm", "accuracy", "precision", "recall", "f1",
        "false_positive", "false_negative"
    ]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    best_f1 = threshold_results.loc[threshold_results["f1"].idxmax()]
    print(
        f"\nHighest validation F1: threshold {best_f1['threshold_mm']:.1f} mm "
        f"(F1={best_f1['f1']:.4f}, precision={best_f1['precision']:.4f}, "
        f"recall={best_f1['recall']:.4f})"
    )
    print(f"Full threshold table: {threshold_path}")


if __name__ == "__main__":
    main()
