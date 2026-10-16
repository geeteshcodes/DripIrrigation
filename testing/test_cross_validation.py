"""Five-fold CV for the original ExtraTreesReg proposed7 configuration.

Uses training rows only. Writes testing/cv_results.csv and does not alter any
source data, trained model, or existing result file.

Run from the project root:
    python .\testing\test_cross_validation.py
"""

import json
from pathlib import Path

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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
IRRIGATION_THRESHOLD_MM = 2.0
FOLDS = 5
SEED = 42


def regression_metrics(y_true, y_pred):
    return {
        "mae_mm": mean_absolute_error(y_true, y_pred),
        "rmse_mm": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan,
        "mean_error_mm": float(np.mean(y_pred - y_true)),
    }


def irrigation_metrics(y_true, y_pred, irrigate_true):
    irrigate_pred = (y_pred >= IRRIGATION_THRESHOLD_MM).astype(int)
    return {
        "irrigation_accuracy": accuracy_score(irrigate_true, irrigate_pred),
        "irrigation_precision": precision_score(
            irrigate_true, irrigate_pred, zero_division=0
        ),
        "irrigation_recall": recall_score(
            irrigate_true, irrigate_pred, zero_division=0
        ),
        "irrigation_f1": f1_score(
            irrigate_true, irrigate_pred, zero_division=0
        ),
        "false_alarms": int(((irrigate_true == 0) & (irrigate_pred == 1)).sum()),
        "missed_irrigations": int(((irrigate_true == 1) & (irrigate_pred == 0)).sum()),
        "predicted_irrigation_rate": float(np.mean(irrigate_pred)),
    }


def main():
    data = pd.read_csv(ROOT / "primary" / "primary_model_ready.csv")
    with (ROOT / "feature_sets.json").open(encoding="utf-8") as f:
        features = json.load(f)["proposed7"]

    train = data.loc[data["split"] == "train"].reset_index(drop=True)
    X = train[features].to_numpy()
    y = train["water_amount_mm"].to_numpy(dtype=float)
    irrigate = train["irrigate"].to_numpy(dtype=int)

    # Balance each crop x recorded irrigation-label combination across folds.
    # The label is used only to form comparable folds, never as a model input.
    strata = train["meta_crop"].astype(str) + "|" + train["irrigate"].astype(str)
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    rows = []

    for fold, (fit_idx, valid_idx) in enumerate(splitter.split(X, strata), start=1):
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X[fit_idx])
        X_valid = scaler.transform(X[valid_idx])

        # Preserve the original proposed7 ExtraTreesReg configuration.
        model = ExtraTreesRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=2,
            random_state=SEED,
            n_jobs=1,
        )
        model.fit(X_fit, y[fit_idx])
        y_pred = np.maximum(model.predict(X_valid), 0.0)
        y_valid = y[valid_idx]
        irrigate_valid = irrigate[valid_idx]
        predicted_irrigate = (y_pred >= IRRIGATION_THRESHOLD_MM).astype(int)

        # `irrigate` is 1 for any positive target (>0), while the predicted
        # decision is positive only at >=2 mm. Count the records in that gap.
        label_threshold_mismatch = (y_valid > 0.0) & (y_valid < IRRIGATION_THRESHOLD_MM)

        overall = {
            "record_type": "fold_overall",
            "fold": fold,
            "scenario": "all",
            "n": len(valid_idx),
            **regression_metrics(y_valid, y_pred),
            **irrigation_metrics(y_valid, y_pred, irrigate_valid),
            "positive_labels_below_2mm": int(label_threshold_mismatch.sum()),
            "positive_labels_below_2mm_rate": float(label_threshold_mismatch.mean()),
        }
        rows.append(overall)

        scenarios = {
            "no_irrigation_target_0mm": y_valid == 0.0,
            "positive_target_below_2mm": (y_valid > 0.0) & (y_valid < IRRIGATION_THRESHOLD_MM),
            "target_at_least_2mm": y_valid >= IRRIGATION_THRESHOLD_MM,
        }
        for scenario, mask in scenarios.items():
            scenario_row = {
                "record_type": "fold_scenario",
                "fold": fold,
                "scenario": scenario,
                "n": int(mask.sum()),
                "predicted_irrigation_rate": (
                    float(predicted_irrigate[mask].mean()) if mask.any() else np.nan
                ),
                "positive_labels_below_2mm": int(
                    (label_threshold_mismatch & mask).sum()
                ),
                "positive_labels_below_2mm_rate": (
                    float((label_threshold_mismatch & mask).sum() / mask.sum())
                    if mask.any() else np.nan
                ),
            }
            if mask.any():
                scenario_row.update(regression_metrics(y_valid[mask], y_pred[mask]))
            rows.append(scenario_row)

        irrigation = irrigation_metrics(y_valid, y_pred, irrigate_valid)
        regression = regression_metrics(y_valid, y_pred)
        print(
            f"Fold {fold}/{FOLDS} (n={len(valid_idx)}): "
            f"MAE={regression['mae_mm']:.4f} mm, "
            f"RMSE={regression['rmse_mm']:.4f} mm, R2={regression['r2']:.4f}; "
            f"irrigation accuracy={irrigation['irrigation_accuracy']:.4f}, "
            f"precision={irrigation['irrigation_precision']:.4f}, "
            f"recall={irrigation['irrigation_recall']:.4f}, "
            f"F1={irrigation['irrigation_f1']:.4f}; "
            f"false alarms={irrigation['false_alarms']}, "
            f"missed={irrigation['missed_irrigations']}, "
            f"0<actual<2 mm label mismatch={int(label_threshold_mismatch.sum())}"
        )
        for scenario, mask in scenarios.items():
            rate = float(predicted_irrigate[mask].mean()) if mask.any() else float("nan")
            print(
                f"  {scenario}: n={int(mask.sum())}, "
                f"MAE={mean_absolute_error(y_valid[mask], y_pred[mask]):.4f} mm, "
                f"predicted-irrigation rate={rate:.4f}"
                if mask.any() else f"  {scenario}: n=0"
            )

    fold_results = pd.DataFrame(rows)
    overall_folds = fold_results[fold_results["record_type"] == "fold_overall"]
    summary_metrics = [
        "mae_mm", "rmse_mm", "r2", "mean_error_mm", "irrigation_accuracy",
        "irrigation_precision", "irrigation_recall", "irrigation_f1",
        "false_alarms", "missed_irrigations", "predicted_irrigation_rate",
        "positive_labels_below_2mm", "positive_labels_below_2mm_rate",
    ]
    for stat_name, values in (
        ("mean", overall_folds[summary_metrics].mean()),
        ("std", overall_folds[summary_metrics].std(ddof=1)),
    ):
        rows.append({
            "record_type": f"fold_overall_{stat_name}",
            "fold": stat_name,
            "scenario": "all",
            "n": int(overall_folds["n"].sum()),
            **values.to_dict(),
        })

    # Add per-scenario means and standard deviations across folds as well.
    per_fold_scenarios = pd.DataFrame(
        [row for row in rows if row["record_type"] == "fold_scenario"]
    )
    scenario_metrics = [
        "mae_mm", "rmse_mm", "r2", "mean_error_mm", "predicted_irrigation_rate",
        "positive_labels_below_2mm", "positive_labels_below_2mm_rate",
    ]
    for scenario, group in per_fold_scenarios.groupby("scenario", sort=False):
        for stat_name, values in (
            ("mean", group[scenario_metrics].mean()),
            ("std", group[scenario_metrics].std(ddof=1)),
        ):
            rows.append({
                "record_type": f"scenario_{stat_name}",
                "fold": stat_name,
                "scenario": scenario,
                "n": int(group["n"].sum()),
                **values.to_dict(),
            })

    output = ROOT / "testing" / "cv_results.csv"
    pd.DataFrame(rows).to_csv(output, index=False, float_format="%.6f")

    print("\nFive-fold mean ± standard deviation (overall):")
    for metric in summary_metrics:
        mean = overall_folds[metric].mean()
        std = overall_folds[metric].std(ddof=1)
        print(f"  {metric}: {mean:.4f} ± {std:.4f}")
    print("\nLabel definition check:")
    print("  Recorded irrigate label: 1 when water_amount_mm > 0.")
    print("  Predicted irrigation decision: 1 when clipped prediction >= 2.0 mm.")
    print(
        "  Positive target amounts between 0 and 2 mm are recorded as irrigate=1 "
        "but fall below the prediction decision threshold."
    )
    print(
        "  Such rows in validation folds: "
        f"{int(overall_folds['positive_labels_below_2mm'].sum())} / "
        f"{int(overall_folds['n'].sum())}"
    )
    print(f"\nSaved new CV results to: {output}")


if __name__ == "__main__":
    main()
