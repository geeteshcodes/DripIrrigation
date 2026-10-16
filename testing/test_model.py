"""Test the selected proposed7 ExtraTrees model and log bad predictions.

Runs each predefined test-split row through the model and writes:
  results/model_test_predictions.csv  - every row and prediction
  results/model_test_bad_cases.csv   - decision errors or amount errors > 2 mm

Run from the project folder with: python .\test_model.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
BAD_AMOUNT_ERROR_MM = 2.0
IRRIGATION_THRESHOLD_MM = 2.0


def category_from_one_hot(row, prefix):
    selected = [name[len(prefix):] for name in row.index
                if name.startswith(prefix) and row[name] == 1]
    return selected[0] if selected else "unknown"


def main():
    data = pd.read_csv(BASE_DIR / "primary" / "primary_model_ready.csv")
    with (BASE_DIR / "feature_sets.json").open(encoding="utf-8") as f:
        features = json.load(f)["proposed7"]

    train = data[data["split"] == "train"].copy()
    test = data[data["split"] == "test"].copy()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[features])
    X_test = scaler.transform(test[features])

    # Best original proposed7 configuration from the training report.
    model = ExtraTreesRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=2,
        random_state=42,
        n_jobs=1,
    )
    model.fit(X_train, train["water_amount_mm"].to_numpy())

    # Predict in one call for compatibility with scikit-learn's forest internals,
    # then iterate through each input and prediction to build the row-by-row log.
    # Calling predict separately 2,000 times is slow and repeatedly enters its
    # parallel worker path.
    test_predictions = np.maximum(model.predict(X_test), 0.0)
    records = []
    for position, (source_index, row) in enumerate(test.iterrows()):
        prediction = float(test_predictions[position])
        actual = float(row["water_amount_mm"])
        error = prediction - actual
        actual_irrigate = int(row["irrigate"])
        predicted_irrigate = int(prediction >= IRRIGATION_THRESHOLD_MM)
        decision_wrong = predicted_irrigate != actual_irrigate

        records.append({
            "source_row_index": source_index,
            "crop": row.get("meta_crop", "unknown"),
            "soil_type": category_from_one_hot(row, "soil_type_"),
            "growth_stage": category_from_one_hot(row, "growth_stage_"),
            "soil_moisture": row["soil_moisture"],
            "temperature_c": row["temperature_c"],
            "rainfall_mm": row["rainfall_mm"],
            "humidity": row["humidity"],
            "actual_water_amount_mm": actual,
            "predicted_water_amount_mm": prediction,
            "error_mm": error,
            "absolute_error_mm": abs(error),
            "actual_irrigate": actual_irrigate,
            "predicted_irrigate": predicted_irrigate,
            "irrigation_decision_wrong": decision_wrong,
            "is_edge_case": int(row["is_edge_case"]),
            "bad_amount_error_over_2mm": abs(error) > BAD_AMOUNT_ERROR_MM,
        })

    predictions = pd.DataFrame(records)
    predictions["is_bad_case"] = (
        predictions["irrigation_decision_wrong"]
        | predictions["bad_amount_error_over_2mm"]
    )
    results_dir = BASE_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    all_path = results_dir / "model_test_predictions.csv"
    bad_path = results_dir / "model_test_bad_cases.csv"
    predictions.to_csv(all_path, index=False)
    bad_cases = predictions[predictions["is_bad_case"]].sort_values(
        ["irrigation_decision_wrong", "absolute_error_mm"],
        ascending=[False, False],
    )
    bad_cases.to_csv(bad_path, index=False)

    y_true = predictions["actual_water_amount_mm"]
    y_pred = predictions["predicted_water_amount_mm"]
    print(f"Model: ExtraTreesReg, proposed7, original settings")
    print(f"Test rows processed: {len(predictions)}")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.4f} mm")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f} mm")
    print(f"R²: {r2_score(y_true, y_pred):.4f}")
    print(f"Bad cases: {len(bad_cases)} (decision wrong OR absolute amount error > {BAD_AMOUNT_ERROR_MM:.1f} mm)")
    print(f"Decision errors: {int(predictions['irrigation_decision_wrong'].sum())}")
    print(f"Amount errors > {BAD_AMOUNT_ERROR_MM:.1f} mm: {int(predictions['bad_amount_error_over_2mm'].sum())}")
    print(f"All predictions: {all_path}")
    print(f"Bad-case log: {bad_path}")
    print("\nLargest errors:")
    print(bad_cases.head(10)[[
        "source_row_index", "crop", "soil_type", "growth_stage",
        "actual_water_amount_mm", "predicted_water_amount_mm",
        "absolute_error_mm", "irrigation_decision_wrong", "is_edge_case",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
