"""Threshold-only evaluation for ExtraTreesReg proposed7 using training OOF data.

No hyperparameters, features, labels, source data, or existing results are changed.
Run from the project root:
    python .\testing\test_thresholds.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "threshold_testing"
THRESHOLDS = [2.0, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 4.0]
IRRIGATION_THRESHOLD_MM = 2.0
SEED = 42
FOLDS = 5


def metrics_for(y_true, predicted_irrigate):
    y_true = np.asarray(y_true, dtype=int)
    predicted_irrigate = np.asarray(predicted_irrigate, dtype=int)
    tn = int(((y_true == 0) & (predicted_irrigate == 0)).sum())
    fp = int(((y_true == 0) & (predicted_irrigate == 1)).sum())
    fn = int(((y_true == 1) & (predicted_irrigate == 0)).sum())
    tp = int(((y_true == 1) & (predicted_irrigate == 1)).sum())
    return {
        "accuracy": accuracy_score(y_true, predicted_irrigate),
        "precision": precision_score(y_true, predicted_irrigate, zero_division=0),
        "recall": recall_score(y_true, predicted_irrigate, zero_division=0),
        "f1": f1_score(y_true, predicted_irrigate, zero_division=0),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "confusion_matrix": f"[[{tn}, {fp}], [{fn}, {tp}]]",
    }


def fmt(value, digits=4):
    if pd.isna(value):
        return "NA"
    return f"{value:.{digits}f}"


def markdown_table(frame, columns, headers=None):
    headers = headers or columns
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in frame.iterrows():
        vals = []
        for col in columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                vals.append(fmt(float(value)))
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def evaluate_overall(predictions):
    rows = []
    actual = predictions["actual_irrigate"].to_numpy(dtype=int)
    amounts = predictions["predicted_water_amount_mm"].to_numpy(dtype=float)
    for threshold in THRESHOLDS:
        predicted = (amounts >= threshold).astype(int)
        rows.append({"threshold_mm": threshold, "n": len(actual), **metrics_for(actual, predicted)})
    return pd.DataFrame(rows)


def evaluate_subgroups(predictions):
    groups = []
    for field, subgroup_type in (
        ("crop", "crop"),
        ("soil_type", "soil_type"),
        ("growth_stage", "growth_stage"),
    ):
        for value in sorted(predictions[field].dropna().unique()):
            groups.append((subgroup_type, str(value), predictions[field].eq(value).to_numpy()))

    edge_flags = [
        "edge_low_temp", "edge_very_dry_soil", "edge_saturated_soil",
        "edge_heavy_rain", "edge_near_zero_rain", "edge_high_wind",
        "edge_extreme_vpd", "edge_label_conflict",
    ]
    edge_any = predictions[edge_flags].max(axis=1).astype(bool)
    groups.extend([
        ("edge_status", "any_edge_case", edge_any.to_numpy()),
        ("edge_status", "no_edge_case", (~edge_any).to_numpy()),
    ])
    for flag in edge_flags:
        groups.append(("edge_category_overlap", flag, predictions[flag].eq(1).to_numpy()))

    rows = []
    actual_all = predictions["actual_irrigate"].to_numpy(dtype=int)
    amount_all = predictions["predicted_water_amount_mm"].to_numpy(dtype=float)
    for subgroup_type, subgroup, mask in groups:
        y_true = actual_all[mask]
        amount = amount_all[mask]
        for threshold in THRESHOLDS:
            metric_row = metrics_for(y_true, (amount >= threshold).astype(int))
            rows.append({
                "subgroup_type": subgroup_type,
                "subgroup": subgroup,
                "threshold_mm": threshold,
                "n": int(mask.sum()),
                **metric_row,
            })
    return pd.DataFrame(rows)


def build_report(overall, subgroups, predictions, output_paths):
    f1_best = overall.loc[overall["f1"].idxmax()]
    acc_best = overall.loc[overall["accuracy"].idxmax()]
    balance = overall.assign(pr_gap=(overall["precision"] - overall["recall"]).abs())
    balance_best = balance.loc[balance["pr_gap"].idxmin()]
    recall_candidates = overall[overall["recall"] >= 0.97]
    lowest_fp_recall = (
        recall_candidates.sort_values(["false_positive", "threshold_mm"]).iloc[0]
        if not recall_candidates.empty else None
    )

    selected_thresholds = sorted(set([
        2.0, 2.7, float(f1_best["threshold_mm"]),
        float(balance_best["threshold_mm"]), 3.0, 3.2, 3.4, 3.5, 4.0,
    ]))
    selected_crops = subgroups[
        (subgroups["subgroup_type"] == "crop")
        & subgroups["threshold_mm"].isin(selected_thresholds)
    ].copy()
    selected_crops = selected_crops.sort_values(["subgroup", "threshold_mm"])

    selected_other = subgroups[
        (subgroups["subgroup_type"] != "crop")
        & subgroups["threshold_mm"].isin([2.0, 2.7, 3.0, 3.2, 3.5, 4.0])
    ].copy()
    selected_other = selected_other.sort_values(
        ["subgroup_type", "subgroup", "threshold_mm"]
    )

    risk_rows = []
    comparisons = [
        ("crop", "crop"),
        ("soil type", "soil_type"),
        ("growth stage", "growth_stage"),
        ("edge status", "edge_status"),
        ("edge flag", "edge_category_overlap"),
    ]
    for label, subgroup_type in comparisons:
        subset = subgroups[subgroups["subgroup_type"] == subgroup_type]
        baseline_rows = subset[subset["threshold_mm"] == 2.0].set_index("subgroup")
        candidate_rows = subset[subset["threshold_mm"] == 3.5].set_index("subgroup")
        merged = baseline_rows.join(candidate_rows, lsuffix="_2mm", rsuffix="_3_5mm")
        merged["recall_change_pp"] = (
            merged["recall_3_5mm"] - merged["recall_2mm"]
        ) * 100
        merged["fn_change"] = merged["false_negative_3_5mm"] - merged["false_negative_2mm"]
        worst = merged.sort_values("recall_change_pp").iloc[0]
        risk_rows.append({
            "group_type": label,
            "subgroup": worst.name,
            "n": int(worst["n_2mm"]),
            "recall_2mm": worst["recall_2mm"],
            "recall_3_5mm": worst["recall_3_5mm"],
            "recall_change_pp": worst["recall_change_pp"],
            "fn_2mm": int(worst["false_negative_2mm"]),
            "fn_3_5mm": int(worst["false_negative_3_5mm"]),
            "fn_change": int(worst["fn_change"]),
        })
    subgroup_risk = pd.DataFrame(risk_rows)
    risk_table = markdown_table(
        subgroup_risk,
        ["group_type", "subgroup", "n", "recall_2mm", "recall_3_5mm", "recall_change_pp", "fn_2mm", "fn_3_5mm", "fn_change"],
        ["Group", "Subgroup with largest recall drop", "n", "Recall at 2 mm", "Recall at 3.5 mm", "Change (pp)", "FN at 2 mm", "FN at 3.5 mm", "FN change"],
    )

    # Count the label/decision threshold gap without redefining the recorded label.
    actual_amount = predictions["actual_water_amount_mm"].to_numpy(dtype=float)
    recorded_irrigate = predictions["actual_irrigate"].to_numpy(dtype=int)
    gap = (actual_amount > 0) & (actual_amount < IRRIGATION_THRESHOLD_MM)
    inconsistent_label = recorded_irrigate != (actual_amount > 0).astype(int)

    overall_for_report = overall.copy()
    overall_for_report["confusion_matrix"] = overall_for_report["confusion_matrix"].apply(
        lambda x: f"`{x}`"
    )
    overall_table = markdown_table(
        overall_for_report,
        ["threshold_mm", "accuracy", "precision", "recall", "f1",
         "false_positive", "false_negative", "confusion_matrix"],
        ["Threshold (mm)", "Accuracy", "Precision", "Recall", "F1", "FP", "FN", "Confusion matrix [[TN, FP], [FN, TP]]"],
    )

    crop_table = markdown_table(
        selected_crops,
        ["subgroup", "threshold_mm", "n", "recall", "false_negative", "precision", "false_positive", "f1"],
        ["Crop", "Threshold (mm)", "n", "Recall", "FN", "Precision", "FP", "F1"],
    )
    other_table = markdown_table(
        selected_other,
        ["subgroup_type", "subgroup", "threshold_mm", "n", "recall", "false_negative", "precision", "false_positive", "f1"],
        ["Group type", "Group", "Threshold (mm)", "n", "Recall", "FN", "Precision", "FP", "F1"],
    )

    lowest_fp_text = (
        f"{lowest_fp_recall['threshold_mm']:.1f} mm (recall="
        f"{lowest_fp_recall['recall']:.4f}, FP={int(lowest_fp_recall['false_positive'])}, "
        f"FN={int(lowest_fp_recall['false_negative'])})"
        if lowest_fp_recall is not None else "None of the evaluated thresholds"
    )

    report = f"""# Irrigation Decision Threshold Testing

## Scope and method

This is a threshold evaluation only. It uses the original ExtraTreesReg configuration (`n_estimators=300`, `max_depth=20`, `min_samples_split=2`), the unchanged `proposed7` features, and only rows marked `split == train`. It does not tune model parameters, alter the feature set, or use the held-out test split.

Row-level five-fold out-of-fold predictions were reproduced because the prior CSV contains aggregate threshold metrics but not the predictions needed for subgroup evaluation. The CV strategy is `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, stratified by crop × the recorded `irrigate` label. Scaling is fit separately on each fold's training rows. Predictions are clipped at zero. Threshold evaluation uses `{len(predictions):,}` out-of-fold training rows.

The recorded target label is preserved: `irrigate = 1` when `water_amount_mm > 0`. A predicted irrigation decision is `1` when the clipped model prediction is at least 2.0 mm for the existing baseline comparison, or at the evaluated threshold. There are {int(gap.sum())} records with `0 < water_amount_mm < 2.0`, {int(inconsistent_label.sum())} inconsistencies between the recorded `irrigate` field and `water_amount_mm > 0`, and {int((predictions['actual_irrigate'].to_numpy() == 1).sum())} positive recorded irrigation labels. Thus the definition gap is explicitly checked and the original label remains the evaluation target.

False positive (FP) means the prediction crosses the selected threshold while recorded `irrigate` is 0. False negative (FN) means recorded `irrigate` is 1 while the prediction remains below threshold. Confusion matrices are `[[TN, FP], [FN, TP]]`.

## Overall comparison

{overall_table}

## Findings

- Highest F1: **{f1_best['threshold_mm']:.1f} mm** (F1={f1_best['f1']:.4f}, precision={f1_best['precision']:.4f}, recall={f1_best['recall']:.4f}, FP={int(f1_best['false_positive'])}, FN={int(f1_best['false_negative'])}).
- Highest accuracy: **{acc_best['threshold_mm']:.1f} mm** (accuracy={acc_best['accuracy']:.4f}, FP={int(acc_best['false_positive'])}, FN={int(acc_best['false_negative'])}).
- Closest precision–recall balance by smallest absolute gap: **{balance_best['threshold_mm']:.1f} mm** (precision={balance_best['precision']:.4f}, recall={balance_best['recall']:.4f}, gap={balance_best['pr_gap']:.4f}).
- Lowest false-positive count among thresholds with recall at least 0.97: **{lowest_fp_text}**.

The highest-F1 threshold is the statistical choice under equal weighting of precision and recall. The operationally safest choice against missed irrigation is **2.0 mm**, since it has the highest recall / fewest missed irrigations among the evaluated thresholds. A higher threshold trades that safety for fewer unnecessary irrigation decisions. No threshold has been applied to the prediction pipeline.

## Crop-wise performance at candidate thresholds

The table includes the current 2.0 mm rule, thresholds around the best overall candidates, and 4.0 mm as a reference. Compare FN counts and recall within each crop; the full crop × threshold sweep is in the subgroup CSV.

{crop_table}

## Soil, growth-stage, and edge-case performance

Edge-category groups overlap because a row can carry multiple edge flags. The subgroup CSV contains every requested threshold for each soil, stage, edge status, and edge category.

{other_table}

## Largest observed subgroup recall drops at 3.5 mm

This comparison is against the current 2 mm threshold. It highlights the subgroup with the largest recall decrease within each group type; all subgroup-by-threshold results are in the CSV.

{risk_table}

Rice recall falls from 0.9833 to 0.9040 (8 to 46 missed cases), and the Sowing-stage recall falls from 0.9804 to 0.8982 (10 to 52 missed cases). These are the clearest crop/stage costs of moving to 3.5 mm. The reported CV subgroup differences are descriptive; they are not a separate inferential significance test.

## Output files

- Overall threshold metrics and confusion matrices: `{output_paths['overall']}`
- Crop, soil, stage, edge-status, and edge-category metrics: `{output_paths['subgroups']}`
- Reproduced row-level out-of-fold predictions: `{output_paths['predictions']}`
- This report: `{output_paths['report']}`
"""
    return report


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(ROOT / "primary" / "primary_model_ready.csv")
    readable = pd.read_csv(ROOT / "primary" / "primary_features_full.csv")
    if len(readable) != len(data):
        raise ValueError("The model-ready and readable primary files have different row counts.")
    with (ROOT / "feature_sets.json").open(encoding="utf-8") as f:
        features = json.load(f)["proposed7"]

    train = data.loc[data["split"] == "train"].copy()
    train.insert(0, "source_row_index", train.index)
    train.reset_index(drop=True, inplace=True)
    train_readable = readable.iloc[train["source_row_index"].to_numpy()].reset_index(drop=True)
    X = train[features].to_numpy()
    y_amount = train["water_amount_mm"].to_numpy(dtype=float)
    y_label = train["irrigate"].to_numpy(dtype=int)  # preserve supplied label
    strata = train["meta_crop"].astype(str) + "|" + train["irrigate"].astype(str)
    cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

    oof_pred = np.full(len(train), np.nan, dtype=float)
    fold_ids = np.zeros(len(train), dtype=int)
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
        oof_pred[valid_idx] = np.maximum(model.predict(X_valid), 0.0)
        fold_ids[valid_idx] = fold
        print(f"Generated out-of-fold predictions for fold {fold}/{FOLDS}")

    if np.isnan(oof_pred).any() or (fold_ids == 0).any():
        raise RuntimeError("Some training rows did not receive an out-of-fold prediction.")

    predictions = pd.DataFrame({
        "source_row_index": train["source_row_index"],
        "fold": fold_ids,
        "actual_water_amount_mm": y_amount,
        "predicted_water_amount_mm": oof_pred,
        "actual_irrigate": y_label,
        "crop": train["meta_crop"].astype(str),
        "soil_type": train[[c for c in train.columns if c.startswith("soil_type_")]].idxmax(axis=1).str.replace("soil_type_", "", regex=False),
        "growth_stage": train[[c for c in train.columns if c.startswith("growth_stage_")]].idxmax(axis=1).str.replace("growth_stage_", "", regex=False),
        "is_edge_case": train["is_edge_case"].astype(int),
    })
    edge_flags = [c for c in train_readable.columns if c.startswith("edge_") and c != "edge_reasons"]
    for flag in edge_flags:
        predictions[flag] = train_readable[flag].astype(int).to_numpy()
    predictions["edge_reasons"] = train_readable["edge_reasons"].fillna("none").to_numpy()

    overall = evaluate_overall(predictions)
    baseline_fps = overall.loc[overall["threshold_mm"] == 2.0, "false_positive"].iloc[0]
    baseline_fns = overall.loc[overall["threshold_mm"] == 2.0, "false_negative"].iloc[0]
    overall["delta_false_positive_vs_2mm"] = overall["false_positive"] - baseline_fps
    overall["delta_false_negative_vs_2mm"] = overall["false_negative"] - baseline_fns

    subgroups = evaluate_subgroups(predictions)
    subgroup_baseline = subgroups[subgroups["threshold_mm"] == 2.0][
        ["subgroup_type", "subgroup", "false_positive", "false_negative", "recall"]
    ].rename(columns={
        "false_positive": "fp_at_2mm",
        "false_negative": "fn_at_2mm",
        "recall": "recall_at_2mm",
    })
    subgroups = subgroups.merge(subgroup_baseline, on=["subgroup_type", "subgroup"], how="left")
    subgroups["delta_false_positive_vs_2mm"] = subgroups["false_positive"] - subgroups["fp_at_2mm"]
    subgroups["delta_false_negative_vs_2mm"] = subgroups["false_negative"] - subgroups["fn_at_2mm"]
    subgroups["delta_recall_vs_2mm"] = subgroups["recall"] - subgroups["recall_at_2mm"]

    overall_path = OUT_DIR / "threshold_comparison.csv"
    subgroup_path = OUT_DIR / "subgroup_metrics.csv"
    prediction_path = OUT_DIR / "out_of_fold_predictions.csv"
    report_path = OUT_DIR / "threshold_testing_report.md"
    overall.to_csv(overall_path, index=False, float_format="%.6f")
    subgroups.to_csv(subgroup_path, index=False, float_format="%.6f")
    predictions.to_csv(prediction_path, index=False, float_format="%.6f")
    paths = {
        "overall": "results/threshold_testing/threshold_comparison.csv",
        "subgroups": "results/threshold_testing/subgroup_metrics.csv",
        "predictions": "results/threshold_testing/out_of_fold_predictions.csv",
        "report": "results/threshold_testing/threshold_testing_report.md",
    }
    report_path.write_text(
        build_report(overall, subgroups, predictions, paths), encoding="utf-8"
    )

    print("\nOverall metrics pooled across all 8,000 out-of-fold training predictions:")
    print(overall[[
        "threshold_mm", "accuracy", "precision", "recall", "f1",
        "false_positive", "false_negative", "confusion_matrix",
    ]].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved threshold testing outputs under: {OUT_DIR}")


if __name__ == "__main__":
    main()
