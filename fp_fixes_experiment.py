"""
fp_fixes_experiment.py
----------------------
Evaluates three targeted post-hoc fixes for the ExtraTreesReg false positives
identified in the cluster investigation, WITHOUT retraining.

Fix 1 – Post-hoc rain suppression rule (targets heavy-rain cluster)
    Suppress the irrigation decision if:
        rain_covers_etc == 1  AND  demand_minus_rain_mm < 0
    Applied after the model predicts; no model weights changed.

Fix 2 – Label-conflict exclusion from evaluation
    Rows flagged edge_label_conflict == 1 are excluded from FP/FN counts
    and metrics. The model still runs on them; they just don't count as errors.

Fix 3 – Sugarcane Veg/Sow rain-covers suppressor
    Same rule as Fix 1, restricted to (crop==sugarcane AND stage in Veg/Sow):
        if rain_covers_etc == 1 -> suppress irrigation decision.
    More conservative version for the sugarcane cluster (68% have rain_covers=0,
    so the rule is intentionally narrow).

Each fix is evaluated separately and in combination at threshold = 2.7 mm.
All results are compared against the baseline ExtraTreesReg @ 2.7 mm.

Outputs -> results/fp_fixes/
    fixes_summary.csv
    fixes_report.md

Usage:
    python fp_fixes_experiment.py
"""

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    mean_absolute_error, mean_squared_error,
    precision_score, recall_score, r2_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT        = Path(__file__).resolve().parent
SEED        = 42
THRESHOLD   = 2.7
TREE_PARAMS = dict(n_estimators=300, max_depth=20, min_samples_split=2)
OUT_DIR     = ROOT / "results" / "fp_fixes"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── feature sets ─────────────────────────────────────────────────────────────
with (ROOT / "feature_sets.json").open(encoding="utf-8") as fh:
    FEATURE_SETS = json.load(fh)

# The extra raw features needed for post-hoc rules (not model inputs)
RULE_FEATURES = [
    "rain_covers_etc",       # 1 if rainfall ≥ ETc
    "demand_minus_rain_mm",  # ETc − rainfall (negative = surplus)
    "meta_crop",             # crop identity
]

# ── helpers ──────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred_binary, threshold_mm=THRESHOLD,
                    exclude_mask=None, label=""):
    """Compute FP/FN/precision/recall/F1. Optionally exclude rows (label conflict)."""
    y_true  = np.asarray(y_true).astype(int)
    y_pred  = np.asarray(y_pred_binary).astype(int)
    if exclude_mask is not None:
        keep = ~np.asarray(exclude_mask, dtype=bool)
        y_true, y_pred = y_true[keep], y_pred[keep]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "strategy":   label,
        "threshold":  threshold_mm,
        "n_eval":     int(len(y_true)),
        "FP":         int(fp),
        "FN":         int(fn),
        "TP":         int(tp),
        "TN":         int(tn),
        "precision":  round(precision_score(y_true, y_pred, zero_division=0), 6),
        "recall":     round(recall_score(y_true, y_pred, zero_division=0), 6),
        "f1":         round(f1_score(y_true, y_pred, zero_division=0), 6),
        "accuracy":   round(accuracy_score(y_true, y_pred), 6),
    }


def apply_rain_suppression(pred_mm, rain_covers_etc, demand_minus_rain_mm, threshold=THRESHOLD):
    """
    Fix 1 / Fix 3 core rule:
    If model predicts irrigate AND rain_covers_etc==1 AND demand_minus_rain_mm < 0
    → override to no-irrigate (set effective pred to 0).
    Returns binary decision array.
    """
    base_decision = (pred_mm >= threshold).astype(int)
    surplus        = (rain_covers_etc == 1) & (demand_minus_rain_mm < 0)
    # Suppress: was 1 (irrigate), surplus → force to 0
    corrected = base_decision.copy()
    corrected[surplus & (base_decision == 1)] = 0
    return corrected


def apply_sugarcane_veg_sow_suppression(pred_mm, rain_covers_etc, crop_series,
                                         stage_series, threshold=THRESHOLD):
    """
    Fix 3 (narrow): suppress only for sugarcane × (Vegetative | Sowing) if
    rain_covers_etc == 1 (more conservative; demand_minus_rain check omitted
    because 68% of this cluster still lacks coverage).
    """
    base_decision = (pred_mm >= threshold).astype(int)
    is_target = (
        (crop_series == "sugarcane") &
        (stage_series.isin(["Vegetative", "Sowing"])) &
        (rain_covers_etc == 1)
    )
    corrected = base_decision.copy()
    corrected[is_target & (base_decision == 1)] = 0
    return corrected


# ── main CV loop ─────────────────────────────────────────────────────────────

def run():
    print("Loading data …")
    data = pd.read_csv(ROOT / "primary" / "primary_model_ready.csv")

    # ── decode one-hot back to string labels ──────────────────────────────
    crop_cols  = [c for c in data.columns if c.startswith("crop_type_")]
    soil_cols  = [c for c in data.columns if c.startswith("soil_type_")]
    stage_cols = [c for c in data.columns if c.startswith("growth_stage_")]

    if crop_cols:
        data["_crop"]  = data[crop_cols].idxmax(axis=1).str.removeprefix("crop_type_")
    else:
        data["_crop"]  = data.get("meta_crop", "unknown")

    if soil_cols:
        data["_soil"]  = data[soil_cols].idxmax(axis=1).str.removeprefix("soil_type_")
    else:
        data["_soil"]  = "unknown"

    if stage_cols:
        data["_stage"] = data[stage_cols].idxmax(axis=1).str.removeprefix("growth_stage_")
    else:
        data["_stage"] = "unknown"

    # ── training rows only ────────────────────────────────────────────────
    train = data.loc[data["split"] == "train"].reset_index(drop=True)

    feature_names = FEATURE_SETS["proposed7"]
    X      = train[feature_names].to_numpy()
    amount = train["water_amount_mm"].to_numpy()
    labels = train["irrigate"].to_numpy(int)

    # Rule features (not model inputs; used for post-hoc decisions)
    rain_cov = train["rain_covers_etc"].to_numpy()           # 0 or 1
    dmr      = train["demand_minus_rain_mm"].to_numpy()      # mm, can be negative
    crop_s   = train["_crop"]
    stage_s  = train["_stage"]

    # Label-conflict flag
    conflict_flag = train["edge_label_conflict"].to_numpy() \
                    if "edge_label_conflict" in train.columns \
                    else np.zeros(len(train), dtype=int)

    # ── 5-fold CV ────────────────────────────────────────────────────────
    folds   = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    strat_y = crop_s.astype(str) + "_" + pd.Series(labels).astype(str)

    all_preds  = np.full(len(train), np.nan)
    all_actual = labels.copy()
    all_conf   = conflict_flag.copy()
    all_rcov   = rain_cov.copy()
    all_dmr    = dmr.copy()
    all_crop   = crop_s.copy()
    all_stage  = stage_s.copy()

    for fold, (fit_idx, val_idx) in enumerate(folds.split(X, strat_y), 1):
        print(f"  Fold {fold}/5 …", end=" ", flush=True)
        scaler = StandardScaler().fit(X[fit_idx])
        xf = scaler.transform(X[fit_idx])
        xv = scaler.transform(X[val_idx])

        model = ExtraTreesRegressor(**TREE_PARAMS, random_state=SEED, n_jobs=-1)
        model.fit(xf, amount[fit_idx])
        preds_val = np.maximum(model.predict(xv), 0)
        all_preds[val_idx] = preds_val
        mae = mean_absolute_error(amount[val_idx], preds_val)
        print(f"MAE={mae:.4f}")

    print(f"\nOOF complete. Total predicted: {(~np.isnan(all_preds)).sum()}")

    # ── baseline binary decision ──────────────────────────────────────────
    base_decision = (all_preds >= THRESHOLD).astype(int)

    # ── Fix 1: full rain suppression (all crops) ──────────────────────────
    fix1_decision = apply_rain_suppression(
        all_preds, all_rcov, all_dmr, threshold=THRESHOLD
    )

    # ── Fix 2: label-conflict exclusion (no change to decisions) ─────────
    # Same decisions as baseline; conflicts excluded from metrics only
    fix2_decision = base_decision.copy()

    # ── Fix 3: sugarcane Veg/Sow narrow suppression ───────────────────────
    fix3_decision = apply_sugarcane_veg_sow_suppression(
        all_preds, all_rcov, all_crop, all_stage, threshold=THRESHOLD
    )

    # ── Fix 1+2: rain suppression + conflict exclusion ────────────────────
    fix12_decision = fix1_decision.copy()

    # ── Fix 1+2+3: all three fixes combined ──────────────────────────────
    fix123_decision = fix3_decision.copy()
    # apply rain suppression on top of fix3
    fix123_decision = apply_rain_suppression(
        all_preds, all_rcov, all_dmr, threshold=THRESHOLD
    )
    # also apply sugarcane narrow (union of both rules)
    surplus = (all_rcov == 1) & (all_dmr < 0)
    sug_cov = (all_crop.to_numpy() == "sugarcane") & \
              (all_stage.isin(["Vegetative", "Sowing"]).to_numpy()) & \
              (all_rcov == 1)
    fix123_decision[sug_cov & (base_decision == 1)] = 0

    # ── evaluate all strategies ───────────────────────────────────────────
    results = []
    conf_mask = all_conf.astype(bool)

    # Baseline
    results.append(compute_metrics(all_actual, base_decision,
                                   label="baseline_2.7mm"))
    # Baseline excluding label conflicts
    results.append(compute_metrics(all_actual, base_decision,
                                   exclude_mask=conf_mask,
                                   label="baseline_excl_conflicts"))
    # Fix 1
    results.append(compute_metrics(all_actual, fix1_decision,
                                   label="fix1_rain_suppression"))
    # Fix 1 + exclude conflicts
    results.append(compute_metrics(all_actual, fix1_decision,
                                   exclude_mask=conf_mask,
                                   label="fix1_rain_suppression_excl_conflicts"))
    # Fix 3
    results.append(compute_metrics(all_actual, fix3_decision,
                                   label="fix3_sugarcane_narrow"))
    # Fix 3 + exclude conflicts
    results.append(compute_metrics(all_actual, fix3_decision,
                                   exclude_mask=conf_mask,
                                   label="fix3_sugarcane_narrow_excl_conflicts"))
    # All three combined
    results.append(compute_metrics(all_actual, fix123_decision,
                                   label="fix1+fix3_combined"))
    results.append(compute_metrics(all_actual, fix123_decision,
                                   exclude_mask=conf_mask,
                                   label="fix1+fix3_combined_excl_conflicts"))

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "fixes_summary.csv", index=False)
    print(f"\nSummary saved → {OUT_DIR / 'fixes_summary.csv'}")
    print("\n" + df[["strategy", "FP", "FN", "precision", "recall", "f1", "n_eval"]].to_string(index=False))

    # ── per-rule suppression counts ───────────────────────────────────────
    fix1_suppressed = int(((base_decision == 1) & (fix1_decision == 0)).sum())
    fix3_suppressed = int(((base_decision == 1) & (fix3_decision == 0)).sum())
    fix1_true_fp_removed = int(
        ((base_decision == 1) & (fix1_decision == 0) & (all_actual == 0)).sum()
    )
    fix1_tp_lost = int(
        ((base_decision == 1) & (fix1_decision == 0) & (all_actual == 1)).sum()
    )
    fix3_true_fp_removed = int(
        ((base_decision == 1) & (fix3_decision == 0) & (all_actual == 0)).sum()
    )
    fix3_tp_lost = int(
        ((base_decision == 1) & (fix3_decision == 0) & (all_actual == 1)).sum()
    )

    # label-conflict breakdown
    conf_fp = int(((base_decision == 1) & (all_actual == 0) & conf_mask).sum())
    conf_fn = int(((base_decision == 0) & (all_actual == 1) & conf_mask).sum())

    # ── markdown report ───────────────────────────────────────────────────
    md = []
    md.append("# FP Fix Experiment — Results")
    md.append("")
    md.append(f"**Model**: ExtraTreesReg (baseline, no retraining)  |  "
              f"**Threshold**: {THRESHOLD} mm  |  **OOF n**: {len(train):,}")
    md.append("")

    md.append("## Summary Table")
    md.append("")
    md.append("| Strategy | FP | FN | F1 | Precision | Recall | n_eval |")
    md.append("|---|---|---|---|---|---|---|")
    for row in results:
        md.append(f"| {row['strategy']} | {row['FP']} | {row['FN']} | "
                  f"{row['f1']:.4f} | {row['precision']:.4f} | "
                  f"{row['recall']:.4f} | {row['n_eval']} |")
    md.append("")

    md.append("---")
    md.append("")
    md.append("## Fix 1 — Post-hoc Rain Suppression Rule")
    md.append("")
    md.append("**Rule**: if `rain_covers_etc == 1` AND `demand_minus_rain_mm < 0` "
              "→ override irrigation decision to **No Irrigate**.")
    md.append("")
    md.append(f"- Total decisions suppressed by rule: **{fix1_suppressed}**")
    md.append(f"- Of those, genuine FPs removed: **{fix1_true_fp_removed}** ✓")
    md.append(f"- Of those, correct irrigations lost (FP→FN): **{fix1_tp_lost}** ✗")
    if fix1_suppressed > 0:
        precision_rule = fix1_true_fp_removed / fix1_suppressed
        md.append(f"- Rule precision (FP / total suppressed): **{precision_rule:.2f}** "
                  f"({'good' if precision_rule > 0.8 else 'moderate — some TPs lost'})")
    md.append("")
    md.append("**How the rule works**: The model predicts high ETc demand (high `dr_mm`) "
              "but `rain_covers_etc=1` means rainfall ≥ ETc in that window, "
              "and `demand_minus_rain_mm < 0` confirms a net rain surplus. "
              "Suppressing in this case is physically justified — rain is sufficient.")
    md.append("")

    md.append("---")
    md.append("")
    md.append("## Fix 2 — Label-Conflict Exclusion from Evaluation")
    md.append("")
    md.append("**Rule**: rows where `edge_label_conflict == 1` are excluded from "
              "FP/FN counts.")
    md.append("")
    md.append(f"- Label-conflict FPs excluded: **{conf_fp}** "
              f"(were counted as model errors; actual `water_amount_mm` is 2.0–2.6 mm)")
    md.append(f"- Label-conflict FNs excluded: **{conf_fn}**")
    md.append(f"- Total conflict rows in OOF: **{int(conf_mask.sum())}**")
    md.append("")
    md.append("**Why**: These rows have `actual_water_amount_mm` between 2.0–2.6 mm "
              "(just below 2.7 mm threshold) but binary label `irrigate=0`. "
              "The model's prediction of ≥ 2.7 mm is consistent with the continuous FAO "
              "signal — the error is in the label, not the prediction. "
              "Excluding them from evaluation gives a cleaner picture of true model error.")
    md.append("")
    md.append("> **Note**: This does NOT change model behaviour in deployment. "
              "It only changes what we count as an error during evaluation. "
              "In production, you may still want to flag conflict rows and alert "
              "a domain expert rather than auto-suppressing.")
    md.append("")

    md.append("---")
    md.append("")
    md.append("## Fix 3 — Sugarcane Veg/Sow Narrow Suppression")
    md.append("")
    md.append("**Rule**: if `crop == sugarcane` AND `growth_stage in [Vegetative, Sowing]` "
              "AND `rain_covers_etc == 1` → override to **No Irrigate**.")
    md.append("")
    md.append(f"- Total decisions suppressed by rule: **{fix3_suppressed}**")
    md.append(f"- Genuine FPs removed: **{fix3_true_fp_removed}** ✓")
    md.append(f"- Correct irrigations lost: **{fix3_tp_lost}** ✗")
    if fix3_suppressed > 0:
        prec3 = fix3_true_fp_removed / fix3_suppressed
        md.append(f"- Rule precision: **{prec3:.2f}**")
    md.append("")
    md.append("**Why narrow**: 68% of sugarcane Veg/Sow FPs have `rain_covers_etc=0` "
              "(rain does NOT cover ETc), so a full suppression would miss those. "
              "This rule targets only the 32% where rain provably covers demand.")
    md.append("")

    md.append("---")
    md.append("")
    md.append("## Combined Fix (Fix 1 + Fix 3)")
    md.append("")
    md.append("Fix 1 and Fix 3 are complementary and can be stacked. "
              "The combined rule is:")
    md.append("")
    md.append("```")
    md.append("if (rain_covers_etc == 1 AND demand_minus_rain_mm < 0):")
    md.append("    decision = No Irrigate          # Fix 1 — all crops")
    md.append("elif (crop == 'sugarcane'")
    md.append("      AND stage in ['Vegetative', 'Sowing']")
    md.append("      AND rain_covers_etc == 1):")
    md.append("    decision = No Irrigate          # Fix 3 — sugarcane narrow")
    md.append("```")
    md.append("")

    md.append("---")
    md.append("")
    md.append("## Interpretation Guide")
    md.append("")
    md.append("| Metric direction | Meaning |")
    md.append("|---|---|")
    md.append("| FP ↓ | Fewer false irrigation alarms |")
    md.append("| FN ↑ | More missed irrigations (bad) |")
    md.append("| FN same | FP reduction without hurting recall |")
    md.append("| F1 ↑ | Overall improvement |")
    md.append("| n_eval < 8000 | Label-conflict rows excluded |")
    md.append("")
    md.append("> **Best outcome**: FP decreases, FN stays flat or decreases, F1 increases.")
    md.append("")

    md.append("---")
    md.append("")
    md.append("## Output Files")
    md.append("")
    md.append("All in `results/fp_fixes/`:")
    md.append("- `fixes_summary.csv` — FP/FN/F1 for all strategies")
    md.append("- `fp_fixes_report.md` — this report")

    report_path = OUT_DIR / "fp_fixes_report.md"
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))

    print(f"\nReport  →  {report_path}")
    print("Done.")


if __name__ == "__main__":
    run()
