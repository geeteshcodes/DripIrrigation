"""
analyze_false_alarms.py
-----------------------
Deep-dive analysis of false-positive irrigation alarms for the baseline
ExtraTreesReg model at the 2.7 mm threshold.

Inputs (all relative to this script's location):
  results/threshold_testing/out_of_fold_predictions.csv   – OOF predictions
  results/false_alarm_review.csv                          – FP rows from test set

Outputs written to  results/false_alarm_analysis/:
  false_alarm_analysis_report.md     – human-readable findings
  fp_by_crop.csv
  fp_by_soil.csv
  fp_by_stage.csv
  fp_by_edge_reason.csv
  fp_prediction_bins.csv             – how close pred was to 2.7 mm
  fp_crop_x_soil.csv
  fp_crop_x_stage.csv
  fp_full.csv                        – full FP rows sorted by pred distance

Usage:
  python analyze_false_alarms.py
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

THRESHOLD = 2.7
OUT_DIR = os.path.join("results", "false_alarm_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1.  Load data
# ---------------------------------------------------------------------------

OOF_PATH = os.path.join("results", "threshold_testing", "out_of_fold_predictions.csv")
FAR_PATH = os.path.join("results", "false_alarm_review.csv")

print(f"Loading OOF predictions from {OOF_PATH} ...")
oof = pd.read_csv(OOF_PATH)
print(f"  rows: {len(oof):,}  columns: {list(oof.columns)}")

# ---------------------------------------------------------------------------
# 2.  Resolve column names flexibly
# ---------------------------------------------------------------------------

def _find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

oof_pred_col        = _find_col(oof, ["pred", "predicted", "predicted_water_amount_mm", "y_pred"])
oof_actual_col      = _find_col(oof, ["actual", "actual_water_amount_mm", "y_true", "target"])
oof_crop_col        = _find_col(oof, ["crop", "crop_type"])
oof_soil_col        = _find_col(oof, ["soil_type", "soil_type_x", "soil"])
oof_stage_col       = _find_col(oof, ["growth_stage", "growth_stage_x", "stage"])
oof_edge_col        = _find_col(oof, ["is_edge_case", "is_edge_case_x", "edge_case"])
oof_edge_reason_col = _find_col(oof, ["edge_reasons", "edge_reason"])

missing = [n for n, c in [("pred", oof_pred_col), ("actual", oof_actual_col)] if c is None]
if missing:
    print(f"ERROR: could not find required columns {missing} in OOF file.")
    print(f"  Available: {list(oof.columns)}")
    sys.exit(1)

print(f"\nResolved columns -> pred='{oof_pred_col}', actual='{oof_actual_col}', "
      f"crop='{oof_crop_col}', soil='{oof_soil_col}', stage='{oof_stage_col}', "
      f"edge='{oof_edge_col}', edge_reason='{oof_edge_reason_col}'")

# ---------------------------------------------------------------------------
# 3.  Derive FP / FN decisions at THRESHOLD
# ---------------------------------------------------------------------------

oof["_pred_irrigate"]   = (oof[oof_pred_col]   >= THRESHOLD).astype(int)
oof["_actual_irrigate"] = (oof[oof_actual_col] >= THRESHOLD).astype(int)
oof["_fp"] = ((oof["_pred_irrigate"] == 1) & (oof["_actual_irrigate"] == 0)).astype(int)
oof["_fn"] = ((oof["_pred_irrigate"] == 0) & (oof["_actual_irrigate"] == 1)).astype(int)

fp = oof[oof["_fp"] == 1].copy()
fn = oof[oof["_fn"] == 1].copy()

total_fp = len(fp)
total_fn = len(fn)

print(f"\nAt threshold {THRESHOLD} mm (OOF n={len(oof):,}):")
print(f"  FP = {total_fp}")
print(f"  FN = {total_fn}")

# ---------------------------------------------------------------------------
# 4.  Breakdown helpers
# ---------------------------------------------------------------------------

def breakdown_vs_base(df_all, col):
    if col is None or col not in df_all.columns:
        return pd.DataFrame()
    all_c  = df_all.groupby(col).size().reset_index(name="n_total")
    fp_c   = df_all[df_all["_fp"] == 1].groupby(col).size().reset_index(name="FP_count")
    fn_c   = df_all[df_all["_fn"] == 1].groupby(col).size().reset_index(name="FN_count")
    result = all_c.merge(fp_c, on=col, how="left").merge(fn_c, on=col, how="left")
    result["FP_count"] = result["FP_count"].fillna(0).astype(int)
    result["FN_count"] = result["FN_count"].fillna(0).astype(int)
    result["FP_rate_%"] = (result["FP_count"] / result["n_total"] * 100).round(2)
    result["FN_rate_%"] = (result["FN_count"] / result["n_total"] * 100).round(2)
    return result.sort_values("FP_count", ascending=False).reset_index(drop=True)

# ---------------------------------------------------------------------------
# 5.  Category breakdowns
# ---------------------------------------------------------------------------

fp_crop  = breakdown_vs_base(oof, oof_crop_col)
fp_soil  = breakdown_vs_base(oof, oof_soil_col)
fp_stage = breakdown_vs_base(oof, oof_stage_col)

fp_crop.to_csv(os.path.join(OUT_DIR, "fp_by_crop.csv"), index=False)
fp_soil.to_csv(os.path.join(OUT_DIR, "fp_by_soil.csv"), index=False)
fp_stage.to_csv(os.path.join(OUT_DIR, "fp_by_stage.csv"), index=False)

# ---------------------------------------------------------------------------
# 6.  Edge-reason breakdown
# ---------------------------------------------------------------------------

fp_reasons = None
if oof_edge_reason_col and oof_edge_reason_col in fp.columns:
    reason_rows = []
    for _, row in fp.iterrows():
        reasons = str(row.get(oof_edge_reason_col, "none"))
        if reasons.strip() in ("", "nan", "none"):
            reason_rows.append("no_edge_flag")
        else:
            for r in reasons.split(";"):
                reason_rows.append(r.strip())
    rs = pd.Series(reason_rows).value_counts().reset_index()
    rs.columns = ["edge_reason", "FP_count"]
    rs["FP_pct"] = (rs["FP_count"] / total_fp * 100).round(1)
    rs.to_csv(os.path.join(OUT_DIR, "fp_by_edge_reason.csv"), index=False)
    fp_reasons = rs

# ---------------------------------------------------------------------------
# 7.  Prediction proximity bins
# ---------------------------------------------------------------------------

pred_vals = fp[oof_pred_col]
bins     = [THRESHOLD, 3.0, 3.5, 4.0, 5.0, 10.0, 999.0]
bin_labs = ["2.7-3.0", "3.0-3.5", "3.5-4.0", "4.0-5.0", "5.0-10.0", ">10.0"]
fp["_pred_bin"] = pd.cut(pred_vals, bins=bins, labels=bin_labs, right=True)

fp_bins = (fp["_pred_bin"]
           .value_counts()
           .sort_index()
           .reset_index())
fp_bins.columns = ["pred_range_mm", "FP_count"]
fp_bins["FP_pct"]        = (fp_bins["FP_count"] / total_fp * 100).round(1)
fp_bins["cumulative_pct"] = fp_bins["FP_pct"].cumsum().round(1)
fp_bins.to_csv(os.path.join(OUT_DIR, "fp_prediction_bins.csv"), index=False)

near_thresh = int(fp_bins[fp_bins["pred_range_mm"] == "2.7-3.0"]["FP_count"].sum())
near_pct    = near_thresh / total_fp * 100 if total_fp else 0
median_fp_pred = pred_vals.median()
mean_fp_pred   = pred_vals.mean()
max_fp_pred    = pred_vals.max()

# ---------------------------------------------------------------------------
# 8.  Cross-tabulations
# ---------------------------------------------------------------------------

if oof_crop_col and oof_soil_col:
    base = oof.groupby([oof_crop_col, oof_soil_col]).size().reset_index(name="n_total")
    fpc  = fp.groupby([oof_crop_col, oof_soil_col]).size().reset_index(name="FP_count")
    cs = base.merge(fpc, on=[oof_crop_col, oof_soil_col], how="left")
    cs["FP_count"]   = cs["FP_count"].fillna(0).astype(int)
    cs["FP_rate_%"]  = (cs["FP_count"] / cs["n_total"] * 100).round(2)
    cs.sort_values("FP_count", ascending=False).to_csv(
        os.path.join(OUT_DIR, "fp_crop_x_soil.csv"), index=False)

if oof_crop_col and oof_stage_col:
    base = oof.groupby([oof_crop_col, oof_stage_col]).size().reset_index(name="n_total")
    fpc  = fp.groupby([oof_crop_col, oof_stage_col]).size().reset_index(name="FP_count")
    cst = base.merge(fpc, on=[oof_crop_col, oof_stage_col], how="left")
    cst["FP_count"]   = cst["FP_count"].fillna(0).astype(int)
    cst["FP_rate_%"]  = (cst["FP_count"] / cst["n_total"] * 100).round(2)
    cst.sort_values("FP_count", ascending=False).to_csv(
        os.path.join(OUT_DIR, "fp_crop_x_stage.csv"), index=False)

# ---------------------------------------------------------------------------
# 9.  Full FP table with distance column
# ---------------------------------------------------------------------------

fp["pred_distance_above_threshold"] = (fp[oof_pred_col] - THRESHOLD).round(4)
save_cols = [c for c in [
    oof_crop_col, oof_soil_col, oof_stage_col,
    oof_edge_col, oof_edge_reason_col,
    oof_actual_col, oof_pred_col,
    "pred_distance_above_threshold", "_pred_bin"
] if c is not None and c in fp.columns]
fp[save_cols].sort_values("pred_distance_above_threshold").to_csv(
    os.path.join(OUT_DIR, "fp_full.csv"), index=False)

# ---------------------------------------------------------------------------
# 10.  Write markdown report
# ---------------------------------------------------------------------------

def table_from_df(df, max_rows=20):
    lines = []
    if df is None or df.empty:
        return ["*No data.*"]
    cols = list(df.columns)
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in df.head(max_rows).iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return lines

edge_fp     = int(fp[oof_edge_col].sum()) if oof_edge_col and oof_edge_col in fp.columns else None
edge_fp_pct = edge_fp / total_fp * 100 if edge_fp is not None else None

md = []

md.append("# False Positive Analysis — ExtraTreesReg @ 2.7 mm")
md.append("")
md.append(f"**Dataset**: OOF predictions on {len(oof):,} training rows (5-fold CV)")
md.append(f"**Threshold**: {THRESHOLD} mm  |  **Total FP**: {total_fp}  |  **Total FN**: {total_fn}")
md.append("")

md.append("## 1. Executive Summary")
md.append("")
md.append(f"- **{near_thresh} / {total_fp} FPs ({near_pct:.0f}%)** have predictions between "
          f"2.7 and 3.0 mm — they sit within 0.3 mm of the threshold.")
if edge_fp is not None:
    md.append(f"- **{edge_fp} / {total_fp} FPs ({edge_fp_pct:.0f}%)** occur in edge-case flagged rows.")
md.append(f"- FP prediction range: {pred_vals.min():.2f} – {max_fp_pred:.2f} mm  "
          f"(median {median_fp_pred:.2f} mm, mean {mean_fp_pred:.2f} mm).")
md.append("")

md.append("## 2. By Crop")
md.append("")
md += table_from_df(fp_crop)
md.append("")

md.append("## 3. By Soil Type")
md.append("")
md += table_from_df(fp_soil)
md.append("")

md.append("## 4. By Growth Stage")
md.append("")
md += table_from_df(fp_stage)
md.append("")

md.append("## 5. Prediction Proximity to Threshold")
md.append("")
md += table_from_df(fp_bins)
md.append("")
md.append(f"> {near_thresh} FPs ({near_pct:.0f}%) are in the 2.7–3.0 mm band.")
md.append("")

if fp_reasons is not None:
    md.append("## 6. Edge-Case Flag Distribution in FPs")
    md.append("")
    md += table_from_df(fp_reasons)
    md.append("")

md.append("## 7. Crop x Soil Cross-Tab (top 15 by FP count)")
md.append("")
if os.path.exists(os.path.join(OUT_DIR, "fp_crop_x_soil.csv")):
    md += table_from_df(pd.read_csv(os.path.join(OUT_DIR, "fp_crop_x_soil.csv")).head(15))
md.append("")

md.append("## 8. Crop x Stage Cross-Tab (top 15 by FP count)")
md.append("")
if os.path.exists(os.path.join(OUT_DIR, "fp_crop_x_stage.csv")):
    md += table_from_df(pd.read_csv(os.path.join(OUT_DIR, "fp_crop_x_stage.csv")).head(15))
md.append("")

md.append("## 9. Recommended Next Steps")
md.append("")
md.append("1. **Identify the dominant (crop, stage) combo** in `fp_crop_x_stage.csv` "
          "with the highest FP rate — check if the FAO-56 target is structurally near-zero "
          "for that combination (e.g. cotton at Harvest after heavy rain).")
md.append("2. **Examine `fp_full.csv`** sorted by `pred_distance_above_threshold` — "
          "the lowest values are borderline cases; look for shared feature patterns "
          "(high rainfall_mm, saturated soil_moisture) that could justify a rule-based override.")
md.append("3. **Check rainfall_mm among FPs** — if heavy rain rows dominate, "
          "the model may not fully suppress irrigation when rainfall alone should suffice.")
md.append("4. If a high-FP subgroup is found, consider a targeted post-hoc rule "
          "(e.g. 'suppress irrigation if rainfall_mm > X') rather than full retraining.")
md.append("")

md.append("## 10. Output Files")
md.append("")
md.append("All in `results/false_alarm_analysis/`:")
md.append("")
for fname, desc in [
    ("fp_by_crop.csv",         "FP/FN count and rate by crop"),
    ("fp_by_soil.csv",         "FP/FN count and rate by soil type"),
    ("fp_by_stage.csv",        "FP/FN count and rate by growth stage"),
    ("fp_by_edge_reason.csv",  "FP count by edge-case flag"),
    ("fp_prediction_bins.csv", "FP distribution by predicted mm band"),
    ("fp_crop_x_soil.csv",     "Crop x soil FP cross-tab"),
    ("fp_crop_x_stage.csv",    "Crop x growth stage FP cross-tab"),
    ("fp_full.csv",            "All FP rows sorted by distance to threshold"),
]:
    md.append(f"- `{fname}` — {desc}")

report_path = os.path.join(OUT_DIR, "false_alarm_analysis_report.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md))

print(f"\nReport  -> {report_path}")
print(f"CSVs    -> {OUT_DIR}/")
print("Done.")
