"""
investigate_fp_clusters.py
--------------------------
Targeted investigation of the three main false-positive clusters identified
in analyze_false_alarms.py:

  1. Sugarcane + Vegetative/Sowing  (highest FP rate subgroups)
  2. Heavy-rain FPs                 (edge_heavy_rain == 1 among FPs)
  3. Label-conflict FPs             (edge_label_conflict == 1 among FPs)

For each cluster this script:
  - Shows the feature distribution (rainfall_mm, soil_moisture, soil_moisture vs
    field_capacity, demand_minus_rain, etc.) compared to TRUE-NEGATIVE rows
    in the same subgroup.
  - Prints summary statistics and writes per-cluster CSVs.
  - Writes a consolidated markdown report.

Outputs -> results/fp_cluster_investigation/
  cluster1_sugarcane_veg_sow.csv
  cluster2_heavy_rain.csv
  cluster3_label_conflict.csv
  cluster_investigation_report.md

Usage:
  python investigate_fp_clusters.py
"""

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

THRESHOLD   = 2.7
OOF_PATH    = os.path.join("results", "threshold_testing", "out_of_fold_predictions.csv")
FEAT_PATH   = os.path.join("primary", "primary_model_ready.csv")
OUT_DIR     = os.path.join("results", "fp_cluster_investigation")
os.makedirs(OUT_DIR, exist_ok=True)

# ── key raw features to inspect ────────────────────────────────────────────
INSPECT_COLS = [
    "rainfall_mm", "soil_moisture", "temperature_c", "wind_speed_kmh",
    "humidity", "et0_mm", "etc_mm", "demand_minus_rain_mm",
    "rain_covers_etc", "eff_rain_window_mm", "rain_per_day_mm",
    "rel_moisture", "dr_mm", "below_mad", "vpd_kpa",
    "field_capacity", "wilting_point", "kc",
]

# ── load & join ─────────────────────────────────────────────────────────────
print("Loading OOF predictions …")
oof = pd.read_csv(OOF_PATH)
oof.rename(columns={"source_row_index": "row_idx"}, inplace=True)
# If no explicit source_row_index, use the DataFrame's natural index
if "row_idx" not in oof.columns:
    oof["row_idx"] = oof.index

print(f"Loading primary features from {FEAT_PATH} …")
feat = pd.read_csv(FEAT_PATH)
feat["row_idx"] = feat.index

# join on row index – keep only training rows (split == 'train') for OOF validity
feat_train = feat[feat["split"] == "train"].copy() if "split" in feat.columns else feat.copy()
data = oof.merge(feat_train[["row_idx"] + [c for c in INSPECT_COLS if c in feat_train.columns]],
                  on="row_idx", how="left")

print(f"Joined shape: {data.shape}")

# ── derive FP/FN/TN labels ──────────────────────────────────────────────────
data["_pred_irr"]   = (data["predicted_water_amount_mm"] >= THRESHOLD).astype(int)
data["_actual_irr"] = (data["actual_water_amount_mm"]    >= THRESHOLD).astype(int)
data["_fp"] = ((data["_pred_irr"] == 1) & (data["_actual_irr"] == 0)).astype(int)
data["_tn"] = ((data["_pred_irr"] == 0) & (data["_actual_irr"] == 0)).astype(int)
data["_fn"] = ((data["_pred_irr"] == 0) & (data["_actual_irr"] == 1)).astype(int)
data["_tp"] = ((data["_pred_irr"] == 1) & (data["_actual_irr"] == 1)).astype(int)

fp_all = data[data["_fp"] == 1].copy()
tn_all = data[data["_tn"] == 1].copy()

avail = [c for c in INSPECT_COLS if c in data.columns]
print(f"Inspection features available: {avail}")

# ── helper: compare FP vs TN for a subgroup ─────────────────────────────────

def compare_fp_tn(fp_sub, tn_sub, label, out_csv):
    """Write a side-by-side median / mean comparison of FP vs TN for key features."""
    rows = []
    for col in avail:
        fp_vals = fp_sub[col].dropna()
        tn_vals = tn_sub[col].dropna()
        rows.append({
            "feature":      col,
            "FP_n":         len(fp_vals),
            "FP_median":    round(fp_vals.median(), 4) if len(fp_vals) else None,
            "FP_mean":      round(fp_vals.mean(),   4) if len(fp_vals) else None,
            "FP_p25":       round(fp_vals.quantile(0.25), 4) if len(fp_vals) else None,
            "FP_p75":       round(fp_vals.quantile(0.75), 4) if len(fp_vals) else None,
            "TN_n":         len(tn_vals),
            "TN_median":    round(tn_vals.median(), 4) if len(tn_vals) else None,
            "TN_mean":      round(tn_vals.mean(),   4) if len(tn_vals) else None,
            "delta_median": round(fp_vals.median() - tn_vals.median(), 4)
                            if len(fp_vals) and len(tn_vals) else None,
        })
    result = pd.DataFrame(rows)
    result.to_csv(out_csv, index=False)
    print(f"  [{label}] FP n={len(fp_sub)}, TN n={len(tn_sub)}  →  {out_csv}")
    return result


# ── Cluster 1 : Sugarcane × Vegetative  and  Sugarcane × Sowing ─────────────
print("\n── Cluster 1: Sugarcane + Vegetative/Sowing ──")

c1_mask = (data["crop"] == "sugarcane") & (data["growth_stage"].isin(["Vegetative", "Sowing"]))
c1_fp   = data[c1_mask & (data["_fp"] == 1)].copy()
c1_tn   = data[c1_mask & (data["_tn"] == 1)].copy()

# Save full FP rows
c1_save_cols = ["crop", "soil_type", "growth_stage", "is_edge_case", "edge_reasons",
                "actual_water_amount_mm", "predicted_water_amount_mm"] + avail
c1_fp[[c for c in c1_save_cols if c in c1_fp.columns]].sort_values(
    "predicted_water_amount_mm", ascending=False
).to_csv(os.path.join(OUT_DIR, "cluster1_sugarcane_veg_sow.csv"), index=False)

c1_comp = compare_fp_tn(c1_fp, c1_tn,
                         "Sugarcane Veg/Sow",
                         os.path.join(OUT_DIR, "cluster1_feature_comparison.csv"))

# breakdown by soil within this cluster
c1_by_soil = (c1_fp.groupby("soil_type").size()
                    .reset_index(name="FP_count")
                    .sort_values("FP_count", ascending=False))


# ── Cluster 2 : Heavy-rain FPs ──────────────────────────────────────────────
print("\n── Cluster 2: Heavy Rain FPs ──")

c2_fp = fp_all[fp_all["edge_heavy_rain"] == 1].copy() if "edge_heavy_rain" in fp_all.columns else \
        fp_all[fp_all["edge_reasons"].str.contains("heavy_rain", na=False)].copy()
# Match TN rows that also have heavy_rain flag for fair comparison
if "edge_heavy_rain" in tn_all.columns:
    c2_tn = tn_all[tn_all["edge_heavy_rain"] == 1].copy()
else:
    c2_tn = tn_all[tn_all["edge_reasons"].str.contains("heavy_rain", na=False)].copy()

c2_fp[[c for c in c1_save_cols if c in c2_fp.columns]].sort_values(
    "predicted_water_amount_mm", ascending=False
).to_csv(os.path.join(OUT_DIR, "cluster2_heavy_rain.csv"), index=False)

c2_comp = compare_fp_tn(c2_fp, c2_tn,
                         "Heavy Rain",
                         os.path.join(OUT_DIR, "cluster2_feature_comparison.csv"))


# ── Cluster 3 : Label-conflict FPs ──────────────────────────────────────────
print("\n── Cluster 3: Label Conflict FPs ──")

c3_fp = fp_all[fp_all["edge_label_conflict"] == 1].copy() if "edge_label_conflict" in fp_all.columns else \
        fp_all[fp_all["edge_reasons"].str.contains("label_conflict", na=False)].copy()
c3_tn = tn_all[tn_all["edge_label_conflict"] == 1].copy() if "edge_label_conflict" in tn_all.columns else \
        tn_all[tn_all["edge_reasons"].str.contains("label_conflict", na=False)].copy()

c3_fp[[c for c in c1_save_cols if c in c3_fp.columns]].sort_values(
    "predicted_water_amount_mm", ascending=False
).to_csv(os.path.join(OUT_DIR, "cluster3_label_conflict.csv"), index=False)

# For label conflicts, also show the actual vs predicted spread
c3_actual_dist = c3_fp["actual_water_amount_mm"].describe()

c3_comp = compare_fp_tn(c3_fp, c3_tn,
                         "Label Conflict",
                         os.path.join(OUT_DIR, "cluster3_feature_comparison.csv"))


# ── helper: top-N rows of a comparison table for report ─────────────────────

def sig_rows(comp_df, n=8):
    """Return rows sorted by abs(delta_median), top n."""
    if comp_df.empty:
        return comp_df
    return (comp_df.dropna(subset=["delta_median"])
                   .assign(_abs_delta=lambda d: d["delta_median"].abs())
                   .sort_values("_abs_delta", ascending=False)
                   .drop(columns=["_abs_delta"])
                   .head(n))


# ── Write markdown report ────────────────────────────────────────────────────

def df_to_md(df, max_rows=12):
    if df is None or df.empty:
        return ["*No data.*"]
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.head(max_rows).iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return lines


md = []

md.append("# FP Cluster Investigation Report")
md.append("")
md.append(f"**Threshold**: {THRESHOLD} mm  |  **Total FP (OOF)**: {len(fp_all)}")
md.append("")

# ────────────────────────────────────────────────────────────────────────────
md.append("---")
md.append("")
md.append("## Cluster 1 — Sugarcane × Vegetative / Sowing")
md.append("")
md.append(f"**FP count**: {len(c1_fp)}  |  **Comparable TN count**: {len(c1_tn)}")
md.append("")

# breakdown by stage
c1_by_stage = (c1_fp.groupby("growth_stage").size()
                     .reset_index(name="FP_count")
                     .sort_values("FP_count", ascending=False))
md.append("### FP count by stage (within Sugarcane)")
md.append("")
md += df_to_md(c1_by_stage)
md.append("")

md.append("### FP count by soil type (within Sugarcane Veg/Sow)")
md.append("")
md += df_to_md(c1_by_soil)
md.append("")

md.append("### Feature comparison: FP vs TN rows (top differentiators)")
md.append("")
md.append("Columns: feature | FP median | TN median | delta (FP−TN)")
md.append("")
c1_sig = sig_rows(c1_comp)[["feature", "FP_n", "FP_median", "TN_median", "delta_median"]]
md += df_to_md(c1_sig, max_rows=12)
md.append("")

# interpretation
if not c1_comp.empty and "rainfall_mm" in c1_comp["feature"].values:
    rain_row = c1_comp[c1_comp["feature"] == "rainfall_mm"].iloc[0]
    md.append(f"> **Rainfall_mm**: FP median = {rain_row['FP_median']}, "
              f"TN median = {rain_row['TN_median']} "
              f"(delta = {rain_row['delta_median']})")
if "demand_minus_rain_mm" in c1_comp["feature"].values:
    dmr_row = c1_comp[c1_comp["feature"] == "demand_minus_rain_mm"].iloc[0]
    md.append(f"> **demand_minus_rain_mm**: FP median = {dmr_row['FP_median']}, "
              f"TN median = {dmr_row['TN_median']}")
md.append("")

md.append("### What this tells us")
md.append("")
# dynamic interpretation
fp_rain_med = c1_fp["rainfall_mm"].median() if "rainfall_mm" in c1_fp.columns else None
tn_rain_med = c1_tn["rainfall_mm"].median() if "rainfall_mm" in c1_tn.columns else None
fp_sm_med   = c1_fp["soil_moisture"].median() if "soil_moisture" in c1_fp.columns else None
tn_sm_med   = c1_tn["soil_moisture"].median() if "soil_moisture" in c1_tn.columns else None
fp_rc       = c1_fp["rain_covers_etc"].mean() if "rain_covers_etc" in c1_fp.columns else None
tn_rc       = c1_tn["rain_covers_etc"].mean() if "rain_covers_etc" in c1_tn.columns else None
fp_rr       = c1_fp["rain_per_day_mm"].median() if "rain_per_day_mm" in c1_fp.columns else None
tn_rr       = c1_tn["rain_per_day_mm"].median() if "rain_per_day_mm" in c1_tn.columns else None

if fp_rain_med is not None:
    if fp_rain_med > (tn_rain_med or 0) * 1.5:
        md.append(f"- FP rows have **much higher rainfall** (median {fp_rain_med:.0f} mm) "
                  f"vs TN rows ({tn_rain_med:.0f} mm). "
                  f"Despite sufficient rain, the model still predicts irrigation — "
                  f"suggesting it underweights `rainfall_mm` for sugarcane in early/active stages.")
    elif fp_rain_med < (tn_rain_med or 999) * 0.7:
        md.append(f"- FP rows have **lower rainfall** (median {fp_rain_med:.0f} mm) "
                  f"than TN rows ({tn_rain_med:.0f} mm) — "
                  f"these are genuinely dry conditions where the model correctly senses "
                  f"high demand but the FAO-56 target is zero (possibly due to crop stage/Kc).")
    else:
        md.append(f"- FP and TN rows have similar rainfall (FP {fp_rain_med:.0f} mm, "
                  f"TN {tn_rain_med:.0f} mm). The cause is likely in other features (VPD, Kc, soil).")

if fp_rc is not None:
    md.append(f"- `rain_covers_etc` (flag: rain ≥ ETc): FP mean = {fp_rc:.2f}, "
              f"TN mean = {tn_rc:.2f}. "
              + ("**Rain does NOT cover ETc for most FPs** — model behaviour is consistent "
                 "with observed demand." if fp_rc < 0.4 else
                 "**Rain covers ETc for many FPs** — the model is ignoring the rain-covers flag."))
md.append("")

# ────────────────────────────────────────────────────────────────────────────
md.append("---")
md.append("")
md.append("## Cluster 2 — Heavy Rain FPs")
md.append("")
md.append(f"**FP count**: {len(c2_fp)}  |  **Heavy-rain TN count**: {len(c2_tn)}")
md.append("")

md.append("### FP breakdown by crop")
md.append("")
c2_by_crop = (c2_fp.groupby("crop").size()
                    .reset_index(name="FP_count")
                    .sort_values("FP_count", ascending=False))
md += df_to_md(c2_by_crop)
md.append("")

md.append("### Feature comparison: heavy-rain FP vs heavy-rain TN (top differentiators)")
md.append("")
c2_sig = sig_rows(c2_comp)[["feature", "FP_n", "FP_median", "TN_median", "delta_median"]]
md += df_to_md(c2_sig, max_rows=12)
md.append("")

# rainfall stats for FP heavy-rain rows
hr_rain_fp = c2_fp["rainfall_mm"].describe().round(2) if "rainfall_mm" in c2_fp.columns else None
hr_rain_tn = c2_tn["rainfall_mm"].describe().round(2) if "rainfall_mm" in c2_tn.columns else None
if hr_rain_fp is not None:
    md.append(f"> **rainfall_mm in heavy-rain FPs**: "
              f"min={hr_rain_fp['min']}, median={hr_rain_fp['50%']}, "
              f"max={hr_rain_fp['max']}")
    if hr_rain_tn is not None:
        md.append(f"> **rainfall_mm in heavy-rain TNs**: "
                  f"min={hr_rain_tn['min']}, median={hr_rain_tn['50%']}, "
                  f"max={hr_rain_tn['max']}")
md.append("")

md.append("### What this tells us")
md.append("")
if "rain_covers_etc" in c2_fp.columns:
    fp_rc2 = c2_fp["rain_covers_etc"].mean()
    md.append(f"- `rain_covers_etc` mean for heavy-rain FPs = **{fp_rc2:.2f}** "
              + ("→ Even in flagged heavy-rain rows, ETc demand often still exceeds rainfall. "
                 "These may not be pure 'rain suppresses irrigation' failures; "
                 "the FAO-56 ETc for that crop/stage genuinely exceeds what fell."
                 if fp_rc2 < 0.5 else
                 "→ Rain does cover ETc for most heavy-rain FPs. "
                 "The model is **not learning the rain-covers-etc signal** for these rows."))
md.append("- Check `demand_minus_rain_mm` median: if negative for FPs, model ignores surplus rain.")
if "demand_minus_rain_mm" in c2_fp.columns:
    dmr = c2_fp["demand_minus_rain_mm"].median()
    md.append(f"  - Median demand_minus_rain_mm for heavy-rain FPs = **{dmr:.2f} mm** "
              + ("(negative → rain surplus; model fired anyway)" if dmr < 0
                 else "(positive → genuine deficit; FAO target=0 may be a label artifact)"))
md.append("")

# ────────────────────────────────────────────────────────────────────────────
md.append("---")
md.append("")
md.append("## Cluster 3 — Label Conflict FPs")
md.append("")
md.append(f"**FP count**: {len(c3_fp)}  |  **Label-conflict TN count**: {len(c3_tn)}")
md.append("")

md.append("### What is a label conflict?")
md.append("")
md.append("A label conflict is flagged when the FAO-56 derived `water_amount_mm` target is > 0 "
          "but the binary `irrigate` label is 0 (or vice-versa). "
          "In such rows the regression target and binary label disagree — "
          "the model may produce predictions that reflect the continuous FAO signal "
          "while the binary ground truth says 'no irrigation'.")
md.append("")

md.append("### Distribution of actual_water_amount_mm for label-conflict FPs")
md.append("")
md.append("*(These rows are classed as FP because predicted ≥ 2.7 mm and actual < 2.7 mm, "
          "but the actual value is non-zero — borderline ground truth.)*")
md.append("")
if not c3_fp.empty:
    desc = c3_fp["actual_water_amount_mm"].describe().round(4)
    md.append(f"- count: {int(desc['count'])}  min: {desc['min']}  "
              f"median: {desc['50%']}  mean: {desc['mean']:.4f}  max: {desc['max']}")
    actual_nonzero = (c3_fp["actual_water_amount_mm"] > 0).sum()
    md.append(f"- Rows where actual > 0 mm: **{actual_nonzero} / {len(c3_fp)}** "
              f"({actual_nonzero/len(c3_fp)*100:.0f}%) — genuine borderline cases.")
md.append("")

md.append("### FP breakdown by crop")
md.append("")
c3_by_crop = (c3_fp.groupby("crop").size()
                    .reset_index(name="FP_count")
                    .sort_values("FP_count", ascending=False))
md += df_to_md(c3_by_crop)
md.append("")

md.append("### Feature comparison: label-conflict FP vs label-conflict TN (top differentiators)")
md.append("")
c3_sig = sig_rows(c3_comp)[["feature", "FP_n", "FP_median", "TN_median", "delta_median"]]
md += df_to_md(c3_sig, max_rows=12)
md.append("")

md.append("### What this tells us")
md.append("")
md.append("Label conflicts fall into two sub-types:")
md.append("")
md.append("| Sub-type | Description | Model behaviour |")
md.append("|---|---|---|")
md.append("| **Ambiguous zero** | actual = 0 but FAO physics suggest partial need | "
          "Model picks up the physics signal; binary ground truth may be wrong |")
md.append("| **Soft borderline** | actual > 0 but < 2.7 mm threshold | "
          "Model and FAO agree there's some need; only the threshold matters |")
md.append("")
md.append("These are the **least actionable** FPs — they stem from inherent label noise, "
          "not model error. Removing or down-weighting label-conflict rows during training "
          "is the most principled fix (not post-hoc rule-based).")
md.append("")

# ────────────────────────────────────────────────────────────────────────────
md.append("---")
md.append("")
md.append("## Summary & Recommended Actions")
md.append("")

md.append("| Cluster | FP Count | Root Cause | Recommended Fix |")
md.append("|---|---|---|---|")

c1_root = "Model predicts high demand for sugarcane Veg/Sow regardless of rain"
c2_root = "demand_minus_rain_mm positive (ETc > rain) even in heavy-rain rows"
c3_root = "Ground truth label noise — FAO physics and binary label disagree"

c1_fix = "Add `rain_covers_etc` or `demand_minus_rain_mm` as explicit suppressor features; or post-hoc rule"
c2_fix = "Review ETc calculation for these crops; consider `eff_rain_window_mm` feature weight"
c3_fix = "Remove/down-weight label-conflict rows in training, or set a 'conflict confidence' flag"

md.append(f"| Sugarcane Veg/Sow | {len(c1_fp)} | {c1_root} | {c1_fix} |")
md.append(f"| Heavy Rain | {len(c2_fp)} | {c2_root} | {c2_fix} |")
md.append(f"| Label Conflict | {len(c3_fp)} | {c3_root} | {c3_fix} |")
md.append("")

total_clustered = len(c1_fp) + len(c2_fp) + len(c3_fp)
# note some rows overlap (e.g., heavy_rain + label_conflict)
c1_ids = set(c1_fp.index)
c2_ids = set(c2_fp.index)
c3_ids = set(c3_fp.index)
overlap = c1_ids & c2_ids | c1_ids & c3_ids | c2_ids & c3_ids
unique_clustered = len(c1_ids | c2_ids | c3_ids)
unclustered = len(fp_all) - unique_clustered

md.append(f"**Unique FPs covered by these 3 clusters**: {unique_clustered} / {len(fp_all)} "
          f"({unique_clustered/len(fp_all)*100:.0f}%)")
md.append(f"**Remaining unexplained FPs**: {unclustered} — worth checking with feature SHAP values "
          f"on a per-row basis to find any other systematic drivers.")
md.append("")

md.append("## Output Files")
md.append("")
md.append("All in `results/fp_cluster_investigation/`:")
md.append("")
for f, d in [
    ("cluster1_sugarcane_veg_sow.csv",   "Full FP rows for cluster 1"),
    ("cluster1_feature_comparison.csv",   "FP vs TN feature stats for cluster 1"),
    ("cluster2_heavy_rain.csv",           "Full FP rows for cluster 2"),
    ("cluster2_feature_comparison.csv",   "FP vs TN feature stats for cluster 2"),
    ("cluster3_label_conflict.csv",       "Full FP rows for cluster 3"),
    ("cluster3_feature_comparison.csv",   "FP vs TN feature stats for cluster 3"),
    ("cluster_investigation_report.md",   "This report"),
]:
    md.append(f"- `{f}` — {d}")

report_path = os.path.join(OUT_DIR, "cluster_investigation_report.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md))

print(f"\nReport  ->  {report_path}")
print(f"CSVs    ->  {OUT_DIR}/")
print("\nDone.")
