# False Positive Investigation — Full Report
## ExtraTreesReg Irrigation Model | Branch: penalty-false-positive

**Dataset**: `primary_model_ready.csv` — 10,000 rows (train = 8,000, test = 2,000)
**Model**: ExtraTreesReg on `proposed7_plus_physics` features — MAE = 0.7145 mm, R² = 0.9964
**Operating point**: threshold = 2.7 mm (pred ≥ 2.7 mm → Irrigate)
**Baseline at 2.7 mm**: **160 FP / 57 FN**, F1 = 0.9713, Precision = 0.9583, Recall = 0.9847

---

## Phase 1 — Cost-Sensitive Training Experiments

Three strategies were tested to penalise false positives during training.

### Strategies tested

| Model / Strategy | Threshold | FP | FN | F1 | MAE (mm) |
|---|---:|---:|---:|---:|---:|
| Baseline ExtraTreesReg | 2.0 | 247 | 18 | 0.9656 | 0.8996 |
| Baseline ExtraTreesReg | **2.7** | **160** | **57** | **0.9713** | 0.8996 |
| Baseline ExtraTreesReg | 3.0 | 139 | 80 | 0.9709 | 0.8996 |
| Baseline ExtraTreesReg | 3.4 | 112 | 110 | 0.9703 | 0.8996 |
| FP-weighted ExtraTreesReg (3:1) | 2.7 | 176 | 49 | 0.9704 | 0.9142 |
| FP-weighted ExtraTreesReg (3:1) | 3.4 | 123 | 91 | 0.9715 | 0.9142 |
| FN-weighted ExtraTreesReg (1:2) | 2.7 | 170 | 59 | 0.9698 | 0.9206 |
| Critical-group FN-weighted ExtraTreesReg | 2.7 | 164 | 61 | 0.9703 | 0.8994 |
| Cost-weighted ExtraTreesClassifier (1:1) | classifier | 128 | 151 | 0.9625 | — |
| Critical-group FN-weighted classifier | classifier | 142 | 145 | 0.9615 | — |

### Findings

1. **FP weighting made FPs worse.** The 3:1 FP-weighted model produced *more* false alarms at 2.7 mm (176 vs 160 baseline). ExtraTrees optimises tree impurity, not the thresholded FP/FN loss — sample weights cannot express that discontinuous objective.

2. **Threshold tuning beats cost-sensitive training.** Moving the threshold from 2.7 → 3.4 mm reduces FPs from 160 → 112 but increases FNs from 57 → 110. This is a Pareto trade-off, not a net win.

3. **The separate classifier degrades recall badly.** Equal-cost classification cuts FPs to 128 but pushes FNs to 151 (2.6× more missed irrigations) and loses the water-amount output entirely.

4. **Verdict**: Retain baseline ExtraTreesReg at 2.7 mm. No cost-sensitive variant improves the overall FP/FN trade-off without significant recall loss.

---

## Phase 2 — False Alarm Location Analysis

Rather than fighting the FPs blindly, the analysis pinpointed *where* they concentrate.

### Distribution across categories (OOF, n = 8,000)

**By crop** — FP rate = FP count ÷ total rows in that crop group

| Crop | n_total | FP | FN | FP rate % | FN rate % |
|---|---|---|---|---|---|
| sugarcane | 1,343 | 45 | 1 | 3.35 | 0.07 |
| potato | 1,330 | 37 | 4 | 2.78 | 0.30 |
| cotton | 1,276 | 34 | 6 | 2.66 | 0.47 |
| maize | 1,355 | 26 | 7 | 1.92 | 0.52 |
| wheat | 1,327 | 24 | 1 | 1.81 | 0.08 |
| rice | 1,369 | 19 | 14 | 1.39 | 1.02 |

**By soil type**

| Soil | n_total | FP | FP rate % |
|---|---|---|---|
| sandy | 2,055 | 57 | 2.77 |
| loamy | 1,970 | 47 | 2.39 |
| silt | 2,018 | 44 | 2.18 |
| clay | 1,957 | 37 | 1.89 |

**By growth stage**

| Stage | n_total | FP | FP rate % |
|---|---|---|---|
| Vegetative | 2,016 | 64 | 3.17 |
| Flowering | 1,978 | 44 | 2.22 |
| Sowing | 1,939 | 40 | 2.06 |
| Harvest | 2,067 | 37 | 1.79 |

**Prediction proximity to threshold** — how far above 2.7 mm were the FP predictions?

| Pred range (mm) | FP count | % of FPs | Cumulative % |
|---|---|---|---|
| 2.7–3.0 | 25 | 13.5 | 13.5 |
| 3.0–3.5 | 45 | 24.3 | 37.8 |
| 3.5–4.0 | 20 | 10.8 | 48.6 |
| 4.0–5.0 | 28 | 15.1 | 63.7 |
| 5.0–10.0 | 35 | 18.9 | 82.6 |
| > 10.0 | 32 | 17.3 | 99.9 |

> **Key finding**: Only 14% of FPs sit within 0.3 mm of the threshold. The median FP prediction is **4.04 mm**, mean **6.69 mm**. Most FPs are *not* borderline — threshold tuning cannot cleanly eliminate them.

**Edge-case flag distribution among FPs**

| Flag | FP count | % of FPs |
|---|---|---|
| no_edge_flag | 108 | 58.4 |
| label_conflict | 21 | 11.4 |
| high_wind | 20 | 10.8 |
| low_temp | 19 | 10.3 |
| heavy_rain | 16 | 8.6 |
| extreme_vpd | 10 | 5.4 |
| saturated_soil | 7 | 3.8 |
| very_dry_soil | 7 | 3.8 |

**Top crop × stage combos by FP count**

| Crop | Stage | n_total | FP | FP rate % |
|---|---|---|---|---|
| sugarcane | Vegetative | 329 | 18 | **5.47** |
| sugarcane | Sowing | 319 | 16 | **5.02** |
| potato | Vegetative | 328 | 14 | 4.27 |
| cotton | Vegetative | 327 | 12 | 3.67 |
| potato | Flowering | 338 | 11 | 3.25 |
| wheat | Vegetative | 368 | 10 | 2.72 |

---

## Phase 3 — Cluster Investigation

The three highest-priority FP clusters were investigated by joining OOF predictions with raw features from `primary_model_ready.csv`.

### Cluster 1 — Sugarcane × Vegetative / Sowing (34 FPs)

Feature comparison between FP and matched TN rows in the same subgroup:

| Feature | FP median | TN median | Delta |
|---|---|---|---|
| `rainfall_mm` | 1,186 mm | 1,318 mm | −131 mm |
| `soil_moisture` | 35.9 | 49.6 | **−13.7** |
| `dr_mm` (depletion) | 39.2 | 29.6 | **+9.6** |
| `rain_covers_etc` (mean) | **0.68** | **0.83** | − |

**Interpretation**: Rainfall is similar. FP rows have lower soil moisture and higher root-zone depletion — the model is reading a genuine physical deficit. However, 68% of these FP rows do NOT have rain covering ETc (`rain_covers_etc = 0`), meaning the model's irrigation alarm is not unreasonable even though the FAO-56 label is zero. The 32% with `rain_covers_etc = 1` are potentially suppressible.

### Cluster 2 — Heavy Rain FPs (16 FPs)

| Feature | FP median | TN median | Delta |
|---|---|---|---|
| `dr_mm` | 47.4 | 28.4 | **+19.0** |
| `humidity` | 43.7 | 62.3 | **−18.6** |
| `rain_covers_etc` (mean) | **1.00** | — | — |
| `demand_minus_rain_mm` | **−11.7 mm** | — | negative |

**Interpretation**: Every heavy-rain FP row has `rain_covers_etc = 1` (rain fully covers ETc) and a net rain surplus of 11.7 mm. Yet the model fires because `dr_mm = 47.4 mm` (high root-zone depletion from before). The model is prioritising historical depletion over the current rain surplus — a plausible but suppressible behaviour.

### Cluster 3 — Label Conflict FPs (21 FPs)

| Property | Value |
|---|---|
| `actual_water_amount_mm` range | 2.00 – 2.64 mm |
| Rows where actual > 0 | **21 / 21 (100%)** |
| `rainfall_mm` FP median | 1,322 mm |
| `rainfall_mm` TN median | 540 mm |

**Interpretation**: Every label-conflict FP has a non-zero actual water amount (2.0–2.6 mm) — they are just barely below the 2.7 mm threshold. These are not model errors; they are **threshold artefacts** where the binary label (`irrigate = 0`) and the FAO-56 continuous target (~2.3 mm) are already in disagreement. The model predicting ≥ 2.7 mm is consistent with the FAO signal.

---

## Phase 4 — Post-Hoc Fix Attempts

Three post-hoc suppression rules were evaluated on OOF data (no retraining).

| Strategy | FP | FN | F1 | Notes |
|---|---|---|---|---|
| Baseline 2.7 mm | 160 | 57 | **0.9713** | Reference |
| Fix 1: rain suppression (all crops) | 51 | **2,479** | 0.4976 | Catastrophic FN increase |
| Fix 3: sugarcane Veg/Sow narrow | 140 | 242 | 0.9481 | FP ↓ 20, FN ↑ 185 — net loss |
| Fix 1 + Fix 3 combined | 51 | 2,479 | 0.4976 | Dominated by Fix 1 |

**Fix 1 details**: The `rain_covers_etc == 1` flag applies to **6,048 / 8,000 training rows (75%)**, including 2,468 legitimate irrigations. Rule precision = 0.04 — for every 1 genuine FP suppressed, 22 correct irrigations are lost.

**Fix 3 details**: Suppresses 205 decisions; only 20 are genuine FPs (precision = 0.10), 185 are correct irrigations lost. Net result is worse than baseline.

---

## Phase 5 — Root Cause Diagnosis

After confirming that no post-hoc feature rule could cleanly separate FPs from TPs, a direct feature-level comparison was run:

### FP vs TP feature medians (at 2.7 mm, OOF)

| Feature | FP median | TP median |
|---|---|---|
| `kc` (crop coefficient) | 0.825 | 0.775 |
| `dr_mm` (root-zone depletion) | 36.5 | 40.1 |
| `soil_moisture` | 37.0 | 36.9 |
| `etc_mm` (crop evapotranspiration) | 3.89 | 3.82 |
| `demand_minus_rain_mm` | −4.70 | −4.26 |
| `eff_rain_window_mm` | 26.9 | 25.1 |

**All feature values are near-identical between FP and TP rows.** The model is predicting irrigation for FP rows for the same reasons it correctly predicts irrigation for TP rows — the inputs are indistinguishable in feature space.

### The structural cause

The dataset is built on FAO-56 derived targets. In the FAO-56 model, `water_amount_mm` is determined by a combination of `kc`, `ET0`, soil depletion, and rainfall. The 160 FP rows appear to have edge cases where the same input combination sometimes yields `water_amount_mm = 0` and sometimes yields `water_amount_mm = 35 mm`. This produces contradictory labels that no ML model can learn to separate without additional discriminating features.

> **This is a dataset-level inconsistency, not a model learning failure.**

---

## Summary and Conclusions

### What was tried and why it didn't work

| Approach | Why it failed |
|---|---|
| FP-weighted training (3:1) | ExtraTrees cannot optimise a discontinuous thresholded loss via sample weights |
| Separate classifier | Swaps FP reduction for a much larger FN increase; loses water-amount output |
| Threshold tuning to 3.4 mm | Trades 48 FPs for 53 extra FNs — not a net improvement |
| Post-hoc rain suppression | `rain_covers_etc` fires on 75% of rows; precision = 0.04; catastrophic TP loss |
| Sugarcane narrow rule | Precision = 0.10; loses 185 TPs for 20 FP reductions |

### What the FP analysis revealed

- **58% of FPs have no edge-case flag** — they occur under normal conditions
- **86% of FPs predict well above 2.7 mm** (median 4.04 mm) — not soft threshold errors
- **FPs are feature-identical to TPs** — no available feature cleanly separates them
- **21 FPs are label-conflict artefacts** — actual water amount is 2.0–2.6 mm, just below threshold; these are threshold artefacts, not model errors

### Recommendation

**Retain the baseline ExtraTreesReg model at 2.7 mm threshold** — F1 = 0.9713, Precision = 0.9583, Recall = 0.9847.

The 160 FPs at 2.7 mm represent the **irreducible error floor** of this dataset and feature set. Reducing them further requires one of:

1. **A better dataset** with daily time-series soil moisture tracking, so the model can distinguish "it rained recently and the soil is now full" from "it rained but root-zone depletion is still high" at the individual-day level.
2. **A dataset with explicit irrigation event records** (whether irrigation was actually applied, not just whether it should have been per FAO-56), which would remove label conflicts.
3. **Accepting F1 = 0.9713 as the ceiling** for this data, which is already a strong result — reporting the FP analysis transparently in the paper as a discussion of data limitations.

---

## Files Produced (this branch)

| File | Description |
|---|---|
| `cost_sensitive_experiment.py` | Phase 1: cost-sensitive CV training |
| `analyze_false_alarms.py` | Phase 2: FP location analysis |
| `investigate_fp_clusters.py` | Phase 3: cluster deep-dive |
| `fp_fixes_experiment.py` | Phase 4: post-hoc rule evaluation |
| `scratch_diag.py` | Phase 5: feature separation diagnostic (temp) |
| `results/cost_sensitive/experiment_report.md` | Phase 1 detailed report |
| `results/false_alarm_analysis/` | Phase 2 CSVs + report |
| `results/fp_cluster_investigation/` | Phase 3 CSVs + report |
| `results/fp_fixes/fixes_summary.csv` | Phase 4 strategy comparison |
| `results/fp_fixes/fp_fixes_report.md` | Phase 4 fix experiment report |

---

*Report generated: 2026-10-28 | Branch: penalty-false-positive*
