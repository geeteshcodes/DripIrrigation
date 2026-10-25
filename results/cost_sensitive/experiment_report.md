# Five-fold cost-sensitive experiment results

## Comparison at requested checkpoints

The following are pooled outer-fold OOF results over 8,000 training rows. Baseline decision cost is shown at 1:1 unless the ratio is stated otherwise. Weighted-regression costs use the same ratio named by the training strategy.

| Model/strategy | Threshold | FP | FN | Precision | Recall | F1 | MAE (mm) | Decision cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline ExtraTreesReg | 2.0 | 247 | 18 | 0.9376 | 0.9952 | 0.9656 | 0.8996 | 265 (1:1) |
| Baseline ExtraTreesReg | 2.7 | 160 | 57 | 0.9583 | 0.9847 | 0.9713 | 0.8996 | 217 (1:1) |
| Baseline ExtraTreesReg | 3.0 | 139 | 80 | 0.9633 | 0.9786 | 0.9709 | 0.8996 | 219 (1:1) |
| FP-weighted ExtraTreesReg (3:1) | 2.7 | 176 | 49 | 0.9544 | 0.9869 | 0.9704 | 0.9142 | 577 (3:1) |
| FP-weighted ExtraTreesReg (3:1) | 3.0 | 155 | 67 | 0.9594 | 0.9820 | 0.9706 | 0.9142 | 532 (3:1) |
| FP-weighted ExtraTreesReg (3:1) | 3.4 | 123 | 91 | 0.9673 | 0.9756 | 0.9715 | 0.9142 | 460 (3:1) |
| FN-weighted ExtraTreesReg (1:2) | 2.7 | 170 | 59 | 0.9558 | 0.9842 | 0.9698 | 0.9206 | 288 (1:2) |
| Critical-group FN-weighted ExtraTreesReg | 2.7 | 164 | 61 | 0.9572 | 0.9837 | 0.9703 | 0.8994 | 225 (1:1) |
| Cost-weighted ExtraTreesClassifier (1:1) | classifier | 128 | 151 | 0.9655 | 0.9595 | 0.9625 | — | 279 (1:1) |
| Critical-group FN-weighted classifier | classifier | 142 | 145 | 0.9619 | 0.9611 | 0.9615 | — | 287 (1:1) |

For a fair ratio-specific baseline comparison, applying the same 3:1 evaluation cost to the baseline counts gives 537 at 2.7 mm (3×160 + 57), 497 at 3.0 mm (3×139 + 80), and 446 at 3.4 mm (3×112 + 110). The 3:1 weighted model therefore does not reduce total 3:1 cost at those same thresholds (577, 532, and 460). The FN-weighted 1:2 baseline costs at 2.7 mm 274 (160 + 2×57), versus 288 for the FN-weighted regressor.

The complete metrics include accuracy, FP/FN rates, confusion counts, R²/RMSE, and cost per sample in `pooled_metrics.csv`. All requested thresholds (2.0, 2.5, 2.7, 3.0, 3.2, 3.4, 3.5) are present for each regression strategy. `fold_metrics.csv` preserves fold results. Classifier rows use 0.5 as the decision probability cutoff and have no regression metric.

## Crop, soil, stage, and fold behavior

At 2.7 mm, baseline recall is 95.20% for rice (23 FN / 1,369), 97.45% for sandy soil (17 FN / 2,055), and 94.91% for Sowing (26 FN / 1,939). The critical-group FN-weighted regressor has 25 rice FN, 17 sandy FN, and 26 Sowing FN; recall is 94.78%, 97.45%, and 94.91%, respectively. The broad FP 3:1 regressor has 23, 13, and 23 FN in those groups, with recall 95.20%, 98.05%, and 95.50%, but it increases the overall FP total to 176. These changes are small and do not establish that group weighting is better. `subgroup_metrics.csv` contains every crop, soil category, and stage across all thresholds/strategies, and classifier subgroups.

At 2.7 mm, fold-average counts (mean ± standard deviation across five folds) are baseline 32.0 ± 5.3 FP and 11.4 ± 0.9 FN per fold. FP 3:1 regression gives 35.2 ± 5.8 FP and 9.8 ± 1.3 FN, with fold MAE 0.9142 ± 0.0537 mm versus baseline 0.8996 ± 0.0736. Critical-group FN weighting gives 32.8 ± 6.1 FP and 12.2 ± 3.0 FN; it does not improve the pooled errors. The model fold results and the full threshold rows are in `fold_metrics.csv`.

## Findings and recommendation

1. **False-alarm reduction from cost-sensitive training is not demonstrated.** The explicit FP 2:1 and 3:1 regression weights increase FP at 2.7 mm (172 and 176 versus 160 baseline), despite reducing FN somewhat. Their 2.7 mm F1 values are lower than baseline. FN 1:2 weighting also does not improve FN or MAE.
2. **Threshold tuning remains the stronger, simpler control.** The unweighted model at 3.4 mm has 112 FP / 110 FN and F1 0.9703, but its known tradeoff includes weaker critical-group recall; at 2.7 mm it has 160/57 and F1 0.9713. Baseline 3.0 mm has 139/80. These are Pareto tradeoffs, not a universally best operating point.
3. **The separate classifier changes the policy and harms recall.** Equal-cost classification gives 128 FP but 151 FN and F1 0.9625, with no water-amount output. Reweighting its classes did not yield a better overall tradeoff than regression thresholding.
4. **Recommendation:** retain ExtraTreesReg as amount model and do not adopt these cost-weighted fits as a replacement. Keep the baseline 2.7 mm validation point as the current F1-focused candidate; a lower FP policy could consider 3.4 mm only after explicit crop-protection review and independent field validation. Its reduced FP count relative to 2.7 mm comes from threshold movement, not cost-sensitive training. Avoid a deployment claim because this dataset has synthetic derived targets and the existing held-out split has already informed prior repository analyses.

No custom combined differentiable regression/decision objective was applied: ExtraTrees optimizes tree impurity for regression/classification, and its sample weights cannot express the discontinuous thresholded FP/FN loss. A genuinely joint objective would need an estimator designed for that loss or a separate policy/model selection procedure conducted inside nested training CV. The experiments here hold ratios and thresholds fixed in advance and make no nested-selection claim.
