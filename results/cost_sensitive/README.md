# Cost-sensitive irrigation experiment

## Read-only RCI summary

- `water_amount_mm` is derived in `feature_engineering.py`: root-zone depletion minus effective rain, floored at zero, capped at 75 mm, and set to zero below the 2 mm minimum event amount. `irrigate` is exactly `water_amount_mm > 0`; existing analysis reports zero disagreements between that label and target-derived positivity.
- The existing `proposed7` set contains four weather/soil moisture measurements and one-hot crop (six), soil (four), and growth-stage (four) indicators: 18 features. It omits the target-derived/leaky columns listed in `feature_sets.json`.
- `train_all_models.py` uses ExtraTreesRegressor with `n_estimators=300`, `max_depth=20`, `min_samples_split=2`; its general hyperparameter selection uses MAE. The prior tuning script also tunes by MAE, but these experiments retain the original configuration.
- `analyze_model_testing.py` produces OOF predictions from only `split == train`, using 5-fold `StratifiedKFold(shuffle=True, random_state=42)` stratified by crop × `irrigate`. Scaling is fit inside each fold, regression predictions are clipped at zero, and decisions are `prediction >= threshold`. Confusion matrices use `[[TN, FP], [FN, TP]]`.
- No sample weighting or cost-sensitive training existed in this pipeline. ExtraTreesRegressor accepts `sample_weight` in `fit`, but this changes the weighted squared-error tree fitting criterion. It does not optimize FP/FN counts, which are discontinuous thresholded outcomes. In particular, labeling an observation negative does not mean every positive regression residual is an irrigation false positive.
- Crop, soil, and stage are available from the proposed7 one-hot columns; `meta_crop` is used only to stratify and report subgroups, never as an input. Soil and stage categories are decoded from one-hot values for subgroup reporting.

## Experiment design and safeguards

Run from the repository root with `python cost_sensitive_experiment.py`. The script reads only `split == train`, uses the exact five outer folds and fixed ExtraTrees configuration above, fits preprocessing/models on each outer training partition, and saves row-level OOF predictions, per-fold and pooled metrics, and critical subgroup metrics here. It never reads the held-out test data for model fitting or selection. Thresholds and ratios are fixed experiments, not selected using outer validation outcomes; there is no claimed nested-CV-selected winner.

The configurable ratios are in `COST_RATIOS`: equal (1:1), FP (2:1, 3:1), FN (1:2), and balanced (2:2). Training weights are class-based and computed from each training fold's recorded `irrigate` labels. Uniform weights are normalized to mean one; consequently 1:1 and 2:2 produce the same weights by design. The optional group FN multipliers are in `GROUP_FN_MULTIPLIERS`; the current exploratory setting doubles training weight for positive-label rice, Sowing, and sandy-soil observations. These are sensitivity experiments, not agricultural cost estimates.

The regression strategies use the configured FP:FN ratio for cost reporting, while unweighted baseline cost is reported at 1:1. Classifier strategies train ExtraTreesClassifier on the same features and apply class-cost sample weights; decisions use its 0.5 probability cutoff. Their predictions are an independent irrigation decision and do not define a water amount. MAE/RMSE/R² are therefore not applicable to classifier-only rows. Classification may improve one objective while losing regression amount consistency.

## Outputs

- `oof_predictions.csv`: one outer-fold prediction/decision per training row and strategy, with crop, soil, and stage metadata.
- `fold_metrics.csv`: fold-wise stability metrics. FP/FN values are per fold, so sum across five folds for pooled counts.
- `pooled_metrics.csv`: pooled OOF metrics, including requested thresholds and total/average decision cost under the indicated ratio.
- `subgroup_metrics.csv`: pooled rice, sandy-soil, and Sowing subgroup FP/FN/recall results.

Confusion counts use the original binary target, and rates are FP/(FP+TN) and FN/(FN+TP). Re-run `python -m unittest testing.test_cost_sensitive` for cost arithmetic, weighting, and fold-separation checks.

## Interpretation limits

This is validation on the project's existing training partition, not independent field evidence. The target itself is a synthetic FAO-56-style derived amount and irrigate flag. Costs are relative experimental units and do not encode water, energy, yield, or crop-specific monetary values. The fixed decision threshold sweep compares model training effects against threshold-only tuning, but no threshold/ratio is selected from these OOF results for final deployment. Since the held-out test split has already been used in earlier repository analyses, it is not a fresh final test set.
