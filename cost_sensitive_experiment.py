"""Training-only cost-sensitive CV experiments for proposed7 irrigation models.

All rows are from the predefined training split. Outer folds produce metrics;
no test split is loaded. Penalty weights are based on recorded labels, never
held-out residuals. See results/cost_sensitive/README.md for interpretation.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, recall_score, r2_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
SEED = 42
THRESHOLDS = (2.0, 2.5, 2.7, 3.0, 3.2, 3.4, 3.5)
COST_RATIOS = {"equal_1_1": (1, 1), "fp_2_1": (2, 1), "fp_3_1": (3, 1),
               "fn_1_2": (1, 2), "balanced_2_2": (2, 2)}
TREE_PARAMS = dict(n_estimators=300, max_depth=20, min_samples_split=2)


def decision_cost(fp, fn, c_fp, c_fn):
    return c_fp * fp + c_fn * fn


def label_weights(y, c_fp, c_fn, groups=None, sensitive_multiplier=1.0):
    """Class-label cost weights; optional multiplier targets rice/sowing/sandy."""
    y = np.asarray(y).astype(int)
    weights = np.where(y == 0, c_fp, c_fn).astype(float)
    if groups is not None and sensitive_multiplier != 1:
        groups = pd.DataFrame(groups).reset_index(drop=True)
        sensitive = ((groups.get("crop", "") == "rice") |
                     (groups.get("growth_stage", "") == "Sowing") |
                     (groups.get("soil_type", "") == "sandy"))
        weights[sensitive.to_numpy()] *= sensitive_multiplier
    return weights / weights.mean()


GROUP_FN_MULTIPLIERS = {"crop:rice": 2.0, "growth_stage:Sowing": 2.0,
                        "soil_type:sandy": 2.0}


def group_sensitive_weights(y, groups, multipliers=GROUP_FN_MULTIPLIERS):
    """Apply extra FN cost to positive labels in configurable critical groups."""
    y = np.asarray(y).astype(int)
    groups = pd.DataFrame(groups).reset_index(drop=True)
    weights = np.ones(len(y), dtype=float)
    for key, factor in multipliers.items():
        column, value = key.split(":", 1)
        weights[(y == 1) & (groups[column].astype(str).to_numpy() == value)] *= factor
    return weights / weights.mean()


def metrics(y, pred, threshold, c_fp=1, c_fn=1):
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int) if threshold is None else (np.asarray(pred) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    n = len(y)
    return dict(threshold_mm=threshold, accuracy=accuracy_score(y, pred),
                precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred, zero_division=0),
                f1=f1_score(y, pred, zero_division=0), tn=int(tn), fp=int(fp),
                fn=int(fn), tp=int(tp), false_positive_rate=fp / (fp + tn),
                false_negative_rate=fn / (fn + tp),
                total_decision_cost=decision_cost(fp, fn, c_fp, c_fn),
                average_decision_cost=decision_cost(fp, fn, c_fp, c_fn) / n)


def run():
    data = pd.read_csv(ROOT / "primary/primary_model_ready.csv")
    with (ROOT / "feature_sets.json").open(encoding="utf-8") as stream:
        features = json.load(stream)["proposed7"]
    train = data.loc[data.split == "train"].reset_index().rename(columns={"index": "source_row_index"})
    X, amount, labels = train[features].to_numpy(), train.water_amount_mm.to_numpy(), train.irrigate.to_numpy(int)
    groups = pd.DataFrame({"crop": train.meta_crop.astype(str),
                           "soil_type": train[[c for c in train if c.startswith("soil_type_")]].idxmax(axis=1).str.removeprefix("soil_type_"),
                           "growth_stage": train[[c for c in train if c.startswith("growth_stage_")]].idxmax(axis=1).str.removeprefix("growth_stage_")})
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    records, predictions = [], []
    for fold, (fit, valid) in enumerate(folds.split(X, groups.crop.astype(str) + "_" + pd.Series(labels).astype(str)), 1):
        scaler = StandardScaler().fit(X[fit])
        xf, xv = scaler.transform(X[fit]), scaler.transform(X[valid])
        yv, av = labels[valid], amount[valid]
        models = {"baseline": ExtraTreesRegressor(**TREE_PARAMS, random_state=SEED, n_jobs=1)}
        models["baseline"].fit(xf, amount[fit])
        for name, (cfp, cfn) in COST_RATIOS.items():
            model = ExtraTreesRegressor(**TREE_PARAMS, random_state=SEED, n_jobs=1)
            model.fit(xf, amount[fit], sample_weight=label_weights(labels[fit], cfp, cfn))
            models[f"weighted_regression:{name}"] = model
            clf = ExtraTreesClassifier(**TREE_PARAMS, random_state=SEED, n_jobs=1)
            clf.fit(xf, labels[fit], sample_weight=label_weights(labels[fit], cfp, cfn))
            prob = clf.predict_proba(xv)[:, list(clf.classes_).index(1)]
            predictions.extend(dict(fold=fold, strategy=f"classifier:{name}", idx=int(i),
                                    actual=labels[i], amount=amount[i], predicted_amount=np.nan,
                                    decision=float(p >= .5)) for i, p in zip(valid, prob))
        group_reg = ExtraTreesRegressor(**TREE_PARAMS, random_state=SEED, n_jobs=1)
        group_reg.fit(xf, amount[fit], sample_weight=group_sensitive_weights(labels[fit], groups.iloc[fit]))
        gp = np.maximum(group_reg.predict(xv), 0)
        predictions.extend(dict(fold=fold, strategy="group_fn_weighted_regression", idx=int(i),
                                actual=labels[i], amount=amount[i], predicted_amount=float(p), decision=np.nan)
                           for i, p in zip(valid, gp))
        group_clf = ExtraTreesClassifier(**TREE_PARAMS, random_state=SEED, n_jobs=1)
        group_clf.fit(xf, labels[fit], sample_weight=group_sensitive_weights(labels[fit], groups.iloc[fit]))
        gprob = group_clf.predict_proba(xv)[:, list(group_clf.classes_).index(1)]
        predictions.extend(dict(fold=fold, strategy="group_fn_weighted_classifier", idx=int(i),
                                actual=labels[i], amount=amount[i], predicted_amount=np.nan,
                                decision=float(p >= .5)) for i, p in zip(valid, gprob))
        for strategy, model in models.items():
            pred = np.maximum(model.predict(xv), 0)
            predictions.extend(dict(fold=fold, strategy=strategy, idx=int(i), actual=labels[i],
                                    amount=amount[i], predicted_amount=float(p), decision=np.nan)
                               for i, p in zip(valid, pred))
        # Score validation predictions per fold, retaining fold-level stability evidence.
        fold_rows = []
        for strategy in ["baseline"] + [f"weighted_regression:{k}" for k in COST_RATIOS] + ["group_fn_weighted_regression"]:
            p = np.array([r["predicted_amount"] for r in predictions if r["fold"] == fold and r["strategy"] == strategy])
            cfp, cfn = COST_RATIOS[strategy.split(":")[1]] if strategy.startswith("weighted_regression:") else (1, 1)
            for t in THRESHOLDS:
                row = metrics(yv, p, t, cfp, cfn)
                row.update(fold=fold, strategy=strategy, c_fp=cfp, c_fn=cfn,
                           mae=mean_absolute_error(av, p), rmse=np.sqrt(mean_squared_error(av, p)),
                           r2=r2_score(av, p))
                fold_rows.append(row)
        for name, (cfp, cfn) in COST_RATIOS.items():
            d = np.array([r["decision"] for r in predictions if r["fold"] == fold and r["strategy"] == f"classifier:{name}"])
            row = metrics(yv, d, None, cfp, cfn)
            row.update(fold=fold, strategy=f"classifier:{name}", c_fp=cfp, c_fn=cfn,
                       mae=np.nan, rmse=np.nan, r2=np.nan)
            fold_rows.append(row)
        gd = np.array([r["decision"] for r in predictions if r["fold"] == fold and r["strategy"] == "group_fn_weighted_classifier"])
        row = metrics(yv, gd, None)
        row.update(fold=fold, strategy="group_fn_weighted_classifier", c_fp=1, c_fn=1,
                   mae=np.nan, rmse=np.nan, r2=np.nan)
        fold_rows.append(row)
        records.extend(fold_rows)
    out = ROOT / "results/cost_sensitive"
    out.mkdir(parents=True, exist_ok=True)
    pred_frame = pd.DataFrame(predictions).merge(groups.reset_index().rename(columns={"index":"idx"}), on="idx")
    pred_frame.to_csv(out / "oof_predictions.csv", index=False)
    fold_frame = pd.DataFrame(records)
    fold_frame.to_csv(out / "fold_metrics.csv", index=False, float_format="%.6f")
    # Pooled OOF metrics are the headline; fold metrics remain available for stability.
    pooled = []
    for strategy, sub in pred_frame.groupby("strategy"):
        if strategy.startswith("classifier:"):
            ratio = strategy.split(":")[1]
            cfp, cfn = COST_RATIOS[ratio]
            decision = sub.decision.to_numpy()
            reg = (np.nan, np.nan, np.nan)
            ts = (None,)
        elif strategy == "group_fn_weighted_classifier":
            cfp = cfn = 1
            decision = sub.decision.to_numpy()
            reg = (np.nan, np.nan, np.nan)
            ts = (None,)
        else:
            cfp, cfn = COST_RATIOS[strategy.split(":")[1]] if strategy.startswith("weighted_regression:") else (1, 1)
            decision = sub.predicted_amount.to_numpy()
            reg = (mean_absolute_error(sub.amount, decision), np.sqrt(mean_squared_error(sub.amount, decision)), r2_score(sub.amount, decision))
            ts = THRESHOLDS
        for t in ts:
            row = metrics(sub.actual, decision, t, cfp, cfn)
            row.update(strategy=strategy, c_fp=cfp, c_fn=cfn, mae=reg[0], rmse=reg[1], r2=reg[2])
            pooled.append(row)
    pd.DataFrame(pooled).to_csv(out / "pooled_metrics.csv", index=False, float_format="%.6f")
    subgroup = []
    subgroup_definitions = []
    for col, kind in [("crop", "crop"), ("soil_type", "soil"), ("growth_stage", "growth_stage")]:
        subgroup_definitions.extend((kind, value, pred_frame[col] == value)
                                    for value in sorted(pred_frame[col].dropna().unique()))
    for strategy, sub in pred_frame.groupby("strategy"):
        thresholds = (None,) if "classifier" in strategy else THRESHOLDS
        for kind, value, mask in subgroup_definitions:
            part = sub.loc[mask.loc[sub.index]]
            if part.empty: continue
            cfp, cfn = COST_RATIOS[strategy.split(":")[1]] if strategy.startswith("classifier:") else (1, 1)
            for t in thresholds:
                row = metrics(part.actual, part.decision if t is None else part.predicted_amount, t, cfp, cfn)
                row.update(strategy=strategy, group_type=kind, group_value=value, n=len(part), c_fp=cfp,c_fn=cfn)
                subgroup.append(row)
    pd.DataFrame(subgroup).to_csv(out / "subgroup_metrics.csv", index=False, float_format="%.6f")
    print(f"Saved training-only OOF results to {out}")


if __name__ == "__main__":
    run()
