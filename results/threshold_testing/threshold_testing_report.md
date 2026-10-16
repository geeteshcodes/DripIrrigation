# Irrigation Decision Threshold Testing

## Scope and method

This is a threshold evaluation only. It uses the original ExtraTreesReg configuration (`n_estimators=300`, `max_depth=20`, `min_samples_split=2`), the unchanged `proposed7` features, and only rows marked `split == train`. It does not tune model parameters, alter the feature set, or use the held-out test split.

Row-level five-fold out-of-fold predictions were reproduced because the prior CSV contains aggregate threshold metrics but not the predictions needed for subgroup evaluation. The CV strategy is `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, stratified by crop × the recorded `irrigate` label. Scaling is fit separately on each fold's training rows. Predictions are clipped at zero. Threshold evaluation uses `8,000` out-of-fold training rows.

The recorded target label is preserved: `irrigate = 1` when `water_amount_mm > 0`. A predicted irrigation decision is `1` when the clipped model prediction is at least 2.0 mm for the existing baseline comparison, or at the evaluated threshold. There are 0 records with `0 < water_amount_mm < 2.0`, 0 inconsistencies between the recorded `irrigate` field and `water_amount_mm > 0`, and 3732 positive recorded irrigation labels. Thus the definition gap is explicitly checked and the original label remains the evaluation target.

False positive (FP) means the prediction crosses the selected threshold while recorded `irrigate` is 0. False negative (FN) means recorded `irrigate` is 1 while the prediction remains below threshold. Confusion matrices are `[[TN, FP], [FN, TP]]`.

## Overall comparison

| Threshold (mm) | Accuracy | Precision | Recall | F1 | FP | FN | Confusion matrix [[TN, FP], [FN, TP]] |
|---|---|---|---|---|---|---|---|
| 2.0000 | 0.9669 | 0.9376 | 0.9952 | 0.9656 | 247 | 18 | `[[4021, 247], [18, 3714]]` |
| 2.7000 | 0.9729 | 0.9583 | 0.9847 | 0.9713 | 160 | 57 | `[[4108, 160], [57, 3675]]` |
| 2.8000 | 0.9728 | 0.9602 | 0.9823 | 0.9711 | 152 | 66 | `[[4116, 152], [66, 3666]]` |
| 2.9000 | 0.9725 | 0.9621 | 0.9796 | 0.9708 | 144 | 76 | `[[4124, 144], [76, 3656]]` |
| 3.0000 | 0.9726 | 0.9633 | 0.9786 | 0.9709 | 139 | 80 | `[[4129, 139], [80, 3652]]` |
| 3.1000 | 0.9729 | 0.9653 | 0.9770 | 0.9711 | 131 | 86 | `[[4137, 131], [86, 3646]]` |
| 3.2000 | 0.9725 | 0.9665 | 0.9748 | 0.9707 | 126 | 94 | `[[4142, 126], [94, 3638]]` |
| 3.3000 | 0.9720 | 0.9677 | 0.9724 | 0.9701 | 121 | 103 | `[[4147, 121], [103, 3629]]` |
| 3.4000 | 0.9722 | 0.9700 | 0.9705 | 0.9703 | 112 | 110 | `[[4156, 112], [110, 3622]]` |
| 3.5000 | 0.9731 | 0.9723 | 0.9700 | 0.9712 | 103 | 112 | `[[4165, 103], [112, 3620]]` |
| 4.0000 | 0.9706 | 0.9763 | 0.9603 | 0.9683 | 87 | 148 | `[[4181, 87], [148, 3584]]` |

## Findings

- Highest F1: **2.7 mm** (F1=0.9713, precision=0.9583, recall=0.9847, FP=160, FN=57).
- Highest accuracy: **3.5 mm** (accuracy=0.9731, FP=103, FN=112).
- Closest precision–recall balance by smallest absolute gap: **3.4 mm** (precision=0.9700, recall=0.9705, gap=0.0005).
- Lowest false-positive count among thresholds with recall at least 0.97: **3.4 mm (recall=0.9705, FP=112, FN=110)**.

The highest-F1 threshold is the statistical choice under equal weighting of precision and recall. The operationally safest choice against missed irrigation is **2.0 mm**, since it has the highest recall / fewest missed irrigations among the evaluated thresholds. A higher threshold trades that safety for fewer unnecessary irrigation decisions. No threshold has been applied to the prediction pipeline.

## Crop-wise performance at candidate thresholds

The table includes the current 2.0 mm rule, thresholds around the best overall candidates, and 4.0 mm as a reference. Compare FN counts and recall within each crop; the full crop × threshold sweep is in the subgroup CSV.

| Crop | Threshold (mm) | n | Recall | FN | Precision | FP | F1 |
|---|---|---|---|---|---|---|---|
| cotton | 2.0000 | 1276 | 0.9962 | 2 | 0.9238 | 43 | 0.9586 |
| cotton | 2.7000 | 1276 | 0.9885 | 6 | 0.9400 | 33 | 0.9637 |
| cotton | 3.0000 | 1276 | 0.9866 | 7 | 0.9433 | 31 | 0.9645 |
| cotton | 3.2000 | 1276 | 0.9866 | 7 | 0.9485 | 28 | 0.9672 |
| cotton | 3.4000 | 1276 | 0.9847 | 8 | 0.9502 | 27 | 0.9671 |
| cotton | 3.5000 | 1276 | 0.9847 | 8 | 0.9502 | 27 | 0.9671 |
| cotton | 4.0000 | 1276 | 0.9809 | 10 | 0.9553 | 24 | 0.9679 |
| maize | 2.0000 | 1355 | 0.9959 | 3 | 0.9568 | 33 | 0.9760 |
| maize | 2.7000 | 1355 | 0.9864 | 10 | 0.9705 | 22 | 0.9784 |
| maize | 3.0000 | 1355 | 0.9823 | 13 | 0.9743 | 19 | 0.9783 |
| maize | 3.2000 | 1355 | 0.9796 | 15 | 0.9756 | 18 | 0.9776 |
| maize | 3.4000 | 1355 | 0.9782 | 16 | 0.9782 | 16 | 0.9782 |
| maize | 3.5000 | 1355 | 0.9782 | 16 | 0.9795 | 15 | 0.9789 |
| maize | 4.0000 | 1355 | 0.9728 | 20 | 0.9835 | 12 | 0.9781 |
| potato | 2.0000 | 1330 | 0.9947 | 4 | 0.9365 | 51 | 0.9647 |
| potato | 2.7000 | 1330 | 0.9815 | 14 | 0.9636 | 28 | 0.9725 |
| potato | 3.0000 | 1330 | 0.9749 | 19 | 0.9672 | 25 | 0.9710 |
| potato | 3.2000 | 1330 | 0.9696 | 23 | 0.9747 | 19 | 0.9721 |
| potato | 3.4000 | 1330 | 0.9630 | 28 | 0.9798 | 15 | 0.9713 |
| potato | 3.5000 | 1330 | 0.9616 | 29 | 0.9864 | 10 | 0.9739 |
| potato | 4.0000 | 1330 | 0.9431 | 43 | 0.9930 | 5 | 0.9674 |
| rice | 2.0000 | 1369 | 0.9833 | 8 | 0.9308 | 35 | 0.9563 |
| rice | 2.7000 | 1369 | 0.9520 | 23 | 0.9723 | 13 | 0.9620 |
| rice | 3.0000 | 1369 | 0.9332 | 32 | 0.9824 | 8 | 0.9572 |
| rice | 3.2000 | 1369 | 0.9207 | 38 | 0.9822 | 8 | 0.9504 |
| rice | 3.4000 | 1369 | 0.9061 | 45 | 0.9819 | 8 | 0.9425 |
| rice | 3.5000 | 1369 | 0.9040 | 46 | 0.9841 | 7 | 0.9423 |
| rice | 4.0000 | 1369 | 0.8789 | 58 | 0.9906 | 4 | 0.9314 |
| sugarcane | 2.0000 | 1343 | 0.9981 | 1 | 0.9094 | 53 | 0.9517 |
| sugarcane | 2.7000 | 1343 | 0.9981 | 1 | 0.9268 | 42 | 0.9612 |
| sugarcane | 3.0000 | 1343 | 0.9962 | 2 | 0.9349 | 37 | 0.9646 |
| sugarcane | 3.2000 | 1343 | 0.9925 | 4 | 0.9379 | 35 | 0.9644 |
| sugarcane | 3.4000 | 1343 | 0.9906 | 5 | 0.9462 | 30 | 0.9679 |
| sugarcane | 3.5000 | 1343 | 0.9906 | 5 | 0.9462 | 30 | 0.9679 |
| sugarcane | 4.0000 | 1343 | 0.9869 | 7 | 0.9495 | 28 | 0.9678 |
| wheat | 2.0000 | 1327 | 1.0000 | 0 | 0.9567 | 32 | 0.9779 |
| wheat | 2.7000 | 1327 | 0.9958 | 3 | 0.9697 | 22 | 0.9826 |
| wheat | 3.0000 | 1327 | 0.9901 | 7 | 0.9736 | 19 | 0.9818 |
| wheat | 3.2000 | 1327 | 0.9901 | 7 | 0.9749 | 18 | 0.9825 |
| wheat | 3.4000 | 1327 | 0.9887 | 8 | 0.9776 | 16 | 0.9831 |
| wheat | 3.5000 | 1327 | 0.9887 | 8 | 0.9804 | 14 | 0.9845 |
| wheat | 4.0000 | 1327 | 0.9859 | 10 | 0.9803 | 14 | 0.9831 |

## Largest observed subgroup recall drops at 3.5 mm

Compared with the current 2 mm threshold, the largest recall drops by group type were:

| Group | Subgroup | n | Recall at 2 mm | Recall at 3.5 mm | Change (percentage points) | FN at 2 mm | FN at 3.5 mm |
|---|---|---:|---:|---:|---:|---:|---:|
| Crop | Rice | 1,369 | 0.9833 | 0.9040 | -7.93 | 8 | 46 |
| Soil type | Sandy | 2,055 | 0.9910 | 0.9490 | -4.20 | 6 | 34 |
| Growth stage | Sowing | 1,939 | 0.9804 | 0.8982 | -8.22 | 10 | 52 |
| Edge status | Any edge case | 4,449 | 0.9945 | 0.9640 | -3.05 | 14 | 92 |
| Edge flag | Saturated soil | 723 | 1.0000 | 0.8871 | -11.29 | 0 | 7 |

The `edge_label_conflict` group also has a notable increase in missed cases: 7 at 2 mm versus 70 at 3.5 mm (recall 0.9958 to 0.9579). These cross-validation subgroup counts are descriptive, not a separate inferential significance test.

## Soil, growth-stage, and edge-case performance

Edge-category groups overlap because a row can carry multiple edge flags. The subgroup CSV contains every requested threshold for each soil, stage, edge status, and edge category.

| Group type | Group | Threshold (mm) | n | Recall | FN | Precision | FP | F1 |
|---|---|---|---|---|---|---|---|---|
| edge_category_overlap | edge_extreme_vpd | 2.0000 | 404 | 1.0000 | 0 | 0.9293 | 14 | 0.9634 |
| edge_category_overlap | edge_extreme_vpd | 2.7000 | 404 | 0.9837 | 3 | 0.9577 | 8 | 0.9705 |
| edge_category_overlap | edge_extreme_vpd | 3.0000 | 404 | 0.9674 | 6 | 0.9622 | 7 | 0.9648 |
| edge_category_overlap | edge_extreme_vpd | 3.2000 | 404 | 0.9620 | 7 | 0.9620 | 7 | 0.9620 |
| edge_category_overlap | edge_extreme_vpd | 3.5000 | 404 | 0.9620 | 7 | 0.9620 | 7 | 0.9620 |
| edge_category_overlap | edge_extreme_vpd | 4.0000 | 404 | 0.9457 | 10 | 0.9667 | 6 | 0.9560 |
| edge_category_overlap | edge_heavy_rain | 2.0000 | 593 | 1.0000 | 0 | 0.8995 | 20 | 0.9471 |
| edge_category_overlap | edge_heavy_rain | 2.7000 | 593 | 1.0000 | 0 | 0.9323 | 13 | 0.9650 |
| edge_category_overlap | edge_heavy_rain | 3.0000 | 593 | 1.0000 | 0 | 0.9323 | 13 | 0.9650 |
| edge_category_overlap | edge_heavy_rain | 3.2000 | 593 | 0.9944 | 1 | 0.9418 | 11 | 0.9674 |
| edge_category_overlap | edge_heavy_rain | 3.5000 | 593 | 0.9888 | 2 | 0.9516 | 9 | 0.9699 |
| edge_category_overlap | edge_heavy_rain | 4.0000 | 593 | 0.9832 | 3 | 0.9617 | 7 | 0.9724 |
| edge_category_overlap | edge_high_wind | 2.0000 | 810 | 0.9918 | 3 | 0.9333 | 26 | 0.9617 |
| edge_category_overlap | edge_high_wind | 2.7000 | 810 | 0.9782 | 8 | 0.9573 | 16 | 0.9677 |
| edge_category_overlap | edge_high_wind | 3.0000 | 810 | 0.9700 | 11 | 0.9648 | 13 | 0.9674 |
| edge_category_overlap | edge_high_wind | 3.2000 | 810 | 0.9564 | 16 | 0.9669 | 12 | 0.9616 |
| edge_category_overlap | edge_high_wind | 3.5000 | 810 | 0.9537 | 17 | 0.9777 | 8 | 0.9655 |
| edge_category_overlap | edge_high_wind | 4.0000 | 810 | 0.9482 | 19 | 0.9831 | 6 | 0.9653 |
| edge_category_overlap | edge_label_conflict | 2.0000 | 1685 | 0.9958 | 7 | 0.9994 | 1 | 0.9976 |
| edge_category_overlap | edge_label_conflict | 2.7000 | 1685 | 0.9808 | 32 | 1.0000 | 0 | 0.9903 |
| edge_category_overlap | edge_label_conflict | 3.0000 | 1685 | 0.9717 | 47 | 1.0000 | 0 | 0.9857 |
| edge_category_overlap | edge_label_conflict | 3.2000 | 1685 | 0.9651 | 58 | 1.0000 | 0 | 0.9823 |
| edge_category_overlap | edge_label_conflict | 3.5000 | 1685 | 0.9579 | 70 | 1.0000 | 0 | 0.9785 |
| edge_category_overlap | edge_label_conflict | 4.0000 | 1685 | 0.9441 | 93 | 1.0000 | 0 | 0.9712 |
| edge_category_overlap | edge_low_temp | 2.0000 | 791 | 0.9894 | 4 | 0.9375 | 25 | 0.9628 |
| edge_category_overlap | edge_low_temp | 2.7000 | 791 | 0.9815 | 7 | 0.9588 | 16 | 0.9700 |
| edge_category_overlap | edge_low_temp | 3.0000 | 791 | 0.9789 | 8 | 0.9636 | 14 | 0.9712 |
| edge_category_overlap | edge_low_temp | 3.2000 | 791 | 0.9736 | 10 | 0.9660 | 13 | 0.9698 |
| edge_category_overlap | edge_low_temp | 3.5000 | 791 | 0.9683 | 12 | 0.9787 | 8 | 0.9735 |
| edge_category_overlap | edge_low_temp | 4.0000 | 791 | 0.9578 | 16 | 0.9784 | 8 | 0.9680 |
| edge_category_overlap | edge_near_zero_rain | 2.0000 | 138 | 1.0000 | 0 | 0.9592 | 4 | 0.9792 |
| edge_category_overlap | edge_near_zero_rain | 2.7000 | 138 | 1.0000 | 0 | 0.9691 | 3 | 0.9843 |
| edge_category_overlap | edge_near_zero_rain | 3.0000 | 138 | 1.0000 | 0 | 0.9691 | 3 | 0.9843 |
| edge_category_overlap | edge_near_zero_rain | 3.2000 | 138 | 1.0000 | 0 | 0.9691 | 3 | 0.9843 |
| edge_category_overlap | edge_near_zero_rain | 3.5000 | 138 | 1.0000 | 0 | 0.9691 | 3 | 0.9843 |
| edge_category_overlap | edge_near_zero_rain | 4.0000 | 138 | 1.0000 | 0 | 0.9691 | 3 | 0.9843 |
| edge_category_overlap | edge_saturated_soil | 2.0000 | 723 | 1.0000 | 0 | 0.8267 | 13 | 0.9051 |
| edge_category_overlap | edge_saturated_soil | 2.7000 | 723 | 0.9194 | 5 | 0.9194 | 5 | 0.9194 |
| edge_category_overlap | edge_saturated_soil | 3.0000 | 723 | 0.9032 | 6 | 0.9492 | 3 | 0.9256 |
| edge_category_overlap | edge_saturated_soil | 3.2000 | 723 | 0.9032 | 6 | 0.9655 | 2 | 0.9333 |
| edge_category_overlap | edge_saturated_soil | 3.5000 | 723 | 0.8871 | 7 | 0.9821 | 1 | 0.9322 |
| edge_category_overlap | edge_saturated_soil | 4.0000 | 723 | 0.8548 | 9 | 0.9815 | 1 | 0.9138 |
| edge_category_overlap | edge_very_dry_soil | 2.0000 | 558 | 0.9911 | 4 | 0.9889 | 5 | 0.9900 |
| edge_category_overlap | edge_very_dry_soil | 2.7000 | 558 | 0.9799 | 9 | 0.9910 | 4 | 0.9854 |
| edge_category_overlap | edge_very_dry_soil | 3.0000 | 558 | 0.9732 | 12 | 0.9932 | 3 | 0.9831 |
| edge_category_overlap | edge_very_dry_soil | 3.2000 | 558 | 0.9643 | 16 | 0.9931 | 3 | 0.9785 |
| edge_category_overlap | edge_very_dry_soil | 3.5000 | 558 | 0.9576 | 19 | 0.9954 | 2 | 0.9761 |
| edge_category_overlap | edge_very_dry_soil | 4.0000 | 558 | 0.9487 | 23 | 0.9953 | 2 | 0.9714 |
| edge_status | any_edge_case | 2.0000 | 4449 | 0.9945 | 14 | 0.9658 | 90 | 0.9800 |
| edge_status | any_edge_case | 2.7000 | 4449 | 0.9828 | 44 | 0.9794 | 53 | 0.9811 |
| edge_status | any_edge_case | 3.0000 | 4449 | 0.9750 | 64 | 0.9819 | 46 | 0.9784 |
| edge_status | any_edge_case | 3.2000 | 4449 | 0.9699 | 77 | 0.9837 | 41 | 0.9768 |
| edge_status | any_edge_case | 3.5000 | 4449 | 0.9640 | 92 | 0.9876 | 31 | 0.9757 |
| edge_status | any_edge_case | 4.0000 | 4449 | 0.9527 | 121 | 0.9890 | 27 | 0.9705 |
| edge_status | no_edge_case | 2.0000 | 3551 | 0.9966 | 4 | 0.8817 | 157 | 0.9356 |
| edge_status | no_edge_case | 2.7000 | 3551 | 0.9889 | 13 | 0.9156 | 107 | 0.9509 |
| edge_status | no_edge_case | 3.0000 | 3551 | 0.9864 | 16 | 0.9257 | 93 | 0.9551 |
| edge_status | no_edge_case | 3.2000 | 3551 | 0.9855 | 17 | 0.9316 | 85 | 0.9578 |
| edge_status | no_edge_case | 3.5000 | 3551 | 0.9830 | 20 | 0.9413 | 72 | 0.9617 |
| edge_status | no_edge_case | 4.0000 | 3551 | 0.9770 | 27 | 0.9503 | 60 | 0.9635 |
| growth_stage | Flowering | 2.0000 | 1978 | 0.9964 | 4 | 0.9508 | 57 | 0.9731 |
| growth_stage | Flowering | 2.7000 | 1978 | 0.9919 | 9 | 0.9648 | 40 | 0.9782 |
| growth_stage | Flowering | 3.0000 | 1978 | 0.9901 | 11 | 0.9725 | 31 | 0.9812 |
| growth_stage | Flowering | 3.2000 | 1978 | 0.9892 | 12 | 0.9742 | 29 | 0.9816 |
| growth_stage | Flowering | 3.5000 | 1978 | 0.9819 | 20 | 0.9793 | 23 | 0.9806 |
| growth_stage | Flowering | 4.0000 | 1978 | 0.9801 | 22 | 0.9819 | 20 | 0.9810 |
| growth_stage | Harvest | 2.0000 | 2067 | 0.9991 | 1 | 0.9563 | 53 | 0.9773 |
| growth_stage | Harvest | 2.7000 | 2067 | 0.9905 | 11 | 0.9721 | 33 | 0.9812 |
| growth_stage | Harvest | 3.0000 | 2067 | 0.9862 | 16 | 0.9736 | 31 | 0.9799 |
| growth_stage | Harvest | 3.2000 | 2067 | 0.9845 | 18 | 0.9753 | 29 | 0.9799 |
| growth_stage | Harvest | 3.5000 | 2067 | 0.9828 | 20 | 0.9794 | 24 | 0.9811 |
| growth_stage | Harvest | 4.0000 | 2067 | 0.9733 | 31 | 0.9835 | 19 | 0.9784 |
| growth_stage | Sowing | 2.0000 | 1939 | 0.9804 | 10 | 0.8995 | 56 | 0.9382 |
| growth_stage | Sowing | 2.7000 | 1939 | 0.9491 | 26 | 0.9454 | 28 | 0.9473 |
| growth_stage | Sowing | 3.0000 | 1939 | 0.9276 | 37 | 0.9556 | 22 | 0.9414 |
| growth_stage | Sowing | 3.2000 | 1939 | 0.9100 | 46 | 0.9607 | 19 | 0.9347 |
| growth_stage | Sowing | 3.5000 | 1939 | 0.8982 | 52 | 0.9684 | 15 | 0.9320 |
| growth_stage | Sowing | 4.0000 | 1939 | 0.8650 | 69 | 0.9693 | 14 | 0.9142 |
| growth_stage | Vegetative | 2.0000 | 2016 | 0.9969 | 3 | 0.9215 | 81 | 0.9577 |
| growth_stage | Vegetative | 2.7000 | 2016 | 0.9885 | 11 | 0.9411 | 59 | 0.9642 |
| growth_stage | Vegetative | 3.0000 | 2016 | 0.9832 | 16 | 0.9446 | 55 | 0.9635 |
| growth_stage | Vegetative | 3.2000 | 2016 | 0.9811 | 18 | 0.9503 | 49 | 0.9654 |
| growth_stage | Vegetative | 3.5000 | 2016 | 0.9790 | 20 | 0.9579 | 41 | 0.9684 |
| growth_stage | Vegetative | 4.0000 | 2016 | 0.9727 | 26 | 0.9647 | 34 | 0.9687 |
| soil_type | clay | 2.0000 | 1957 | 0.9970 | 3 | 0.9552 | 46 | 0.9756 |
| soil_type | clay | 2.7000 | 1957 | 0.9919 | 8 | 0.9692 | 31 | 0.9804 |
| soil_type | clay | 3.0000 | 1957 | 0.9848 | 15 | 0.9709 | 29 | 0.9778 |
| soil_type | clay | 3.2000 | 1957 | 0.9776 | 22 | 0.9747 | 25 | 0.9762 |
| soil_type | clay | 3.5000 | 1957 | 0.9746 | 25 | 0.9786 | 21 | 0.9766 |
| soil_type | clay | 4.0000 | 1957 | 0.9644 | 35 | 0.9804 | 19 | 0.9723 |
| soil_type | loamy | 2.0000 | 1970 | 0.9970 | 3 | 0.9483 | 54 | 0.9720 |
| soil_type | loamy | 2.7000 | 1970 | 0.9940 | 6 | 0.9639 | 37 | 0.9787 |
| soil_type | loamy | 3.0000 | 1970 | 0.9919 | 8 | 0.9676 | 33 | 0.9796 |
| soil_type | loamy | 3.2000 | 1970 | 0.9879 | 12 | 0.9684 | 32 | 0.9781 |
| soil_type | loamy | 3.5000 | 1970 | 0.9869 | 13 | 0.9780 | 22 | 0.9825 |
| soil_type | loamy | 4.0000 | 1970 | 0.9778 | 22 | 0.9808 | 19 | 0.9793 |
| soil_type | sandy | 2.0000 | 2055 | 0.9910 | 6 | 0.8825 | 88 | 0.9336 |
| soil_type | sandy | 2.7000 | 2055 | 0.9745 | 17 | 0.9272 | 51 | 0.9503 |
| soil_type | sandy | 3.0000 | 2055 | 0.9655 | 23 | 0.9374 | 43 | 0.9513 |
| soil_type | sandy | 3.2000 | 2055 | 0.9625 | 25 | 0.9455 | 37 | 0.9539 |
| soil_type | sandy | 3.5000 | 2055 | 0.9490 | 34 | 0.9519 | 32 | 0.9505 |
| soil_type | sandy | 4.0000 | 2055 | 0.9340 | 44 | 0.9629 | 24 | 0.9482 |
| soil_type | silt | 2.0000 | 2018 | 0.9945 | 6 | 0.9483 | 59 | 0.9708 |
| soil_type | silt | 2.7000 | 2018 | 0.9761 | 26 | 0.9628 | 41 | 0.9694 |
| soil_type | silt | 3.0000 | 2018 | 0.9688 | 34 | 0.9688 | 34 | 0.9688 |
| soil_type | silt | 3.2000 | 2018 | 0.9678 | 35 | 0.9705 | 32 | 0.9692 |
| soil_type | silt | 3.5000 | 2018 | 0.9632 | 40 | 0.9740 | 28 | 0.9686 |
| soil_type | silt | 4.0000 | 2018 | 0.9568 | 47 | 0.9765 | 25 | 0.9666 |

## Output files

- Overall threshold metrics and confusion matrices: `results/threshold_testing/threshold_comparison.csv`
- Crop, soil, stage, edge-status, and edge-category metrics: `results/threshold_testing/subgroup_metrics.csv`
- Reproduced row-level out-of-fold predictions: `results/threshold_testing/out_of_fold_predictions.csv`
- This report: `results/threshold_testing/threshold_testing_report.md`
