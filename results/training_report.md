# 📊 Multi-Model Training Results Report

**Generated**: 2026-10-12 01:52:50

**Dataset**: `primary_model_ready.csv` (50,000 rows: train=40,000, test=10,000, edge-case test=1,127)

---

## 1. Regression Results (`water_amount_mm`)

> 🏆 **Best Overall Regressor**: **ExtraTreesReg** on `proposed7_plus_physics` — **MAE = 0.7145**, RMSE = 1.6075, **R² = 0.9964**


### Feature Set: `baseline4`

| Rank | Model | MAE | RMSE | R² | Irrigate Acc | Time (s) |
|------|-------|-----|------|----|--------------|----------|
| 1 | WideDeep_Regressor | 15.0208 | 22.2594 | 0.3091 | 0.7495 | 25.47 |
| 2 | DNN_Regressor | 15.2694 | 23.0318 | 0.2603 | 0.7405 | 23.35 |
| 3 | ExtraTreesReg | 15.5462 | 21.4839 | 0.3564 | 0.6720 | 4.3 |
| 4 | HistGradientBoostingReg | 15.5646 | 21.4512 | 0.3584 | 0.6705 | 4.74 |
| 5 | RandomForestReg | 15.5751 | 21.6710 | 0.3452 | 0.6905 | 14.66 |
| 6 | DecisionTreeReg | 15.5823 | 21.7867 | 0.3382 | 0.6950 | 0.21 |
| 7 | KNNReg | 15.6354 | 21.7978 | 0.3375 | 0.6510 | 0.07 |
| 8 | LinearSVR | 15.7883 | 23.6278 | 0.2216 | 0.6415 | 0.06 |
| 9 | AdaBoostReg | 15.8442 | 21.6666 | 0.3454 | 0.6995 | 2.15 |
| 10 | LinearRegression | 16.5043 | 21.9348 | 0.3291 | 0.6130 | 0.0 |
| 11 | Ridge | 16.5328 | 21.9464 | 0.3284 | 0.6085 | 3.47 |
| 12 | ElasticNet | 16.6306 | 21.9899 | 0.3257 | 0.5970 | 0.07 |
| 13 | Lasso | 16.7249 | 22.0525 | 0.3219 | 0.5920 | 0.03 |


### Feature Set: `proposed7`

| Rank | Model | MAE | RMSE | R² | Irrigate Acc | Time (s) |
|------|-------|-----|------|----|--------------|----------|
| 1 | ExtraTreesReg | 0.7456 | 1.8334 | 0.9953 | 0.9710 | 9.86 |
| 2 | RandomForestReg | 1.1465 | 2.2427 | 0.9930 | 0.9665 | 20.21 |
| 3 | DNN_Regressor | 1.1772 | 3.0599 | 0.9869 | 0.9675 | 57.19 |
| 4 | WideDeep_Regressor | 1.1970 | 3.1731 | 0.9860 | 0.9710 | 50.39 |
| 5 | DecisionTreeReg | 1.3808 | 2.9973 | 0.9875 | 0.9585 | 0.25 |
| 6 | HistGradientBoostingReg | 1.7930 | 2.8238 | 0.9889 | 0.9000 | 3.13 |
| 7 | KNNReg | 4.7548 | 10.7111 | 0.8400 | 0.8715 | 0.54 |
| 8 | AdaBoostReg | 9.8660 | 13.6233 | 0.7412 | 0.7255 | 4.54 |
| 9 | LinearSVR | 12.1290 | 16.9283 | 0.6004 | 0.6635 | 0.34 |
| 10 | LinearRegression | 12.1670 | 16.0796 | 0.6395 | 0.6410 | 0.01 |
| 11 | Ridge | 12.2251 | 16.1364 | 0.6369 | 0.6390 | 0.21 |
| 12 | Lasso | 12.2424 | 16.1448 | 0.6366 | 0.6410 | 0.38 |
| 13 | ElasticNet | 12.4357 | 16.3479 | 0.6274 | 0.6325 | 0.26 |


### Feature Set: `proposed7_plus_physics`

| Rank | Model | MAE | RMSE | R² | Irrigate Acc | Time (s) |
|------|-------|-----|------|----|--------------|----------|
| 1 | ExtraTreesReg | 0.7145 | 1.6075 | 0.9964 | 0.9710 | 16.49 |
| 2 | RandomForestReg | 0.7789 | 1.6194 | 0.9963 | 0.9795 | 45.2 |
| 3 | HistGradientBoostingReg | 0.8967 | 1.6163 | 0.9964 | 0.9650 | 4.66 |
| 4 | DNN_Regressor | 1.0699 | 3.5342 | 0.9826 | 0.9675 | 55.44 |
| 5 | DecisionTreeReg | 1.2026 | 2.6167 | 0.9905 | 0.9715 | 0.42 |
| 6 | WideDeep_Regressor | 1.2353 | 3.0756 | 0.9868 | 0.9545 | 55.03 |
| 7 | KNNReg | 5.8685 | 12.1485 | 0.7942 | 0.8385 | 0.54 |
| 8 | AdaBoostReg | 6.9900 | 10.1687 | 0.8558 | 0.5775 | 8.62 |
| 9 | LinearRegression | 11.8116 | 15.8987 | 0.6475 | 0.6590 | 0.01 |
| 10 | LinearSVR | 11.8227 | 17.5286 | 0.5716 | 0.6605 | 0.61 |
| 11 | Ridge | 11.8497 | 15.9175 | 0.6467 | 0.6560 | 0.19 |
| 12 | Lasso | 11.9001 | 15.9643 | 0.6446 | 0.6560 | 0.41 |
| 13 | ElasticNet | 11.9441 | 15.9865 | 0.6436 | 0.6565 | 0.36 |


### Feature Set: `extended_raw`

| Rank | Model | MAE | RMSE | R² | Irrigate Acc | Time (s) |
|------|-------|-----|------|----|--------------|----------|
| 1 | ExtraTreesReg | 0.9821 | 2.2223 | 0.9931 | 0.9570 | 15.34 |
| 2 | DNN_Regressor | 1.0545 | 3.0079 | 0.9874 | 0.9700 | 81.18 |
| 3 | WideDeep_Regressor | 1.2587 | 3.3760 | 0.9841 | 0.9645 | 55.1 |
| 4 | RandomForestReg | 1.3290 | 2.5289 | 0.9911 | 0.9510 | 32.34 |
| 5 | DecisionTreeReg | 1.6419 | 3.4153 | 0.9837 | 0.9520 | 0.37 |
| 6 | HistGradientBoostingReg | 1.8621 | 2.9239 | 0.9881 | 0.8975 | 4.41 |
| 7 | AdaBoostReg | 9.8660 | 13.6233 | 0.7412 | 0.7255 | 6.24 |
| 8 | KNNReg | 11.3608 | 18.1683 | 0.5397 | 0.6930 | 0.47 |
| 9 | LinearSVR | 12.1262 | 16.9274 | 0.6005 | 0.6610 | 0.47 |
| 10 | LinearRegression | 12.1803 | 16.0935 | 0.6389 | 0.6415 | 0.01 |
| 11 | Ridge | 12.2383 | 16.1499 | 0.6363 | 0.6410 | 0.21 |
| 12 | Lasso | 12.2461 | 16.1472 | 0.6364 | 0.6400 | 0.49 |
| 13 | ElasticNet | 12.4429 | 16.3540 | 0.6271 | 0.6340 | 0.35 |


## 2. Classification Results (`need_ord`: Low / Medium / High)

> 🏆 **Best Overall Classifier**: **HistGradientBoostingClf** on `extended_raw` — **Macro-F1 = 0.9877**, **Accuracy = 0.9960**


### Feature Set: `baseline4`

| Rank | Model | Macro-F1 | Accuracy | Precision | Recall | Time (s) |
|------|-------|----------|----------|-----------|--------|----------|
| 1 | HistGradientBoostingClf | 0.6176 | 0.7370 | 0.6966 | 0.5838 | 2.95 |
| 2 | AdaBoostClf | 0.6116 | 0.7380 | 0.7125 | 0.5757 | 2.85 |
| 3 | DecisionTreeClf | 0.6063 | 0.7325 | 0.7086 | 0.5700 | 0.13 |
| 4 | RandomForestClf | 0.6022 | 0.7310 | 0.6700 | 0.5719 | 9.58 |
| 5 | ExtraTreesClf | 0.5887 | 0.7150 | 0.6915 | 0.5537 | 4.27 |
| 6 | KNNClf | 0.5687 | 0.6760 | 0.6136 | 0.5441 | 0.15 |
| 7 | DNN_Classifier | 0.5498 | 0.7220 | 0.6665 | 0.5244 | 44.14 |
| 8 | LogisticRegression | 0.4807 | 0.6855 | 0.5918 | 0.4710 | 0.09 |
| 9 | LinearSVC | 0.4498 | 0.6935 | 0.4490 | 0.4558 | 0.04 |


### Feature Set: `proposed7`

| Rank | Model | Macro-F1 | Accuracy | Precision | Recall | Time (s) |
|------|-------|----------|----------|-----------|--------|----------|
| 1 | DecisionTreeClf | 0.7501 | 0.8505 | 0.8442 | 0.7066 | 0.28 |
| 2 | AdaBoostClf | 0.7484 | 0.8520 | 0.8523 | 0.7036 | 4.31 |
| 3 | HistGradientBoostingClf | 0.7360 | 0.8445 | 0.8152 | 0.6970 | 6.83 |
| 4 | RandomForestClf | 0.7147 | 0.8350 | 0.8241 | 0.6725 | 7.28 |
| 5 | DNN_Classifier | 0.6900 | 0.8150 | 0.7385 | 0.6628 | 45.71 |
| 6 | LogisticRegression | 0.5957 | 0.7620 | 0.6994 | 0.5692 | 0.42 |
| 7 | ExtraTreesClf | 0.5562 | 0.7840 | 0.6484 | 0.5488 | 4.9 |
| 8 | KNNClf | 0.5151 | 0.7155 | 0.5435 | 0.5064 | 0.44 |
| 9 | LinearSVC | 0.4994 | 0.7540 | 0.4926 | 0.5068 | 0.23 |


### Feature Set: `proposed7_plus_physics`

| Rank | Model | Macro-F1 | Accuracy | Precision | Recall | Time (s) |
|------|-------|----------|----------|-----------|--------|----------|
| 1 | DecisionTreeClf | 0.7503 | 0.8390 | 0.7628 | 0.7395 | 0.39 |
| 2 | AdaBoostClf | 0.7490 | 0.8440 | 0.8472 | 0.7071 | 6.43 |
| 3 | HistGradientBoostingClf | 0.7440 | 0.8505 | 0.7914 | 0.7143 | 8.32 |
| 4 | RandomForestClf | 0.7125 | 0.8390 | 0.7770 | 0.6802 | 8.55 |
| 5 | DNN_Classifier | 0.6978 | 0.8230 | 0.7489 | 0.6705 | 27.45 |
| 6 | LogisticRegression | 0.6332 | 0.7835 | 0.6932 | 0.6072 | 0.85 |
| 7 | ExtraTreesClf | 0.6310 | 0.8170 | 0.6936 | 0.6110 | 6.04 |
| 8 | KNNClf | 0.5438 | 0.7295 | 0.5613 | 0.5341 | 0.52 |
| 9 | LinearSVC | 0.5140 | 0.7750 | 0.5076 | 0.5214 | 0.67 |


### Feature Set: `extended_raw`

| Rank | Model | Macro-F1 | Accuracy | Precision | Recall | Time (s) |
|------|-------|----------|----------|-----------|--------|----------|
| 1 | HistGradientBoostingClf | 0.9877 | 0.9960 | 0.9920 | 0.9836 | 9.24 |
| 2 | DecisionTreeClf | 0.9668 | 0.9930 | 0.9708 | 0.9629 | 0.37 |
| 3 | RandomForestClf | 0.9371 | 0.9885 | 0.9906 | 0.9006 | 9.41 |
| 4 | DNN_Classifier | 0.9075 | 0.9550 | 0.9431 | 0.8803 | 61.24 |
| 5 | AdaBoostClf | 0.7888 | 0.9740 | 0.9786 | 0.7497 | 5.62 |
| 6 | LogisticRegression | 0.7488 | 0.8375 | 0.8010 | 0.7172 | 0.5 |
| 7 | ExtraTreesClf | 0.6960 | 0.8905 | 0.9222 | 0.6623 | 8.21 |
| 8 | LinearSVC | 0.5567 | 0.8320 | 0.5473 | 0.5663 | 0.48 |
| 9 | KNNClf | 0.5206 | 0.7230 | 0.5577 | 0.5104 | 0.79 |


## 3. Edge-Case Performance (Stress, Drought, Heavy Rain)

Edge-case test rows: 1,127


### Top Regressors on Edge Cases

| Rank | Model | Feature Set | MAE | RMSE | R² | Irrigate Acc |
|------|-------|-------------|-----|------|----|--------------|
| 1 | ExtraTreesReg | proposed7_plus_physics | 0.7606 | 1.6614 | 0.9964 | 0.9858 |
| 2 | ExtraTreesReg | proposed7 | 0.7741 | 1.8440 | 0.9955 | 0.9867 |
| 3 | RandomForestReg | proposed7_plus_physics | 0.9478 | 1.7760 | 0.9958 | 0.9831 |
| 4 | HistGradientBoostingReg | proposed7_plus_physics | 1.0083 | 1.7645 | 0.9959 | 0.9760 |
| 5 | ExtraTreesReg | extended_raw | 1.0231 | 2.3271 | 0.9928 | 0.9778 |
| 6 | DNN_Regressor | extended_raw | 1.1055 | 2.8067 | 0.9896 | 0.9831 |
| 7 | DNN_Regressor | proposed7_plus_physics | 1.1501 | 3.4538 | 0.9842 | 0.9752 |
| 8 | WideDeep_Regressor | proposed7_plus_physics | 1.2454 | 2.8373 | 0.9894 | 0.9725 |
| 9 | DNN_Regressor | proposed7 | 1.2519 | 2.9551 | 0.9885 | 0.9760 |
| 10 | RandomForestReg | proposed7 | 1.2947 | 2.3546 | 0.9927 | 0.9760 |
| 11 | WideDeep_Regressor | proposed7 | 1.3101 | 3.1535 | 0.9869 | 0.9823 |
| 12 | WideDeep_Regressor | extended_raw | 1.3349 | 3.2552 | 0.9860 | 0.9796 |
| 13 | DecisionTreeReg | proposed7_plus_physics | 1.4368 | 2.7836 | 0.9898 | 0.9725 |
| 14 | RandomForestReg | extended_raw | 1.4793 | 2.6114 | 0.9910 | 0.9618 |
| 15 | DecisionTreeReg | proposed7 | 1.6246 | 3.1739 | 0.9867 | 0.9512 |


### Top Classifiers on Edge Cases

| Rank | Model | Feature Set | Macro-F1 | Accuracy | Precision | Recall |
|------|-------|-------------|----------|----------|-----------|--------|
| 1 | HistGradientBoostingClf | extended_raw | 0.9812 | 0.9938 | 0.9849 | 0.9777 |
| 2 | DecisionTreeClf | extended_raw | 0.9607 | 0.9938 | 0.9831 | 0.9419 |
| 3 | DNN_Classifier | extended_raw | 0.9249 | 0.9610 | 0.9442 | 0.9094 |
| 4 | RandomForestClf | extended_raw | 0.9092 | 0.9867 | 0.9863 | 0.8657 |
| 5 | AdaBoostClf | extended_raw | 0.7900 | 0.9760 | 0.9754 | 0.7519 |
| 6 | AdaBoostClf | proposed7 | 0.7446 | 0.8563 | 0.8625 | 0.6981 |
| 7 | HistGradientBoostingClf | proposed7 | 0.7418 | 0.8509 | 0.8262 | 0.7008 |
| 8 | DecisionTreeClf | proposed7 | 0.7402 | 0.8554 | 0.8429 | 0.6966 |
| 9 | AdaBoostClf | proposed7_plus_physics | 0.7369 | 0.8412 | 0.8506 | 0.6982 |
| 10 | HistGradientBoostingClf | proposed7_plus_physics | 0.7339 | 0.8527 | 0.7890 | 0.7027 |
| 11 | DecisionTreeClf | proposed7_plus_physics | 0.7321 | 0.8367 | 0.7389 | 0.7262 |
| 12 | ExtraTreesClf | extended_raw | 0.7256 | 0.9095 | 0.9252 | 0.6880 |
| 13 | LogisticRegression | extended_raw | 0.7204 | 0.8190 | 0.7519 | 0.7014 |
| 14 | RandomForestClf | proposed7 | 0.7177 | 0.8385 | 0.8269 | 0.6748 |
| 15 | RandomForestClf | proposed7_plus_physics | 0.6952 | 0.8394 | 0.7511 | 0.6683 |


## 4. Best Hyperparameters

| Model Tag | Best Tuned Parameters |
|-----------|-----------------------|
| AdaBoostClf__baseline4 | `learning_rate`: 1.0, `n_estimators`: 200 |
| AdaBoostClf__extended_raw | `learning_rate`: 1.0, `n_estimators`: 200 |
| AdaBoostClf__proposed7 | `learning_rate`: 1.0, `n_estimators`: 200 |
| AdaBoostClf__proposed7_plus_physics | `learning_rate`: 1.0, `n_estimators`: 100 |
| AdaBoostReg__baseline4 | `learning_rate`: 0.01, `n_estimators`: 50 |
| AdaBoostReg__extended_raw | `learning_rate`: 0.1, `n_estimators`: 200 |
| AdaBoostReg__proposed7 | `learning_rate`: 0.1, `n_estimators`: 200 |
| AdaBoostReg__proposed7_plus_physics | `learning_rate`: 0.1, `n_estimators`: 100 |
| DNN_Classifier__baseline4 | `learning_rate`: 0.001, `batch_size`: 128, `epochs_run`: 33 |
| DNN_Classifier__extended_raw | `learning_rate`: 0.001, `batch_size`: 256, `epochs_run`: 99 |
| DNN_Classifier__proposed7 | `learning_rate`: 0.001, `batch_size`: 128, `epochs_run`: 48 |
| DNN_Classifier__proposed7_plus_physics | `learning_rate`: 0.001, `batch_size`: 128, `epochs_run`: 42 |
| DNN_Regressor__baseline4 | `learning_rate`: 0.0005, `batch_size`: 256, `epochs_run`: 20 |
| DNN_Regressor__extended_raw | `learning_rate`: 0.001, `batch_size`: 256, `epochs_run`: 127 |
| DNN_Regressor__proposed7 | `learning_rate`: 0.001, `batch_size`: 256, `epochs_run`: 80 |
| DNN_Regressor__proposed7_plus_physics | `learning_rate`: 0.001, `batch_size`: 128, `epochs_run`: 85 |
| DecisionTreeClf__baseline4 | `min_samples_split`: 2, `max_depth`: 5 |
| DecisionTreeClf__extended_raw | `min_samples_split`: 5, `max_depth`: None |
| DecisionTreeClf__proposed7 | `min_samples_split`: 2, `max_depth`: 5 |
| DecisionTreeClf__proposed7_plus_physics | `min_samples_split`: 2, `max_depth`: 5 |
| DecisionTreeReg__baseline4 | `min_samples_split`: 2, `max_depth`: 5 |
| DecisionTreeReg__extended_raw | `min_samples_split`: 10, `max_depth`: 20 |
| DecisionTreeReg__proposed7 | `min_samples_split`: 2, `max_depth`: None |
| DecisionTreeReg__proposed7_plus_physics | `min_samples_split`: 5, `max_depth`: None |
| ElasticNet__baseline4 | `alpha`: 0.1, `l1_ratio`: 0.5 |
| ElasticNet__extended_raw | `alpha`: 0.1, `l1_ratio`: 0.5 |
| ElasticNet__proposed7 | `alpha`: 0.1, `l1_ratio`: 0.5 |
| ElasticNet__proposed7_plus_physics | `alpha`: 0.1, `l1_ratio`: 0.8 |
| ExtraTreesClf__baseline4 | `max_depth`: None, `n_estimators`: 200 |
| ExtraTreesClf__extended_raw | `max_depth`: 20, `n_estimators`: 300 |
| ExtraTreesClf__proposed7 | `max_depth`: None, `n_estimators`: 100 |
| ExtraTreesClf__proposed7_plus_physics | `max_depth`: None, `n_estimators`: 300 |
| ExtraTreesReg__baseline4 | `n_estimators`: 300, `min_samples_split`: 5, `max_depth`: 10 |
| ExtraTreesReg__extended_raw | `n_estimators`: 300, `min_samples_split`: 2, `max_depth`: 20 |
| ExtraTreesReg__proposed7 | `n_estimators`: 300, `min_samples_split`: 2, `max_depth`: 20 |
| ExtraTreesReg__proposed7_plus_physics | `n_estimators`: 300, `min_samples_split`: 2, `max_depth`: 20 |
| HistGradientBoostingClf__baseline4 | `max_iter`: 200, `max_depth`: 3, `learning_rate`: 0.01 |
| HistGradientBoostingClf__extended_raw | `max_iter`: 200, `max_depth`: 5, `learning_rate`: 0.1 |
| HistGradientBoostingClf__proposed7 | `max_iter`: 200, `max_depth`: 7, `learning_rate`: 0.01 |
| HistGradientBoostingClf__proposed7_plus_physics | `max_iter`: 100, `max_depth`: 5, `learning_rate`: 0.05 |
| HistGradientBoostingReg__baseline4 | `max_iter`: 100, `max_depth`: 3, `learning_rate`: 0.05 |
| HistGradientBoostingReg__extended_raw | `max_iter`: 300, `max_depth`: 7, `learning_rate`: 0.05 |
| HistGradientBoostingReg__proposed7 | `max_iter`: 300, `max_depth`: 7, `learning_rate`: 0.05 |
| HistGradientBoostingReg__proposed7_plus_physics | `max_iter`: 300, `max_depth`: 7, `learning_rate`: 0.05 |
| KNNClf__baseline4 | `n_neighbors`: 5, `weights`: distance |
| KNNClf__extended_raw | `n_neighbors`: 3, `weights`: distance |
| KNNClf__proposed7 | `n_neighbors`: 3, `weights`: distance |
| KNNClf__proposed7_plus_physics | `n_neighbors`: 3, `weights`: distance |
| KNNReg__baseline4 | `n_neighbors`: 11, `weights`: distance |
| KNNReg__extended_raw | `n_neighbors`: 3, `weights`: distance |
| KNNReg__proposed7 | `n_neighbors`: 3, `weights`: distance |
| KNNReg__proposed7_plus_physics | `n_neighbors`: 3, `weights`: distance |
| Lasso__baseline4 | `alpha`: 1.0 |
| Lasso__extended_raw | `alpha`: 0.1 |
| Lasso__proposed7 | `alpha`: 0.1 |
| Lasso__proposed7_plus_physics | `alpha`: 0.1 |
| LinearRegression__baseline4 | default |
| LinearRegression__extended_raw | default |
| LinearRegression__proposed7 | default |
| LinearRegression__proposed7_plus_physics | default |
| LinearSVC__baseline4 | `C`: 0.01 |
| LinearSVC__extended_raw | `C`: 1.0 |
| LinearSVC__proposed7 | `C`: 0.01 |
| LinearSVC__proposed7_plus_physics | `C`: 0.01 |
| LinearSVR__baseline4 | `epsilon`: 0.1, `C`: 1.0 |
| LinearSVR__extended_raw | `epsilon`: 0.5, `C`: 10.0 |
| LinearSVR__proposed7 | `epsilon`: 0.5, `C`: 10.0 |
| LinearSVR__proposed7_plus_physics | `epsilon`: 0.5, `C`: 10.0 |
| LogisticRegression__baseline4 | `C`: 10.0, `solver`: lbfgs |
| LogisticRegression__extended_raw | `C`: 10.0, `solver`: saga |
| LogisticRegression__proposed7 | `C`: 10.0, `solver`: saga |
| LogisticRegression__proposed7_plus_physics | `C`: 10.0, `solver`: lbfgs |
| RandomForestClf__baseline4 | `n_estimators`: 100, `min_samples_split`: 5, `max_depth`: 10 |
| RandomForestClf__extended_raw | `n_estimators`: 300, `min_samples_split`: 2, `max_depth`: 20 |
| RandomForestClf__proposed7 | `n_estimators`: 200, `min_samples_split`: 2, `max_depth`: None |
| RandomForestClf__proposed7_plus_physics | `n_estimators`: 100, `min_samples_split`: 5, `max_depth`: None |
| RandomForestReg__baseline4 | `n_estimators`: 100, `min_samples_split`: 2, `max_depth`: 10 |
| RandomForestReg__extended_raw | `n_estimators`: 300, `min_samples_split`: 2, `max_depth`: 20 |
| RandomForestReg__proposed7 | `n_estimators`: 300, `min_samples_split`: 2, `max_depth`: 20 |
| RandomForestReg__proposed7_plus_physics | `n_estimators`: 300, `min_samples_split`: 2, `max_depth`: 20 |
| Ridge__baseline4 | `alpha`: 100.0 |
| Ridge__extended_raw | `alpha`: 100.0 |
| Ridge__proposed7 | `alpha`: 100.0 |
| Ridge__proposed7_plus_physics | `alpha`: 100.0 |
| WideDeep_Regressor__baseline4 | `learning_rate`: 0.0005, `batch_size`: 128, `epochs_run`: 16 |
| WideDeep_Regressor__extended_raw | `learning_rate`: 0.001, `batch_size`: 128, `epochs_run`: 85 |
| WideDeep_Regressor__proposed7 | `learning_rate`: 0.0005, `batch_size`: 256, `epochs_run`: 119 |
| WideDeep_Regressor__proposed7_plus_physics | `learning_rate`: 0.0005, `batch_size`: 128, `epochs_run`: 118 |


## 5. Saved Models & Artifacts

| Directory / Path | Contents |
|------------------|----------|
| `trained_models/sklearn_regressors/` | 44 files/models |
| `trained_models/sklearn_classifiers/` | 32 files/models |
| `trained_models/tensorflow/` | 12 files/models |
| `trained_models/*.joblib` | 4 Feature Scalers (StandardScaler per feature set) |
| `results/*.csv` | Full metric dumps (all models x feature sets x subsets) |
| `results/best_hyperparams.json` | JSON dictionary of best tuned hyperparameters |


## 6. Key Takeaways & Recommendations

1. **Feature Engineering Impact**: Moving from `baseline4` to `proposed7_plus_physics` reduces regression MAE from ~15.0 mm down to **0.71 mm** (a **>95% error reduction**). Water balance physics (`dr_mm`, `raw_mm`, `vpd_kpa`) give tree models explicit domain signals.
2. **Top Regressor**: `ExtraTreesReg` and `RandomForestReg` on `proposed7_plus_physics` achieve **R² > 0.996** and MAE < 0.78 mm.
3. **Top Classifier**: `HistGradientBoostingClf` and `DecisionTreeClf` on `extended_raw` achieve near-perfect classification (**Macro-F1 = 0.9877**, **Accuracy = 99.6%**).
4. **Neural Networks**: `DNN_Regressor` and `WideDeep_Regressor` reached MAE ~1.05-1.17 mm (R² ~ 0.987).
5. **Deployment Recommendation**: For production edge inference, use `HistGradientBoostingClf` (instant inference, low memory) and `ExtraTreesReg` or `RandomForestReg` with the saved `scaler_proposed7_plus_physics.joblib`.
