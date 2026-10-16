\test_model.py
Model: ExtraTreesReg, proposed7, original settings
Test rows processed: 2000
MAE: 0.7456 mm
RMSE: 1.8334 mm
R²: 0.9953
Bad cases: 200 (decision wrong OR absolute amount error > 2.0 mm)
Decision errors: 58
Amount errors > 2.0 mm: 198
All predictions: C:\Users\KIIT0001\DripIrrigation\results\model_test_predictions.csv
Bad-case log: C:\Users\KIIT0001\DripIrrigation\results\model_test_bad_cases.csv

Largest errors:
 source_row_index      crop soil_type growth_stage  actual_water_amount_mm  predicted_water_amount_mm  absolute_error_mm  irrigation_decision_wrong  is_edge_case
             5808    cotton     loamy      Harvest                     0.0                  21.020669          21.020669                       True             0
              455 sugarcane      clay    Flowering                     0.0                  15.912937          15.912937                       True             0
             1411     wheat     loamy      Harvest                     0.0                  13.417654          13.417654                       True             0
             4651 sugarcane      silt   Vegetative                     0.0                  13.331897          13.331897                       True             0
             9035     wheat     sandy    Flowering                     0.0                  12.231482          12.231482                       True             0
             9256     wheat      silt       Sowing                     0.0                  12.117707          12.117707                       True             0
             6688     wheat     loamy       Sowing                     0.0                  12.067016          12.067016                       True             0
             9047    cotton     sandy   Vegetative                     0.0                  10.307102          10.307102                       True             1
             6656 sugarcane     sandy    Flowering                     0.0                   9.262982           9.262982                       True             0
             7675 sugarcane     sandy   Vegetative                     0.0                   7.926354           7.926354                       True             1


Test false alarms reviewed: 56
False alarms by native need class:
need_class
Low       38
Medium    18

False alarms by edge-case reason:
edge_reasons
none                             43
low_temp                          4
high_wind                         3
near_zero_rain                    1
near_zero_rain;high_wind          1
saturated_soil                    1
extreme_vpd                       1
saturated_soil;near_zero_rain     1
low_temp;heavy_rain               1

Detailed review: C:\Users\KIIT0001\DripIrrigation\results\false_alarm_review.csv
Completed threshold validation fold 1/5
Completed threshold validation fold 2/5
Completed threshold validation fold 3/5
Completed threshold validation fold 4/5
Completed threshold validation fold 5/5

Training-only out-of-fold threshold comparison:
 threshold_mm  accuracy  precision  recall     f1  true_positive  false_positive  false_negative  true_negative
       0.5000    0.9032     0.8282  1.0000 0.9060           3732             774               0           3494
       1.0000    0.9407     0.8875  0.9997 0.9403           3731             473               1           3795
       1.5000    0.9573     0.9169  0.9989 0.9561           3728             338               4           3930
       2.0000    0.9669     0.9376  0.9952 0.9656           3714             247              18           4021
       2.5000    0.9709     0.9515  0.9879 0.9694           3687             188              45           4080
       3.0000    0.9726     0.9633  0.9786 0.9709           3652             139              80           4129
       4.0000    0.9706     0.9763  0.9603 0.9683           3584              87             148           4181
       5.0000    0.9627     0.9823  0.9370 0.9591           3497              63             235           4205
       7.5000    0.9413     0.9857  0.8869 0.9337           3310              48             422           4220
      10.0000    0.9227     0.9899  0.8430 0.9106           3146              32             586           4236

Highest validation F1: threshold 3.0 mm (F1=0.9709, precision=0.9633, recall=0.9786)
Threshold table
