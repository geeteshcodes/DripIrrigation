import pandas as pd, numpy as np

d = pd.read_csv('primary/primary_model_ready.csv')
t = d[d.split=='train'].reset_index(drop=True)
oof = pd.read_csv('results/threshold_testing/out_of_fold_predictions.csv')
oof = oof.rename(columns={'source_row_index':'row_idx'})
t['row_idx'] = t.index
m = oof.merge(
    t[['row_idx','rainfall_mm','soil_moisture','demand_minus_rain_mm',
       'rain_covers_etc','eff_rain_window_mm','water_amount_mm','kc',
       'dr_mm','below_mad','etc_mm']],
    on='row_idx', how='left')
m['_fp'] = ((m['predicted_water_amount_mm']>=2.7)&(m['actual_water_amount_mm']<2.7)).astype(int)
m['_tp'] = ((m['predicted_water_amount_mm']>=2.7)&(m['actual_water_amount_mm']>=2.7)).astype(int)
fp = m[m['_fp']==1]
tp = m[m['_tp']==1]

print('=== Key separator features (FP vs TP medians) ===')
for col in ['water_amount_mm','kc','dr_mm','below_mad','soil_moisture','etc_mm']:
    fp_med = fp[col].median()
    tp_med = tp[col].median()
    print(f'{col:30s}  FP={fp_med:8.2f}  TP={tp_med:8.2f}  ratio={tp_med/(fp_med+0.001):.1f}x')

print()
print('=== FP actual water_amount_mm distribution ===')
print(fp['actual_water_amount_mm'].describe().round(4))

print()
print('=== TPs with rain surplus: why still need irrigation? ===')
tp_rain = tp[(tp['eff_rain_window_mm']>=25)&(tp['demand_minus_rain_mm']<-3)]
print('n =', len(tp_rain))
print('kc median =', round(tp_rain['kc'].median(), 2))
print('etc_mm median =', round(tp_rain['etc_mm'].median(), 2))
print('water_amount_mm median =', round(tp_rain['water_amount_mm'].median(), 1))
