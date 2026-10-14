# How to use this feature pack (for the modelling team)

Files
- primary/primary_model_ready.csv   all 10,000 rows, numeric, one-hot encoded, with targets + `split` column
- primary/per_crop/<crop>_model_ready.csv   same, one crop per file (rice, maize, potato, wheat, cotton, sugarcane)
- primary/primary_features_full.csv human-readable version (strings, every engineered column)
- primary/edge_cases.csv            rows flagged as edge cases (low temperature, very dry, saturated, ...)
- feature_sets.json                 exact column lists for baseline4 / proposed7 / extended_raw / proposed7_plus_physics
- tomato/tomato_features.csv        tomato is NOT in the primary dataset; use this (json_real_sensor rows are the trustworthy ones)
- sensor/iot_sensor_kalman.csv      noisy IoT stream with Kalman-filtered soil moisture (Q=0.01, R=0.05)
- feature_report.md, data_dictionary.md

Train ONE regressor (baseline vs proposed, as in the plan)
```python
import json, pandas as pd
df   = pd.read_csv("primary/primary_model_ready.csv")
sets = json.load(open("feature_sets.json"))
tr, te = df[df.split == "train"], df[df.split == "test"]

for name in ["baseline4", "proposed7"]:
    cols = sets[name]
    model.fit(tr[cols], tr["water_amount_mm"])
    pred = model.predict(te[cols]).clip(min=0)      # never negative
    irrigate_pred = pred >= 2.0                     # 0 / <2 mm = no irrigation
```
Rules
- Never use `meta_crop`, `split`, `irrigate`, `need_ord`, `is_edge_case` or anything in
  `leaky_for_amount_do_not_use_as_inputs` as model inputs.
- Do not re-split randomly; use the `split` column so everyone is comparable.
- Report results on the edge-case rows separately: te[te.is_edge_case == 1].
- `water_amount_mm` is DERIVED (FAO-56 style), not measured. A near-perfect score with crop/soil/stage inputs is
  partly by construction. For an honest benchmark also report the native label `need_ord` (Low/Medium/High).
