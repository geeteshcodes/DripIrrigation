#!/usr/bin/env python3
"""
Feature engineering pipeline - Smart Irrigation major project (Hydro Intelligence Pro)

Builds, from the project's datasets:
  1. PRIMARY  (Kaggle irrigation_prediction.csv, 10,000 x 20): cleaned + engineered features,
     two targets (native Low/Medium/High class AND a derived FAO-56 water amount in mm),
     train/test split, edge-case flags, per-crop files.
  2. TOMATO   (Mendeley tomato JSON sensor data + tomato CSV): harmonised tomato table
     (tomato is NOT in the primary dataset).
  3. IOT      (noisy sensor stream, data.csv): cleaned + Kalman-filtered soil moisture
     (Q=0.01, R=0.05 exactly as in the mini-project report).
  4. Reports: feature_report.md, data_dictionary.md, feature_sets.json
  (Raw_Data.csv / Filtered_Data.csv are drip-line pressure/flow logs, profiled but not used for features.)

Usage (from the folder that holds your files):
  python feature_engineering.py --primary archive__3_.zip --tomato-json tomato_water_sensor_data.json \
      --tomato-csv tomato_irrigation_dataset.csv --iot archive__4_.zip \
      --hydraulic-raw Raw_Data.csv --hydraulic-filtered Filtered_Data.csv --outdir out

Every input except --primary is optional. Inputs may be .csv or a .zip holding one csv.
Requires: pandas, numpy. Optional: scikit-learn (for the ablation study in the report).
"""
import argparse
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
TEST_FRACTION = 0.20

# =========================================================================== #
# AGRONOMY TABLES  (textbook midpoints - verify against FAO-56 before citing)
# =========================================================================== #
# FAO-56 Table 12 crop coefficients: (Kc_ini, Kc_mid, Kc_end)
KC = {"rice": (1.05, 1.20, 0.90), "maize": (0.30, 1.20, 0.35), "potato": (0.50, 1.15, 0.75),
      "wheat": (0.30, 1.15, 0.30), "cotton": (0.35, 1.20, 0.70), "sugarcane": (0.40, 1.25, 0.75),
      "tomato": (0.60, 1.15, 0.80)}
ROOT_MM = {"rice": 300, "maize": 900, "potato": 500, "wheat": 900, "cotton": 1000,
           "sugarcane": 1200, "tomato": 700}                       # mature effective root depth
DEPLETION_P = {"rice": 0.20, "maize": 0.55, "potato": 0.35, "wheat": 0.55, "cotton": 0.65,
               "sugarcane": 0.65, "tomato": 0.40}                  # allowable depletion (MAD)
T_OPT_MAX = {"rice": 35, "maize": 32, "potato": 25, "wheat": 25, "cotton": 35,
             "sugarcane": 38, "tomato": 30}                        # heat-stress onset, deg C
T_BASE = {"rice": 15, "maize": 10, "potato": 5, "wheat": 0, "cotton": 15,
          "sugarcane": 15, "tomato": 10}                           # cold-stress below, deg C
STAGE_ROOT_FACTOR = {"ini": 0.30, "dev": 0.60, "mid": 1.00, "late": 1.00}
STAGE_MAP = {"sowing": "ini", "vegetative": "dev", "flowering": "mid", "harvest": "late",
             "initial stage": "ini", "development stage": "dev", "mid stage": "mid",
             "last stage": "late"}
# volumetric field capacity / wilting point by soil texture
SOIL_WATER = {"sandy": (0.12, 0.05), "silt": (0.30, 0.12), "loamy": (0.27, 0.12),
              "clay": (0.36, 0.20), "unknown": (0.25, 0.12)}
RA_BY_SEASON = {"rabi": 29.0, "kharif": 38.0, "zaid": 40.0}        # MJ/m2/day, ~20N latitude
RA_DEFAULT = 34.0
DAYLIGHT_HOURS = 12.0

# ----- ASSUMPTIONS for the derived water-amount target (all configurable) ----- #
SEASON_DAYS = 120          # Rainfall_mm looks like a season total -> daily = total / SEASON_DAYS
RAIN_WINDOW_DAYS = 3       # effective rain that can offset one irrigation event
EFF_RAIN_FRACTION = 0.8
MAX_EVENT_MM = 75.0        # cap on one irrigation event
MIN_EVENT_MM = 2.0         # smaller amounts are not practical -> treated as "no irrigation"
# Soil_Moisture is read as % of plant-available water (0-100 scale); see report.

# Kalman filter constants from the mini-project report
KALMAN_Q = 0.01
KALMAN_R = 0.05
IOT_ADC_DRY = 480.0

STAGE_ORDER = ["ini", "dev", "mid", "late"]


# =========================================================================== #
# IO helpers
# =========================================================================== #
def snake(s):
    return re.sub(r"[^0-9a-z]+", "_", str(s).strip().lower()).strip("_")


def read_csv_any(path, **kw):
    p = Path(path)
    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not names:
                raise SystemExit(f"No csv inside {p}")
            with z.open(names[0]) as f:
                return pd.read_csv(io.BytesIO(f.read()), **kw)
    return pd.read_csv(p, **kw)


# =========================================================================== #
# Physics helpers
# =========================================================================== #
def vpd_kpa(t, rh):
    es = 0.6108 * np.exp(17.27 * t / (t + 237.3))
    return es * (1 - np.clip(rh, 0, 100) / 100.0)


def et0_penman_monteith(t, rh, wind_kmh, sun_hours, ra):
    """FAO-56 Penman-Monteith daily ET0 (mm/day) from daily-mean inputs.
    Assumes wind measured at 2 m, elevation ~0 m, G = 0, N = 12 h."""
    u2 = np.asarray(wind_kmh, float) / 3.6
    es = 0.6108 * np.exp(17.27 * t / (t + 237.3))
    ea = es * np.clip(rh, 0, 100) / 100.0
    delta = 4098 * es / (t + 237.3) ** 2
    gamma = 0.066
    rs = (0.25 + 0.5 * np.clip(sun_hours / DAYLIGHT_HOURS, 0, 1)) * ra
    rso = 0.75 * ra
    rns = 0.77 * rs
    rnl = (4.903e-9 * (t + 273.16) ** 4 * (0.34 - 0.14 * np.sqrt(ea))
           * (1.35 * np.clip(rs / rso, 0, 1) - 0.35))
    rn = rns - rnl
    num = 0.408 * delta * rn + gamma * 900 / (t + 273) * u2 * (es - ea)
    den = delta + gamma * (1 + 0.34 * u2)
    return np.clip(num / den, 0, None)


def kc_series(crop, stage):
    ini = crop.map(lambda c: KC[c][0]); mid = crop.map(lambda c: KC[c][1])
    end = crop.map(lambda c: KC[c][2])
    return pd.Series(np.select([stage == "ini", stage == "dev", stage == "mid"],
                               [ini, (ini + mid) / 2, mid], default=end), index=crop.index)


def kalman_1d(z, q=KALMAN_Q, r=KALMAN_R):
    """Scalar Kalman filter exactly as written in the mini-project report:
       predict: P = P + Q ; gain: K = P / (P + R) ; update: x = x + K (z - x) ; P = (1 - K) P.
       NaN measurements skip the update (prediction only) = missing-sensor handling."""
    z = np.asarray(z, float)
    x = np.zeros_like(z)
    first = np.where(~np.isnan(z))[0]
    if len(first) == 0:
        return np.full_like(z, np.nan)
    x_prev, p = z[first[0]], 1.0
    for i, zi in enumerate(z):
        p = p + q
        if not np.isnan(zi):
            k = p / (p + r)
            x_prev = x_prev + k * (zi - x_prev)
            p = (1 - k) * p
        x[i] = x_prev
    return x


# =========================================================================== #
# 1. PRIMARY
# =========================================================================== #
def build_primary(raw, log):
    df = raw.copy()
    df.columns = [snake(c) for c in df.columns]
    n0 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    log.append(f"primary: {n0} rows loaded, {n0 - len(df)} duplicates dropped, "
               f"{int(df.isna().sum().sum())} missing cells.")
    for c in ("crop_type", "soil_type", "crop_growth_stage", "season", "mulching_used",
              "irrigation_need"):
        df[c] = df[c].astype(str).str.strip()
    df["crop_type"] = df["crop_type"].str.lower()
    df["soil_type"] = df["soil_type"].str.lower()
    df = df.rename(columns={"crop_growth_stage": "growth_stage"})
    df["growth_stage"] = df["growth_stage"].str.title()

    # physical-range clipping (reported)
    for c, lo, hi in [("humidity", 0, 100), ("rainfall_mm", 0, None), ("temperature_c", -10, 55),
                      ("soil_moisture", 0, 100), ("wind_speed_kmh", 0, None)]:
        bad = int(((df[c] < lo) | ((df[c] > hi) if hi is not None else False)).sum())
        if bad:
            log.append(f"primary: clipped {bad} out-of-range values in {c}.")
        df[c] = df[c].clip(lo, hi)
    missing_crops = sorted(set(df.crop_type) - set(KC))
    if missing_crops:
        raise SystemExit(f"No agronomy table for crops: {missing_crops}")

    # ---- stage / crop / soil parameters ---------------------------------- #
    df["stage_group"] = df["growth_stage"].str.lower().map(STAGE_MAP)
    df["stage_ord"] = df["stage_group"].map({s: i for i, s in enumerate(STAGE_ORDER)})
    df["stage_water_sensitive"] = df["growth_stage"].isin(["Vegetative", "Flowering"]).astype(int)
    df["kc"] = kc_series(df["crop_type"], df["stage_group"])
    wp = df["soil_type"].map(lambda s: SOIL_WATER.get(s, SOIL_WATER["unknown"]))
    df["field_capacity"] = wp.map(lambda t: t[0])
    df["wilting_point"] = wp.map(lambda t: t[1])

    # ---- atmospheric demand ---------------------------------------------- #
    ra = df["season"].str.lower().map(RA_BY_SEASON).fillna(RA_DEFAULT)
    df["vpd_kpa"] = vpd_kpa(df["temperature_c"], df["humidity"])
    df["et0_mm"] = et0_penman_monteith(df["temperature_c"], df["humidity"],
                                       df["wind_speed_kmh"], df["sunlight_hours"], ra)
    df["etc_mm"] = df["et0_mm"] * df["kc"]

    # ---- soil water status ----------------------------------------------- #
    df["rel_moisture"] = df["soil_moisture"] / 100.0
    root = df["crop_type"].map(ROOT_MM) * df["stage_group"].map(STAGE_ROOT_FACTOR)
    df["taw_mm"] = (df["field_capacity"] - df["wilting_point"]) * root
    df["dr_mm"] = (1 - df["rel_moisture"]) * df["taw_mm"]                  # depletion
    df["raw_mm"] = df["crop_type"].map(DEPLETION_P) * df["taw_mm"]         # readily available water
    df["below_mad"] = (df["dr_mm"] > df["raw_mm"]).astype(int)
    df["rain_per_day_mm"] = df["rainfall_mm"] / SEASON_DAYS
    df["eff_rain_window_mm"] = EFF_RAIN_FRACTION * df["rain_per_day_mm"] * RAIN_WINDOW_DAYS
    df["rain_covers_etc"] = (EFF_RAIN_FRACTION * df["rain_per_day_mm"] >= df["etc_mm"]).astype(int)

    # ---- stress + interaction features ----------------------------------- #
    stage_shift = np.where(df["growth_stage"] == "Flowering", 2.0, 0.0)   # flowering is more heat-sensitive
    df["heat_stress_idx"] = (df["temperature_c"] - (df["crop_type"].map(T_OPT_MAX) - stage_shift)).clip(lower=0)
    df["cold_stress_idx"] = (df["crop_type"].map(T_BASE) - df["temperature_c"]).clip(lower=0)
    df["dry_hot"] = ((df["rel_moisture"] < 0.5) & (df["temperature_c"] > 30)).astype(int)
    df["moisture_x_vpd"] = (1 - df["rel_moisture"]) * df["vpd_kpa"]
    df["demand_minus_rain_mm"] = df["etc_mm"] - EFF_RAIN_FRACTION * df["rain_per_day_mm"]

    # ---- targets ---------------------------------------------------------- #
    need = (df["dr_mm"] - df["eff_rain_window_mm"]).clip(lower=0)
    amt = np.where(df["below_mad"] == 1, np.minimum(need, MAX_EVENT_MM), 0.0)
    df["water_amount_mm"] = np.where(amt < MIN_EVENT_MM, 0.0, amt)
    df["irrigate"] = (df["water_amount_mm"] > 0).astype(int)
    df["need_class"] = df["irrigation_need"].str.title()
    df["need_ord"] = df["need_class"].map({"Low": 0, "Medium": 1, "High": 2})
    assert (df["water_amount_mm"] >= 0).all()

    # ---- split: stratified by crop x native class -------------------------- #
    rng = np.random.default_rng(SEED)
    df["split"] = "train"
    for _, idx in df.groupby(["crop_type", "need_class"]).groups.items():
        idx = np.array(list(idx))
        test = rng.choice(idx, size=int(round(len(idx) * TEST_FRACTION)), replace=False)
        df.loc[test, "split"] = "test"

    # ---- edge cases (from the team's "edge cases / low temperature testing" list)
    q95_vpd = df["vpd_kpa"].quantile(0.95)
    flags = {
        "low_temp": df["temperature_c"] <= 15,
        "very_dry_soil": df["soil_moisture"] <= 12,
        "saturated_soil": df["soil_moisture"] >= 60,
        "heavy_rain": df["rainfall_mm"] >= 2300,
        "near_zero_rain": df["rainfall_mm"] <= 50,
        "high_wind": df["wind_speed_kmh"] >= 18,
        "extreme_vpd": df["vpd_kpa"] >= q95_vpd,
        "label_conflict": ((df["irrigate"] == 1) & (df["need_class"] == "Low")) |
                          ((df["irrigate"] == 0) & (df["need_class"] == "High")),
    }
    for k, v in flags.items():
        df[f"edge_{k}"] = v.astype(int)
    reasons = [";".join(k for k in flags if flags[k].iloc[i]) for i in range(len(df))]
    df["is_edge_case"] = [int(r != "") for r in reasons]
    df["edge_reasons"] = [r if r else "none" for r in reasons]
    return df


FEATURE_SETS_RAW = {
    "baseline4": ["soil_moisture", "temperature_c", "rainfall_mm", "humidity"],
    "proposed7": ["soil_moisture", "temperature_c", "rainfall_mm", "humidity",
                  "crop_type", "soil_type", "growth_stage"],
    "extended_raw": ["soil_moisture", "temperature_c", "rainfall_mm", "humidity", "crop_type",
                     "soil_type", "growth_stage", "wind_speed_kmh", "sunlight_hours",
                     "mulching_used", "previous_irrigation_mm", "season"],
}
PHYSICS = ["kc", "et0_mm", "etc_mm", "vpd_kpa", "rel_moisture", "taw_mm", "stage_water_sensitive",
           "heat_stress_idx", "cold_stress_idx", "dry_hot", "moisture_x_vpd",
           "demand_minus_rain_mm", "rain_covers_etc"]
LEAKY_FOR_AMOUNT = ["dr_mm", "raw_mm", "below_mad", "eff_rain_window_mm", "rain_per_day_mm",
                    "field_capacity", "wilting_point"]


def primary_model_ready(df):
    cat = ["crop_type", "soil_type", "growth_stage", "season", "mulching_used"]
    num = ["soil_moisture", "temperature_c", "rainfall_mm", "humidity", "wind_speed_kmh",
           "sunlight_hours", "previous_irrigation_mm", "stage_ord"] + PHYSICS + LEAKY_FOR_AMOUNT
    num = list(dict.fromkeys(num))
    out = pd.concat([df[num], pd.get_dummies(df[cat], dtype=int)], axis=1)
    out["water_amount_mm"] = df["water_amount_mm"]
    out["irrigate"] = df["irrigate"]
    out["need_ord"] = df["need_ord"]
    out["is_edge_case"] = df["is_edge_case"]
    out["split"] = df["split"]
    out["meta_crop"] = df["crop_type"]                   # id only - do not feed to a model
    return out


def model_ready_sets(mr):
    def cols(prefixes, plain):
        c = [x for x in plain if x in mr.columns]
        for p in prefixes:
            c += [x for x in mr.columns if x.startswith(p + "_")]
        return c
    base = ["soil_moisture", "temperature_c", "rainfall_mm", "humidity"]
    proposed = base + cols(["crop_type", "soil_type", "growth_stage"], [])
    sets = {
        "baseline4": base,
        "proposed7": proposed,
        "extended_raw": proposed + ["wind_speed_kmh", "sunlight_hours", "previous_irrigation_mm"]
                        + [c for c in mr.columns if c.startswith(("mulching_used_", "season_"))],
        "proposed7_plus_physics": proposed + PHYSICS,
        "leaky_for_amount_do_not_use_as_inputs": LEAKY_FOR_AMOUNT,
    }
    return {k: list(dict.fromkeys(v)) for k, v in sets.items()}


# =========================================================================== #
# Ablation (for the report)
# =========================================================================== #
def run_ablation(df, mr, sets, log):
    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.metrics import f1_score, mean_absolute_error, r2_score
        from sklearn.model_selection import StratifiedKFold, cross_val_score
    except ImportError:
        log.append("ablation skipped: scikit-learn not installed.")
        return None
    y_cls = mr["need_ord"]
    cv = StratifiedKFold(3, shuffle=True, random_state=SEED)

    def cls_score(cols):
        m = RandomForestClassifier(100, random_state=SEED, n_jobs=-1)
        return cross_val_score(m, mr[cols], y_cls, cv=cv, scoring="f1_macro").mean()

    def sub(cols_plain, extra_prefix=()):
        c = list(cols_plain)
        for p in extra_prefixes(extra_prefix):
            c += [x for x in mr.columns if x.startswith(p)]
        return c

    def extra_prefixes(p):
        return p

    base = sets["baseline4"]
    one = lambda pfx: [c for c in mr.columns if c.startswith(pfx + "_")]
    rows = [("baseline4 (moisture, temp, rain, humidity)", base),
            ("baseline4 + growth_stage", base + one("growth_stage")),
            ("baseline4 + crop_type", base + one("crop_type")),
            ("baseline4 + soil_type", base + one("soil_type")),
            ("proposed7 (+crop, soil, stage)", sets["proposed7"]),
            ("extended_raw (+wind, sunlight, mulching, ...)", sets["extended_raw"]),
            ("proposed7 + physics features", sets["proposed7_plus_physics"])]
    clf = [(n, round(cls_score(c), 4)) for n, c in rows]

    tr, te = mr["split"] == "train", mr["split"] == "test"
    reg = []
    for n, c in [("baseline4", sets["baseline4"]), ("proposed7", sets["proposed7"]),
                 ("proposed7 + physics", sets["proposed7_plus_physics"])]:
        m = RandomForestRegressor(100, random_state=SEED, n_jobs=-1, min_samples_leaf=2)
        m.fit(mr.loc[tr, c], mr.loc[tr, "water_amount_mm"])
        p = np.clip(m.predict(mr.loc[te, c]), 0, None)
        yt = mr.loc[te, "water_amount_mm"]
        reg.append((n, round(mean_absolute_error(yt, p), 3), round(r2_score(yt, p), 4)))
    return {"classification": clf, "regression": reg}


# =========================================================================== #
# 2. TOMATO
# =========================================================================== #
def build_tomato(json_path, csv_path, log):
    frames = []
    cal = None
    if json_path:
        j = pd.DataFrame(json.load(open(json_path, encoding="utf-8")))
        rule = np.where(j["moisturePercentage"] < j["minTomatoMoisture"],
                        j["evapotranspirationValue"] - j["moisturePercentage"], 0.0)
        rule_ok = bool(np.allclose(np.clip(rule, 0, None), j["irrigation_needed_value"], atol=1e-3))
        etc_ok = bool(np.allclose(j["evapotranspirationValue"], j["reference_eto"] * j["cropCoffiecient"], rtol=1e-3))
        log.append(f"tomato json: {len(j)} rows; irrigation_needed = ETc - moisture% when moisture% < "
                   f"min threshold else 0 -> rule reproduced exactly: {rule_ok}; ETc = ETo x Kc holds: {etc_ok}.")
        cal = np.polyfit(j["moistureValue"], j["moisturePercentage"], 1)
        a = pd.DataFrame({
            "source": "json_real_sensor", "crop_type": "tomato", "growth_stage": "Mid stage (assumed, Kc=1.15)",
            "stage_group": "mid", "soil_type": "unknown", "temperature_c": j["temperature"],
            "humidity": j["humidity"], "soil_moisture_pct": j["moisturePercentage"],
            "wind_speed_ms": j["windSpeedVal"], "ghi_wm2": j["solarRadiation_ghi_val"],
            "kc": j["cropCoffiecient"], "et0_source": j["reference_eto"], "etc_source": j["evapotranspirationValue"],
            "water_amount_mm_source": j["irrigation_needed_value"], "motor_on": j["motorValue"],
            "day_of_year": j["dayOfYearNumber"], "quality_flag": "ok"})
        frames.append(a)
    if csv_path:
        c = read_csv_any(csv_path)
        c.columns = [snake(x) for x in c.columns]
        n0 = len(c)
        c = c.drop_duplicates().reset_index(drop=True)
        log.append(f"tomato csv: {n0} rows, {n0 - len(c)} exact duplicates dropped.")
        pct = (cal[0] * c["soil_moisture"] + cal[1]) if cal is not None else np.nan
        stage = c["crop_coefficient_stage"].str.strip().str.lower()
        et_bad = (c["evapotranspiration"] / c["reference_evapotranspiration"] - c["crop_coefficient"]).abs() > 0.05
        ph_bad = (c["ph"] < 3.5) | (c["ph"] > 9.5)
        log.append(f"tomato csv: ETc/ETo disagrees with Kc in {et_bad.mean():.0%} of rows; "
                   f"pH outside 3.5-9.5 in {ph_bad.mean():.0%}; ET values reach {c['reference_evapotranspiration'].max():.0f} "
                   f"(implausible as mm/day) -> flagged low quality, no derived target created.")
        b = pd.DataFrame({
            "source": "csv_augmented", "crop_type": "tomato", "growth_stage": c["crop_coefficient_stage"].str.strip(),
            "stage_group": stage.map(STAGE_MAP), "soil_type": "unknown",
            "temperature_c": c["temperature_c"], "humidity": c["humidity"],
            "soil_moisture_pct": np.clip(pct, 0, 100), "wind_speed_ms": c["wind_speed"],
            "ghi_wm2": c["solar_radiation_ghi"], "kc": c["crop_coefficient"],
            "et0_source": c["reference_evapotranspiration"], "etc_source": c["evapotranspiration"],
            "water_amount_mm_source": np.nan, "motor_on": np.nan, "day_of_year": np.nan,
            "days_planted": c["days_of_planted"], "nitrogen_mgkg": c["nitrogen_mg_kg"],
            "phosphorus_mgkg": c["phosphorus_mg_kg"], "potassium": c["potassium"], "ph": c["ph"],
            "quality_flag": np.where(et_bad | ph_bad, "low_quality_et_or_ph", "ok")})
        frames.append(b)
    if not frames:
        return None
    t = pd.concat(frames, ignore_index=True)
    t["stage_ord"] = t["stage_group"].map({s: i for i, s in enumerate(STAGE_ORDER)})
    t["vpd_kpa"] = vpd_kpa(t["temperature_c"], t["humidity"])
    stage_shift = np.where(t["stage_group"] == "mid", 2.0, 0.0)
    t["heat_stress_idx"] = (t["temperature_c"] - (T_OPT_MAX["tomato"] - stage_shift)).clip(lower=0)
    t["cold_stress_idx"] = (T_BASE["tomato"] - t["temperature_c"]).clip(lower=0)
    t["irrigate"] = np.where(t["water_amount_mm_source"].isna(), np.nan,
                             (t["water_amount_mm_source"] > 0).astype(float))
    return t


# =========================================================================== #
# 3. IOT sensor stream + Kalman
# =========================================================================== #
def build_iot(raw, log):
    d = raw.copy()
    d.columns = [snake(c) for c in d.columns]
    d = d.rename(columns={"soilmiosture": "soil_adc_raw"})
    alt = pd.to_numeric(d["altitude"].astype(str).str.replace("-", "", regex=False), errors="coerce")
    glitch_t = (d["temperature"] < -10) | (d["temperature"] > 60)
    glitch_p = (d["pressure"] < 8000) | (d["pressure"] > 12000)
    glitch_s = (d["soil_adc_raw"] < 0) | (d["soil_adc_raw"] > 1023)
    d["is_glitch"] = (glitch_t | glitch_p | glitch_s | alt.isna()).astype(int)
    out = pd.DataFrame({"id": d["id"], "temperature_c_raw": d["temperature"],
                        "temperature_c": d["temperature"].where(~glitch_t),
                        "pressure_raw": d["pressure"], "pressure": d["pressure"].where(~glitch_p),
                        "altitude_m": alt, "soil_adc_raw": d["soil_adc_raw"],
                        "soil_adc_clean": d["soil_adc_raw"].where(~glitch_s),
                        "class": d["class"], "note": d["note"], "status": d["status"],
                        "is_glitch": d["is_glitch"]})
    z_raw = np.clip(d["soil_adc_raw"], 0, IOT_ADC_DRY) / IOT_ADC_DRY          # dryness 0..1 (raw, glitches kept)
    z_clean = out["soil_adc_clean"].clip(0, IOT_ADC_DRY) / IOT_ADC_DRY
    out["dryness_raw"] = z_raw
    out["dryness_kalman"] = kalman_1d(z_clean.values)
    out["soil_moisture_pct_kalman"] = 100 * (1 - out["dryness_kalman"])
    dr_raw = np.abs(np.diff(z_raw)); dr_k = np.abs(np.diff(out["dryness_kalman"]))
    log.append(f"iot: {len(out)} rows, {int(out['is_glitch'].sum())} glitch rows (temp 178.7, pressure -2120, "
               f"negative ADC, unparsable altitude) masked -> Kalman predicts through them; "
               f"mean |step| of dryness {dr_raw.mean():.4f} raw -> {dr_k.mean():.4f} filtered; "
               f"max step {dr_raw.max():.3f} -> {dr_k.max():.3f}. Timestamp columns are constant "
               f"(single date/time for all rows), so row order (id) is treated as time order.")
    return out


# =========================================================================== #
# Hydraulic logs (profile only)
# =========================================================================== #
def profile_hydraulic(raw_path, filt_path, log):
    r = pd.read_csv(raw_path)
    f = pd.read_csv(filt_path)
    f["id"] = pd.to_numeric(f["id"], errors="coerce")
    stats_rows = int(f["id"].isna().sum())
    kept = r.merge(f.dropna(subset=["id"]).astype({"id": int}), on="id", suffixes=("_r", "_f"))
    same = bool(np.allclose(kept["Flow rate_r"], kept["Flow rate_f"]))
    log.append(f"hydraulic: Raw {len(r)} rows (pressure 1/2, flow rate); Filtered keeps {len(kept)} rows with "
               f"identical values (it only trims start-up/shut-down transients, same values: {same}) and has "
               f"{stats_rows} summary rows (MEAN/MEDIAN/STD DEV) appended. These describe drip-line hydraulics, "
               f"not crop water need -> not used as features.")


# =========================================================================== #
# Reports
# =========================================================================== #
def write_reports(out, prim, mr, sets, tomato, iot, ablation, log):
    L = ["# Feature engineering report", ""]
    L += ["## 1. Run log"] + [f"- {x}" for x in log] + [""]

    L += ["## 2. What the primary dataset can and cannot support", ""]
    L += ["Rows per crop: " + ", ".join(f"{k}={v}" for k, v in prim["crop_type"].value_counts().items()),
          "", "Tomato is NOT in the primary dataset; see the separate tomato table.", ""]
    share = prim.groupby("crop_type")["need_class"].value_counts(normalize=True).unstack().round(3)
    L += ["Native irrigation-need class share by crop (flat = crop carries no signal):", "",
          share.to_markdown(), ""]
    share = prim.groupby("soil_type")["need_class"].value_counts(normalize=True).unstack().round(3)
    L += ["By soil type:", "", share.to_markdown(), ""]
    share = prim.groupby("growth_stage")["need_class"].value_counts(normalize=True).unstack().round(3)
    L += ["By growth stage (large differences = real signal):", "", share.to_markdown(), ""]

    if ablation:
        L += ["### Ablation: native Low/Medium/High class (RandomForest, 3-fold CV, macro-F1)", "",
              "| feature set | macro-F1 |", "|---|---|"]
        L += [f"| {n} | {s} |" for n, s in ablation["classification"]]
        L += ["", "### Ablation: derived water amount (RandomForest, held-out test split)", "",
              "CAUTION: the derived target is built from crop, soil and stage parameters, so any gain from those "
              "columns here is partly by construction (circular).", "",
              "| feature set | MAE (mm) | R2 |", "|---|---|---|"]
        L += [f"| {n} | {m} | {r} |" for n, m, r in ablation["regression"]]
        L += [""]

    agree = pd.crosstab(prim["need_class"], prim["irrigate"], normalize="index").round(3)
    L += ["## 3. Target design", "",
          "- `need_class` / `need_ord`: the dataset's own Low/Medium/High label (0/1/2). The dataset has NO "
          "water-amount column.",
          "- `water_amount_mm` (>= 0) and `irrigate` (1 if > 0): derived FAO-56 style. Irrigate when root-zone "
          "depletion exceeds the readily-available water (MAD); amount = depletion - effective rain over "
          f"{RAIN_WINDOW_DAYS} days, capped at {MAX_EVENT_MM:.0f} mm; amounts under {MIN_EVENT_MM:.0f} mm count as no irrigation. Soil_Moisture is read as % of "
          f"plant-available water; Rainfall_mm is read as a season total ({SEASON_DAYS} days).",
          "- Agreement with the native label (row = native class, column = derived irrigate flag):", "",
          agree.to_markdown(), "",
          f"- Derived irrigate share overall: {prim['irrigate'].mean():.1%}; mean positive amount "
          f"{prim.loc[prim['irrigate'] == 1, 'water_amount_mm'].mean():.1f} mm; min amount {prim['water_amount_mm'].min():.2f}.",
          ""]

    L += ["## 4. Edge-case rows (for 'edge cases / low temperature testing')", "",
          "`is_edge_case` and `edge_reasons` mark: low_temp (<=15 C), very_dry_soil, saturated_soil, heavy_rain, "
          "near_zero_rain, high_wind, extreme_vpd (top 5%), label_conflict (derived flag disagrees strongly with the "
          "native label). Counts:", ""]
    cnt = {c[5:]: int(prim[c].sum()) for c in prim.columns if c.startswith("edge_") and c != "edge_reasons"}
    L += ["| flag | rows |", "|---|---|"] + [f"| {k} | {v} |" for k, v in cnt.items()]
    L += ["", f"Total edge-case rows: {int(prim['is_edge_case'].sum())} "
              f"({int(((prim['is_edge_case']==1)&(prim['split']=='test')).sum())} in the test split). "
              "Negative outputs: the target is floored at 0; clip model predictions with max(0, pred).", ""]

    if tomato is not None:
        L += ["## 5. Tomato table", "", tomato.groupby("source").size().to_frame("rows").to_markdown(), "",
              "Only `json_real_sensor` has a source-provided irrigation amount and it is extremely imbalanced "
              f"({int((tomato['water_amount_mm_source'] > 0).sum())} positive rows). "
              "The csv rows are flagged where ET/pH values are implausible.", ""]
    if iot is not None:
        L += ["## 6. IoT sensor stream (Kalman)", "",
              f"Q={KALMAN_Q}, R={KALMAN_R}. Dryness = ADC/{IOT_ADC_DRY:.0f} (higher ADC = drier). Use this file to "
              "demonstrate noise handling; it has no crop/soil/stage columns so it is not a model-training set.", ""]
    L += ["## 7. Assumptions to state in the paper", "",
          "- FAO-56 Kc, root depth, MAD, field capacity and wilting point are textbook midpoints.",
          "- ET0 uses Penman-Monteith with assumed 12 h day length, sea-level pressure, 2 m wind and season-based Ra.",
          "- The primary dataset behaves like a rule-generated (synthetic) dataset: its label follows thresholds on "
          "soil moisture, rainfall, temperature, wind, growth stage and mulching. State this in the limitations.", ""]
    Path(out, "feature_report.md").write_text("\n".join(L), encoding="utf-8")

    desc = {
        "soil_moisture": "Soil moisture, % (read as % of plant-available water)",
        "temperature_c": "Air temperature, C", "humidity": "Relative humidity, %",
        "rainfall_mm": "Rainfall, mm (season total in source)", "stage_group": "ini/dev/mid/late bucket",
        "stage_ord": "0-3 ordinal of stage_group", "stage_water_sensitive": "1 if Vegetative or Flowering",
        "kc": "FAO-56 crop coefficient for crop x stage", "field_capacity": "Volumetric FC by soil texture",
        "wilting_point": "Volumetric WP by soil texture", "vpd_kpa": "Vapour pressure deficit, kPa",
        "et0_mm": "FAO-56 Penman-Monteith reference ET, mm/day", "etc_mm": "ET0 x Kc, mm/day",
        "rel_moisture": "soil_moisture / 100", "taw_mm": "Total available water in root zone, mm",
        "dr_mm": "Root-zone depletion, mm (LEAKS the derived target)", "raw_mm": "Readily available water, mm (leaks)",
        "below_mad": "1 if depletion > RAW (leaks)", "rain_per_day_mm": "rainfall_mm / season days (leaks)",
        "eff_rain_window_mm": "Effective rain over the window, mm (leaks)",
        "rain_covers_etc": "1 if effective daily rain >= ETc",
        "heat_stress_idx": "deg C above crop heat threshold (lower in flowering)",
        "cold_stress_idx": "deg C below crop base temperature", "dry_hot": "1 if rel_moisture<0.5 and T>30",
        "moisture_x_vpd": "(1 - rel_moisture) x VPD", "demand_minus_rain_mm": "ETc - effective daily rain",
        "water_amount_mm": "TARGET (derived): irrigation depth, mm, >= 0", "irrigate": "TARGET: 1 if amount > 0",
        "need_class": "Dataset's own label Low/Medium/High", "need_ord": "0/1/2 version of need_class",
        "is_edge_case": "1 if any edge flag", "edge_reasons": "Which edge flags fired",
        "split": "train/test (80/20 stratified by crop x native class, seed 42)",
        "meta_crop": "Crop id for grouping only - never a model input"}
    D = ["# Data dictionary (primary_features_full.csv / primary_model_ready.csv)", "",
         "| column | meaning |", "|---|---|"]
    for c in list(prim.columns):
        D.append(f"| {c} | {desc.get(c, '')} |")
    D += ["", "One-hot columns in primary_model_ready.csv are named <category>_<value> (crop_type_rice, ...)."]
    Path(out, "data_dictionary.md").write_text("\n".join(D), encoding="utf-8")
    Path(out, "feature_sets.json").write_text(json.dumps(sets, indent=2), encoding="utf-8")


# =========================================================================== #
# Main
# =========================================================================== #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--primary", required=True)
    ap.add_argument("--tomato-json"); ap.add_argument("--tomato-csv"); ap.add_argument("--iot")
    ap.add_argument("--hydraulic-raw"); ap.add_argument("--hydraulic-filtered")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--no-ablation", action="store_true")
    a = ap.parse_args()

    out = Path(a.outdir)
    (out / "primary" / "per_crop").mkdir(parents=True, exist_ok=True)
    log = []

    prim = build_primary(read_csv_any(a.primary), log)
    mr = primary_model_ready(prim)
    sets = model_ready_sets(mr)
    readable = prim.copy()
    readable.to_csv(out / "primary" / "primary_features_full.csv", index=False)
    mr.to_csv(out / "primary" / "primary_model_ready.csv", index=False)
    prim[prim["is_edge_case"] == 1].to_csv(out / "primary" / "edge_cases.csv", index=False)
    for crop, g in readable.groupby("crop_type"):
        g.to_csv(out / "primary" / "per_crop" / f"{crop}.csv", index=False)
        mr[mr["meta_crop"] == crop].to_csv(out / "primary" / "per_crop" / f"{crop}_model_ready.csv", index=False)

    tomato = None
    if a.tomato_json or a.tomato_csv:
        (out / "tomato").mkdir(exist_ok=True)
        tomato = build_tomato(a.tomato_json, a.tomato_csv, log)
        tomato.to_csv(out / "tomato" / "tomato_features.csv", index=False)
    iot = None
    if a.iot:
        (out / "sensor").mkdir(exist_ok=True)
        iot = build_iot(read_csv_any(a.iot), log)
        iot.to_csv(out / "sensor" / "iot_sensor_kalman.csv", index=False)
    if a.hydraulic_raw and a.hydraulic_filtered:
        profile_hydraulic(a.hydraulic_raw, a.hydraulic_filtered, log)

    ablation = None if a.no_ablation else run_ablation(prim, mr, sets, log)
    write_reports(out, prim, mr, sets, tomato, iot, ablation, log)

    print("\n".join(log))
    print(f"\nPrimary rows per crop:\n{prim['crop_type'].value_counts().to_string()}")
    print(f"\nDerived irrigate share: {prim['irrigate'].mean():.1%}")
    if ablation:
        print("\nAblation (native class, macro-F1):")
        for n, s in ablation["classification"]:
            print(f"  {s:.4f}  {n}")
    print(f"\nWrote everything to {out.resolve()}")


if __name__ == "__main__":
    main()
