# Data dictionary (primary_features_full.csv / primary_model_ready.csv)

| column | meaning |
|---|---|
| soil_type |  |
| soil_ph |  |
| soil_moisture | Soil moisture, % (read as % of plant-available water) |
| organic_carbon |  |
| electrical_conductivity |  |
| temperature_c | Air temperature, C |
| humidity | Relative humidity, % |
| rainfall_mm | Rainfall, mm (season total in source) |
| sunlight_hours |  |
| wind_speed_kmh |  |
| crop_type |  |
| growth_stage |  |
| season |  |
| irrigation_type |  |
| water_source |  |
| field_area_hectare |  |
| mulching_used |  |
| previous_irrigation_mm |  |
| region |  |
| irrigation_need |  |
| stage_group | ini/dev/mid/late bucket |
| stage_ord | 0-3 ordinal of stage_group |
| stage_water_sensitive | 1 if Vegetative or Flowering |
| kc | FAO-56 crop coefficient for crop x stage |
| field_capacity | Volumetric FC by soil texture |
| wilting_point | Volumetric WP by soil texture |
| vpd_kpa | Vapour pressure deficit, kPa |
| et0_mm | FAO-56 Penman-Monteith reference ET, mm/day |
| etc_mm | ET0 x Kc, mm/day |
| rel_moisture | soil_moisture / 100 |
| taw_mm | Total available water in root zone, mm |
| dr_mm | Root-zone depletion, mm (LEAKS the derived target) |
| raw_mm | Readily available water, mm (leaks) |
| below_mad | 1 if depletion > RAW (leaks) |
| rain_per_day_mm | rainfall_mm / season days (leaks) |
| eff_rain_window_mm | Effective rain over the window, mm (leaks) |
| rain_covers_etc | 1 if effective daily rain >= ETc |
| heat_stress_idx | deg C above crop heat threshold (lower in flowering) |
| cold_stress_idx | deg C below crop base temperature |
| dry_hot | 1 if rel_moisture<0.5 and T>30 |
| moisture_x_vpd | (1 - rel_moisture) x VPD |
| demand_minus_rain_mm | ETc - effective daily rain |
| water_amount_mm | TARGET (derived): irrigation depth, mm, >= 0 |
| irrigate | TARGET: 1 if amount > 0 |
| need_class | Dataset's own label Low/Medium/High |
| need_ord | 0/1/2 version of need_class |
| split | train/test (80/20 stratified by crop x native class, seed 42) |
| edge_low_temp |  |
| edge_very_dry_soil |  |
| edge_saturated_soil |  |
| edge_heavy_rain |  |
| edge_near_zero_rain |  |
| edge_high_wind |  |
| edge_extreme_vpd |  |
| edge_label_conflict |  |
| is_edge_case | 1 if any edge flag |
| edge_reasons | Which edge flags fired |

One-hot columns in primary_model_ready.csv are named <category>_<value> (crop_type_rice, ...).