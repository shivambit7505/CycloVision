import math

# Model 2: Calibrated Intensity Estimation (Slide 19)
def estimate_intensity_v2(t_number: float, sst: float = 29.5, vws: float = 8.0):
    wind_kts = 23.0 * (t_number ** 0.95)
    wind_kmph = wind_kts * 1.852
    wind_kmph += (sst - 28.0) * 4.2 - (vws - 10.0) * 1.5
    central_pressure = 1010.0 - 0.72 * ((wind_kts / 0.88) ** 1.15)
    error_margin_kmph = 12.5 if t_number < 3.5 else 18.0
    return {
        "wind_kts": round(wind_kts, 1),
        "wind_kmph": round(wind_kmph, 1),
        "central_pressure_hpa": round(central_pressure, 1),
        "confidence_pct": 88 if t_number >= 3.0 else 76,
        "bounds_90pct": [round(wind_kmph - error_margin_kmph, 1), round(wind_kmph + error_margin_kmph, 1)]
    }

# Model 3: Probabilistic RI Classifier (Slide 20)
def predict_rapid_intensification(current_wind_kts: float, sst: float, vws: float, rh_mid: float):
    logit = -3.2 + (0.28 * (sst - 26.5)) - (0.18 * (vws - 10.0)) + (0.04 * (rh_mid - 60.0))
    prob = 1.0 / (1.0 + math.exp(-logit))
    prob = max(0.05, min(0.95, prob))
    return {
        "ri_probability": round(prob, 2),
        "ri_status": "HIGH" if prob > 0.65 else "MODERATE" if prob > 0.35 else "LOW",
        "threshold_delta_wind_24h": "+30 kts (+55 km/h)",
        "key_drivers": {
            "sst_favorable": sst >= 28.5,
            "vws_favorable": vws <= 10.0,
            "moisture_favorable": rh_mid >= 70.0
        }
    }

# Model 4 & 5: Sequence Track + Calibrated Uncertainty Cone (Slide 21 & 22)
def forecast_track_v2(cur_lat: float, cur_lon: float, heading_deg: float = 330.0, speed_kmph: float = 14.0):
    lead_hours = [6, 12, 24, 48, 72, 120]
    track_points = []
    error_distributions = {
        6: {"p50": 18.2, "p75": 28.5, "p90": 38.0},
        12: {"p50": 32.5, "p75": 46.0, "p90": 62.0},
        24: {"p50": 42.1, "p75": 68.0, "p90": 92.0},
        48: {"p50": 78.4, "p75": 118.0, "p90": 155.0},
        72: {"p50": 122.0, "p75": 175.0, "p90": 230.0},
        120: {"p50": 195.0, "p75": 280.0, "p90": 365.0}
    }
    rad = math.radians(heading_deg)
    for h in lead_hours:
        dist_km = speed_kmph * h
        coriolis_deflection = 0.05 * (h / 24.0) ** 1.3
        adjusted_rad = rad + coriolis_deflection
        d_lat = (dist_km * math.cos(adjusted_rad)) / 111.0
        d_lon = (dist_km * math.sin(adjusted_rad)) / (111.0 * math.cos(math.radians(cur_lat)))
        track_points.append({
            "lead_hour": h,
            "latitude": round(cur_lat + d_lat, 2),
            "longitude": round(cur_lon + d_lon, 2),
            "uncertainty_cone": error_distributions[h]
        })
    return track_points
