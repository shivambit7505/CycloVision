"""
CycloVision AI - Canonical Meteorological Physics Engine
Consolidates empirical and thermodynamic formulas for cyclone estimation:
- Dvorak Atkinson-Holliday wind formulation
- Barometric central pressure deficit balance
- Logistic Rapid Intensification (RI) classifier
- IMD standard classification stages
"""

import math

def compute_dvorak_wind(t_number: float, sst: float = 29.5, vws: float = 8.0):
    """Calculates maximum sustained surface wind in knots and km/h."""
    wind_kts = 23.0 * (t_number ** 0.95)
    wind_kmph = wind_kts * 1.852
    wind_kmph += (sst - 28.0) * 4.0 - (vws - 10.0) * 1.5
    wind_kts = wind_kmph / 1.852
    return round(wind_kts, 1), round(wind_kmph, 1)

def compute_central_pressure(wind_kts: float) -> float:
    """Estimates central barometric pressure (hPa) using pressure-wind deficit balance."""
    central_pres = 1010.0 - 0.72 * ((wind_kts / 0.88) ** 1.15)
    return round(central_pres, 1)

def compute_ri_probability(sst: float, vws: float, rh_mid: float) -> float:
    """Evaluates thermodynamic Rapid Intensification probability."""
    logit = -3.2 + (0.28 * (sst - 26.5)) - (0.18 * (vws - 10.0)) + (0.04 * (rh_mid - 60.0))
    prob = 1.0 / (1.0 + math.exp(-logit))
    return round(max(0.05, min(0.95, prob)), 2)

def classify_imd_stage(wind_kmph: float):
    """Classifies wind speed into official IMD RSMC development categories."""
    if wind_kmph < 51:
        return 'Depression', 'D'
    elif wind_kmph < 62:
        return 'Deep Depression', 'DD'
    elif wind_kmph < 88:
        return 'Cyclonic Storm', 'CS'
    elif wind_kmph < 117:
        return 'Severe Cyclonic Storm', 'SCS'
    elif wind_kmph < 166:
        return 'Very Severe Cyclonic Storm', 'VSCS'
    elif wind_kmph < 221:
        return 'Extremely Severe Cyclonic Storm', 'ESCS'
    else:
        return 'Super Cyclonic Storm', 'SuCS'
