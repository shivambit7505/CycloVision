# Structured Bulletin Generator (Slide 31)
def generate_imd_bulletin_v2(storm_name, stage, wind_kmph, pressure_hpa, landfall_sec, eta_hours):
    bulletin = f"""
================================================================================
                    INDIA METEOROLOGICAL DEPARTMENT (IMD)
                   TROPICAL CYCLONE ADVISORY BULLETIN (DRAFT)
================================================================================
NOTICE: AI-GENERATED DECISION-SUPPORT DRAFT — VERIFY AGAINST OFFICIAL RSMC GUIDANCE.

1. STORM IDENTIFICATION:
   - System Name: {storm_name}
   - Current Intensity Stage: {stage}
   - Maximum Sustained Surface Wind: {wind_kmph} km/h (Gusting up to {round(wind_kmph*1.15, 1)} km/h)
   - Estimated Central Pressure: {pressure_hpa} hPa

2. COASTAL IMPACT & LANDFALL HAZARD:
   - Projected Landfall Sector: {landfall_sec}
   - Estimated Time to Coastal Crossing: Next {eta_hours} Hours
   - Expected Storm Surge: 3.5 to 4.2 meters above astronomical tide

3. ACTION ADVISORY & WARNING:
   - Total suspension of fishing operations over deep-sea sectors.
   - Evacuation of low-lying coastal population to multi-purpose cyclone shelters.
   - State Disaster Management Authority (SDMA) Helpline: 1070

SOURCE PROVENANCE: Multi-spectral INSAT-3D/3DS + NOAA IBTrACS v4 + MERRA-2
================================================================================
"""
    return bulletin.strip()
