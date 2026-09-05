from datetime import datetime

class IMDBulletinDispatcher:
    @staticmethod
    def generate_bulletin(storm_data):
        now_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        return f"""================================================================================
INDIA METEOROLOGICAL DEPARTMENT (IMD) / CYCLOVISION AI EMERGENCY ADVISORY
SPECIAL TROPICAL CYCLONE BULLETIN NO.: {storm_data.get('bulletin_no', 'CV-01')}
TIME OF ISSUE: {now_utc}
================================================================================

1. CURRENT LOCATION & INTENSITY:
   THE SYSTEM IS CURRENTLY CENTERED AT LATITUDE {storm_data['current_lat']}°N AND 
   LONGITUDE {storm_data['current_lon']}°E OVER {storm_data['basin'].upper()}.
   
   - ESTIMATED DVORAK T-NUMBER: T{storm_data['dvorak_t_number']}
   - MAXIMUM SUSTAINED SURFACE WIND: {storm_data['wind_knots']} KNOTS ({round(storm_data['wind_knots'] * 1.852)} KMPH)
   - ESTIMATED CENTRAL PRESSURE: {storm_data['pressure_hpa']} HPA
   - CURRENT CLASSIFICATION: {storm_data['imd_category'].upper()}
   - ALERT LEVEL: {storm_data['alert_level']}

2. INTENSIFICATION DYNAMICS:
   - 24-HOUR RAPID INTENSIFICATION PROBABILITY: {round(storm_data['ri_risk']['ri_probability'] * 100, 1)}%
   - STATUS: {storm_data['ri_risk']['ri_alert']}

3. LANDFALL & TRAJECTORY FORECAST:
   - EXPECTED LANDFALL SECTOR: {storm_data['landfall']['coastal_sector']}
   - ESTIMATED TIME OF LANDFALL: {storm_data['landfall']['estimated_time']}
   - WIND AT LANDFALL: {storm_data['landfall']['estimated_wind_at_landfall_kmph']} KMPH
   - ESTIMATED STORM SURGE: {storm_data['landfall']['expected_surge_meters']} METERS

4. ACTIONABLE WARNINGS:
   (A) FISHERMEN ARE ADVISED NOT TO VENTURE INTO CENTRAL & NORTH BAY OF BENGAL.
   (B) HOIST LOCAL CAUTIONARY SIGNAL NO. 3 AT VISAKHAPATNAM AND GOPALPUR PORTS.
   (C) EVACUATION PLANNING RECOMMENDED FOR LOW-LYING AREAS IN COASTAL DISTRICTS.
================================================================================
"""
