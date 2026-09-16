import math
from datetime import datetime, timedelta

class CycloneDataAlignmentPipeline:
    def __init__(self, tolerance_minutes=30):
        self.tolerance_minutes = tolerance_minutes

    def align_observation(self, storm_id, obs_time, best_track_lat, best_track_lon, satellite_timestamp):
        t_obs = datetime.strptime(obs_time, '%Y-%m-%d %H:%M')
        t_sat = datetime.strptime(satellite_timestamp, '%Y-%m-%d %H:%M')
        delta_mins = abs((t_sat - t_obs).total_seconds()) / 60.0
        
        is_aligned = delta_mins <= self.tolerance_minutes
        
        # 256x256 storm-centered bounding box crop (Slide 16)
        crop_bbox = {
            'lat_min': round(best_track_lat - 2.5, 2),
            'lat_max': round(best_track_lat + 2.5, 2),
            'lon_min': round(best_track_lon - 2.5, 2),
            'lon_max': round(best_track_lon + 2.5, 2),
            'resolution_deg': 0.04
        }
        
        return {
            'storm_id': storm_id,
            'is_aligned': is_aligned,
            'time_delta_minutes': round(delta_mins, 1),
            'crop_center': {'lat': best_track_lat, 'lon': best_track_lon},
            'bounding_box': crop_bbox,
            'qc_status': 'PASSED_QC' if is_aligned else 'EXCEEDS_TOLERANCE'
        }
