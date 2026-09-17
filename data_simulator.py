import numpy as np

class SatelliteDataSimulator:
    def __init__(self):
        self.grid_size = (256, 256)

    def generate_cyclone_frame(self, center=(128, 128), intensity=4.5, eye_radius=12):
        """
        Synthesizes a 4-channel satellite tensor:
        Channel 0: Thermal Infrared 1 (TIR-1 in Kelvin [190K - 300K])
        Channel 1: Water Vapor (WV 6.8um)
        Channel 2: Visible / Cloud Albedo (0.0 to 1.0)
        Channel 3: Scatterometer Surface Wind Field (knots)
        """
        h, w = self.grid_size
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)

        tir1 = np.ones((h, w), dtype=np.float32) * 285.0
        theta = np.arctan2(y - center[1], x - center[0])
        spiral = np.sin(2.5 * np.log(dist_from_center + 1e-5) - theta * 2)

        eyewall_mask = (dist_from_center >= eye_radius) & (dist_from_center <= eye_radius + 35)
        tir1[eyewall_mask] = 205.0 - (intensity * 3.5) + (spiral[eyewall_mask] * 12.0)

        eye_mask = dist_from_center < eye_radius
        tir1[eye_mask] = 245.0 + (intensity * 2.0)

        outer_mask = (dist_from_center > eye_radius + 35) & (dist_from_center < 110)
        tir1[outer_mask] = 250.0 - (spiral[outer_mask] * 20.0)

        noise = np.random.normal(0, 1.5, (h, w))
        tir1 = np.clip(tir1 + noise, 185.0, 310.0)

        wv = 260.0 - (285.0 - tir1) * 0.75 + np.random.normal(0, 1.0, (h, w))
        wv = np.clip(wv, 200.0, 275.0)

        vis = np.clip((300.0 - tir1) / 110.0, 0.0, 1.0)

        v_max = intensity * 18.0 + 15.0
        wind_field = np.zeros((h, w), dtype=np.float32)
        inner = dist_from_center < eye_radius
        outer = ~inner
        wind_field[inner] = v_max * (dist_from_center[inner] / max(eye_radius, 1))
        wind_field[outer] = v_max * np.sqrt(eye_radius / (dist_from_center[outer] + 1e-5))
        wind_field = np.clip(wind_field, 0.0, 180.0)

        return {
            "tir1": tir1,
            "wv": wv,
            "vis": vis,
            "wind": wind_field,
            "tensor": np.stack([tir1, wv, vis, wind_field], axis=0)
        }

    def get_environmental_parameters(self, basin="Bay of Bengal"):
        if basin == "Bay of Bengal":
            return {
                "sst_celsius": 30.4,
                "ocean_heat_content_kj_cm2": 95.0,
                "vertical_wind_shear_knots": 7.5,
                "relative_humidity_700hpa": 84.0,
                "steering_flow_direction_deg": 315.0,
                "steering_flow_speed_kmph": 14.5
            }
        else:
            return {
                "sst_celsius": 29.2,
                "ocean_heat_content_kj_cm2": 72.0,
                "vertical_wind_shear_knots": 11.2,
                "relative_humidity_700hpa": 76.0,
                "steering_flow_direction_deg": 340.0,
                "steering_flow_speed_kmph": 11.0
            }
