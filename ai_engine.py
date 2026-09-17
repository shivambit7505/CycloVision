import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime, timezone

class ObjectiveDvorakNetwork(nn.Module):
    def __init__(self):
        super(ObjectiveDvorakNetwork, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.fc_shared = nn.Sequential(
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        self.fc_t_number = nn.Linear(256, 1)
        self.fc_wind = nn.Linear(256, 1)
        self.fc_pressure = nn.Linear(256, 1)

    def forward(self, x):
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        shared = self.fc_shared(feat)
        return self.fc_t_number(shared), self.fc_wind(shared), self.fc_pressure(shared)

class PredictionResult:
    def __init__(self, value, unit, uncertainty, source, model_version='v2.1-production'):
        self.value = value
        self.unit = unit
        self.uncertainty = uncertainty
        self.timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
        self.source = source
        self.model_version = model_version

    def to_dict(self):
        return {
            'value': self.value,
            'unit': self.unit,
            'uncertainty': self.uncertainty,
            'timestamp': self.timestamp,
            'source': self.source,
            'model_version': self.model_version
        }

from v2.models.physics import compute_dvorak_wind, compute_central_pressure, compute_ri_probability, classify_imd_stage

def predict_cyclone_v2(t_number, sst=29.8, vws=7.5, rh_mid=78.0, lat=19.8, lon=85.8):
    wind_kts, wind_kmph = compute_dvorak_wind(t_number, sst, vws)
    central_pres = compute_central_pressure(wind_kts)
    ri_prob = compute_ri_probability(sst, vws, rh_mid)
    stage, stage_code = classify_imd_stage(wind_kmph)

    return {
        'intensity': PredictionResult(round(wind_kmph, 1), 'km/h', {'ci_90': [round(wind_kmph-12, 1), round(wind_kmph+12, 1)]}, 'NOAA IBTrACS / IMD RSMC').to_dict(),
        'pressure': PredictionResult(round(central_pres, 1), 'hPa', {'ci_90': [round(central_pres-5, 1), round(central_pres+5, 1)]}, 'Barometric Physics Balance').to_dict(),
        'stage': stage,
        'stage_code': stage_code,
        'rapid_intensification': {
            'probability': ri_prob,
            'status': 'HIGH' if ri_prob > 0.65 else 'MODERATE' if ri_prob > 0.35 else 'LOW',
            'features': {'sst': f'{sst}°C', 'vws': f'{vws} kts', 'rh': f'{rh_mid}%'}
        }
    }
