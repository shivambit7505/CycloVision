# -*- coding: utf-8 -*-
"""
CycloVision AI V2 — Weather Service & Provider Abstraction
Provides pluggable weather integrations (Open-Meteo as default) with:
- Abstract Base Class WeatherProvider
- Timeout & Exponential Backoff Retry mechanism
- Coordinate & Parameter Validation
- In-Memory TTL Caching (5-minute window)
- Local SQLite Observation Persistence
- Resilient Error Handling & Local Cache Fallback
"""

import os
import time
import json
import logging
import requests
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timezone

from v2.db.database import save_weather_observation, get_latest_weather_observation

logger = logging.getLogger("cyclovision.weather")

CARDINAL_DIRECTIONS = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"
]

def degrees_to_cardinal(deg: Optional[float]) -> str:
    """Converts degrees [0, 360) into standard compass direction abbreviation."""
    if deg is None:
        return "N/A"
    idx = int((deg + 11.25) / 22.5) % 16
    return CARDINAL_DIRECTIONS[idx]

class WeatherProvider(ABC):
    """Abstract Base Class for pluggable weather providers."""

    @abstractmethod
    def get_current_weather(self, lat: float, lon: float) -> Dict[str, Any]:
        """Fetches current atmospheric & marine weather observation for given coordinates."""
        pass

    @abstractmethod
    def get_historical_weather(
        self, lat: float, lon: float, start_date: str, end_date: str
    ) -> Dict[str, Any]:
        """Fetches historical hourly weather observation archive for given coordinates and date range."""
        pass

class OpenMeteoWeatherProvider(WeatherProvider):
    """
    Open-Meteo Weather Provider (Default Hackathon Implementation).
    - Open access, no API key required
    - High-resolution WMO surface and atmospheric data
    """

    FORECAST_URL = os.getenv("OPEN_METEO_BASE_URL", "https://api.open-meteo.com/v1/forecast")
    ARCHIVE_URL = os.getenv("OPEN_METEO_ARCHIVE_URL", "https://archive-api.open-meteo.com/v1/archive")
    TIMEOUT_SECONDS = 6.0
    MAX_RETRIES = 3
    CACHE_TTL_SECONDS = 300  # 5 minutes

    def __init__(self):
        self._current_cache: Dict[Tuple[float, float], Tuple[float, Dict[str, Any]]] = {}
        self._archive_cache: Dict[Tuple[float, float, str, str], Tuple[float, Dict[str, Any]]] = {}

    def _validate_coordinates(self, lat: float, lon: float) -> Tuple[float, float]:
        if not (-90.0 <= lat <= 90.0):
            raise ValueError(f"Latitude must be between -90.0 and 90.0, got {lat}")
        if not (-180.0 <= lon <= 180.0):
            raise ValueError(f"Longitude must be between -180.0 and 180.0, got {lon}")
        return round(float(lat), 4), round(float(lon), 4)

    def _validate_date_string(self, date_str: str) -> str:
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            return dt.strftime("%Y-%m-%d")
        except Exception:
            raise ValueError(f"Invalid date format: '{date_str}'. Expected YYYY-MM-DD.")

    def _execute_with_retry(self, url: str, params: Dict[str, Any]) -> requests.Response:
        last_err = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                res = requests.get(url, params=params, timeout=self.TIMEOUT_SECONDS)
                if res.status_code == 200:
                    return res
                elif res.status_code in [429, 500, 502, 503, 504]:
                    logger.warning(f"Open-Meteo HTTP {res.status_code} on attempt {attempt}/{self.MAX_RETRIES}")
                    time.sleep(0.4 * (2 ** (attempt - 1)))
                else:
                    res.raise_for_status()
            except requests.RequestException as e:
                last_err = e
                logger.warning(f"Open-Meteo request error on attempt {attempt}/{self.MAX_RETRIES}: {e}")
                time.sleep(0.4 * (2 ** (attempt - 1)))

        raise RuntimeError(f"Open-Meteo request failed after {self.MAX_RETRIES} attempts: {last_err}")

    def get_current_weather(self, lat: float, lon: float) -> Dict[str, Any]:
        lat, lon = self._validate_coordinates(lat, lon)
        cache_key = (round(lat, 2), round(lon, 2))
        now = time.time()

        # 1. In-memory TTL Cache check
        if cache_key in self._current_cache:
            cache_time, cached_val = self._current_cache[cache_key]
            if now - cache_time < self.CACHE_TTL_SECONDS:
                result = dict(cached_val)
                result["is_cached"] = True
                result["cache_age_seconds"] = round(now - cache_time, 1)
                return result

        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,precipitation",
            "wind_speed_unit": "kmh"
        }

        try:
            res = self._execute_with_retry(self.FORECAST_URL, params)
            data = res.json()
            curr = data.get("current", {})

            temp = curr.get("temperature_2m")
            rh = curr.get("relative_humidity_2m")
            pres = curr.get("surface_pressure")
            wind_kmph = curr.get("wind_speed_10m")
            wind_kts = round(wind_kmph / 1.852, 1) if wind_kmph is not None else None
            wind_dir = curr.get("wind_direction_10m")
            precip = curr.get("precipitation", 0.0)
            obs_time = curr.get("time", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))

            result = {
                "status": "SUCCESS",
                "source": "Open-Meteo",
                "coordinates": {"latitude": lat, "longitude": lon},
                "timestamp": obs_time,
                "temperature": temp,
                "temperature_unit": "°C",
                "relative_humidity": rh,
                "relative_humidity_unit": "%",
                "surface_pressure": pres,
                "surface_pressure_unit": "hPa",
                "wind_speed": wind_kmph,
                "wind_speed_unit": "km/h",
                "wind_speed_knots": wind_kts,
                "wind_direction": wind_dir,
                "wind_direction_unit": "°",
                "wind_direction_cardinal": degrees_to_cardinal(wind_dir),
                "precipitation": precip,
                "precipitation_unit": "mm",
                "is_cached": False
            }

            # 2. Update In-memory cache
            self._current_cache[cache_key] = (now, result)

            # 3. Store in local SQLite observation ledger
            try:
                save_weather_observation(
                    latitude=lat,
                    longitude=lon,
                    observation_type="CURRENT",
                    timestamp=obs_time,
                    temperature=temp,
                    relative_humidity=rh,
                    surface_pressure=pres,
                    wind_speed=wind_kmph,
                    wind_direction=wind_dir,
                    precipitation=precip,
                    source="Open-Meteo",
                    raw_json=json.dumps(curr)
                )
            except Exception as dbe:
                logger.error(f"Failed to persist weather observation: {dbe}")

            return result

        except Exception as e:
            logger.warning(f"Live Open-Meteo query failed: {e}. Checking local database cache.")
            # Local SQLite fallback
            cached_db = get_latest_weather_observation(lat, lon, max_distance_deg=0.75)
            if cached_db:
                wind_kmph = cached_db.get("wind_speed")
                wind_kts = round(wind_kmph / 1.852, 1) if wind_kmph is not None else None
                wind_dir = cached_db.get("wind_direction")
                return {
                    "status": "LOCAL_CACHE_FALLBACK",
                    "source": "Open-Meteo (Local DB Cache)",
                    "coordinates": {"latitude": lat, "longitude": lon},
                    "timestamp": cached_db.get("timestamp"),
                    "temperature": cached_db.get("temperature"),
                    "temperature_unit": "°C",
                    "relative_humidity": cached_db.get("relative_humidity"),
                    "relative_humidity_unit": "%",
                    "surface_pressure": cached_db.get("surface_pressure"),
                    "surface_pressure_unit": "hPa",
                    "wind_speed": wind_kmph,
                    "wind_speed_unit": "km/h",
                    "wind_speed_knots": wind_kts,
                    "wind_direction": wind_dir,
                    "wind_direction_unit": "°",
                    "wind_direction_cardinal": degrees_to_cardinal(wind_dir),
                    "precipitation": cached_db.get("precipitation"),
                    "precipitation_unit": "mm",
                    "is_cached": True,
                    "note": f"Retrieved from local SQLite cache due to network event: {e}"
                }
            raise RuntimeError(f"Open-Meteo query failed and no local cache available for ({lat}, {lon}): {e}")

    def get_historical_weather(
        self, lat: float, lon: float, start_date: str, end_date: str
    ) -> Dict[str, Any]:
        lat, lon = self._validate_coordinates(lat, lon)
        start_date = self._validate_date_string(start_date)
        end_date = self._validate_date_string(end_date)
        cache_key = (round(lat, 2), round(lon, 2), start_date, end_date)
        now = time.time()

        # Check in-memory cache
        if cache_key in self._archive_cache:
            cache_time, cached_val = self._archive_cache[cache_key]
            if now - cache_time < 3600:  # 1-hour cache for static historical records
                result = dict(cached_val)
                result["is_cached"] = True
                return result

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,precipitation",
            "wind_speed_unit": "kmh"
        }

        res = self._execute_with_retry(self.ARCHIVE_URL, params)
        data = res.json()
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])

        records = []
        for i, t in enumerate(times):
            w_spd = hourly.get("wind_speed_10m", [])[i] if i < len(hourly.get("wind_speed_10m", [])) else None
            w_dir = hourly.get("wind_direction_10m", [])[i] if i < len(hourly.get("wind_direction_10m", [])) else None
            records.append({
                "time": t,
                "temperature_2m": hourly.get("temperature_2m", [])[i] if i < len(hourly.get("temperature_2m", [])) else None,
                "relative_humidity_2m": hourly.get("relative_humidity_2m", [])[i] if i < len(hourly.get("relative_humidity_2m", [])) else None,
                "surface_pressure": hourly.get("surface_pressure", [])[i] if i < len(hourly.get("surface_pressure", [])) else None,
                "wind_speed_kmph": w_spd,
                "wind_speed_knots": round(w_spd / 1.852, 1) if w_spd is not None else None,
                "wind_direction": w_dir,
                "wind_direction_cardinal": degrees_to_cardinal(w_dir),
                "precipitation_mm": hourly.get("precipitation", [])[i] if i < len(hourly.get("precipitation", [])) else None
            })

        result = {
            "status": "SUCCESS",
            "source": "Open-Meteo",
            "coordinates": {"latitude": lat, "longitude": lon},
            "time_range": {"start_date": start_date, "end_date": end_date},
            "total_hourly_records": len(records),
            "hourly_records": records,
            "is_cached": False
        }

        self._archive_cache[cache_key] = (now, result)

        # Store sample peak observation locally in SQLite
        if records:
            # Pick observation with maximum wind speed as significant event
            max_wind_rec = max(records, key=lambda x: (x["wind_speed_kmph"] or 0))
            try:
                save_weather_observation(
                    latitude=lat,
                    longitude=lon,
                    observation_type="HISTORICAL_PEAK",
                    timestamp=max_wind_rec["time"],
                    temperature=max_wind_rec["temperature_2m"],
                    relative_humidity=max_wind_rec["relative_humidity_2m"],
                    surface_pressure=max_wind_rec["surface_pressure"],
                    wind_speed=max_wind_rec["wind_speed_kmph"],
                    wind_direction=max_wind_rec["wind_direction"],
                    precipitation=max_wind_rec["precipitation_mm"],
                    source="Open-Meteo Archive",
                    raw_json=json.dumps(max_wind_rec)
                )
            except Exception as dbe:
                logger.error(f"Failed to persist historical peak weather observation: {dbe}")

        return result

# Default Weather Service Singleton
default_weather_provider: WeatherProvider = OpenMeteoWeatherProvider()

def get_weather_provider() -> WeatherProvider:
    """Factory getter to allow swapping the weather provider easily in future."""
    return default_weather_provider
