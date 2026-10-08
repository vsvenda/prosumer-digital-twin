"""Shared defaults. All dates and timestamps in this project are UTC."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / 'data'
LATITUDE = 45.7742
LONGITUDE = 19.1122
LOCATION = 'sombor'
START_DATE = '2023-04-07'
END_DATE = '2026-03-31'
DAY1_START_DATE = '2024-01-20'  # First complete ICON day found by a live January-2024 probe.
TILT = 40.0
AZIMUTH = 25.0  # Open-Meteo: 0=south, positive=west.
HISTORICAL_MODELS = {'era5': 'era5', 'ifs': 'ecmwf_ifs'}
FORECAST_MODEL = 'icon_global'  # Same explicit model for both forecast sets.
WEATHER_VARIABLES = [
    'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'cloud_cover',
    'wind_speed_10m', 'wind_direction_10m', 'precipitation', 'snowfall', 'pressure_msl',
]
SOLAR_VARIABLES = [
    'shortwave_radiation', 'direct_radiation', 'diffuse_radiation',
    'direct_normal_irradiance', 'global_tilted_irradiance',
]
SOLAR_ENERGY_COLUMNS = {
    'shortwave_radiation': 'ghi_wh_m2',
    'direct_radiation': 'bhi_wh_m2',
    'diffuse_radiation': 'dhi_wh_m2',
    'direct_normal_irradiance': 'bni_wh_m2',
    'global_tilted_irradiance': 'gti_wh_m2',
}
ENDPOINTS = {
    'historical': 'https://archive-api.open-meteo.com/v1/archive',
    'historical_forecast': 'https://historical-forecast-api.open-meteo.com/v1/forecast',
    'previous_runs_day1': 'https://previous-runs-api.open-meteo.com/v1/forecast',
}
