# Prosumer Digital Twin

Utilities for downloading and preparing external weather and solar-radiation data for PV/prosumer modelling.

The pipeline supports arbitrary locations and time periods. Main settings are defined in `pipeline_config.py`, including location, date range, weather models, requested variables, and panel geometry.

## Setup

Python 3.10+ is recommended.

```bash
python -m pip install -r requirements.txt
```

CAMS downloads also require a valid `.cdsapirc` configuration and acceptance of the relevant Copernicus dataset terms.

## Pipelines

### Historical data

Downloads historical Open-Meteo weather data and CAMS solar-radiation data.

```bash
python download_historical_openmeteo.py
python download_historical_cams.py
python build_10min_historical_datasets.py
```

### Historical Forecast

Downloads historical forecast data from Open-Meteo.

```bash
python download_historical_forecast_openmeteo.py
python build_10min_historical_forecast_datasets.py
```

### Previous Runs Day-1

Downloads Day-1 previous-run data from Open-Meteo.

```bash
python download_previous_runs_day1_openmeteo.py
python build_10min_previous_runs_day1_datasets.py
```

## Configuration

Edit `pipeline_config.py` before running a pipeline.

Typical settings include:

```python
LATITUDE = ...
LONGITUDE = ...
LOCATION = "site_name"

START_DATE = "YYYY-MM-DD"
END_DATE = "YYYY-MM-DD"
```

Command-line arguments can also override dates, models, and data directories where supported.

## Data

Downloaded and processed datasets are stored under:

```text
data/raw/
data/processed/
```

Actual data files are ignored by Git. Only `.gitkeep` placeholders are versioned.

## Notes

All processed timestamps use UTC.

Raw downloads are preserved at their native temporal resolution. Separate build scripts create aligned 10-minute datasets for downstream analysis and modelling.

## Sources

- Open-Meteo Historical Weather API
- Open-Meteo Historical Forecast API
- Open-Meteo Previous Runs API
- CAMS Solar Radiation Time-Series