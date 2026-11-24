# Prosumer Digital Twin

Python tools for preparing photovoltaic (PV) production, weather and solar radiation datasets, and evaluating physical PV models under different training-data availability scenarios.

## Setup

Python 3.10+ is recommended. Run all commands from the repository root.

```bash
python -m pip install -r requirements.txt
```

CAMS downloads require a configured `.cdsapirc` file and acceptance of the relevant dataset terms in the Copernicus Atmosphere Data Store (ADS).

## 1. PV data preprocessing

The repository includes dataset-specific preprocessing scripts (`*-dataset-preprocessing.py`) that can be used as starting points for integrating new PV datasets. They demonstrate:

- Extracting PV power measurements from CSV files and ZIP archives.
- Converting cumulative energy (kWh) to average power (W).
- Selecting individual PV systems and aligning their timestamps.
- Handling missing or interpolated measurements.

Adapt the input format, column names, source time zone and sampling resolution to your dataset. Store the resulting PV power series in `data/solar/<location_start_end>/`, using UTC timestamps and power in W. Preserve missing measurements as NaN rather than replacing them with zeros.

## 2. Weather and solar radiation data

The existing download and build scripts are reusable across locations without code changes, subject to data availability from the selected providers.

Set the following parameters in `pipeline_config.py`:

| Parameter | Description |
|---|---|
| `LOCATION` | Dataset/location identifier |
| `LATITUDE`, `LONGITUDE` | Geographic coordinates |
| `START_DATE`, `END_DATE` | Inclusive UTC date range |
| `HISTORICAL_MODELS`, `FORECAST_MODEL` | Weather model selection |
| `DAY1_START_DATE` | Start date for day-1 previous runs |
| `TILT`, `AZIMUTH` | Panel geometry for forecast pipelines (Open-Meteo convention: 0° south, positive west) |

### Historical data

Download hourly weather data from Open-Meteo (ERA5 or IFS) and 1-minute observed-cloud solar irradiation from CAMS (GHI, DHI, BNI and reliability). Build aligned 10-minute datasets:

```bash
python download_historical_openmeteo.py --models era5
python download_historical_cams.py
python build_10min_historical_datasets.py --models era5
```

### Historical forecasts

Download archived weather and solar radiation forecasts from Open-Meteo (ICON Global by default).

```bash
python download_historical_forecast_openmeteo.py
python build_10min_historical_forecast_datasets.py
```

### Previous runs (day-1)

Download day-1 weather and solar radiation forecasts from Open-Meteo (ICON Global by default).

```bash
python download_previous_runs_day1_openmeteo.py
python build_10min_previous_runs_day1_datasets.py
```

Use `--help` to inspect available command-line options. Native-resolution downloads are preserved, and the build scripts generate 10-minute datasets and quality reports. Open-Meteo download and build scripts enforce completeness checks by default; `--allow-missing` explicitly permits incomplete outputs.

## 3. Data organization

```text
data/
  raw/<pipeline>/<location_start_end>/
  processed/<pipeline>/<location_start_end>/
    cams/               # Historical only: 1-min and 10-min irradiation
    openmeteo/<model>/  # Hourly and 10-min weather/forecast data
    combined/           # Historical CAMS + Open-Meteo, 10 min
  solar/<location_start_end>/  # Measured PV power
results/pvlib/<site>/<system>/
```

Data conventions:

- Standard pipeline timestamps use UTC; 10-minute intervals are labelled by their start time.
- PV power is expressed in W; solar irradiation in Wh/m² per interval. Multiply 10-minute irradiation by 6 to obtain mean irradiance in W/m².
- CAMS irradiation is summed over complete intervals. Hourly instantaneous weather variables are interpolated (historical and historical forecast) or held from the preceding hour (day-1).
- Wind speed is stored in km/h. Generated datasets and experiment results are excluded from Git.

## 4. Calibrated pvlib baseline

`pvlib_calibrated.py` implements a PVWatts baseline for comparing physical and data-driven PV models using different input variables and training durations (1/2 weeks and 1/3/6/12 months).

It supports partial and full calibration of panel tilt, azimuth and effective DC capacity, using predefined training/test splits. The current implementation uses dataset-specific site definitions, input column conventions and a fixed UTC+1 input clock. Adapt the site configuration, data loader, time handling and coverage criteria when integrating a new dataset.

```bash
python -m pip install scipy pvlib
python pvlib_calibrated.py --help
```

The baseline expects a 5-minute combined PV/weather dataset (`output/solar_production_5min.csv.gz`) and split definitions (`training_periods/<system>/periods.csv`) in a site archive or directory. Outputs include `experiments.csv` (calibrated parameters and metrics such as MAE, nMAE, RMSE and R²) and `training_features.csv` (training-period descriptors).

## Data sources

- [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
- [Open-Meteo Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api)
- [Open-Meteo Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api)
- [CAMS Solar Radiation Time-Series](https://ads.atmosphere.copernicus.eu/datasets/cams-solar-radiation-timeseries)
- [pvlib Python](https://pvlib-python.readthedocs.io/)
