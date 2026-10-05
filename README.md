# Prosumer Digital Twin

This repository contains tools and models for developing a digital twin of an electricity prosumer with photovoltaic (PV) generation.

The digital twin combines historical prosumer measurements with external data sources such as solar irradiation and meteorological conditions. The goal is to create a data-driven representation of the prosumer that can be used for tasks such as PV generation forecasting, consumption forecasting, energy-flow modelling, and future energy-management applications.

The repository is organized as a modular pipeline. Raw data are first downloaded and preserved at their native temporal resolution. Separate processing scripts then transform and align the data into common time-series datasets suitable for analysis and machine-learning models.


## 1. Meteorological and Solar Radiation Data

Historical weather and solar radiation data are obtained from two external sources:

- **CAMS Solar Radiation Time-Series** — solar irradiation
- **Open-Meteo Historical Weather API** — meteorological variables

All timestamps are represented in UTC to provide a common time reference across data sources.


### 1.1 CAMS Solar Radiation Data

Solar radiation data are obtained from the Copernicus Atmosphere Monitoring Service (CAMS) Solar Radiation Time-Series dataset.

The following variables are retained:

| Variable | Description | Unit |
|---|---|---|
| GHI | Global Horizontal Irradiation | Wh/m² |
| DHI | Diffuse Horizontal Irradiation | Wh/m² |
| BNI | Beam Normal Irradiation | Wh/m² |
| Reliability | CAMS reliability / quality indicator | - |

The data are downloaded at **1-minute resolution**.

To make long downloads more robust, the requested period is divided into monthly chunks. Original CAMS responses are preserved under:

```text
data/raw/cams/
```

The monthly files are parsed, concatenated, sorted by timestamp, and reduced to the variables required by the project.

The resulting native-resolution dataset is stored as:

```text
data/processed/cams_*_1min.csv
```

Each timestamp represents the beginning of the corresponding 1-minute interval.

GHI, DHI, and BNI represent irradiation integrated over the time interval rather than instantaneous irradiance. Their units therefore remain `Wh/m²` when the data are subsequently aggregated to longer intervals.


### 1.2 Open-Meteo Weather Data

Historical meteorological data are obtained through the Open-Meteo Historical Weather API.

Two weather datasets are retained independently:

- **ERA5**
- **ECMWF IFS**

Keeping both datasets makes it possible to evaluate whether the choice of meteorological data source affects downstream forecasting performance.

The following hourly variables are used:

| Variable | Description |
|---|---|
| `temperature_2m` | Air temperature at 2 m |
| `relative_humidity_2m` | Relative humidity at 2 m |
| `dew_point_2m` | Dew-point temperature at 2 m |
| `cloud_cover` | Total cloud cover |
| `wind_speed_10m` | Wind speed at 10 m |
| `wind_direction_10m` | Wind direction at 10 m |
| `precipitation` | Precipitation amount |
| `snowfall` | Snowfall amount |
| `pressure_msl` | Mean sea-level pressure |

The data are downloaded at **1-hour resolution**.

The original Open-Meteo API responses are stored under:

```text
data/raw/openmeteo/
    era5/
    ifs/
```

Processed native-resolution datasets are stored as:

```text
data/processed/openmeteo_*_era5_1h.csv
data/processed/openmeteo_*_ifs_1h.csv
```

ERA5 and IFS are kept as separate datasets throughout the processing pipeline so that their effect on downstream forecasting performance can be evaluated independently.


### 1.3 Conversion to 10-Minute Resolution

CAMS and Open-Meteo have different native temporal resolutions. To create a common dataset for modelling, all variables are converted to a **10-minute temporal resolution**.

The processing depends on the physical meaning of each variable:

| Variable | Source | Native resolution | 10-minute processing |
|---|---|---:|---|
| GHI | CAMS | 1 min | Sum |
| DHI | CAMS | 1 min | Sum |
| BNI | CAMS | 1 min | Sum |
| Reliability | CAMS | 1 min | Mean |
| Temperature 2 m | Open-Meteo | 1 h | Linear interpolation |
| Relative humidity 2 m | Open-Meteo | 1 h | Linear interpolation |
| Dew point 2 m | Open-Meteo | 1 h | Linear interpolation |
| Cloud cover | Open-Meteo | 1 h | Linear interpolation |
| Pressure MSL | Open-Meteo | 1 h | Linear interpolation |
| Wind speed / direction 10 m | Open-Meteo | 1 h | Vector interpolation |
| Precipitation | Open-Meteo | 1 h | Hourly amount divided across six 10-minute intervals |
| Snowfall | Open-Meteo | 1 h | Hourly amount divided across six 10-minute intervals |

The transformation is performed by:

```text
build_10min_datasets.py
```


#### Solar Irradiation

CAMS GHI, DHI, and BNI are energy quantities integrated over each 1-minute interval. They are therefore **summed** when converting to 10-minute resolution:

```text
GHI_10min = sum(GHI_1min)
DHI_10min = sum(DHI_1min)
BNI_10min = sum(BNI_1min)
```

For example, a 10-minute GHI value at `12:00 UTC` represents the sum of the 1-minute irradiation values from `12:00` through `12:09`.

The resulting value represents the total irradiation over the interval:

```text
[12:00, 12:10)
```

The unit remains:

```text
Wh/m²
```

This aggregation preserves the total irradiation over the complete dataset.


#### Continuous Meteorological Variables

Temperature, relative humidity, dew point, cloud cover, and pressure are treated as continuous meteorological quantities.

Their hourly values are linearly interpolated to obtain values at 10-minute intervals.


#### Wind

Wind direction cannot be interpolated directly in degrees because it is a circular quantity.

For example, interpolating between:

```text
350° -> 10°
```

should pass through approximately:

```text
0°
```

rather than:

```text
180°
```

Wind speed and direction are therefore converted to horizontal vector components. These components are interpolated independently at 10-minute resolution and then converted back to wind speed and direction.


#### Precipitation and Snowfall

Precipitation and snowfall are provided as hourly accumulated quantities.

Each hourly amount is divided equally across the corresponding six 10-minute intervals.

For example:

```text
Hourly precipitation = 1.2 mm

10-minute values:
    0.2 mm
    0.2 mm
    0.2 mm
    0.2 mm
    0.2 mm
    0.2 mm
```

This preserves the total hourly precipitation:

```text
sum(precipitation_10min) = precipitation_1h
```

The same procedure is applied to snowfall.

This is a **temporal downscaling assumption**. It does not imply that precipitation or snowfall actually occurred uniformly throughout the hour. The original information source remains hourly.


### 1.4 Processed 10-Minute Datasets

The temporal conversion produces separate 10-minute datasets for CAMS, ERA5, and IFS:

```text
data/processed/cams_*_10min.csv

data/processed/openmeteo_*_era5_10min.csv
data/processed/openmeteo_*_ifs_10min.csv
```

Each dataset uses the same 10-minute UTC time grid.


### 1.5 Combined Meteorological Datasets

After temporal alignment, CAMS solar-radiation data are combined independently with the ERA5 and IFS meteorological data.

This produces two final datasets:

```text
data/processed/combined_*_era5_10min.csv
data/processed/combined_*_ifs_10min.csv
```

Each row represents one 10-minute interval and contains:

- CAMS solar irradiation
- air temperature
- relative humidity
- dew point
- cloud cover
- atmospheric pressure
- wind speed
- wind direction
- precipitation
- snowfall

The resulting structure is approximately:

```text
timestamp

ghi_wh_m2
dhi_wh_m2
bni_wh_m2
reliability

temperature_2m
relative_humidity_2m
dew_point_2m
cloud_cover
pressure_msl

wind_speed_10m
wind_direction_10m

precipitation
snowfall
```

Maintaining separate ERA5 and IFS datasets allows downstream models to be trained and evaluated using identical prosumer and solar-radiation data but different meteorological inputs.

This makes it possible to quantify the effect of the weather-data source on PV forecasting performance.


### 1.6 Data Processing Pipeline

The complete meteorological data pipeline is:

```text
CAMS API
    |
    | 1-minute solar irradiation
    v
download_cams.py
    |
    v
cams_*_1min.csv
    |
    |
    +-------------------------+
                              |
Open-Meteo API               |
    |                         |
    | hourly ERA5 + IFS       |
    v                         |
download_openmeteo.py         |
    |                         |
    +--> openmeteo_*_era5_1h.csv
    |                         |
    +--> openmeteo_*_ifs_1h.csv
                              |
                              v
                    build_10min_datasets.py
                              |
              +---------------+---------------+
              |               |               |
              v               v               v
        CAMS 10-min      ERA5 10-min      IFS 10-min
              |               |               |
              +-------+-------+-------+-------+
                      |               |
                      v               v
              CAMS + ERA5         CAMS + IFS
                 10-min              10-min
                      |               |
                      v               v
              combined_*_       combined_*_
              era5_10min.csv     ifs_10min.csv
```

These combined datasets provide the meteorological component of the prosumer digital twin and can subsequently be aligned with historical PV generation and electricity-consumption measurements.