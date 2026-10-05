"""
Download and preprocess historical weather data from Open-Meteo.

Sources:
    ERA5 (hourly)
    ECMWF IFS (hourly)

Outputs:
    data/raw/openmeteo/                  Original API responses
    data/processed/openmeteo_*_era5_1h.csv
    data/processed/openmeteo_*_ifs_1h.csv

ERA5 and IFS are kept as separate datasets for later comparison.
All timestamps use UTC.

Temporal interpolation is handled separately by build_10min_datasets.py.
"""

from pathlib import Path
import json

import pandas as pd
import requests


# ============================================================
# Configuration
# ============================================================

LATITUDE = 45.7742
LONGITUDE = 19.1122

START_DATE = "2023-04-07"
END_DATE = "2026-03-31"

URL = "https://archive-api.open-meteo.com/v1/archive"

PROJECT_ROOT = Path(__file__).resolve().parent

RAW_OPENMETEO_DIR = PROJECT_ROOT / "data" / "raw" / "openmeteo"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "snowfall",
    "pressure_msl",
]

MODELS = {
    "era5": "era5",
    "ifs": "ecmwf_ifs",
}


# ============================================================
# Helpers
# ============================================================

def create_directories():
    for model_name in MODELS:
        (
            RAW_OPENMETEO_DIR / model_name
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def download_model(
    model_name,
    model_api_name,
):
    """
    Download one Open-Meteo model.
    """

    raw_dir = (
        RAW_OPENMETEO_DIR / model_name
    )

    raw_file = (
        raw_dir
        / f"openmeteo_sombor_{model_name}_raw.json"
    )

    if raw_file.exists():
        print(
            f"Using existing raw file: "
            f"{raw_file.name}"
        )

        with open(
            raw_file,
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        return data

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(
            HOURLY_VARIABLES
        ),
        "timezone": "UTC",
        "models": model_api_name,
    }

    print()
    print(
        f"Downloading {model_name.upper()}..."
    )

    response = requests.get(
        URL,
        params=params,
        timeout=120,
    )

    response.raise_for_status()

    data = response.json()

    with open(
        raw_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
        )

    print(
        f"Saved raw data: {raw_file}"
    )

    return data


def parse_model(
    data,
    model_name,
):
    """
    Convert Open-Meteo JSON response to a clean DataFrame.
    """

    df = pd.DataFrame(
        data["hourly"]
    )

    df["time"] = pd.to_datetime(
        df["time"],
        utc=True,
    )

    df = df.rename(
        columns={
            "time": "timestamp",
        }
    )

    expected_columns = [
        "timestamp",
        *HOURLY_VARIABLES,
    ]

    df = df[
        expected_columns
    ].copy()

    df = (
        df
        .sort_values("timestamp")
        .drop_duplicates(
            subset="timestamp",
            keep="first",
        )
        .reset_index(drop=True)
    )

    output_file = (
        PROCESSED_DIR
        / f"openmeteo_sombor_{model_name}_1h.csv"
    )

    df.to_csv(
        output_file,
        index=False,
    )

    print(
        f"Saved processed data: "
        f"{output_file}"
    )

    return df


def validate_dataset(
    df,
    model_name,
):
    """
    Basic validation of hourly Open-Meteo data.
    """

    print()
    print("================================")
    print(
        f"{model_name.upper()} validation"
    )
    print("================================")

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Start: {df['timestamp'].min()}"
    )

    print(
        f"End:   {df['timestamp'].max()}"
    )

    duplicate_count = (
        df["timestamp"]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate timestamps: "
        f"{duplicate_count:,}"
    )

    print()
    print("Missing values:")

    print(
        df.isna().sum()
    )

    expected_index = pd.date_range(
        start=pd.Timestamp(
            START_DATE,
            tz="UTC",
        ),
        end=(
            pd.Timestamp(
                END_DATE,
                tz="UTC",
            )
            + pd.Timedelta(days=1)
            - pd.Timedelta(hours=1)
        ),
        freq="1h",
    )

    actual_index = pd.DatetimeIndex(
        df["timestamp"]
    )

    missing_timestamps = (
        expected_index
        .difference(actual_index)
    )

    print()

    print(
        f"Expected rows: "
        f"{len(expected_index):,}"
    )

    print(
        f"Missing timestamps: "
        f"{len(missing_timestamps):,}"
    )

    if len(missing_timestamps) > 0:
        print()
        print(
            "First missing timestamps:"
        )

        print(
            missing_timestamps[:20]
        )

    print()
    print("Summary:")

    print(
        df.describe(
            include="all"
        )
    )


# ============================================================
# Main
# ============================================================

def main():

    create_directories()

    datasets = {}

    for (
        model_name,
        model_api_name,
    ) in MODELS.items():

        data = download_model(
            model_name=model_name,
            model_api_name=model_api_name,
        )

        df = parse_model(
            data=data,
            model_name=model_name,
        )

        validate_dataset(
            df=df,
            model_name=model_name,
        )

        datasets[
            model_name
        ] = df

    print()
    print("================================")
    print("Finished")
    print("================================")

    print()

    for model_name in MODELS:
        print(
            PROCESSED_DIR
            / f"openmeteo_sombor_{model_name}_1h.csv"
        )


if __name__ == "__main__":
    main()