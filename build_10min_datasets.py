"""
Build temporally aligned 10-minute meteorological datasets.

Inputs:
    CAMS solar radiation data (1-minute)
    Open-Meteo ERA5 weather data (hourly)
    Open-Meteo IFS weather data (hourly)

Outputs:
    CAMS 10-minute dataset
    ERA5 10-minute dataset
    IFS 10-minute dataset
    Combined CAMS + ERA5 10-minute dataset
    Combined CAMS + IFS 10-minute dataset

CAMS irradiation is aggregated by summation. Open-Meteo variables are
interpolated or temporally distributed according to variable type.
All datasets are aligned using UTC timestamps.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

START_TIME = pd.Timestamp(
    "2023-04-07 00:00:00",
    tz="UTC",
)

END_TIME = pd.Timestamp(
    "2026-03-31 23:50:00",
    tz="UTC",
)

CAMS_1MIN_FILE = (
    PROCESSED_DIR
    / "cams_sombor_1min.csv"
)

OPENMETEO_FILES = {
    "era5": (
        PROCESSED_DIR
        / "openmeteo_sombor_era5_1h.csv"
    ),
    "ifs": (
        PROCESSED_DIR
        / "openmeteo_sombor_ifs_1h.csv"
    ),
}


# ============================================================
# CAMS
# ============================================================

def build_cams_10min():
    """
    Aggregate CAMS 1-minute irradiation data to 10-minute resolution.

    GHI, DHI and BNI are summed because they represent energy
    accumulated over each 1-minute interval.

    Reliability is aggregated using the arithmetic mean.
    """

    print()
    print("Building CAMS 10-minute dataset...")

    df = pd.read_csv(
        CAMS_1MIN_FILE,
        parse_dates=["timestamp"],
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    df = (
        df
        .sort_values("timestamp")
        .drop_duplicates(
            subset="timestamp",
            keep="first",
        )
    )

    cams_10min = (
        df
        .set_index("timestamp")
        .resample("10min")
        .agg(
            {
                "ghi_wh_m2": "sum",
                "dhi_wh_m2": "sum",
                "bni_wh_m2": "sum",
                "reliability": "mean",
            }
        )
        .reset_index()
    )

    output_file = (
        PROCESSED_DIR
        / "cams_sombor_10min.csv"
    )

    cams_10min.to_csv(
        output_file,
        index=False,
    )

    print(
        f"Saved: {output_file}"
    )

    return cams_10min


# ============================================================
# Open-Meteo loading
# ============================================================

def load_hourly_weather(file_path):
    """
    Load an hourly Open-Meteo dataset.
    """

    df = pd.read_csv(
        file_path,
        parse_dates=["timestamp"],
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    df = (
        df
        .sort_values("timestamp")
        .drop_duplicates(
            subset="timestamp",
            keep="first",
        )
        .set_index("timestamp")
    )

    return df


# ============================================================
# Continuous variables
# ============================================================

def interpolate_continuous_variables(
    hourly_df,
    index_10min,
):
    """
    Interpolate instantaneous meteorological variables
    from hourly to 10-minute resolution.
    """

    variables = [
        "temperature_2m",
        "relative_humidity_2m",
        "dew_point_2m",
        "cloud_cover",
        "pressure_msl",
    ]

    result = pd.DataFrame(
        index=index_10min
    )

    expanded_index = (
        hourly_df.index
        .union(index_10min)
        .sort_values()
    )

    for variable in variables:

        series = (
            hourly_df[variable]
            .reindex(expanded_index)
            .interpolate(
                method="time"
            )
            .ffill()
            .bfill()
        )

        result[variable] = (
            series.reindex(
                index_10min
            )
        )

    return result


# ============================================================
# Wind
# ============================================================

def interpolate_wind(
    hourly_df,
    index_10min,
):
    """
    Interpolate wind speed and direction using vector components.

    Direct interpolation of wind direction in degrees is incorrect
    because direction is circular.

    For example:
        350° -> 10°

    should pass through 0°, not through 180°.
    """

    speed = (
        hourly_df["wind_speed_10m"]
        .astype(float)
    )

    direction_rad = np.deg2rad(
        hourly_df[
            "wind_direction_10m"
        ].astype(float)
    )

    # Meteorological convention:
    # wind direction indicates where the wind comes FROM.
    u = (
        -speed
        * np.sin(direction_rad)
    )

    v = (
        -speed
        * np.cos(direction_rad)
    )

    expanded_index = (
        hourly_df.index
        .union(index_10min)
        .sort_values()
    )

    u_10min = (
        u
        .reindex(expanded_index)
        .interpolate(
            method="time"
        )
        .ffill()
        .bfill()
        .reindex(index_10min)
    )

    v_10min = (
        v
        .reindex(expanded_index)
        .interpolate(
            method="time"
        )
        .ffill()
        .bfill()
        .reindex(index_10min)
    )

    speed_10min = np.sqrt(
        u_10min ** 2
        + v_10min ** 2
    )

    direction_10min = (
        np.degrees(
            np.arctan2(
                -u_10min,
                -v_10min,
            )
        )
        % 360
    )

    result = pd.DataFrame(
        {
            "wind_speed_10m":
                speed_10min,

            "wind_direction_10m":
                direction_10min,
        },
        index=index_10min,
    )

    return result


# ============================================================
# Accumulated variables
# ============================================================

def downscale_accumulated_variables(
    hourly_df,
    index_10min,
):
    """
    Convert hourly precipitation and snowfall to 10-minute values.

    Open-Meteo provides these variables as hourly accumulated amounts.

    Each hourly amount is divided equally across the six 10-minute
    intervals belonging to that hour.

    This is a temporal downscaling assumption and should not be
    interpreted as an actual 10-minute observation.
    """

    result = pd.DataFrame(
        index=index_10min
    )

    for variable in [
        "precipitation",
        "snowfall",
    ]:

        hourly_values = (
            hourly_df[variable]
            .astype(float)
        )

        values_10min = (
            hourly_values
            .reindex(index_10min)
            .ffill()
            / 6.0
        )

        result[variable] = (
            values_10min
        )

    return result


# ============================================================
# Open-Meteo 10-minute dataset
# ============================================================

def build_openmeteo_10min(
    hourly_df,
):
    """
    Build a complete 10-minute Open-Meteo dataset.
    """

    index_10min = pd.date_range(
        start=START_TIME,
        end=END_TIME,
        freq="10min",
    )

    continuous = (
        interpolate_continuous_variables(
            hourly_df,
            index_10min,
        )
    )

    wind = (
        interpolate_wind(
            hourly_df,
            index_10min,
        )
    )

    accumulated = (
        downscale_accumulated_variables(
            hourly_df,
            index_10min,
        )
    )

    df = pd.concat(
        [
            continuous,
            wind,
            accumulated,
        ],
        axis=1,
    )

    df = (
        df
        .reset_index()
        .rename(
            columns={
                "index": "timestamp"
            }
        )
    )

    return df


# ============================================================
# Validation
# ============================================================

def validate_dataset(
    df,
    name,
):
    """
    Basic validation of a 10-minute dataset.
    """

    print()
    print("================================")
    print(f"{name} validation")
    print("================================")

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Start: "
        f"{df['timestamp'].min()}"
    )

    print(
        f"End:   "
        f"{df['timestamp'].max()}"
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
        start=START_TIME,
        end=END_TIME,
        freq="10min",
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


# ============================================================
# Combine CAMS + Open-Meteo
# ============================================================

def combine_with_cams(
    cams,
    weather,
):
    """
    Merge CAMS and Open-Meteo data on the common UTC timestamp.
    """

    combined = pd.merge(
        cams,
        weather,
        on="timestamp",
        how="inner",
        validate="one_to_one",
    )

    return combined


# ============================================================
# Main
# ============================================================

def main():

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # CAMS
    # --------------------------------------------------------

    cams_10min = (
        build_cams_10min()
    )

    validate_dataset(
        cams_10min,
        "CAMS",
    )

    # --------------------------------------------------------
    # Open-Meteo models
    # --------------------------------------------------------

    for (
        model_name,
        file_path,
    ) in OPENMETEO_FILES.items():

        print()
        print(
            f"Processing "
            f"{model_name.upper()}..."
        )

        hourly_df = (
            load_hourly_weather(
                file_path
            )
        )

        weather_10min = (
            build_openmeteo_10min(
                hourly_df
            )
        )

        weather_output = (
            PROCESSED_DIR
            / (
                f"openmeteo_sombor_"
                f"{model_name}_10min.csv"
            )
        )

        weather_10min.to_csv(
            weather_output,
            index=False,
        )

        validate_dataset(
            weather_10min,
            f"Open-Meteo "
            f"{model_name.upper()}",
        )

        # ----------------------------------------------------
        # Combined dataset
        # ----------------------------------------------------

        combined = (
            combine_with_cams(
                cams_10min,
                weather_10min,
            )
        )

        combined_output = (
            PROCESSED_DIR
            / (
                f"combined_sombor_"
                f"{model_name}_10min.csv"
            )
        )

        combined.to_csv(
            combined_output,
            index=False,
        )

        validate_dataset(
            combined,
            f"Combined "
            f"{model_name.upper()}",
        )

        print()
        print("Saved:")

        print(
            weather_output
        )

        print(
            combined_output
        )

    print()
    print("================================")
    print("Finished")
    print("================================")


if __name__ == "__main__":
    main()