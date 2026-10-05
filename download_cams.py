"""
Download and preprocess CAMS solar radiation data.

Source:
    CAMS Solar Radiation Time-Series (1-minute resolution)

Outputs:
    data/raw/cams/                       Original monthly CAMS files
    data/processed/cams_*_1min.csv       Clean 1-minute dataset

The processed dataset contains GHI, DHI and BNI irradiation in Wh/m²,
together with the CAMS reliability indicator. All timestamps use UTC.

Temporal aggregation is handled separately by build_10min_datasets.py.
"""

from pathlib import Path

import cdsapi
import pandas as pd


# ============================================================
# Configuration
# ============================================================

LATITUDE = 45.7742  # Sombor, Serbia
LONGITUDE = 19.1122

START_DATE = pd.Timestamp("2023-04-07")
END_DATE = pd.Timestamp("2026-03-31")

DATASET = "cams-solar-radiation-timeseries"

PROJECT_ROOT = Path(__file__).resolve().parent

RAW_CAMS_DIR = PROJECT_ROOT / "data" / "raw" / "cams"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_FILE = PROCESSED_DIR / "cams_sombor_1min.csv"


CAMS_COLUMNS = [
    "observation_period",
    "toa",
    "clear_sky_ghi",
    "clear_sky_bhi",
    "clear_sky_dhi",
    "clear_sky_bni",
    "ghi",
    "bhi",
    "dhi",
    "bni",
    "reliability",
]


# ============================================================
# Helpers
# ============================================================

def create_directories():
    RAW_CAMS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def get_monthly_periods():
    """
    Split the requested period into monthly chunks.

    Example:
        2023-04-07 -> 2023-04-30
        2023-05-01 -> 2023-05-31
        ...
        2026-03-01 -> 2026-03-31
    """

    periods = []

    current_start = START_DATE

    while current_start <= END_DATE:
        month_end = (
            current_start
            + pd.offsets.MonthEnd(0)
        )

        current_end = min(
            month_end,
            END_DATE,
        )

        periods.append(
            (
                current_start,
                current_end,
            )
        )

        current_start = current_end + pd.Timedelta(
            days=1
        )

    return periods


def download_month(
    client,
    start_date,
    end_date,
):
    """
    Download one CAMS monthly chunk.
    """

    filename = (
        f"cams_"
        f"{start_date.strftime('%Y-%m-%d')}_"
        f"{end_date.strftime('%Y-%m-%d')}.csv"
    )

    output_path = RAW_CAMS_DIR / filename

    if (
        output_path.exists()
        and output_path.stat().st_size > 0
    ):
        print(
            f"Already downloaded: {filename}"
        )

        return output_path

    request = {
        "sky_type": "observed_cloud",
        "location": {
            "longitude": LONGITUDE,
            "latitude": LATITUDE,
        },
        "altitude": ["-999"],
        "date": [
            (
                f"{start_date.strftime('%Y-%m-%d')}/"
                f"{end_date.strftime('%Y-%m-%d')}"
            )
        ],
        "time_step": "1minute",
        "time_reference": "universal_time",
        "data_format": "csv",
    }

    print()
    print(
        f"Downloading "
        f"{start_date.date()} -> "
        f"{end_date.date()}"
    )

    client.retrieve(
        DATASET,
        request,
    ).download(
        str(output_path)
    )

    print(
        f"Saved: {output_path}"
    )

    return output_path


def parse_cams_file(file_path):
    """
    Parse one original CAMS CSV file and keep only
    variables needed for the final dataset.
    """

    df = pd.read_csv(
        file_path,
        sep=";",
        comment="#",
        header=None,
        names=CAMS_COLUMNS,
    )

    df[
        [
            "timestamp_start",
            "timestamp_end",
        ]
    ] = (
        df["observation_period"]
        .str.split(
            "/",
            expand=True,
        )
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp_start"],
        utc=True,
    )

    df = df[
        [
            "timestamp",
            "ghi",
            "dhi",
            "bni",
            "reliability",
        ]
    ].copy()

    df = df.rename(
        columns={
            "ghi": "ghi_wh_m2",
            "dhi": "dhi_wh_m2",
            "bni": "bni_wh_m2",
        }
    )

    numeric_columns = [
        "ghi_wh_m2",
        "dhi_wh_m2",
        "bni_wh_m2",
        "reliability",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    return df


def validate_dataset(df):
    """
    Basic quality-control checks.
    """

    print()
    print("================================")
    print("Dataset validation")
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

    missing_values = (
        df.isna()
        .sum()
    )

    print()
    print("Missing values:")
    print(missing_values)

    expected_index = pd.date_range(
        start=START_DATE.tz_localize("UTC"),
        end=(
            END_DATE
            + pd.Timedelta(days=1)
            - pd.Timedelta(minutes=1)
        ).tz_localize("UTC"),
        freq="1min",
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
    print("Radiation totals:")

    print(
        df[
            [
                "ghi_wh_m2",
                "dhi_wh_m2",
                "bni_wh_m2",
            ]
        ].sum()
    )


# ============================================================
# Main
# ============================================================

def main():

    create_directories()

    client = cdsapi.Client()

    periods = get_monthly_periods()

    print(
        f"Number of CAMS requests: "
        f"{len(periods)}"
    )

    raw_files = []

    for start_date, end_date in periods:
        file_path = download_month(
            client=client,
            start_date=start_date,
            end_date=end_date,
        )

        raw_files.append(
            file_path
        )

    print()
    print("Parsing downloaded files...")

    dataframes = []

    for file_path in raw_files:
        print(
            f"Parsing: {file_path.name}"
        )

        monthly_df = parse_cams_file(
            file_path
        )

        dataframes.append(
            monthly_df
        )

    df = pd.concat(
        dataframes,
        ignore_index=True,
    )

    df = (
        df
        .sort_values("timestamp")
        .drop_duplicates(
            subset="timestamp",
            keep="first",
        )
        .reset_index(drop=True)
    )

    validate_dataset(
        df
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("================================")
    print("Finished")
    print("================================")

    print(
        f"Saved processed dataset:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()