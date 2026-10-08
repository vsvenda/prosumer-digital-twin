"""
Download and preprocess CAMS solar radiation data.

Source:
    CAMS Solar Radiation Time-Series (1-minute resolution)

Outputs:
    data/raw/historical/<location_start_end>/cams/
        Original monthly CAMS files and request metadata.

    data/processed/historical/<location_start_end>/cams/
        Clean combined 1-minute CAMS dataset.

The processed dataset contains GHI, DHI and BNI irradiation in Wh/m²,
together with the CAMS reliability indicator. All timestamps use UTC.

Temporal aggregation is handled separately by build_10min_historical_datasets.py.
"""

from pathlib import Path
import argparse
import hashlib
import json
import pipeline_config as cfg

import cdsapi
import pandas as pd


# ============================================================
# Configuration
# ============================================================

LATITUDE = cfg.LATITUDE
LONGITUDE = cfg.LONGITUDE

START_DATE = pd.Timestamp(cfg.START_DATE)
END_DATE = pd.Timestamp(cfg.END_DATE)

DATASET = "cams-solar-radiation-timeseries"

PROJECT_ROOT = Path(__file__).resolve().parent

RAW_CAMS_DIR = None
PROCESSED_DIR = None
OUTPUT_FILE = None


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


def download_month(client, start_date, end_date):
    """Download a request-hashed chunk; reuse known original-site legacy files."""
    request = {
        'sky_type': 'observed_cloud',
        'location': {'longitude': LONGITUDE, 'latitude': LATITUDE},
        'altitude': ['-999'],
        'date': [f'{start_date:%Y-%m-%d}/{end_date:%Y-%m-%d}'],
        'time_step': '1minute', 'time_reference': 'universal_time', 'data_format': 'csv',
    }
    stem = f'cams_{start_date:%Y-%m-%d}_{end_date:%Y-%m-%d}'
    signature = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:12]
    output_path = RAW_CAMS_DIR / f'{stem}_{signature}.csv'
    legacy = RAW_CAMS_DIR / f'{stem}.csv'
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f'Already downloaded: {output_path.name}')
        return output_path
    # Original repository requests used precisely this site and all other parameters above.
    # A changed site must never reuse these unlabelled legacy chunks.
    if LATITUDE == 45.7742 and LONGITUDE == 19.1122 and legacy.exists() and legacy.stat().st_size > 0:
        print(f'Reusing original-site legacy chunk: {legacy.name}')
        return legacy
    print(f'Downloading CAMS: {start_date.date()} -> {end_date.date()}', flush=True)
    temporary = output_path.with_suffix('.csv.part')
    client.retrieve(DATASET, request).download(str(temporary))
    temporary.replace(output_path)
    output_path.with_suffix('.request.json').write_text(json.dumps(request, indent=2), encoding='utf-8')
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

    ends = pd.to_datetime(df['timestamp_end'], utc=True)
    if not (ends - df['timestamp']).eq(pd.Timedelta(minutes=1)).all():
        raise ValueError('Expected one-minute CAMS observation periods')

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
        df.loc[df[column] < 0, column] = float("nan")

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
    global START_DATE, END_DATE, RAW_CAMS_DIR, PROCESSED_DIR, OUTPUT_FILE
    parser = argparse.ArgumentParser(description="Download historical observed-cloud CAMS radiation.")
    parser.add_argument('--start', default=cfg.START_DATE)
    parser.add_argument('--end', default=cfg.END_DATE)
    parser.add_argument('--data-dir', type=Path, default=cfg.DATA_DIR)
    args = parser.parse_args()
    START_DATE, END_DATE = pd.Timestamp(args.start), pd.Timestamp(args.end)

    if START_DATE > END_DATE:
        parser.error('--start must be before or equal to --end')

    dataset_folder = f"{cfg.LOCATION}_{args.start}_{args.end}"

    RAW_CAMS_DIR = (
            args.data_dir
            / 'raw'
            / 'historical'
            / dataset_folder
            / 'cams'
    )

    PROCESSED_DIR = (
            args.data_dir
            / 'processed'
            / 'historical'
            / dataset_folder
            / 'cams'
    )

    OUTPUT_FILE = (
            PROCESSED_DIR
            / f'cams_{cfg.LOCATION}_1min.csv'
    )

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