"""UTC alignment and physically consistent 10-minute temporal conversion."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import pipeline_config as cfg


def load_csv(path):
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    if df['timestamp'].duplicated().any():
        raise ValueError(f'Duplicate timestamps: {path}')
    return df.sort_values('timestamp').set_index('timestamp')


def instantaneous(series, target, hold=False):
    """No extrapolation, no interpolation across missing hourly samples."""
    left = target.floor('h')
    a = pd.Series(series.reindex(left).to_numpy(), index=target)
    if hold:
        return a
    right = target.ceil('h')
    b = pd.Series(series.reindex(right).to_numpy(), index=target)
    weight = (target - left) / pd.Timedelta(hours=1)
    return a * (1 - weight) + b * weight


def trailing_hour(series, target):
    """A bin [t,t+10min) belongs to the hourly interval ending at ceil(t+10min)."""
    ends = (target + pd.Timedelta(minutes=10)).ceil('h')
    return pd.Series(series.reindex(ends).to_numpy(), index=target)


def build_weather(hourly, target, mode):
    required = cfg.WEATHER_VARIABLES + ([] if mode == 'historical' else cfg.SOLAR_VARIABLES)
    absent = set(required) - set(hourly.columns)
    if absent:
        raise ValueError(f'Missing columns: {sorted(absent)}')
    if not hourly.index.equals(hourly.index.floor('h')):
        raise ValueError('Open-Meteo timestamps must be aligned to whole UTC hours')
    hold = mode == 'previous_runs_day1'
    result = pd.DataFrame(index=target)
    for variable in ['temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'cloud_cover', 'pressure_msl']:
        result[variable] = instantaneous(hourly[variable], target, hold)
    speed = hourly['wind_speed_10m']
    direction = np.deg2rad(hourly['wind_direction_10m'])
    u = instantaneous(-speed * np.sin(direction), target, hold)
    v = instantaneous(-speed * np.cos(direction), target, hold)
    result['wind_speed_10m'] = np.hypot(u, v)
    result['wind_direction_10m'] = np.degrees(np.arctan2(-u, -v)) % 360
    for variable in ['precipitation', 'snowfall']:
        result[variable] = trailing_hour(hourly[variable], target) / 6
    if mode != 'historical':
        for source, dest in cfg.SOLAR_ENERGY_COLUMNS.items():
            # W/m² (mean over preceding hour) * 1/6 h = Wh/m² per 10-minute bin.
            result[dest] = trailing_hour(hourly[source], target) / 6
    return result.rename_axis('timestamp').reset_index()


def build_cams(minute, target):
    """Require all ten valid minute samples separately for each energy column."""
    if not minute.index.equals(minute.index.floor('min')):
        raise ValueError('CAMS samples must align to minute boundaries')
    result = pd.DataFrame(index=target)
    for variable in ['ghi_wh_m2', 'dhi_wh_m2', 'bni_wh_m2']:
        result[variable] = minute[variable].resample('10min').sum(min_count=10).reindex(target)
    # A quality value is meaningful only if all ten samples have a quality flag.
    flags = minute['reliability'].resample('10min')
    result['reliability'] = flags.mean().where(flags.count() == 10).reindex(target)
    return result.rename_axis('timestamp').reset_index()


def save_dataset(df, path, mode, args):
    missing = {c: int(df[c].isna().sum()) for c in df if c != 'timestamp'}
    report = dict(mode=mode, start=args.start, end=args.end, rows=len(df),
                  missing_values=missing, incomplete_rows=int(df.drop(columns='timestamp').isna().any(axis=1).sum()),
                  timestamp_convention='UTC start of [timestamp, timestamp+10min)',
                  radiation_unit='Wh/m2 per 10-minute interval',
                  native_weather_resolution='hourly',
                  native_radiation_resolution='1 minute' if mode == 'historical' else 'hourly',
                  instant_weather_resampling='backward hold' if mode == 'previous_runs_day1' else 'linear interpolation',
                  exact_24h_lead_at_10min=False if mode == 'previous_runs_day1' else None)
    report['csv_updated'] = False
    path.with_suffix('.quality.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if report['incomplete_rows'] and not args.allow_missing:
        raise ValueError(f'{path.name}: {report["incomplete_rows"]} incomplete rows. '
                         'Inspect .quality.json; adjust range, download buffers, or pass --allow-missing. '
                         'No new CSV written.')
    temporary = path.with_suffix('.csv.part')
    df.to_csv(temporary, index=False)
    temporary.replace(path)
    report['csv_updated'] = True
    path.with_suffix('.quality.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Saved {path}: {len(df):,} rows; {report["incomplete_rows"]:,} incomplete rows')


def build_main(mode):
    parser = argparse.ArgumentParser(description=f'Build 10-minute {mode} datasets.')
    parser.add_argument('--start', default=cfg.DAY1_START_DATE if mode == 'previous_runs_day1' else cfg.START_DATE)
    parser.add_argument('--end', default=cfg.END_DATE)
    parser.add_argument('--models', nargs='+', default=list(cfg.HISTORICAL_MODELS) if mode == 'historical' else [cfg.FORECAST_MODEL])
    parser.add_argument('--data-dir', type=Path, default=cfg.DATA_DIR)
    parser.add_argument('--allow-missing', action='store_true', help='Keep NaNs and save the coverage report.')
    args = parser.parse_args()
    start = pd.Timestamp(args.start, tz='UTC')
    stop = pd.Timestamp(args.end, tz='UTC') + pd.Timedelta(days=1)
    if start >= stop:
        parser.error('--start must be before or equal to --end')
    target = pd.date_range(start, stop - pd.Timedelta(minutes=10), freq='10min')
    dataset_folder = f"{cfg.LOCATION}_{args.start}_{args.end}"

    folder = (
            args.data_dir
            / 'processed'
            / mode
            / dataset_folder
    )

    folder.mkdir(parents=True, exist_ok=True)
    cams = None
    if mode == 'historical':
        cams_folder = folder / 'cams'
        cams_folder.mkdir(parents=True, exist_ok=True)

        cams = build_cams(
            load_csv(
                cams_folder
                / f'cams_{cfg.LOCATION}_1min.csv'
            ),
            target
        )

        save_dataset(
            cams,
            cams_folder
            / f'cams_{cfg.LOCATION}_10min.csv',
            mode,
            args
        )
    for model in args.models:
        model_folder = (
                folder
                / 'openmeteo'
                / model
        )

        model_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        weather = build_weather(
            load_csv(
                model_folder
                / f'openmeteo_{cfg.LOCATION}_{model}_1h.csv'
            ),
            target,
            mode
        )
        save_dataset(
            weather,
            model_folder
            / f'openmeteo_{cfg.LOCATION}_{model}_10min.csv',
            mode,
            args
        )
        if cams is not None:
            combined = cams.merge(
                weather,
                on='timestamp',
                how='left',
                validate='one_to_one'
            )

            combined_folder = folder / 'combined'
            combined_folder.mkdir(
                parents=True,
                exist_ok=True
            )

            save_dataset(
                combined,
                combined_folder
                / f'combined_{cfg.LOCATION}_{model}_10min.csv',
                mode,
                args
            )
