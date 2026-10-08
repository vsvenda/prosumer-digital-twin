"""Download helpers shared by the three explicitly separate entry points."""
import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import pipeline_config as cfg


def monthly_periods(start, end):
    current = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if current > end:
        raise ValueError('start must be before or equal to end')
    while current <= end:
        stop = min(current + pd.offsets.MonthEnd(0), end)
        yield current.strftime('%Y-%m-%d'), stop.strftime('%Y-%m-%d')
        current = stop + pd.Timedelta(days=1)


def session():
    client = requests.Session()
    retry = Retry(total=4, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=['GET'])
    client.mount('https://', HTTPAdapter(max_retries=retry))
    return client


def parse_response(data, variables, day1=False):
    if not isinstance(data, dict):
        raise ValueError('Expected one Open-Meteo JSON object')
    if data.get('error') or 'hourly' not in data:
        raise ValueError(f'Invalid Open-Meteo response: {data.get("reason", "no hourly data")}')
    suffix = '_previous_day1' if day1 else ''
    expected = [v + suffix for v in variables]
    missing = set(expected + ['time']) - set(data['hourly'])
    if missing:
        raise ValueError(f'API omitted requested fields: {sorted(missing)}')
    df = pd.DataFrame(data['hourly'])[['time', *expected]].rename(
        columns={'time': 'timestamp', **{v + suffix: v for v in variables}})
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    if df['timestamp'].duplicated().any():
        raise ValueError('Duplicate hourly timestamps in API response')
    for variable in variables:
        df[variable] = pd.to_numeric(df[variable], errors='raise')
    units = data.get('hourly_units', {})
    for variable in cfg.SOLAR_VARIABLES:
        if variable in variables and units.get(variable + suffix) != 'W/m²':
            raise ValueError(f'Unexpected radiation unit for {variable}: {units.get(variable + suffix)}')
    return df.sort_values('timestamp')


def download_main(mode):
    parser = argparse.ArgumentParser(description=f'Download Open-Meteo {mode} data.')
    parser.add_argument('--start', default=cfg.DAY1_START_DATE if mode == 'previous_runs_day1' else cfg.START_DATE)
    parser.add_argument('--end', default=cfg.END_DATE)
    parser.add_argument('--models', nargs='+', default=list(cfg.HISTORICAL_MODELS) if mode == 'historical' else [cfg.FORECAST_MODEL])
    parser.add_argument('--data-dir', type=__import__('pathlib').Path, default=cfg.DATA_DIR)
    parser.add_argument('--refresh', action='store_true', help='Re-download cached requests.')
    parser.add_argument('--allow-missing', action='store_true', help='Save incomplete data with an explicit coverage report.')
    args = parser.parse_args()
    if any(not re.fullmatch(r'[a-z0-9_]+', m) for m in args.models):
        parser.error('Models must be Open-Meteo identifiers containing lowercase letters, digits and underscores')
    start = pd.Timestamp(args.start, tz='UTC')
    stop = pd.Timestamp(args.end, tz='UTC') + pd.Timedelta(days=1)
    dataset_folder = (
        f"{cfg.LOCATION}_"
        f"{args.start}_"
        f"{args.end}"
    )
    if start >= stop:
        parser.error('--start must be before or equal to --end')
    if mode == 'historical' and any(m not in cfg.HISTORICAL_MODELS for m in args.models):
        parser.error('Historical models must be era5 and/or ifs')
    # Buffer days supply the previous/next hour for interpolation and trailing-hour quantities.
    periods = list(monthly_periods((start - pd.Timedelta(days=1)).date(), stop.date()))
    variables = cfg.WEATHER_VARIABLES + ([] if mode == 'historical' else cfg.SOLAR_VARIABLES)
    day1 = mode == 'previous_runs_day1'
    suffix = '_previous_day1' if day1 else ''
    out = (
            args.data_dir
            / 'processed'
            / mode
            / dataset_folder
    )
    out.mkdir(parents=True, exist_ok=True)
    with session() as client:
        for model in args.models:
            api_model = cfg.HISTORICAL_MODELS[model] if mode == 'historical' else model
            raw = (
                    args.data_dir
                    / 'raw'
                    / mode
                    / dataset_folder
                    / 'openmeteo'
                    / model
            )
            raw.mkdir(parents=True, exist_ok=True)
            frames = []
            raw_files = []
            for first, last in periods:
                params = dict(latitude=cfg.LATITUDE, longitude=cfg.LONGITUDE,
                              start_date=first, end_date=last, timezone='UTC',
                              models=api_model, wind_speed_unit='kmh',
                              hourly=','.join(v + suffix for v in variables))
                if mode != 'historical':
                    params.update(tilt=cfg.TILT, azimuth=cfg.AZIMUTH)
                signature = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]
                path = raw / f'{first}_{last}_{signature}.json'
                if path.exists() and not args.refresh:
                    envelope = json.loads(path.read_text(encoding='utf-8'))
                    if envelope['params'] != params or envelope['endpoint'] != cfg.ENDPOINTS[mode]:
                        raise ValueError(f'Cache request mismatch: {path}')
                    data = envelope['response']
                    print(f'Cached: {mode}/{model}/{first}')
                else:
                    print(f'Downloading: {mode}/{model}/{first} -> {last}', flush=True)
                    response = client.get(cfg.ENDPOINTS[mode], params=params, timeout=(15, 120))
                    if not response.ok:
                        raise RuntimeError(f'Open-Meteo HTTP {response.status_code}: {response.text[:500]}')
                    data = response.json()
                    parse_response(data, variables, day1)  # Never cache error responses.
                    envelope = dict(endpoint=cfg.ENDPOINTS[mode], params=params,
                                    downloaded_at=datetime.now(timezone.utc).isoformat(), response=data)
                    temporary = path.with_suffix('.json.part')
                    temporary.write_text(json.dumps(envelope), encoding='utf-8')
                    temporary.replace(path)
                    time.sleep(1)  # Avoid hammering the public service.
                frames.append(parse_response(data, variables, day1))
                raw_files.append(str(path.relative_to(args.data_dir)))
            df = pd.concat(frames).sort_values('timestamp').reset_index(drop=True)
            if df['timestamp'].duplicated().any():
                raise ValueError('Overlapping raw chunks contain duplicate timestamps')
            core_index = pd.date_range(start, stop - pd.Timedelta(hours=1), freq='h')
            core = df.set_index('timestamp').reindex(core_index)
            quality = {v: int(core[v].isna().sum()) for v in variables}
            # Include the following midnight because it supplies radiation for the last hour.
            next_midnight = df.set_index('timestamp').reindex([stop])
            interval_vars = ['precipitation', 'snowfall'] + ([] if mode == 'historical' else cfg.SOLAR_VARIABLES)
            boundary_missing = {v: bool(next_midnight[v].isna().any()) for v in interval_vars}
            model_out = out / 'openmeteo' / model
            model_out.mkdir(parents=True, exist_ok=True)

            base = model_out / f'openmeteo_{cfg.LOCATION}_{model}_1h'
            manifest = dict(mode=mode, model=api_model, start=args.start, end=args.end,
                            nominal_lead_hours=24 if day1 else None,
                            latitude=cfg.LATITUDE, longitude=cfg.LONGITUDE,
                            tilt=cfg.TILT if mode != 'historical' else None,
                            azimuth=cfg.AZIMUTH if mode != 'historical' else None,
                            endpoint=cfg.ENDPOINTS[mode], missing_core_hours=quality,
                            missing_next_midnight=boundary_missing, raw_files=raw_files,
                            units=envelope['response'].get('hourly_units', {}))
            manifest['csv_updated'] = False
            base.with_suffix('.coverage.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
            print('Missing hourly values:', quality)
            if (any(quality.values()) or any(boundary_missing.values())) and not args.allow_missing:
                raise ValueError('Incomplete archive coverage. Inspect the JSON report, adjust dates/model, '
                                 'or use --allow-missing to preserve NaNs. No new hourly CSV was written.')
            temporary = base.with_suffix('.csv.part')
            df.to_csv(temporary, index=False)
            temporary.replace(base.with_suffix('.csv'))
            manifest['csv_updated'] = True
            base.with_suffix('.coverage.json').write_text(
                json.dumps(manifest, indent=2),
                encoding='utf-8'
            )
            print(f'Saved: {base.with_suffix(".csv")} (includes boundary buffer days)')


if __name__ == '__main__':
    raise SystemExit('Use a download_* entry point.')
