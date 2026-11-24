"""Extract three HKUST inverter power series from the original ZIP (W, UTC).

Run from the repository root. No interpolation or night-time zero filling.
The optional 10-minute output requires both native 5-minute measurements.
"""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
LOCATION = 'hongkong'
START_DATE = '2021-07-07'  # UTC date of first local measurement (July 8, 06:00 HKT)
END_DATE = '2023-12-31'
SOURCE_TIMEZONE = 'Asia/Hong_Kong'  # Naive source times interpreted as local HKT
SYSTEMS = {
    'SQ01_12': 'SQ Apartment1-12_Inverter.csv',
    'SQ13_24': 'SQ Apartment13-24_Inverter.csv',
    'SQ25_36': 'SQ Apartment25-36_Inverter.csv',
}


def read_system(archive, filename):
    matches = [n for n in archive.namelist()
               if '/Inverter level dataset/' in n and n.rsplit('/', 1)[-1] == filename]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one inverter file {filename}: {matches}')
    with archive.open(matches[0]) as f:
        df = pd.read_csv(f, usecols=['Time', 'totalActivePower(W)'])
    # Source timestamps are naive and use mixed slash/hyphen date formats.
    stamp = pd.to_datetime(df['Time'], format='mixed', errors='raise')
    if stamp.dt.tz is not None:
        raise ValueError(f'Unexpected timezone-aware source timestamps: {filename}')
    index = pd.DatetimeIndex(stamp.dt.tz_localize(SOURCE_TIMEZONE).dt.tz_convert('UTC'))
    if index.has_duplicates or not index.equals(index.floor('5min')):
        raise ValueError(f'Duplicate or off-grid timestamps in {filename}')
    power = pd.to_numeric(df['totalActivePower(W)'], errors='coerce')
    power = power.where(np.isfinite(power) & (power >= 0))
    return pd.Series(power.to_numpy(), index=index, name=filename).sort_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Original HKUST Dataset ZIP')
    parser.add_argument('--data-dir', type=Path, default=PROJECT_ROOT / 'data')
    parser.add_argument('--start', default=START_DATE, help='Inclusive UTC date')
    parser.add_argument('--end', default=END_DATE, help='Inclusive UTC date')
    parser.add_argument('--write-10min', action='store_true')
    args = parser.parse_args()
    start = pd.Timestamp(args.start, tz='UTC')
    stop = pd.Timestamp(args.end, tz='UTC') + pd.Timedelta(days=1)
    if start >= stop:
        parser.error('--start must be before or equal to --end')
    target = pd.date_range(start, stop, freq='5min', inclusive='left', name='DateTime')
    out = pd.DataFrame(index=target)
    report = {'source': str(args.input), 'source_timezone_assumed': SOURCE_TIMEZONE,
              'output_timezone': 'UTC', 'power_unit': 'W', 'native_resolution': '5min',
              'interpolation': False, 'night_zero_fill': False, 'rows': len(target), 'systems': {}}
    local = target.tz_convert(SOURCE_TIMEZONE)
    window = (local.hour >= 6) & (local.hour < 18)
    with ZipFile(args.input) as archive:
        for key, filename in SYSTEMS.items():
            series = read_system(archive, filename)
            out[key] = series.reindex(target)
            active = (target >= series.index.min()) & (target <= series.index.max()) & window
            valid = out[key].notna()
            daily = out[key].loc[window].groupby(local[window].normalize()).count()
            report['systems'][key] = {
                'source_file': filename, 'first_utc': str(series.index.min()),
                'last_utc': str(series.index.max()), 'source_rows': len(series),
                'valid_in_output': int(valid.sum()), 'missing_in_full_grid': int((~valid).sum()),
                'expected_during_recorded_local_window': int(active.sum()),
                'missing_during_recorded_local_window': int((active & ~valid).sum()),
                'complete_local_06_18_days': int((daily == 144).sum()),
                'max_power_W': float(out[key].max()) if valid.any() else None,
            }
    folder = args.data_dir / 'solar' / f'{LOCATION}_{args.start}_{args.end}'
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / 'hongkong_g1_households.csv'
    out.to_csv(output)
    print(f'Saved {output} ({len(out):,} rows)')
    if args.write_10min:
        bins = out.resample('10min', label='left', closed='left')
        ten = bins.mean().where(bins.count() == 2)
        ten_path = folder / 'hongkong_g1_households_10min.csv'
        ten.to_csv(ten_path)
        report['10min'] = {'aggregation': 'mean of both 5min samples, left-labelled UTC',
                           'rows': len(ten), 'valid_per_system': {k: int(v) for k,v in ten.count().items()}}
        print(f'Saved {ten_path} ({len(ten):,} rows)')
    (folder / 'hongkong_g1_households.quality.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
