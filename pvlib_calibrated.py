"""PVWatts baseline matching colleague's 5-minute training-period CSV files.

Install: python -m pip install numpy pandas scipy pvlib
Run:     python pvlib_calibrated.py
          python pvlib_calibrated.py --site utrecht --household ID003 --max-splits 1
          python pvlib_calibrated.py --downloads "D:/Data" --only-durations 1W 2W

Inputs:  <downloads>/konstanz.zip, <downloads>/utrecht.zip (or extracted dirs).
Outputs: results/pvlib/<site>/<household>/{experiments,training_features}.csv

Both calibration regimes are fitted against training PV production:
  partial = fix verified known geometry/DC rating, fit missing parameters.
  full    = fit tilt, azimuth, effective DC rating; verified values as starts.
Only tilt, azimuth and effective DC rating are candidates for optimization.
AC rating, losses, thermal coefficient etc. follow fixed rules below.
"""
from __future__ import annotations

import argparse
import io
import json
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

try:
    import pvlib
except ImportError as exc:
    raise SystemExit('Install dependencies: python -m pip install numpy pandas scipy pvlib') from exc

ROOT = Path(__file__).resolve().parent
DURATIONS = ('1W', '2W', '1M', '3M', '6M', '12M')
MODES = {
    'A_all': ('ghi', 'dhi', 'bni', 'temperature_2m', 'wind_speed_10m'),
    'B_irradiance': ('ghi', 'dhi', 'bni'),
    'C_ghi_bni': ('ghi', 'bni'),
}
SITES = {
    'konstanz': {'lat': 47.6779, 'lon': 9.1732,
                 'households': ['residential1', 'residential3']},
    'utrecht': {'lat': 52.0907, 'lon': 5.1214,
               'households': ['ID003', 'ID005']},
}
# Only CONFIRMED physical specifications should be entered here. Null means unknown.
# Previous ID003 metadata estimates (40 deg, 203 deg, 2.503 kW) are intentionally
# NOT marked as known. You can enter independently verified specs below.
KNOWN_SPECS = {
    'konstanz/residential1': {
        'tilt': None, 'azimuth': None,
        'dc_kw': None, 'ac_kw': None
    },
    'konstanz/residential3': {
        'tilt': None, 'azimuth': None,
        'dc_kw': None, 'ac_kw': None
    },
    'utrecht/ID003': {
        'tilt': 40.0, 'azimuth': 203.0,
        'dc_kw': 2.503, 'ac_kw': 2.250
    },
    'utrecht/ID005': {
        'tilt': 15.0, 'azimuth': 180.0,
        'dc_kw': 2.413, 'ac_kw': 2.500
    }
}
# Model assumptions: fixed across all experiments.
SYSTEM_LOSS = 0.14
TEMP_COEFF = -0.004
INVERTER_EFFICIENCY = 0.96
UNKNOWN_AC_TO_DC = 0.90   # an explicit assumption, not measured specification
DEFAULT_TILT = 35.0
DEFAULT_AZIMUTH = 180.0
MIN_COVERAGE = 0.80
SUN_THRESHOLD = 90.0  # zenith >= 90 degrees -> night for missing radiation fill


def source_bytes(downloads: Path, site: str, relative: str) -> bytes:
    archive = downloads / f'{site}.zip'
    if archive.is_file():
        with zipfile.ZipFile(archive) as z:
            path = f'{site}/{relative}'
            try:
                return z.read(path)
            except KeyError as exc:
                raise FileNotFoundError(f'{path} is missing inside {archive}') from exc
    path = downloads / site / relative
    if not path.exists():
        raise FileNotFoundError(f'Missing {path} (or archive {archive})')
    return path.read_bytes()


def read_periods(downloads: Path, site: str, household: str) -> pd.DataFrame:
    raw = source_bytes(downloads, site, f'training_periods/{household}/periods.csv')
    frame = pd.read_csv(io.BytesIO(raw))
    required = {'split_id', 'Start', 'End', 'type'}
    if not required <= set(frame.columns):
        raise ValueError(f'periods.csv missing: {required - set(frame.columns)}')
    frame['Start'] = pd.to_datetime(frame['Start'])
    frame['End'] = pd.to_datetime(frame['End'])
    frame['type'] = frame['type'].astype(str)
    return frame


def read_site(downloads: Path, site: str, household: str) -> pd.DataFrame:
    """Read only one household's needed columns from the big, gzipped 5-minute CSV."""
    raw = source_bytes(downloads, site, 'output/solar_production_5min.csv.gz')
    wanted = ['solar_production_w', 'ghi_wh_m2', 'dhi_wh_m2', 'bhi_wh_m2',
              'bni_wh_m2', 'temperature_2m', 'wind_speed_10m',
              'relative_humidity_2m', 'cloud_cover', 'precipitation', 'snowfall']
    import gzip
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
        df = pd.read_csv(gz, usecols=['timestamp_utc_plus_1'] +
                         [f'{col}_{household}' for col in wanted])
    df = df.rename(columns={f'{col}_{household}': col for col in wanted})
    # Data explicitly label this clock as 'UTC+1', regardless of seasonal DST.
    # Periods.csv uses the very same clock labels. Keep a naive clock for splits,
    # convert to true UTC (subtract exactly one hour) only for solar position.
    df['stamp'] = pd.to_datetime(df['timestamp_utc_plus_1'], errors='coerce')
    df = df.dropna(subset=['stamp']).drop_duplicates('stamp').sort_values('stamp')
    df = df.set_index('stamp').drop(columns=['timestamp_utc_plus_1'])
    for name in wanted:
        df[name] = pd.to_numeric(df[name], errors='coerce')
    df['pv_kw'] = df['solar_production_w'].where(df['solar_production_w'] >= 0) / 1000.0
    # CAMS columns are Wh/m^2 over five minutes, not instantaneous W/m^2.
    for name in ('ghi', 'dhi', 'bhi', 'bni'):
        df[name] = (df[f'{name}_wh_m2'] * 12.0).clip(lower=0)
    # Solar position at bin CENTRE; timestamp is bin start.
    utc_at_mid = (df.index - pd.Timedelta(hours=1) + pd.Timedelta(minutes=2, seconds=30)).tz_localize('UTC')
    solar = pvlib.solarposition.get_solarposition(
        utc_at_mid, latitude=SITES[site]['lat'], longitude=SITES[site]['lon'])
    df['zenith'] = np.asarray(solar['apparent_zenith'], dtype=float)
    df['sun_azimuth'] = np.asarray(solar['azimuth'], dtype=float)
    # Only genuine nighttime missing irradiation is filled with zero; otherwise
    # missing weather values cause the individual point to be excluded.
    night = df['zenith'] >= SUN_THRESHOLD
    for name in ('ghi', 'dhi', 'bni'):
        df.loc[night & df[name].isna(), name] = 0.0
    return df


def mode_mask(df: pd.DataFrame, mode: str, require_target: bool = True) -> np.ndarray:
    fields = list(MODES[mode]) + ['zenith', 'sun_azimuth']
    if require_target:
        fields.append('pv_kw')
    values = df[fields].to_numpy(dtype=float)
    return np.isfinite(values).all(axis=1)


def predict(frame: pd.DataFrame, parameters: dict, mode: str) -> np.ndarray:
    """POA isotropic + optional Faiman temperature + PVWatts DC/inverter."""
    zen = frame['zenith'].to_numpy(dtype=float)
    sun_az = frame['sun_azimuth'].to_numpy(dtype=float)
    ghi = frame['ghi'].to_numpy(dtype=float)
    dni = frame['bni'].to_numpy(dtype=float)
    if mode == 'C_ghi_bni':
        cosz = np.maximum(np.cos(np.deg2rad(zen)), 0.0)
        dhi = np.maximum(ghi - dni * cosz, 0.0)
    else:
        dhi = frame['dhi'].to_numpy(dtype=float)
    dni = np.where(zen >= SUN_THRESHOLD, 0.0, np.maximum(dni, 0.0))
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=parameters['tilt'], surface_azimuth=parameters['azimuth'],
        solar_zenith=zen, solar_azimuth=sun_az, dni=dni, ghi=ghi, dhi=dhi,
        model='isotropic')
    plane = np.maximum(np.asarray(poa['poa_global'], dtype=float), 0.0)
    if mode == 'A_all':
        # ERA5 wind_speed_10m is in km/h in this pipeline; Faiman uses m/s.
        cell = np.asarray(pvlib.temperature.faiman(
            plane, frame['temperature_2m'].to_numpy(dtype=float),
            frame['wind_speed_10m'].to_numpy(dtype=float) / 3.6))
    else:
        cell = np.full(len(plane), 25.0)
    dc_w = pvlib.pvsystem.pvwatts_dc(
        plane, cell, pdc0=parameters['dc_kw'] * 1000,
        gamma_pdc=TEMP_COEFF)
    dc_w = np.maximum(np.asarray(dc_w, dtype=float), 0.0) * (1.0 - SYSTEM_LOSS)
    ac_w = pvlib.inverter.pvwatts(
        dc_w, pdc0=parameters['ac_kw'] * 1000 / INVERTER_EFFICIENCY,
        eta_inv_nom=INVERTER_EFFICIENCY)
    return np.maximum(np.asarray(ac_w, dtype=float) / 1000.0, 0.0)


def fit(train: pd.DataFrame, mode: str, regime: str, known: dict,
        max_nfev: int) -> dict:
    """Optimize only the predeclared candidate parameters; never fit test rows."""
    finite_power = train['pv_kw'].to_numpy(dtype=float)
    positive = finite_power[finite_power > 0.01]
    if len(positive) < 20:
        raise ValueError('Fewer than 20 positive PV samples in training data')
    initial_dc = max(float(np.percentile(positive, 98) * 1.25), 0.2)
    defaults = {'tilt': DEFAULT_TILT, 'azimuth': DEFAULT_AZIMUTH,
                'dc_kw': initial_dc}
    # If known, initialize from actual specification, never from test PV values.
    start = {k: (float(known[k]) if known.get(k) is not None else defaults[k])
             for k in defaults}
    free = list(defaults) if regime == 'full' else [k for k in defaults if known.get(k) is None]
    lo = {'tilt': 0.0, 'azimuth': 45.0, 'dc_kw': 0.1}
    hi = {'tilt': 85.0, 'azimuth': 315.0,
          'dc_kw': max(50.0, 4.0 * start['dc_kw'])}
    # Zero-parameter case: all three specs are known, no numerical optimization.
    def unpack(x):
        params = dict(start)
        params.update(zip(free, map(float, x)))
        params['ac_kw'] = (float(known['ac_kw']) if known.get('ac_kw') is not None
                           else params['dc_kw'] * UNKNOWN_AC_TO_DC)
        return params

    if free:
        x0 = np.array([np.clip(start[k], lo[k]+1e-5, hi[k]-1e-5) for k in free])
        bounds = ([lo[k] for k in free], [hi[k] for k in free])
        def residual(x):
            return predict(train, unpack(x), mode) - finite_power
        result = least_squares(residual, x0=x0, bounds=bounds, max_nfev=max_nfev,
                               x_scale='jac', diff_step=1e-3)
        params = unpack(result.x)
        succeeded = bool(result.success)
        cost = float(np.sum(np.square(result.fun)))
        nfev = result.nfev
    else:
        params = unpack([])
        succeeded = True
        cost = float(np.sum((predict(train, params, mode) - finite_power) ** 2))
        nfev = 0
    return {**params, 'fitted_parameters': '|'.join(free) or 'none',
            'optimizer_success': succeeded, 'optimizer_nfev': nfev,
            'train_sse_kw2': cost,
            'ac_source': 'known' if known.get('ac_kw') is not None else 'assumed_0.90_times_dc'}


def metric_dict(truth: np.ndarray, pred: np.ndarray, norm_mean: float) -> dict:
    e = pred - truth
    ss_tot = float(np.sum((truth - np.mean(truth)) ** 2))
    mae = float(np.mean(np.abs(e)))
    return {'mae_kw': mae,
            'rmse_kw': float(np.sqrt(np.mean(e * e))),
            'p90_abs_error_kw': float(np.percentile(np.abs(e), 90)),
            'r2': float(1.0 - np.sum(e * e) / ss_tot) if ss_tot > 0 else np.nan,
            'bias_kw': float(np.mean(e)),
            'nmae_pct': (100.0 * mae / norm_mean) if norm_mean > 0 else np.nan}


def range_mask(index: pd.DatetimeIndex, intervals: pd.DataFrame, kind: str) -> np.ndarray:
    out = np.zeros(len(index), dtype=bool)
    for row in intervals.loc[intervals['type'] == kind].itertuples(index=False):
        out |= (index >= row.Start) & (index < row.End)
    return out


def training_features(train: pd.DataFrame, site: str, household: str, split_id: int,
                      duration: str, start: pd.Timestamp, end: pd.Timestamp,
                      expected: int, coverage: float) -> dict:
    ret = {'training_period_id': f'{site}_{household}_s{split_id}_{duration}',
           'site': site, 'household': household, 'split_id': split_id,
           'duration': duration, 'train_start': start.isoformat(),
           'train_end_exclusive': end.isoformat(), 'n_train_pv_valid': len(train),
           'expected_bins': expected, 'pv_coverage': coverage,
           'n_daylight_points': int((train['zenith'] < 90).sum()),
           'months_present': '|'.join(str(m) for m in sorted(train.index.month.unique()))}
    daylight = train.loc[train['zenith'] < 90]
    for c in ('ghi', 'bni', 'dhi', 'bhi', 'temperature_2m', 'wind_speed_10m',
              'pv_kw', 'relative_humidity_2m', 'cloud_cover', 'precipitation', 'snowfall'):
        # For irradiance, describe daylight variability rather than night zeros.
        series = daylight[c] if c in ('ghi', 'bni', 'dhi', 'bhi') else train[c]
        a = series.dropna().to_numpy(dtype=float)
        ret[f'{c}_n'] = len(a)
        for label, val in [('mean', np.mean(a) if len(a) else np.nan),
                           ('std', np.std(a) if len(a) else np.nan),
                           ('min', np.min(a) if len(a) else np.nan),
                           ('max', np.max(a) if len(a) else np.nan),
                           ('p10', np.percentile(a,10) if len(a) else np.nan),
                           ('p50', np.percentile(a,50) if len(a) else np.nan),
                           ('p90', np.percentile(a,90) if len(a) else np.nan)]:
            ret[f'{c}_{label}'] = float(val)
    ghi = daylight['ghi']
    valid = ghi.notna()
    ret['ghi_low_0_200_n'] = int((valid & (ghi < 200)).sum())
    ret['ghi_mid_200_600_n'] = int((valid & (ghi >= 200) & (ghi < 600)).sum())
    ret['ghi_high_600_plus_n'] = int((valid & (ghi >= 600)).sum())
    for name, mask in [('low', valid & (ghi < 200)),
                       ('mid', valid & (ghi >= 200) & (ghi < 600)),
                       ('high', valid & (ghi >= 600))]:
        ret[f'ghi_{name}_distinct_days'] = int(daylight.loc[mask].index.normalize().nunique())
    # Clear-sky index: only well-defined when clear-sky GHI is sufficiently high.
    # Uses site location; no measured target values from test are used.
    cs_times = (daylight.index - pd.Timedelta(hours=1) + pd.Timedelta(minutes=2, seconds=30)).tz_localize('UTC')
    if len(cs_times):
        cs = pvlib.location.Location(SITES[site]['lat'], SITES[site]['lon']).get_clearsky(cs_times)
        clear_ghi = np.asarray(cs['ghi'], dtype=float)
        measured = daylight['ghi'].to_numpy(dtype=float)
        ok = (clear_ghi >= 100) & np.isfinite(measured)
        kt = np.clip(measured[ok] / clear_ghi[ok], 0, 1.5)
        ret['clearsky_index_mean'] = float(np.mean(kt)) if len(kt) else np.nan
        ret['clearsky_index_std'] = float(np.std(kt)) if len(kt) else np.nan
        ret['clearsky_index_n'] = int(len(kt))
        ret['clearsky_index_cloudy_n'] = int(np.sum(kt < .35))
        ret['clearsky_index_clear_n'] = int(np.sum(kt > .75))
    else:
        ret.update(clearsky_index_mean=np.nan, clearsky_index_std=np.nan,
                   clearsky_index_n=0, clearsky_index_cloudy_n=0,
                   clearsky_index_clear_n=0)
    return ret


def evaluate_household(args, site: str, household: str):
    print(f'\n=== {site}/{household} ===', flush=True)
    periods = read_periods(args.downloads, site, household)
    frame = read_site(args.downloads, site, household)
    known = KNOWN_SPECS[f'{site}/{household}']
    denom = float(frame['pv_kw'].mean())  # constant over complete valid household series, incl. nights
    print(f'  {len(frame):,} input points, normalization mean {denom:.5f} kW; known={known}', flush=True)
    out = ROOT / 'results' / 'pvlib' / site / household
    out.mkdir(parents=True, exist_ok=True)
    result_path, features_path = out / 'experiments.csv', out / 'training_features.csv'
    results, features = [], []
    groups = list(periods.groupby('split_id', sort=True))
    if args.max_splits:
        groups = groups[:args.max_splits]
    all_index = frame.index
    for idx, (split_id, intervals) in enumerate(groups, 1):
        test_mask = range_mask(all_index, intervals, 'test')
        test_source = frame.iloc[np.flatnonzero(test_mask)]
        # Do not include val/pause; no test rows inside reserved 12 months.
        print(f'  split {split_id} ({idx}/{len(groups)}): test={len(test_source):,}', flush=True)
        for dur in DURATIONS:
            if args.only_durations and dur not in args.only_durations:
                continue
            matches = intervals.loc[intervals['type'] == dur]
            # Absence in periods.csv signals INELIGIBLE training period; do not re-enable it.
            if matches.empty:
                continue
            if len(matches) != 1:
                raise ValueError(f'Expected one {dur} for split {split_id}, got {len(matches)}')
            start, end = matches.iloc[0][['Start', 'End']]
            expected = int((end - start) / pd.Timedelta(minutes=5))
            base = frame.loc[(all_index >= start) & (all_index < end)]
            base = base.loc[base['pv_kw'].notna()]
            coverage = len(base) / expected if expected else 0
            if coverage < MIN_COVERAGE:
                warnings.warn(f'{site}/{household} split={split_id} {dur} PV coverage {coverage:.1%} <80%; skipping')
                continue
            period_id = f'{site}_{household}_s{split_id}_{dur}'
            features.append(training_features(base, site, household, int(split_id),
                                              dur, start, end, expected, coverage))
            for mode in MODES:
                train = base.loc[mode_mask(base, mode)]
                test = test_source.loc[mode_mask(test_source, mode)]
                if len(train) < 20 or len(test) < 20:
                    warnings.warn(f'{period_id} {mode}: not enough complete records; skipping')
                    continue
                y = test['pv_kw'].to_numpy(dtype=float)
                for regime in ('partial', 'full'):
                    try:
                        fitted = fit(train, mode, regime, known, args.max_nfev)
                        yhat = predict(test, fitted, mode)
                        if not np.isfinite(yhat).all():
                            raise ValueError('Nonfinite model predictions')
                    except (ValueError, FloatingPointError, RuntimeError) as exc:
                        warnings.warn(f'{period_id} {mode}/{regime}: {exc}')
                        continue
                    row = {'experiment_id': f'{period_id}_{mode}_{regime}',
                           'training_period_id': period_id, 'site': site, 'household': household,
                           'split_id': int(split_id), 'duration': dur, 'input_mode': mode,
                           'calibration_mode': regime, 'train_start': start.isoformat(),
                           'train_end_exclusive': end.isoformat(), 'n_train': len(train),
                           'n_test': len(test), 'training_pv_coverage': coverage,
                           'norm_mean_pv_kw': denom, 'test_scope': 'colleague_test_intervals',
                           'test_start': test.index.min().isoformat(),
                           'test_end': test.index.max().isoformat(),
                           'known_tilt': known.get('tilt'), 'known_azimuth': known.get('azimuth'),
                           'known_dc_kw': known.get('dc_kw'), 'known_ac_kw': known.get('ac_kw'),
                           **fitted, **metric_dict(y, yhat, denom)}
                    results.append(row)
        # Persist after each split, to preserve progress on long runs.
        if results:
            pd.DataFrame(results).to_csv(result_path, index=False)
        if features:
            pd.DataFrame(features).to_csv(features_path, index=False)
    print(f'  wrote {len(results)} experiments and {len(features)} period feature rows to {out}', flush=True)
    return len(results), len(features)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--downloads', type=Path, default=Path.home() / 'Downloads',
                   help='Directory holding konstanz.zip and utrecht.zip OR extracted folders')
    p.add_argument('--site', choices=list(SITES), help='Run one site instead of both')
    p.add_argument('--household', help='Run one household (requires --site)')
    p.add_argument('--only-durations', nargs='+', choices=DURATIONS)
    p.add_argument('--max-splits', type=int, default=0, help='Debug: at most N split IDs per household')
    p.add_argument('--max-nfev', type=int, default=70)
    args = p.parse_args()
    if args.household and not args.site:
        p.error('--household requires --site')
    count = 0
    for site in ([args.site] if args.site else SITES):
        households = [args.household] if args.household else SITES[site]['households']
        for household in households:
            if household not in SITES[site]['households']:
                p.error(f'Unknown household {household!r} for site {site}')
            n, _ = evaluate_household(args, site, household)
            count += n
    print(f'\nDONE: {count} successful experiment rows', flush=True)


if __name__ == '__main__':
    main()
