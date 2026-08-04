#!/usr/bin/env python3
"""
Update the Daily Bidding constraint sheet in GCS from SPP DA/RT mvalues.

Target: gs://spptest/manual/Daily Bidding - daily_constraint_manual.csv

Converted from QCTest/Manual_bidding/update_sheet/update_daily_excel.ipynb. The update
logic is unchanged; only the source and destination move from a local file to the bucket.

The target is an object, not a spreadsheet, so every write replaces the whole file — there
is no cell-level patch, and anything a colleague changed since the read is lost. Versioning
is DISABLED on this bucket, so a timestamped copy is made under manual/backups/ before each
overwrite; that copy is the only way back.

Auth: ambient credentials — GOOGLE_APPLICATION_CREDENTIALS points at
/var/www/python/Common/service-key.json.

Calling:
  /opt/venvs/prod-py312/bin/python spp_update_daily_bidding_sheet.py [start_dt] [end_dt] [--dry-run]
    start_dt, end_dt : optional YYYY-MM-DD; default is yesterday .. tomorrow (America/Chicago)
    --dry-run        : compute and report the changes, write nothing
"""

from __future__ import print_function

import io
import sys

import pandas as pd
from google.cloud import storage

sys.path.append("/var/www/python/Prod/nighthawk/")
from nighthawk.data import Constraint
from nighthawk.data.pipeline.common_functions.load import Load
from nighthawk.data.pipeline.common_functions.wind import Wind
from nighthawk.util import dataframe_functions

# Config
BUCKET      = 'spptest'
BLOB_PATH   = 'manual/Daily Bidding - daily_constraint_manual.csv'
BACKUP_DIR  = 'manual/backups'

MARKET      = 'SPP'
THRESHOLD   = -900
TZ          = 'US/Central'
META_COLS   = ['location', 'physical_condition', 'outage_name',
               'comment on this constraint', 'start_date', 'end_date',
               'wind', 'reserve_zone']
# zones that go in the "today's wind" cell. The forecast tables also carry rz21, but every
# summary written to this sheet so far covers 1-5 only, so rz21 is left out.
SUMMARY_ZONES = (1, 2, 3, 4, 5)

script_name = 'spp_update_daily_bidding_sheet.py'


def read_source():
    """
    Load the bidding table from GCS.

    Returns (df, src_cols, name_col) — src_cols and name_col preserve the file's own
    header spelling and order so the rewrite does not silently rename or reorder columns
    ('constraints' here, but the notebook's local copy used 'constraints ' with a trailing
    space, and either must round-trip unchanged).
    """
    blob = storage.Client().bucket(BUCKET).blob(BLOB_PATH)
    if not blob.exists():
        sys.exit('not found: gs://{}/{}'.format(BUCKET, BLOB_PATH))

    df = pd.read_csv(io.StringIO(blob.download_as_text()))
    df.columns = df.columns.str.strip()
    src_cols = list(df.columns)

    name_col = next((c for c in ('constraints', 'constraints ', 'monitored') if c in src_cols), None)
    if name_col is None:
        sys.exit('no constraint-name column in header: {}'.format(src_cols))

    df = df.rename(columns={name_col: 'monitored'})
    df['monitored'] = df['monitored'].astype(str).str.strip()

    before = len(df)
    # coerce bad/missing-year bid_dates (e.g. "6/15") to NaT, then drop those rows —
    # they disappear from the file, so say so rather than dropping them quietly
    df['bid_date'] = pd.to_datetime(df['bid_date'], format='mixed', errors='coerce')
    df = df.dropna(subset=['bid_date']).reset_index(drop=True)
    dropped = before - len(df)

    print('read {} rows from gs://{}/{}{}'.format(
        len(df), BUCKET, BLOB_PATH,
        ' ({} row(s) dropped for unparseable bid_date)'.format(dropped) if dropped else ''))
    return df, src_cols, name_col


def backup_source():
    """Copy the current object aside; the bucket has no versioning to fall back on."""
    client = storage.Client()
    bucket = client.bucket(BUCKET)
    stamp  = pd.Timestamp.now(tz=TZ).strftime('%Y%m%d_%H%M%S')
    dest   = '{}/{}.{}.csv'.format(BACKUP_DIR, BLOB_PATH.split('/')[-1][:-4], stamp)
    bucket.copy_blob(bucket.blob(BLOB_PATH), bucket, dest)
    print('backup  → gs://{}/{}'.format(BUCKET, dest))
    return dest


def _fetch_mvalues(dt_str):
    """DA and RT mvalues for one day, summed per monitored element."""
    opex  = MARKET
    mv_rt = Constraint(oops_constraint_num_df=None, market=opex).get_mvalues(
        start_dt=dt_str, end_dt=dt_str, type='RT', granularity='daily')
    mv_da = Constraint(oops_constraint_num_df=None, market=opex).get_mvalues(
        start_dt=dt_str, end_dt=dt_str, type='DA', granularity='daily')

    all_cons = pd.DataFrame({'oops_constraint_num':
        pd.concat([mv_rt['oops_constraint_num'], mv_da['oops_constraint_num']]).unique()})

    det_rt = Constraint(oops_constraint_num_df=all_cons, market=opex).get_constraint_details(da_or_rt='RT')
    det_da = Constraint(oops_constraint_num_df=all_cons, market=opex).get_constraint_details(da_or_rt='DA')
    details = (pd.concat([det_rt, det_da])
               .drop_duplicates('oops_constraint_num')
               [['oops_constraint_num', 'monitored_clean']]
               .rename(columns={'monitored_clean': 'monitored'}))

    da_sum = (mv_da.merge(details, on='oops_constraint_num', how='left')
              .groupby('monitored')['mvalue'].sum().rename('DA_mvalue').reset_index())
    rt_sum = (mv_rt.merge(details, on='oops_constraint_num', how='left')
              .groupby('monitored')['mvalue'].sum().rename('RT_mvalue').reset_index())

    merged = pd.merge(da_sum, rt_sum, on='monitored', how='outer').fillna(0)
    if merged.empty:
        return pd.DataFrame(columns=['monitored', 'DA_mvalue', 'RT_mvalue'])
    merged['monitored'] = merged['monitored'].str.strip()
    # coerce to numeric (column can be object dtype if a side was empty) before rounding
    merged['DA_mvalue'] = pd.to_numeric(merged['DA_mvalue'], errors='coerce').fillna(0).round(0).astype(int)
    merged['RT_mvalue'] = pd.to_numeric(merged['RT_mvalue'], errors='coerce').fillna(0).round(0).astype(int)
    return merged


def fetch_fundamentals(start_dt, end_dt):
    """
    Hourly reserve-zone wind and load forecasts, pivoted one column per zone.

    Ported from functions.get_weather_date(), which derives its start date from an
    hourly-mvalue frame this script does not build; the range is passed in instead.
    """
    rz_wind_df = Wind('SPP').get_res_zonal_wind(start_dt, end_dt, var_spec=['f'], pivot=True)
    rz_load_df = Load('SPP').get_res_zonal_load(start_dt, end_dt, var_spec=['f'], pivot=True)

    rz_wind_f_cols = [c for c in rz_wind_df.columns if 'forecast' in c]
    rz_load_f_cols = [c for c in rz_load_df.columns if 'forecast' in c]

    for frame in (rz_wind_df, rz_load_df):
        frame['dt'] = frame['dt'].astype(str)
        frame['hr'] = frame['hr'].astype(int)

    fundamentals = (rz_wind_df[['dt', 'hr'] + rz_wind_f_cols]
                    .merge(rz_load_df[['dt', 'hr'] + rz_load_f_cols], on=['dt', 'hr'], how='outer'))
    zones = sorted(set(int(c.split('_')[0][2:]) for c in rz_wind_f_cols))
    print('fundamentals {} .. {} | reserve zones {}'.format(start_dt, end_dt, zones))
    return fundamentals


def zone_percentiles(fundamentals, bid_dt):
    """
    Where each hour of bid_dt's forecast sits against the prior month, same hour, same zone.

    Ported from functions.predict_tomorrow_percentile() minus its display() calls, which
    are IPython-only and would raise NameError under plain python.
    """
    tmr = fundamentals[fundamentals['dt'] == bid_dt].copy()
    if tmr.empty:
        return pd.DataFrame()

    cutoff  = (pd.Timestamp(bid_dt) - pd.DateOffset(months=1)).strftime('%Y-%m-%d')
    history = fundamentals[(fundamentals['dt'] >= cutoff) & (fundamentals['dt'] < bid_dt)]

    zones = sorted(set(int(c.split('_')[0][2:]) for c in fundamentals.columns
                       if c.startswith('rz') and 'forecast' in c))

    rows = []
    for zone in zones:
        wind_col = 'rz{}_spp_res_zonal_wind_forecast_f'.format(zone)
        load_col = 'rz{}_spp_res_zonal_load_forecast_f'.format(zone)
        for _, row in tmr.iterrows():
            hr = int(row['hr'])
            wind_hist = history[history['hr'] == hr][wind_col].dropna()
            load_hist = history[history['hr'] == hr][load_col].dropna()
            wind_val  = row.get(wind_col)
            load_val  = row.get(load_col)
            rows.append({
                'zone': zone,
                'hr': hr,
                'wind_pct': round((wind_hist < wind_val).mean() * 100, 1) if pd.notna(wind_val) else None,
                'load_pct': round((load_hist < load_val).mean() * 100, 1) if pd.notna(load_val) else None,
            })

    result = pd.DataFrame(rows).sort_values(['zone', 'hr']).reset_index(drop=True)
    result['peak'] = result['hr'].apply(lambda h: 'off' if h <= 7 or h == 24 else 'on')
    return result


def wind_summary(fundamentals, bid_dt):
    """
    The "today's wind" cell text for one bid date — one line per reserve zone:

        1: ow: 51, fw: 16, ol: 83, fl: 86
        2: ow: 26, fw: 20, ol: 93, fl: 91

    ow/ol are the median on-peak wind/load percentiles, fw/fl the off-peak ones
    (off-peak being hr <= 7 or hr 24). Returns '' when the forecast is not published yet.
    """
    pct = zone_percentiles(fundamentals, bid_dt)
    if pct.empty:
        return ''

    lines = []
    for zone, grp in pct[pct['zone'].isin(SUMMARY_ZONES)].groupby('zone'):
        on  = grp[grp['peak'] == 'on']
        off = grp[grp['peak'] == 'off']
        vals = [on['wind_pct'].median(), off['wind_pct'].median(),
                on['load_pct'].median(), off['load_pct'].median()]
        if any(pd.isna(v) for v in vals):
            continue
        lines.append('{}: ow: {}, fw: {}, ol: {}, fl: {}'.format(zone, *[int(v) for v in vals]))
    return '\n'.join(lines)


def _lookup_metadata(monitored_name, df):
    prior = df[df['monitored'] == monitored_name]
    if prior.empty:
        return {c: '' for c in META_COLS}
    latest = prior.sort_values('bid_date').iloc[-1]
    return {c: latest.get(c, '') for c in META_COLS}


def _opportunity(monitored_name, df, before_dt):
    prior = df[(df['monitored'] == monitored_name) & (df['bid_date'] < before_dt)]
    if prior.empty:
        return 'new'
    return pd.Timestamp(prior['bid_date'].max()).strftime('%-m/%-d/%Y')


def update_constraints(df, start_dt, end_dt, fundamentals=None):
    """Refresh mvalues for [start_dt, end_dt], appending constraint/date pairs not yet present."""
    new_rows = []
    updated  = 0

    for dt in pd.date_range(start=start_dt, end=end_dt, freq='D'):
        dt_str = dt.strftime('%Y-%m-%d')
        print('\nFetching {}...'.format(dt_str))

        fetched = _fetch_mvalues(dt_str)
        fetched = fetched[
            (fetched['DA_mvalue'] <= THRESHOLD) | (fetched['RT_mvalue'] <= THRESHOLD)
        ].reset_index(drop=True)

        if fetched.empty:
            print('  no constraints below threshold')
            continue
        print('  {} constraint(s) found'.format(len(fetched)))

        existing_mask = df['bid_date'] == dt

        for _, frow in fetched.iterrows():
            name     = frow['monitored']
            opp      = _opportunity(name, df, before_dt=dt)
            row_mask = existing_mask & (df['monitored'] == name)

            if row_mask.any():
                df.loc[row_mask, 'DA_mvalue']   = frow['DA_mvalue']
                df.loc[row_mask, 'RT_mvalue']   = frow['RT_mvalue']
                df.loc[row_mask, 'opportunity'] = opp
                updated += int(row_mask.sum())
                print('    updated  : {} (opportunity={})'.format(name, opp))
            else:
                meta = _lookup_metadata(name, df)
                new_rows.append({'bid_date': dt, 'monitored': name,
                                 'DA_mvalue': frow['DA_mvalue'], 'RT_mvalue': frow['RT_mvalue'],
                                 **meta, "today's wind": '', 'opportunity': opp})
                print('    appended : {} (opportunity={})'.format(name, opp))

    result = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    result['bid_date'] = pd.to_datetime(result['bid_date'], format='mixed', errors='coerce')
    result = result.dropna(subset=['bid_date']).reset_index(drop=True)

    # Sort: by date first, then within each date by abs(RT - DA) descending
    result['_rank'] = (
        pd.to_numeric(result['RT_mvalue'], errors='coerce').fillna(0) -
        pd.to_numeric(result['DA_mvalue'], errors='coerce').fillna(0)
    ).abs()
    result = (result
              .sort_values(['bid_date', '_rank'], ascending=[True, False])
              .drop(columns='_rank')
              .reset_index(drop=True))

    # zonal wind/load percentile summary, on the first row of each processed date — the
    # sort above already put that date's top-ranked constraint there
    if fundamentals is not None:
        for dt in pd.date_range(start=start_dt, end=end_dt, freq='D'):
            rows = result.index[result['bid_date'] == dt]
            if not len(rows):
                continue
            summary = wind_summary(fundamentals, dt.strftime('%Y-%m-%d'))
            if not summary:
                print("  no zonal forecast for {}, today's wind left as is".format(dt.strftime('%Y-%m-%d')))
                continue
            result.loc[rows[0], "today's wind"] = summary
            print('  wind summary on {} row {}: {}'.format(
                dt.strftime('%Y-%m-%d'), rows[0], summary.replace('\n', ' | ')))

    result['bid_date'] = result['bid_date'].dt.strftime('%-m/%-d/%Y')

    return result, updated, len(new_rows)


def main():
    args    = [a for a in sys.argv[1:] if not a.startswith('--')]
    dry_run = '--dry-run' in sys.argv

    today    = pd.Timestamp.now(tz=TZ).normalize().tz_localize(None)
    start_dt = args[0] if len(args) > 0 else (today - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    end_dt   = args[1] if len(args) > 1 else (today + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    print('{}: {} .. {}{}'.format(script_name, start_dt, end_dt, '  [DRY RUN]' if dry_run else ''))

    df, src_cols, name_col = read_source()
    before = len(df)

    # zone_percentiles ranks each date against the prior month, so pull enough history
    fundamentals = fetch_fundamentals(
        (pd.Timestamp(start_dt) - pd.Timedelta(days=45)).strftime('%Y-%m-%d'), end_dt)

    result, updated, appended = update_constraints(df, start_dt, end_dt, fundamentals)

    # restore the file's own header spelling and column order
    result = result.rename(columns={'monitored': name_col})
    missing = [c for c in src_cols if c not in result.columns]
    for c in missing:
        result[c] = ''
    result = result[src_cols]

    print('\n' + '=' * 60)
    print('rows {} → {}  |  {} updated, {} appended'.format(before, len(result), updated, appended))
    print(result.tail(min(appended + 3, 10)).to_string())

    if dry_run:
        print('\n--dry-run: nothing written')
        return
    if not updated and not appended:
        print('nothing to do; leaving the object untouched')
        return

    backup_source()
    dataframe_functions.upload_df_to_storage(result, BUCKET, BLOB_PATH, type='csv')
    print('wrote {} rows → gs://{}/{}'.format(len(result), BUCKET, BLOB_PATH))

    print('\nProgram done.')


if __name__ == '__main__':
    main()
