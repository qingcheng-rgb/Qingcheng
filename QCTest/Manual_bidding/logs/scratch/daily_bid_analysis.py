import builtins, io, os, sys
builtins.display = print                      # functions.py calls display() in several places
os.environ.setdefault('MPLBACKEND', 'Agg')    # plot helpers must not open a window

import pandas as pd
from google.cloud import storage
sys.path.append('/var/www/python/Prod/nighthawk')
from nighthawk.data import Constraint, Outage
sys.path.append('/var/www/python/Qingcheng/QCTest/Manual_bidding')
from functions import (
    get_recent_constraint_mvalue, get_hourly_mvalue_for_constraint_num,
    get_weather_date, predict_tomorrow_percentile, analyze_constraint_by_zone,
)

# same bid-date rule the notebook uses: after 10am CT we are bidding two days out.
now      = pd.Timestamp.now(tz='US/Central')
bid_dt   = (now + pd.Timedelta(days=2 if now.hour >= 10 else 1)).strftime('%Y-%m-%d')
dt       = (pd.Timestamp(bid_dt) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
today_dt = now.strftime('%Y-%m-%d')
yest_dt  = (now - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
print('bid_dt={}  dt(DA cleared)={}  today={}  yesterday={}'.format(bid_dt, dt, today_dt, yest_dt))

# --- candidates from two sources ------------------------------------------------------
mvalue_da = get_recent_constraint_mvalue(dt, dt, threshold=500)
mvalue_rt = get_recent_constraint_mvalue(yest_dt, today_dt, threshold=500)

mvalue = (pd.concat([mvalue_da, mvalue_rt], ignore_index=True)
            .drop_duplicates('oops_constraint_num'))
da_names = set(mvalue_da['monitored'].dropna())
rt_names     = set(mvalue_rt.loc[mvalue_rt['rt_total'] != 0, 'monitored'].dropna())
recent_names = set(mvalue_rt['monitored'].dropna())
print('\nDA candidates for {}: {}'.format(bid_dt, len(da_names)))
print('RT binders {} to {}: {}'.format(yest_dt, today_dt, len(rt_names)))
print('RT-only (not in tomorrow DA): {}'.format(sorted(rt_names - da_names)))

hourly_mvalue = get_hourly_mvalue_for_constraint_num(mvalue)

h = hourly_mvalue
first8 = (h[(h['dt'] == today_dt) & (h['hr'] <= 8)]
            .groupby('monitored')[['rt_mvalue', 'da_mvalue']].sum()
            .query('rt_mvalue != 0 or da_mvalue != 0')
            .sort_values('rt_mvalue'))
yest = (h[h['dt'] == yest_dt]
          .groupby('monitored')[['rt_mvalue', 'da_mvalue']].sum()
          .query('rt_mvalue != 0 or da_mvalue != 0')
          .sort_values('rt_mvalue'))
print('\n--- today {} hours 1-8, RT ---'.format(today_dt))
print(first8.to_string() if len(first8) else '  nothing bound yet')
print('\n--- yesterday {} full day ---'.format(yest_dt))
print(yest.to_string() if len(yest) else '  nothing bound')

recent = h[h['dt'] >= (pd.Timestamp(today_dt) - pd.Timedelta(days=30)).strftime('%Y-%m-%d')]
freq = (recent[recent['rt_mvalue'] != 0].groupby('monitored')['dt'].nunique()
              .rename('rt_days_last_30').sort_values(ascending=False))
print('\n--- RT binding days in the last 30 ---')
print(freq.to_string() if len(freq) else '  none')

# --- zonal setup for the bid date -----------------------------------------------------
fundamentals = get_weather_date(hourly_mvalue, bid_dt)
tomorrow_pct = predict_tomorrow_percentile(fundamentals, bid_dt)

# --- outages --------------------------------------------------------------------------
con_nums = mvalue[['oops_constraint_num']].drop_duplicates()
linked_outages = Constraint(oops_constraint_num_df=con_nums,
                            market='SPP').get_linked_outage_string_for_frontend(dt=dt)
print('\nlinked outages:')
print(linked_outages.to_string() if len(linked_outages) else '  none')

sched_outages = Outage('SPP').get_scheduled_outages_for_frontend(
    start_date=bid_dt, end_date=bid_dt,
    viewing_time=now.strftime('%Y-%m-%d %H:%M:%S'), latest_schedule=True)
print('\nscheduled outages in effect on {}: {}'.format(bid_dt, len(sched_outages)))
if len(sched_outages):
    print(sched_outages[['outageName', 'eqNum', 'outageStartDtTime',
                         'outageEndDtTime', 'voltage']].to_string(index=False))

# --- the sheet, for reserve_zone ONLY -------------------------------------------------
blob  = storage.Client().bucket('spptest').blob('manual/Daily Bidding - daily_constraint_manual.csv')
sheet = pd.read_csv(io.StringIO(blob.download_as_text()))
sheet['_d'] = pd.to_datetime(sheet['bid_date'], format='mixed', errors='coerce')
zone_of = {}
for name, g in sheet.groupby(sheet['constraints'].astype(str).str.strip()):
    z = g.sort_values('_d')['reserve_zone'].dropna()
    if len(z):
        try:
            zone_of[name] = int(str(z.iloc[-1]).split(',')[0])
        except ValueError:
            pass
print('\nsheet rows: {} | latest bid_date: {} | zone mappings: {}'.format(
    len(sheet), sheet['_d'].max().date(), len(zone_of)))

# --- wind / load attribution per candidate -------------------------------------------
names = sorted(da_names | recent_names)
print('\ncandidates ({}): {}'.format(len(names), names))
for name in names:
    src = ('DA+RT' if name in da_names and name in rt_names
           else 'DA only' if name in da_names else 'RT only' if name in rt_names
           else 'recent DA only')
    print('\n' + '=' * 70)
    print('{}  |  source: {}  |  zone {}  |  RT days last 30: {}'.format(
        name, src, zone_of.get(name, '4 (DEFAULT - attribution unreliable)'),
        int(freq.get(name, 0))))
    try:
        analyze_constraint_by_zone(fundamentals, hourly_mvalue, name, zone_of.get(name, 4), 95)
    except Exception as e:
        print('  uplift failed: {}: {}'.format(type(e).__name__, e))

import json
json.dump(names, open('/var/www/python/Qingcheng/QCTest/Manual_bidding/logs/scratch/names.json', 'w'))
