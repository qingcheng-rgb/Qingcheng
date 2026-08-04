import builtins, io, os, sys
builtins.display = print                      # functions.py calls display() in several places
os.environ.setdefault('MPLBACKEND', 'Agg')    # plot helpers must not open a window

import pandas as pd
from google.cloud import storage
sys.path.append('/var/www/python/Prod/nighthawk')
from nighthawk.data import Constraint, Outage
sys.path.append('/var/www/python/Qingcheng/QCTest/Manual_bidding')
from functions import (
    get_recent_constraint_mvalue, get_node_dfax_from_constraint_num,
    get_hourly_mvalue_for_constraint_num, get_constraints_node_price,
    get_metrics_for_nodes, get_weather_date, predict_tomorrow_percentile,
    analyze_constraint_by_zone,
)

# same bid-date rule the notebook uses: after 10am CT we are bidding two days out
now        = pd.Timestamp.now(tz='US/Central')
bid_dt     = (now + pd.Timedelta(days=2 if now.hour >= 10 else 1)).strftime('%Y-%m-%d')
dt         = (pd.Timestamp(bid_dt) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
print('bid_dt={}  mvalue_dt={}'.format(bid_dt, dt))

mvalue        = get_recent_constraint_mvalue(dt, dt, threshold=500)
dfax          = get_node_dfax_from_constraint_num(mvalue)
hourly_mvalue = get_hourly_mvalue_for_constraint_num(mvalue)
# NOTE: the second argument is the END of the forecast window, so it must be bid_dt.
# Passing dt (as notebook cell 2 does) stops the frame one day short and
# predict_tomorrow_percentile then prints "No forecast data" and returns empty.
fundamentals  = get_weather_date(hourly_mvalue, bid_dt)
tomorrow_pct  = predict_tomorrow_percentile(fundamentals, bid_dt)

# --- outages -------------------------------------------------------------------------
# Same source the Daily Constraints page uses for its linkedOutage column: monitored-side
# linked outages, falling back to contingency-side, and to scheduled times when the RT
# start/end are missing. Format: "__outage_name__ FROM __start__ TO __end__".
con_nums = mvalue[['oops_constraint_num']].drop_duplicates()
linked_outages = Constraint(oops_constraint_num_df=con_nums,
                            market='SPP').get_linked_outage_string_for_frontend(dt=dt)
print('\nlinked outages:')
print(linked_outages.to_string() if len(linked_outages) else '  none')

# Everything scheduled to be in effect on the bid date, as known right now — this is what
# catches an outage that posted after the constraint last bound.
sched_outages = Outage('SPP').get_scheduled_outages_for_frontend(
    start_date=bid_dt, end_date=bid_dt,
    viewing_time=now.strftime('%Y-%m-%d %H:%M:%S'), latest_schedule=True)
print('\nscheduled outages in effect on {}: {}'.format(bid_dt, len(sched_outages)))
if len(sched_outages):
    print(sched_outages[['outageName', 'eqNum', 'outageStartDtTime',
                         'outageEndDtTime', 'voltage']].to_string(index=False))

# the bidding sheet, for cross-reference
blob  = storage.Client().bucket('spptest').blob('manual/Daily Bidding - daily_constraint_manual.csv')
sheet = pd.read_csv(io.StringIO(blob.download_as_text()))
sheet['_d'] = pd.to_datetime(sheet['bid_date'], format='mixed', errors='coerce')
print('\nsheet rows: {} | latest bid_date: {}'.format(len(sheet), sheet['_d'].max().date()))

names = sorted(hourly_mvalue['monitored'].unique())
print('\ncandidate constraints ({}): {}'.format(len(names), names))

for name in names:
    seen = sheet[sheet['constraints'].astype(str).str.strip() == name]
    print('\n' + '=' * 70)
    print('{}  |  {}'.format(
        name,
        'NEW — never in the sheet' if seen.empty else
        'seen {} time(s), last {}'.format(len(seen), seen['_d'].max().date())))
    if not seen.empty:
        last = seen.sort_values('_d').iloc[-1]
        for c in ('location', 'physical_condition', 'outage_name', 'wind', 'reserve_zone'):
            if pd.notna(last.get(c)) and str(last.get(c)).strip():
                print('  {:20s}: {}'.format(c, last[c]))
    zone = 4
    if not seen.empty and pd.notna(last.get('reserve_zone')):
        try:
            zone = int(str(last['reserve_zone']).split(',')[0])
        except ValueError:
            pass
    try:
        analyze_constraint_by_zone(fundamentals, hourly_mvalue, name, zone, 95)
        table = get_constraints_node_price(hourly_mvalue, dfax, [name])
        get_metrics_for_nodes(table)
    except Exception as e:
        print('  analysis failed: {}: {}'.format(type(e).__name__, e))
