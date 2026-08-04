import builtins, io, os, sys
builtins.display = print
os.environ.setdefault('MPLBACKEND', 'Agg')

import pandas as pd
from google.cloud import storage
sys.path.append('/var/www/python/Prod/nighthawk')
from nighthawk.data import Constraint, Outage, Node
sys.path.append('/var/www/python/Qingcheng/QCTest/Manual_bidding')
from functions import (get_recent_constraint_mvalue, get_node_dfax_from_constraint_num,
                       get_hourly_mvalue_for_constraint_num)

pd.set_option('display.max_columns', None); pd.set_option('display.width', 250)

now    = pd.Timestamp.now(tz='US/Central')
bid_dt = (now + pd.Timedelta(days=2 if now.hour >= 10 else 1)).strftime('%Y-%m-%d')
dt     = (pd.Timestamp(bid_dt) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')

mvalue = get_recent_constraint_mvalue(dt, dt, threshold=500)
dfax   = get_node_dfax_from_constraint_num(mvalue)
h      = get_hourly_mvalue_for_constraint_num(mvalue)
h['dt'] = pd.to_datetime(h['dt'])

print('\n\n######## RT hour coverage on dt={} ########'.format(dt))
today = h[h['dt'] == pd.Timestamp(dt)]
for n, g in today.groupby('monitored'):
    rt = g[g['rt_mvalue'] != 0]
    da = g[g['da_mvalue'] != 0]
    print('{:22s} RT hrs={} (min {}, max {})  DA hrs={} (min {}, max {})'.format(
        n, len(rt), rt['hr'].min() if len(rt) else '-', rt['hr'].max() if len(rt) else '-',
        len(da), da['hr'].min() if len(da) else '-', da['hr'].max() if len(da) else '-'))

print('\n\n######## trailing 21 days: daily DA / RT totals by monitored ########')
recent = h[h['dt'] >= pd.Timestamp(dt) - pd.Timedelta(days=21)]
piv = recent.groupby(['monitored', 'dt'])[['rt_mvalue', 'da_mvalue']].sum().round(0)
for n, g in piv.groupby(level=0):
    print('\n--- {} ---'.format(n))
    g = g.droplevel(0)
    g['rt_hrs'] = recent[recent.monitored == n].assign(b=lambda d: d.rt_mvalue != 0).groupby('dt')['b'].sum()
    g['da_hrs'] = recent[recent.monitored == n].assign(b=lambda d: d.da_mvalue != 0).groupby('dt')['b'].sum()
    print(g.to_string())

print('\n\n######## last-year same calendar window (2025-07-20 .. 2025-08-16) ########')
ly = h[(h['dt'] >= '2025-07-20') & (h['dt'] <= '2025-08-16')]
if ly.empty:
    print('  nothing bound in that window for any candidate')
for n, g in ly.groupby('monitored'):
    d = g.groupby('dt')[['rt_mvalue', 'da_mvalue']].sum().round(0)
    d = d[(d.rt_mvalue != 0) | (d.da_mvalue != 0)]
    print('\n--- {} : {} day(s) ---'.format(n, len(d)))
    print(d.to_string())

print('\n\n######## full 104-week footprint by monitored ########')
for n, g in h.groupby('monitored'):
    rtd = g[g.rt_mvalue != 0]['dt'].nunique(); dad = g[g.da_mvalue != 0]['dt'].nunique()
    print('{:22s} first={}  last={}  RT days={}  DA days={}  RT sum={:.0f}  DA sum={:.0f}'.format(
        n, g['dt'].min().date(), g['dt'].max().date(), rtd, dad,
        g.rt_mvalue.sum(), g.da_mvalue.sum()))

print('\n\n######## node names for candidate nodes ########')
nd = Node('SPP').get_node_details()
nmap = nd.set_index('node_num')
name_by_con = mvalue.set_index('oops_constraint_num')['monitored'].to_dict()
dx = dfax.copy()
dx['monitored'] = dx['oops_constraint_num'].map(name_by_con)
for n, g in dx.groupby('monitored'):
    g = g.reindex(g.dfax.abs().sort_values(ascending=False).index).drop_duplicates('node_num').head(12)
    print('\n--- {} ---'.format(n))
    for _, r in g.iterrows():
        info = nmap.loc[r.node_num] if r.node_num in nmap.index else None
        if info is not None and hasattr(info, 'ndim') and info.ndim > 1:
            info = info.iloc[0]
        print('  node {:6d}  dfax {:+.4f}  {}'.format(
            int(r.node_num), r.dfax,
            '{} | zone={} | state={}'.format(info.get('node_name'), info.get('zone'), info.get('state'))
            if info is not None else 'unknown'))

print('\n\n######## scheduled outages on {} that POSTED in the last 10 days ########'.format(bid_dt))
so = Outage('SPP').get_scheduled_outages_for_frontend(
    start_date=bid_dt, end_date=bid_dt, viewing_time=now.strftime('%Y-%m-%d %H:%M:%S'),
    latest_schedule=True)
so['start'] = pd.to_datetime(so['outageStartDtTime'], errors='coerce')
fresh = so[so['start'] >= pd.Timestamp(dt) - pd.Timedelta(days=10)].sort_values('start')
print(fresh[['outageName', 'eqNum', 'outageStartDtTime', 'outageEndDtTime', 'voltage']].to_string(index=False))
print('\ncolumns available on sched_outages:', list(so.columns))
