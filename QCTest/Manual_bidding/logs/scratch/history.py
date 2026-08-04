import builtins, os, sys
builtins.display = print
os.environ.setdefault('MPLBACKEND', 'Agg')

import pandas as pd
sys.path.append('/var/www/python/Prod/nighthawk')
sys.path.append('/var/www/python/Qingcheng/QCTest/Manual_bidding')
from functions import (get_recent_constraint_mvalue,
                       get_hourly_mvalue_for_constraint_num)

now    = pd.Timestamp.now(tz='US/Central')
bid_dt = (now + pd.Timedelta(days=2 if now.hour >= 10 else 1)).strftime('%Y-%m-%d')
dt     = (pd.Timestamp(bid_dt) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')

mvalue = get_recent_constraint_mvalue(dt, dt, threshold=500)
h      = get_hourly_mvalue_for_constraint_num(mvalue)
h['d'] = pd.to_datetime(h['dt'])

DT   = pd.Timestamp(dt)
BID  = pd.Timestamp(bid_dt)
week0, week1 = DT - pd.Timedelta(days=7), DT - pd.Timedelta(days=1)
ly0,  ly1    = BID - pd.Timedelta(days=372), BID - pd.Timedelta(days=358)

print('=' * 100)
print('dt={}  bid_dt={}'.format(dt, bid_dt))
print('trailing week window : {} .. {}'.format(week0.date(), week1.date()))
print('last-year window     : {} .. {}  (bid_dt -365d +/- 7d)'.format(ly0.date(), ly1.date()))
print('=' * 100)

hdr = ('{:<26} {:>7} {:>7} | {:>6} {:>6} {:>9} | {:>6} {:>6} {:>9} {:>9}'
       .format('monitored', 'dt_DA', 'dt_RT', 'wkDA', 'wkRT', 'wk_maxRT',
               'lyDA', 'lyRT', 'ly_maxDA', 'ly_maxRT'))
print(hdr)
print('-' * len(hdr))

rows = []
for name, g in h.groupby('monitored'):
    on_dt = g[g['d'] == DT]
    wk    = g[(g['d'] >= week0) & (g['d'] <= week1)]
    ly    = g[(g['d'] >= ly0)  & (g['d'] <= ly1)]

    def days(f, col):
        return f.loc[f[col] != 0, 'd'].dt.date.nunique()

    rows.append(dict(
        monitored=name,
        dt_da=round(on_dt['da_mvalue'].abs().max() or 0, 1) if len(on_dt) else 0,
        dt_rt=round(on_dt['rt_mvalue'].abs().max() or 0, 1) if len(on_dt) else 0,
        wk_da=days(wk, 'da_mvalue'), wk_rt=days(wk, 'rt_mvalue'),
        wk_maxrt=round(wk['rt_mvalue'].abs().max() or 0, 1) if len(wk) else 0,
        ly_da=days(ly, 'da_mvalue'), ly_rt=days(ly, 'rt_mvalue'),
        ly_maxda=round(ly['da_mvalue'].abs().max() or 0, 1) if len(ly) else 0,
        ly_maxrt=round(ly['rt_mvalue'].abs().max() or 0, 1) if len(ly) else 0))

for r in sorted(rows, key=lambda x: -x['dt_da']):
    print('{monitored:<26} {dt_da:>7} {dt_rt:>7} | {wk_da:>6} {wk_rt:>6} {wk_maxrt:>9} | '
          '{ly_da:>6} {ly_rt:>6} {ly_maxda:>9} {ly_maxrt:>9}'.format(**r))

# how far back does each constraint's record go, and its all-time binding day count
print('\n' + '=' * 100)
print('record depth (all 104 weeks)')
print('=' * 100)
print('{:<26} {:>12} {:>12} {:>8} {:>8}'.format('monitored', 'first_dt', 'last_dt', 'DAdays', 'RTdays'))
for name, g in h.groupby('monitored'):
    print('{:<26} {:>12} {:>12} {:>8} {:>8}'.format(
        name, str(g['d'].min().date()), str(g['d'].max().date()),
        g.loc[g['da_mvalue'] != 0, 'd'].dt.date.nunique(),
        g.loc[g['rt_mvalue'] != 0, 'd'].dt.date.nunique()))

# month-of-year profile for the thin-record constraints, to test seasonality
print('\n' + '=' * 100)
print('binding days by month (DA / RT) — seasonality check')
print('=' * 100)
for name, g in h.groupby('monitored'):
    da = g.loc[g['da_mvalue'] != 0].groupby(g['d'].dt.month)['d'].apply(lambda s: s.dt.date.nunique())
    rt = g.loc[g['rt_mvalue'] != 0].groupby(g['d'].dt.month)['d'].apply(lambda s: s.dt.date.nunique())
    parts = []
    for m in range(1, 13):
        a, b = int(da.get(m, 0)), int(rt.get(m, 0))
        if a or b:
            parts.append('{}:{}/{}'.format(m, a, b))
    print('{:<26} {}'.format(name, ' '.join(parts) if parts else '(none)'))
