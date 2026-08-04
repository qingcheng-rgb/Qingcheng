import builtins, os, sys
builtins.display = print
os.environ.setdefault('MPLBACKEND', 'Agg')
import pandas as pd
sys.path.append('/var/www/python/Prod/nighthawk')
sys.path.append('/var/www/python/Qingcheng/QCTest/Manual_bidding')
from functions import (get_recent_constraint_mvalue, get_hourly_mvalue_for_constraint_num,
                       get_weather_date)
now    = pd.Timestamp.now(tz='US/Central')
bid_dt = (now + pd.Timedelta(days=2 if now.hour >= 10 else 1)).strftime('%Y-%m-%d')
dt     = (pd.Timestamp(bid_dt) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
mvalue = get_recent_constraint_mvalue(dt, dt, threshold=500)
h      = get_hourly_mvalue_for_constraint_num(mvalue)
f      = get_weather_date(h, bid_dt)

tom = f[f['dt'].astype(str) == bid_dt]
print('bid_dt={}  forecast hours available: {}'.format(bid_dt, len(tom)))
print('\n' + '='*104)
print("does tomorrow actually reach the top-5% window these constraints bind in?")
print('='*104)
print('{:<6} {:<6} {:>11} {:>11} {:>11} {:>10} {:>9}  {}'.format(
    'zone','kind','q95_2yr','tom_max','tom_max_hr','tom_med','pct_of_q95','reaches top-5%?'))
for z in [1,2,3,4,5,21]:
    for kind in ('wind','load'):
        c = 'rz{}_spp_res_zonal_{}_forecast_f'.format(z, kind)
        if c not in f.columns: continue
        v = f[c].dropna()
        if not len(v): continue
        q95 = v.quantile(0.95)
        tv  = tom[c].dropna()
        if not len(tv):
            print('{:<6} {:<6} {:>11.1f}  <no forecast for bid_dt>'.format(z, kind, q95)); continue
        mx  = tv.max(); hr = int(tom.loc[tv.idxmax(), 'hr'])
        nrch = int((tv > q95).sum())
        print('{:<6} {:<6} {:>11.1f} {:>11.1f} {:>11} {:>10.1f} {:>8.0f}%  {}'.format(
            z, kind, q95, mx, hr, tv.median(), 100*mx/q95,
            'YES ({}h)'.format(nrch) if nrch else 'no'))
