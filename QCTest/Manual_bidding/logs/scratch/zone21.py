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

print('\n' + '='*95)
print('per-zone forecast density and size of the top-5% conditioning window')
print('='*95)
print('{:<38} {:>8} {:>8} {:>10} {:>12}'.format('column','non_nan','nan','n_above_q95','distinct'))
for z in [1,2,3,4,5,21]:
    for kind in ('wind','load'):
        c = 'rz{}_spp_res_zonal_{}_forecast_f'.format(z, kind)
        if c not in f.columns: continue
        v = f[c].dropna()
        thr = v.quantile(0.95)
        print('{:<38} {:>8} {:>8} {:>10} {:>12}'.format(
            c, len(v), int(f[c].isna().sum()), int((v > thr).sum()), v.nunique()))

# recompute lift for ln6_me4-sw51 honestly, showing the raw counts
print('\n' + '='*95)
print('raw counts behind the lift, for the thin-record names (binding = rt_da != 0)')
print('='*95)
grid = pd.date_range(h['dt'].min(), h['dt'].max(), freq='D')
full = pd.DataFrame([{'dt': d.strftime('%Y-%m-%d'), 'hr': hh} for d in grid for hh in range(1,25)])
for name in ['ln6_me4-sw51','lnbois_tap-commtap4_rev23','lnbrookngs-aurorawa',
             'lnmdwst-frnkln1','lnseminole-maud_tap','xfmrnowst1-nowst1']:
    g = h[h['monitored'] == name]
    b = g[g['rt_da'] != 0][['dt','hr']].drop_duplicates().copy()
    b['dt'] = b['dt'].astype(str); b['hr'] = b['hr'].astype(int); b['binding'] = 1
    tmp = full.merge(f, on=['dt','hr'], how='left').merge(b, on=['dt','hr'], how='left')
    tmp['binding'] = tmp['binding'].fillna(0)
    base = tmp['binding'].mean()
    print('\n{}  total_bind_hrs={}  baseline={:.5f}'.format(name, int(tmp['binding'].sum()), base))
    for z in [4, 21]:
        for kind in ('load','wind'):
            c = 'rz{}_spp_res_zonal_{}_forecast_f'.format(z, kind)
            valid = tmp[[c,'binding']].dropna()
            thr = valid[c].quantile(0.95)
            above = valid[valid[c] > thr]
            k = int(above['binding'].sum())
            print('   z{:<3}{:<5} window={:>5}h  bind_in_window={:>3}  cond={:.4f}  lift={:>7.2f}'
                  .format(z, kind, len(above), k, above['binding'].mean(),
                          above['binding'].mean()/base - 1))
