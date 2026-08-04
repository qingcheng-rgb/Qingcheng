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

print('\n' + '='*90)
print('fundamentals coverage vs the 104-week grid the baseline is averaged over')
print('='*90)
print('mv_merged dt range : {} .. {}'.format(h['dt'].min(), h['dt'].max()))
grid = pd.date_range(h['dt'].min(), h['dt'].max(), freq='D')
print('full_grid hours    : {}  ({} days x 24)'.format(len(grid)*24, len(grid)))
print('fundamentals rows  : {}'.format(len(f)))
print('fundamentals dt rng: {} .. {}'.format(f['dt'].min(), f['dt'].max()))
print('=> cond_prob is estimated on {:.1f}% of the hours the baseline uses'
      .format(100*len(f)/(len(grid)*24)))

# where does each constraint's "binding" (rt_da != 0) mass sit relative to that window?
print('\n' + '='*90)
print('share of each constraint\'s rt_da!=0 hours that fall inside the fundamentals window')
print('='*90)
fk = set(zip(f['dt'].astype(str), f['hr'].astype(int)))
print('{:<26} {:>10} {:>10} {:>9}  {}'.format('monitored','bind_hrs','in_window','share','verdict'))
for name, g in h.groupby('monitored'):
    b = g[g['rt_da'] != 0][['dt','hr']].drop_duplicates()
    if not len(b):
        print('{:<26} {:>10} {:>10} {:>9}'.format(name, 0, 0, 'n/a')); continue
    inw = sum(1 for d,hh in zip(b['dt'].astype(str), b['hr'].astype(int)) if (d,hh) in fk)
    share = inw/len(b)
    verdict = 'LIFT INFLATED' if share > 0.5 else ('caution' if share > 0.2 else 'lift usable')
    print('{:<26} {:>10} {:>10} {:>8.1%}  {}'.format(name, len(b), inw, share, verdict))
