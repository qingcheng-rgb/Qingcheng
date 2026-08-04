import builtins, os, sys
builtins.display = print
os.environ.setdefault('MPLBACKEND', 'Agg')
import pandas as pd
sys.path.append('/var/www/python/Prod/nighthawk')
from nighthawk.data import Outage
sys.path.append('/var/www/python/Qingcheng/QCTest/Manual_bidding')
from functions import get_recent_constraint_mvalue, get_hourly_mvalue_for_constraint_num
pd.set_option('display.max_columns', None); pd.set_option('display.width', 250)

now    = pd.Timestamp.now(tz='US/Central')
bid_dt = (now + pd.Timedelta(days=2 if now.hour >= 10 else 1)).strftime('%Y-%m-%d')
dt     = (pd.Timestamp(bid_dt) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
print('now (CT) =', now)

mvalue = get_recent_constraint_mvalue(dt, dt, threshold=500)
h = get_hourly_mvalue_for_constraint_num(mvalue)
h['dt'] = pd.to_datetime(h['dt'])

print('\n######## last bind BEFORE dt, per candidate ########')
prior = h[h['dt'] < pd.Timestamp(dt)]
for n, g in prior.groupby('monitored'):
    rt = g[g.rt_mvalue != 0]; da = g[g.da_mvalue != 0]
    print('{:22s} last RT={}  last DA={}'.format(
        n, rt['dt'].max().date() if len(rt) else 'never',
        da['dt'].max().date() if len(da) else 'never'))

print('\n######## xfmrover-over: every day it bound in 2026 ########')
x = h[(h.monitored == 'xfmrover-over') & (h.dt >= '2026-01-01')]
print(x.groupby('dt')[['rt_mvalue', 'da_mvalue']].sum().round(0).query('rt_mvalue != 0').to_string())

print('\n######## lnmryvl_sj-midway5: every day it bound in 2026 ########')
m = h[(h.monitored == 'lnmryvl_sj-midway5') & (h.dt >= '2026-01-01')]
print(m.groupby('dt')[['rt_mvalue', 'da_mvalue']].sum().round(0).query('rt_mvalue != 0').to_string())

print('\n######## xfmrover-over hourly profile, historical (hr -> mean RT mvalue, n days) ########')
xa = h[(h.monitored == 'xfmrover-over') & (h.rt_mvalue != 0)]
print(xa.groupby('hr')['rt_mvalue'].agg(['mean', 'count']).round(0).to_string())

print('\n######## outages scheduled on {} that POSTED/STARTED in the last 10 days ########'.format(bid_dt))
so = Outage('SPP').get_scheduled_outages_for_frontend(
    start_date=bid_dt, end_date=bid_dt, viewing_time=now.strftime('%Y-%m-%d %H:%M:%S'),
    latest_schedule=True)
so['start'] = pd.to_datetime(so['outageStartDtTime'], errors='coerce')
fresh = so[so['start'] >= pd.Timestamp(dt) - pd.Timedelta(days=10)].sort_values('start')
print(fresh[['outageName', 'eqNum', 'outageStartDtTime', 'outageEndDtTime', 'voltage']].to_string(index=False))
print('\ntotal scheduled on bid_dt:', len(so), '| >=115kV:', (so.voltage >= 115).sum())
print('\n>=138kV scheduled on bid_dt:')
print(so[so.voltage >= 138].sort_values('start')[
    ['outageName', 'eqNum', 'outageStartDtTime', 'outageEndDtTime', 'voltage']].to_string(index=False))
