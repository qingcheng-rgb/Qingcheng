import builtins, os, sys
builtins.display = print
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np, pandas as pd
sys.path.append('/var/www/python/Prod/nighthawk')
from nighthawk.data import Node
sys.path.append('/var/www/python/Qingcheng/QCTest/Manual_bidding')
from functions import (get_recent_constraint_mvalue, get_node_dfax_from_constraint_num,
                       get_hourly_mvalue_for_constraint_num, get_constraints_node_price)
pd.set_option('display.max_columns', None); pd.set_option('display.width', 250)
pd.set_option('display.max_rows', 60)

now    = pd.Timestamp.now(tz='US/Central')
bid_dt = (now + pd.Timedelta(days=2 if now.hour >= 10 else 1)).strftime('%Y-%m-%d')
dt     = (pd.Timestamp(bid_dt) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
mvalue = get_recent_constraint_mvalue(dt, dt, threshold=500)
dfax   = get_node_dfax_from_constraint_num(mvalue)
h      = get_hourly_mvalue_for_constraint_num(mvalue)

nd   = Node(sorted(set(int(x) for x in dfax.node_num.unique())), 'SPP').get_node_details()
nmap = nd.drop_duplicates('node_num').set_index('node_num')

def stats(g):
    x, y = g.predicted_node_rt_da, g.node_rt_da
    m = x.notna() & y.notna(); x, y = x[m], y[m]
    if len(x) < 3:
        return pd.Series({'dfax': g.dfax.iloc[0], 'corr': np.nan, 'sign_acc': np.nan,
                          'mean_pred': np.nan, 'mean_actual': np.nan, 'n': len(x)})
    return pd.Series({'dfax': g.dfax.iloc[0], 'corr': x.corr(y),
                      'sign_acc': (np.sign(x) == np.sign(y)).mean(),
                      'mean_pred': x.mean(), 'mean_actual': y.mean(), 'n': len(x)})

for name in ['lnnebrcty-sub3456', 'xfmrover-over']:
    t = get_constraints_node_price(h, dfax, [name])
    t['node_rt_da'] = t.node_rt_lmp - t.node_da_lmp
    t['predicted_node_rt_da'] = -t.dfax * t.rt_da
    a = t[t.rt_da != 0]
    # restrict to the last 120 days so the picks reflect the current topology
    a_recent = a[pd.to_datetime(a.dt) >= pd.Timestamp(dt) - pd.Timedelta(days=120)]
    for label, frame in (('all history', a), ('last 120 days', a_recent)):
        s = frame.groupby('node_num').apply(stats).reset_index()
        s = s[s.n >= 30]
        s['node_name'] = s.node_num.map(nmap.node_name)
        s['zone'] = s.node_num.map(nmap.zone)
        s['absdfax'] = s.dfax.abs()
        print('\n\n########## {} — {} — top 15 by corr (n>=30) ##########'.format(name, label))
        print(s.sort_values('corr', ascending=False).head(15).to_string(index=False))
        print('\n---------- {} — {} — top 15 by |dfax| (n>=30) ----------'.format(name, label))
        print(s.sort_values('absdfax', ascending=False).head(15).to_string(index=False))

# and the same recency check for the two small ones
for name in ['lnrussett-sbrown', 'lnmryvl_sj-midway5']:
    t = get_constraints_node_price(h, dfax, [name])
    t['node_rt_da'] = t.node_rt_lmp - t.node_da_lmp
    t['predicted_node_rt_da'] = -t.dfax * t.rt_da
    a = t[(t.rt_da != 0) & (pd.to_datetime(t.dt) >= pd.Timestamp(dt) - pd.Timedelta(days=120))]
    s = a.groupby('node_num').apply(stats).reset_index()
    s['node_name'] = s.node_num.map(nmap.node_name)
    s['zone'] = s.node_num.map(nmap.zone)
    print('\n\n########## {} — last 120 days ##########'.format(name))
    print(s.sort_values('corr', ascending=False).to_string(index=False))
