import builtins, io, os, sys
builtins.display = print
os.environ.setdefault('MPLBACKEND', 'Agg')
import pandas as pd
sys.path.append('/var/www/python/Prod/nighthawk')
from nighthawk.data import Node
sys.path.append('/var/www/python/Qingcheng/QCTest/Manual_bidding')
from functions import (get_recent_constraint_mvalue, get_node_dfax_from_constraint_num,
                       get_hourly_mvalue_for_constraint_num, get_weather_date,
                       analyze_constraint_by_zone)
pd.set_option('display.max_columns', None); pd.set_option('display.width', 250)

now    = pd.Timestamp.now(tz='US/Central')
bid_dt = (now + pd.Timedelta(days=2 if now.hour >= 10 else 1)).strftime('%Y-%m-%d')
dt     = (pd.Timestamp(bid_dt) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')

mvalue = get_recent_constraint_mvalue(dt, dt, threshold=500)
dfax   = get_node_dfax_from_constraint_num(mvalue)
h      = get_hourly_mvalue_for_constraint_num(mvalue)
fund   = get_weather_date(h, bid_dt)

# the six constraints the main run had no sheet reserve_zone for, so defaulted to 4
UNZONED = ['ln6_me4-sw51', 'lnbois_tap-commtap4_rev23', 'lnbrookngs-aurorawa',
           'lnmaurerlk-carroltn', 'lnmdwst-frnkln1', 'lnseminole-maud_tap']
ZONES   = [1, 2, 3, 4, 5, 21]

# --- what utility zone do each constraint's nodes actually sit in? --------------------
want = sorted(set(int(x) for x in dfax['node_num'].unique()))
nd   = Node(want, 'SPP').get_node_details().drop_duplicates('node_num').set_index('node_num')
name_by_con = mvalue.set_index('oops_constraint_num')['monitored'].to_dict()
dx = dfax.copy()
dx['monitored'] = dx['oops_constraint_num'].map(name_by_con)

print('=' * 100)
print('utility / rep zone of each constraint\'s dfax nodes (weighted by |dfax|)')
print('=' * 100)
for c, g in dx.groupby('monitored'):
    g = g.drop_duplicates('node_num').copy()
    g['zone']       = g['node_num'].map(nd['zone'])
    g['rep_zone']   = g['node_num'].map(nd['rep_zone'])
    g['broad_zone'] = g['node_num'].map(nd['broad_zone'])
    g['state']      = g['node_num'].map(nd['state'])
    top = g.reindex(g['dfax'].abs().sort_values(ascending=False).index).head(8)
    print('\n{:26s} nodes={}'.format(c, len(g)))
    print('   zone counts     : {}'.format(g['zone'].value_counts().head(5).to_dict()))
    print('   rep_zone counts : {}'.format(g['rep_zone'].value_counts().head(5).to_dict()))
    print('   broad_zone      : {}'.format(g['broad_zone'].value_counts().head(5).to_dict()))
    print('   states          : {}'.format(g['state'].value_counts().head(5).to_dict()))

# --- which reserve zone's fundamentals actually explain the binding? -----------------
# analyze_constraint_by_zone prints; capture its stdout and pull the lift rows out.
print('\n' + '=' * 100)
print('lift by candidate reserve zone — highest |lift| wind/load wins')
print('=' * 100)
for c in UNZONED:
    print('\n' + '-' * 100)
    print(c)
    print('-' * 100)
    for z in ZONES:
        buf, old = io.StringIO(), sys.stdout
        sys.stdout = buf
        try:
            analyze_constraint_by_zone(fund, h, c, z, 95)
        except Exception as e:
            sys.stdout = old
            print('  zone {:>2}: failed {}: {}'.format(z, type(e).__name__, e))
            continue
        sys.stdout = old
        out = buf.getvalue()
        wind = load = None
        for line in out.splitlines():
            if 'wind_forecast' in line:
                wind = line.split()[-1]
            elif 'load_forecast' in line:
                load = line.split()[-1]
        print('  zone {:>2}:  wind_lift={:>8}  load_lift={:>8}'.format(z, str(wind), str(load)))
