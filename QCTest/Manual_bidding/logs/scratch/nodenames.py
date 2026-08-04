import builtins, os, sys
builtins.display = print
os.environ.setdefault('MPLBACKEND', 'Agg')
import pandas as pd
sys.path.append('/var/www/python/Prod/nighthawk')
from nighthawk.data import Node, Constraint
sys.path.append('/var/www/python/Qingcheng/QCTest/Manual_bidding')
from functions import get_recent_constraint_mvalue, get_node_dfax_from_constraint_num
pd.set_option('display.max_columns', None); pd.set_option('display.width', 250)

now    = pd.Timestamp.now(tz='US/Central')
bid_dt = (now + pd.Timedelta(days=2 if now.hour >= 10 else 1)).strftime('%Y-%m-%d')
dt     = (pd.Timestamp(bid_dt) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
mvalue = get_recent_constraint_mvalue(dt, dt, threshold=500)
dfax   = get_node_dfax_from_constraint_num(mvalue)

want = sorted(set(int(x) for x in dfax['node_num'].unique()))
nd = Node(want, 'SPP').get_node_details()
print('node_details cols:', list(nd.columns), len(nd))
nmap = nd.drop_duplicates('node_num').set_index('node_num')

name_by_con = mvalue.set_index('oops_constraint_num')['monitored'].to_dict()
dx = dfax.copy()
dx['monitored'] = dx['oops_constraint_num'].map(name_by_con)

# nodes the analysis surfaced as best per constraint
picks = {
    'lnbrookngs-aurorawa':       [887, 874, 924, 870],
    'lnmaurerlk-carroltn':       [835, 390, 1030],
    'lnmdwst-frnkln1':           [1588, 535, 1339, 1587, 1329, 749, 1715, 1358],
    'lnmonett-aur1241':          [134, 133, 1220, 145, 144, 1748, 155, 1264],
    'lnosage_og-webbtap4':       [546, 1117, 549, 547, 541, 523, 582],
    'lnrussett-sbrown':          [2094, 1499, 1532, 1533, 1734, 1735, 1522, 1152],
    'lnseminole-maud_tap':       [1736, 112, 113, 1314, 517, 1717],
    'xfmrftsmth-ftsmth':         [2105, 2103, 1298, 2082, 1288],
    'xfmrnowst1-nowst1':         [1442, 1082, 1081, 1577, 1083, 2051, 2091],
}
for con, nodes in picks.items():
    print('\n--- {} ---'.format(con))
    sub = dx[dx.monitored == con].drop_duplicates('node_num').set_index('node_num')
    for nn in nodes:
        d = sub.loc[nn, 'dfax'] if nn in sub.index else float('nan')
        if nn in nmap.index:
            r = nmap.loc[nn]
            print('  {:6d}  dfax {:+.4f}  {:28s} zone={} state={} type={}'.format(
                nn, d, str(r.get('node_name')), r.get('zone'), r.get('state'), r.get('node_type', '')))
        else:
            print('  {:6d}  dfax {:+.4f}  <no node_details row>'.format(nn, d))

# also: how many high-|dfax| nodes each constraint has
print('\n--- dfax spread per constraint ---')
for c, g in dx.groupby('monitored'):
    g = g.drop_duplicates('node_num')
    print('{:22s} nodes={:4d}  max|dfax|={:.3f}  n(|dfax|>0.2)={}'.format(
        c, len(g), g.dfax.abs().max(), (g.dfax.abs() > 0.2).sum()))

# the two constraints the main run could not price: do they have dfax rows at all?
print('\n--- dfax presence for the "could not find the dfax value" pair ---')
for c in ('ln6_me4-sw51', 'lnbois_tap-commtap4_rev23'):
    cons = [k for k, v in name_by_con.items() if v == c]
    sub  = dfax[dfax['oops_constraint_num'].isin(cons)]
    print('{:26s} con_nums={} dfax_rows={}'.format(c, cons, len(sub)))
