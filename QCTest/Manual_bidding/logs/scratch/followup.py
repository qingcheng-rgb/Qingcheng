import builtins, os, re, sys
builtins.display = print
os.environ.setdefault('MPLBACKEND', 'Agg')
import pandas as pd
S = '/var/www/python/Qingcheng/QCTest/Manual_bidding/logs/scratch'
txt = open(S + '/attribution.txt').read()

# --- the _rev23 section for nowst1, plus any section mentioning our RT-only names -------
hdrs = []
for m in re.finditer(r'Monitored Element:\s*', txt):
    nm = re.match(r'[a-z0-9_.\-]+', re.sub(r'\s+', '', txt[m.end():m.end() + 80]))
    if nm:
        hdrs.append((m.start(), nm.group(0)))
sec = {}
for i, (pos, nm) in enumerate(hdrs):
    sec.setdefault(nm, txt[pos:hdrs[i + 1][0] if i + 1 < len(hdrs) else len(txt)])

for name in ['xfmrnowst1-nowst1_rev23']:
    body = sec[name]
    d = re.search(r'Analysis Date:\s*([\d\-]+)', body)
    print('=' * 70 + '\n{}  (analysis date {})'.format(name, d.group(1) if d else '?'))
    for lbl, pat in (('SUMMARY',  r'3\.1 Summary:.*?(?=3\.2|\Z)'),
                     ('OUTAGES',  r'3\.3 Outages Table:(.*?)(?=3\.4|\Z)'),
                     ('WEATHER',  r'3\.4 Weather Drivers:(.*?)(?=3\.5|\Z)')):
        m = re.search(pat, body, re.S)
        if m:
            print(' {}: {}'.format(lbl, re.sub(r'\s+', ' ', m.group(0))[:1400]))

# --- seasonal precedent for candidates with no RT in the last 30 days ------------------
sys.path.append('/var/www/python/Prod/nighthawk')
sys.path.append('/var/www/python/Qingcheng/QCTest/Manual_bidding')
from functions import get_recent_constraint_mvalue, get_hourly_mvalue_for_constraint_num

now      = pd.Timestamp.now(tz='US/Central')
bid_dt   = (now + pd.Timedelta(days=2 if now.hour >= 10 else 1)).strftime('%Y-%m-%d')
dt       = (pd.Timestamp(bid_dt) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
today_dt = now.strftime('%Y-%m-%d')
yest_dt  = (now - pd.Timedelta(days=1)).strftime('%Y-%m-%d')

mvalue = (pd.concat([get_recent_constraint_mvalue(dt, dt, threshold=500),
                     get_recent_constraint_mvalue(yest_dt, today_dt, threshold=500)],
                    ignore_index=True).drop_duplicates('oops_constraint_num'))
h = get_hourly_mvalue_for_constraint_num(mvalue)
h['d'] = pd.to_datetime(h['dt'])

print('\n\n--- seasonal window: Aug 10-25 of each year, RT days by constraint ---')
win = h[(h['d'].dt.month == 8) & (h['d'].dt.day.between(10, 25))]
tab = (win[win['rt_mvalue'] != 0]
       .groupby(['monitored', win['d'].dt.year])['dt'].nunique().unstack(fill_value=0))
print(tab.to_string())

print('\n--- same window, DA days by constraint ---')
tabd = (win[win['da_mvalue'] != 0]
        .groupby(['monitored', win['d'].dt.year])['dt'].nunique().unstack(fill_value=0))
print(tabd.to_string())

print('\n--- last date each candidate bound in RT, and total RT days in 104wk history ---')
rt = h[h['rt_mvalue'] != 0]
print(pd.DataFrame({'last_rt': rt.groupby('monitored')['dt'].max(),
                    'rt_days_all': rt.groupby('monitored')['dt'].nunique()}).to_string())

print('\n--- hourly profile of RT binding, last 30d, by constraint (hour of peak) ---')
r30 = h[(h['d'] >= pd.Timestamp(today_dt) - pd.Timedelta(days=30)) & (h['rt_mvalue'] != 0)]
print(r30.groupby(['monitored', 'hr'])['rt_mvalue'].sum().unstack(fill_value=0)
        .round(0).to_string())
