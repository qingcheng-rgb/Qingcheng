---
description: Load the Daily Bidding sheet from GCS and run the SPP constraint node-selection analysis for the next bid date
---

Run the daily SPP constraint analysis that supports manual bidding, then give long/short calls.

## Environment (all local to this box — this command cannot run as a cloud agent)

- Interpreter: `/opt/venvs/prod-py312/bin/python` (nighthawk, google-cloud-storage, pypdf, matplotlib)
- Analysis helpers: `/var/www/python/Qingcheng/QCTest/Manual_bidding/functions.py`
- Bidding sheet: `gs://spptest/manual/Daily Bidding - daily_constraint_manual.csv`
- Attribution PDFs and logs: `/var/www/python/Qingcheng/QCTest/Manual_bidding/logs/`
- GCS auth: ambient, via `GOOGLE_APPLICATION_CREDENTIALS`

Two things in `functions.py` assume Jupyter and must be neutralised before import:

- several functions call `display()`, which is IPython-only → bind `builtins.display = print`
- the plot helpers open figures → set `MPLBACKEND=Agg`

## Step 0 — housekeeping (run this first, it is cheap)

```bash
cd /var/www/python/Qingcheng/QCTest/Manual_bidding/logs
# intermediate step files from previous days — today's are kept so this run can use them
find scratch -maxdepth 1 -type f ! -newermt "$(date +%Y-%m-%d)" -print -delete
# run logs older than 3 days. today's is being written by the cron redirect right now,
# and -mtime +3 cannot match it, so this is safe. PDFs are never touched.
find . -maxdepth 1 -name 'bid_analysis_*.log' -mtime +3 -print -delete
ls -1 *.pdf 2>/dev/null || echo 'NO ATTRIBUTION PDF PRESENT'
```

Report what was deleted as a one-line count, not a file list.

## Step 1 — run the analysis

Write this to the scratchpad and run it with the interpreter above. Do not paste it into a
`python -c` one-liner; it is long and the quoting will bite.

```python
import builtins, io, os, sys
builtins.display = print                      # functions.py calls display() in several places
os.environ.setdefault('MPLBACKEND', 'Agg')    # plot helpers must not open a window

import pandas as pd
from google.cloud import storage
sys.path.append('/var/www/python/Prod/nighthawk')
from nighthawk.data import Constraint, Outage
sys.path.append('/var/www/python/Qingcheng/QCTest/Manual_bidding')
from functions import (
    get_recent_constraint_mvalue, get_hourly_mvalue_for_constraint_num,
    get_weather_date, predict_tomorrow_percentile, analyze_constraint_by_zone,
)

# same bid-date rule the notebook uses: after 10am CT we are bidding two days out.
# at the 13:00 cron time that makes dt = tomorrow, whose DA has cleared but whose RT has not run.
now      = pd.Timestamp.now(tz='US/Central')
bid_dt   = (now + pd.Timedelta(days=2 if now.hour >= 10 else 1)).strftime('%Y-%m-%d')
dt       = (pd.Timestamp(bid_dt) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
today_dt = now.strftime('%Y-%m-%d')
yest_dt  = (now - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
print('bid_dt={}  dt(DA cleared)={}  today={}  yesterday={}'.format(bid_dt, dt, today_dt, yest_dt))

# --- candidates from two sources ------------------------------------------------------
# (a) tomorrow's cleared DA: what the market already says will bind
mvalue_da = get_recent_constraint_mvalue(dt, dt, threshold=500)
# (b) what has actually been binding in RT: yesterday in full, plus today so far.
#     today's early hours are the freshest real signal there is and catch a constraint
#     that started binding after tomorrow's DA was built.
mvalue_rt = get_recent_constraint_mvalue(yest_dt, today_dt, threshold=500)

mvalue = (pd.concat([mvalue_da, mvalue_rt], ignore_index=True)
            .drop_duplicates('oops_constraint_num'))
da_names = set(mvalue_da['monitored'].dropna())
# get_recent_constraint_mvalue over a range clears its threshold on EITHER side, so the
# recent frame contains DA-only names too. Split them: only rt_total != 0 is real RT.
rt_names     = set(mvalue_rt.loc[mvalue_rt['rt_total'] != 0, 'monitored'].dropna())
recent_names = set(mvalue_rt['monitored'].dropna())
print('\nDA candidates for {}: {}'.format(bid_dt, len(da_names)))
print('RT binders {} to {}: {}'.format(yest_dt, today_dt, len(rt_names)))
print('RT-only (not in tomorrow DA): {}'.format(sorted(rt_names - da_names)))

# 104 weeks of hourly history for the union — this is what momentum and seasonality come from
hourly_mvalue = get_hourly_mvalue_for_constraint_num(mvalue)

# --- the freshest RT signal, stated explicitly ----------------------------------------
# columns: oops_constraint_num, dt, hr, rt_mvalue, da_mvalue, rt_da, monitored, contingency
# NOTE the frame is fillna(0)-ed, so a zero means "no row", not "bound at zero".
h = hourly_mvalue
first8 = (h[(h['dt'] == today_dt) & (h['hr'] <= 8)]
            .groupby('monitored')[['rt_mvalue', 'da_mvalue']].sum()
            .query('rt_mvalue != 0 or da_mvalue != 0')
            .sort_values('rt_mvalue'))
yest = (h[h['dt'] == yest_dt]
          .groupby('monitored')[['rt_mvalue', 'da_mvalue']].sum()
          .query('rt_mvalue != 0 or da_mvalue != 0')
          .sort_values('rt_mvalue'))
print('\n--- today {} hours 1-8, RT ---'.format(today_dt))
print(first8.to_string() if len(first8) else '  nothing bound yet')
print('\n--- yesterday {} full day ---'.format(yest_dt))
print(yest.to_string() if len(yest) else '  nothing bound')

# how often each candidate has bound in the trailing 30 days, for "new AND frequent"
recent = h[h['dt'] >= (pd.Timestamp(today_dt) - pd.Timedelta(days=30)).strftime('%Y-%m-%d')]
freq = (recent[recent['rt_mvalue'] != 0].groupby('monitored')['dt'].nunique()
              .rename('rt_days_last_30').sort_values(ascending=False))
print('\n--- RT binding days in the last 30 ---')
print(freq.to_string() if len(freq) else '  none')

# --- zonal setup for the bid date -----------------------------------------------------
# the second argument is the END of the forecast window, so it must be bid_dt. Passing dt
# stops the frame one day short and predict_tomorrow_percentile returns empty.
fundamentals = get_weather_date(hourly_mvalue, bid_dt)
tomorrow_pct = predict_tomorrow_percentile(fundamentals, bid_dt)

# --- outages --------------------------------------------------------------------------
# Same source the Daily Constraints page uses for its linkedOutage column: monitored-side
# linked outages, falling back to contingency-side, and to scheduled times when the RT
# start/end are missing. Format: "__outage_name__ FROM __start__ TO __end__".
con_nums = mvalue[['oops_constraint_num']].drop_duplicates()
linked_outages = Constraint(oops_constraint_num_df=con_nums,
                            market='SPP').get_linked_outage_string_for_frontend(dt=dt)
print('\nlinked outages:')
print(linked_outages.to_string() if len(linked_outages) else '  none')

# Everything scheduled to be in effect on the bid date, as known right now — this is what
# catches an outage that posted after the constraint last bound.
sched_outages = Outage('SPP').get_scheduled_outages_for_frontend(
    start_date=bid_dt, end_date=bid_dt,
    viewing_time=now.strftime('%Y-%m-%d %H:%M:%S'), latest_schedule=True)
print('\nscheduled outages in effect on {}: {}'.format(bid_dt, len(sched_outages)))
if len(sched_outages):
    print(sched_outages[['outageName', 'eqNum', 'outageStartDtTime',
                         'outageEndDtTime', 'voltage']].to_string(index=False))

# --- the sheet, for reserve_zone ONLY -------------------------------------------------
# Do not read physical_condition / outage_name / location from here. The attribution PDF in
# step 2 is the source for why a constraint binds; the sheet only supplies the zone mapping.
blob  = storage.Client().bucket('spptest').blob('manual/Daily Bidding - daily_constraint_manual.csv')
sheet = pd.read_csv(io.StringIO(blob.download_as_text()))
sheet['_d'] = pd.to_datetime(sheet['bid_date'], format='mixed', errors='coerce')
zone_of = {}
for name, g in sheet.groupby(sheet['constraints'].astype(str).str.strip()):
    z = g.sort_values('_d')['reserve_zone'].dropna()
    if len(z):
        try:
            zone_of[name] = int(str(z.iloc[-1]).split(',')[0])
        except ValueError:
            pass
print('\nsheet rows: {} | latest bid_date: {} | zone mappings: {}'.format(
    len(sheet), sheet['_d'].max().date(), len(zone_of)))

# --- wind / load attribution per candidate -------------------------------------------
names = sorted(da_names | recent_names)
print('\ncandidates ({}): {}'.format(len(names), names))
for name in names:
    src = ('DA+RT' if name in da_names and name in rt_names
           else 'DA only' if name in da_names else 'RT only' if name in rt_names
           else 'recent DA only')
    print('\n' + '=' * 70)
    print('{}  |  source: {}  |  zone {}  |  RT days last 30: {}'.format(
        name, src, zone_of.get(name, '4 (DEFAULT - attribution unreliable)'),
        int(freq.get(name, 0))))
    try:
        analyze_constraint_by_zone(fundamentals, hourly_mvalue, name, zone_of.get(name, 4), 95)
    except Exception as e:
        print('  uplift failed: {}: {}'.format(type(e).__name__, e))
```

## Step 2 — read the attribution PDF

The per-constraint write-ups in the newest PDF in `logs/` are the **primary reason** behind
every call you make in step 3. Extract the text once and slice it per constraint rather than
paging through the PDF — it runs to ~113 pages and the text extracts cleanly.

```python
import glob, os, re, pypdf
pdfs = sorted(glob.glob('/var/www/python/Qingcheng/QCTest/Manual_bidding/logs/*.pdf'))
assert pdfs, 'no attribution PDF in logs/'
pdf = pdfs[-1]                       # newest by filename, which starts with its window
print('using:', os.path.basename(pdf))
txt = '\n'.join((p.extract_text() or '') for p in pypdf.PdfReader(pdf).pages)
open('<scratchpad>/attribution.txt', 'w').write(txt)

# Index the PDF by its own section headers, NOT by searching for the constraint name.
# Names are cross-referenced inside other constraints' write-ups, so the first bare-name
# hit is often somebody else's section and you will report the wrong reason.
hdrs = []
for m in re.finditer(r'Monitored Element:\s*', txt):
    nm = re.match(r'[a-z0-9_.\-]+', re.sub(r'\s+', '', txt[m.end():m.end() + 80]))
    if nm:
        hdrs.append((m.start(), nm.group(0)))
sec = {}
for i, (pos, nm) in enumerate(hdrs):
    sec.setdefault(nm, txt[pos:hdrs[i + 1][0] if i + 1 < len(hdrs) else len(txt)])
print('sections in PDF: {}'.format(len(hdrs)))

for name in names:                   # `names` from step 1
    body = sec.get(name)
    if body is None:
        print('\n{}: NOT IN PDF'.format(name)); continue
    d = re.search(r'Analysis Date:\s*([\d\-]+)', body)
    print('\n' + '=' * 70 + '\n{}  (analysis date {})'.format(name, d.group(1) if d else '?'))
    for lbl, pat in (('SUMMARY',  r'3\.1 Summary:.*?(?=3\.2|\Z)'),
                     ('OUTAGES',  r'3\.3 Outages Table:(.*?)(?=3\.4|\Z)'),
                     ('WEATHER',  r'3\.4 Weather Drivers:(.*?)(?=3\.5|\Z)')):
        m = re.search(pat, body, re.S)
        if m:
            print(' {}: {}'.format(lbl, re.sub(r'\s+', ' ', m.group(0))[:1400]))
```

Each constraint block carries `2. Constraint Topology` (what and where the element is),
`3. Explanation Section` with a `3.1 Summary` naming the drivers, a `3.3 Outages Table` with
LODF / flow-impact / risk-multiplier per outage, and `3.4 Weather Drivers`, then
`4. Historical Summary`.

**Check every outage the PDF names against its own end time.** The write-ups are per analysis
date, so an outage that was the acute trigger last week may already have expired — cross it
against `sched_outages` for `bid_dt`. An expired trigger turns a long into a short.

**State the PDF's window in the briefing, taken from its filename.** If that window does not
reach the last few days, say so plainly and note that the attribution then only helps for the
constraints that appear in it — everything else falls through to the RT and zonal evidence.
Never present a stale write-up as if it described tomorrow.

## Step 3 — report: long/short calls

Write a briefing, not a transcript. The output is a list of constraints to go **long** and a
list to go **short**, each with its reason. No node picks — that selection happens elsewhere.

**Which side of the market is real.** `dt` is the day whose DA has cleared; its RT has not run.
So for `dt`, use DA only — any RT figure there is an artefact of the `fillna(0)`, not a result.
Real RT evidence comes from `yest_dt` in full and `today_dt` hours 1–8.

Work the evidence in this order of trust, and say which rung each call rests on:

1. **Outage-driven.** Any constraint whose path has an outage in effect on `bid_dt`, from
   `sched_outages` and `linked_outages`. This outranks everything else — a scheduled outage is
   a known change to the network, not an inference. Call out any outage on `bid_dt` with no
   constraint attached yet: a new outage on a path that has not bound recently is the most
   common source of a surprise.
2. **PDF condition match.** For each candidate, take the drivers from its `Explanation Section`
   and judge them against tomorrow's zonal setup. Say explicitly whether tomorrow reproduces
   the conditions the write-up describes or contradicts them. Quote the driver, briefly.
3. **New and frequent.** Constraints binding in RT yesterday and in today's hours 1–8, ranked
   by `rt_days_last_30`. A constraint binding repeatedly in RT but absent from tomorrow's DA is
   the highest-value disagreement on the board. For genuinely new ones, check `hourly_mvalue`
   (104 weeks) for the same calendar window last year — a seasonal precedent is a different
   proposition from none.
4. **Momentum.** Bound on several recent days with tomorrow's setup resembling those days.
   Say which zone and direction carries it.
5. **Condition mismatch.** Binding persistently on the DA side while tomorrow's setup is the
   opposite of what the uplift says it binds under. These are where DA is likely wrong.

For every call also give:

- **wind or load** — from the `analyze_constraint_by_zone` lift, name whichever family of
  forecast columns carries the highest lift, and the zone. If neither separates, say so. When
  the constraint fell back to zone 4 because the sheet has no mapping, say the attribution is
  low-confidence: the lift was measured against the wrong zone's forecast. Prefer the PDF's
  named zone in that case, and note the disagreement.
- **zonal setup** — `ow`/`fw` are on/off-peak **wind** percentiles, `ol`/`fl` on/off-peak
  **load**, per reserve zone, against the trailing month at the same hour. Flag any zone above
  ~85 or below ~15; those drive everything above.

Close with **gaps**: any constraint whose uplift raised, any date with no forecast, any
candidate absent from the PDF, any empty result. State them; do not quietly drop them.

Do not edit the sheet. `SPP/Virtual/spp_update_daily_bidding_sheet.py` owns those writes;
this command is read-only analysis.
