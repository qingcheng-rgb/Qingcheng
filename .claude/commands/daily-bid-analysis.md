---
description: Load the Daily Bidding sheet from GCS and run the SPP constraint node-selection analysis for the next bid date
---

Run the daily SPP constraint analysis that supports manual bidding, then report findings.

## Environment (all local to this box — this command cannot run as a cloud agent)

- Interpreter: `/opt/venvs/prod-py312/bin/python` (nighthawk, google-cloud-storage, matplotlib)
- Analysis helpers: `/var/www/python/Qingcheng/QCTest/Manual_bidding/functions.py`
- Bidding sheet: `gs://spptest/manual/Daily Bidding - daily_constraint_manual.csv`
- GCS auth: ambient, via `GOOGLE_APPLICATION_CREDENTIALS`

Two things in `functions.py` assume Jupyter and must be neutralised before import:

- several functions call `display()`, which is IPython-only → bind `builtins.display = print`
- the plot helpers open figures → set `MPLBACKEND=Agg`

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
    get_recent_constraint_mvalue, get_node_dfax_from_constraint_num,
    get_hourly_mvalue_for_constraint_num, get_constraints_node_price,
    get_metrics_for_nodes, get_weather_date, predict_tomorrow_percentile,
    analyze_constraint_by_zone,
)

# same bid-date rule the notebook uses: after 10am CT we are bidding two days out
now        = pd.Timestamp.now(tz='US/Central')
bid_dt     = (now + pd.Timedelta(days=2 if now.hour >= 10 else 1)).strftime('%Y-%m-%d')
dt         = (pd.Timestamp(bid_dt) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
print('bid_dt={}  mvalue_dt={}'.format(bid_dt, dt))

mvalue        = get_recent_constraint_mvalue(dt, dt, threshold=500)
dfax          = get_node_dfax_from_constraint_num(mvalue)
hourly_mvalue = get_hourly_mvalue_for_constraint_num(mvalue)
# NOTE: the second argument is the END of the forecast window, so it must be bid_dt.
# Passing dt (as notebook cell 2 does) stops the frame one day short and
# predict_tomorrow_percentile then prints "No forecast data" and returns empty.
fundamentals  = get_weather_date(hourly_mvalue, bid_dt)
tomorrow_pct  = predict_tomorrow_percentile(fundamentals, bid_dt)

# --- outages -------------------------------------------------------------------------
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

# the bidding sheet, for cross-reference
blob  = storage.Client().bucket('spptest').blob('manual/Daily Bidding - daily_constraint_manual.csv')
sheet = pd.read_csv(io.StringIO(blob.download_as_text()))
sheet['_d'] = pd.to_datetime(sheet['bid_date'], format='mixed', errors='coerce')
print('\nsheet rows: {} | latest bid_date: {}'.format(len(sheet), sheet['_d'].max().date()))

names = sorted(hourly_mvalue['monitored'].unique())
print('\ncandidate constraints ({}): {}'.format(len(names), names))

for name in names:
    seen = sheet[sheet['constraints'].astype(str).str.strip() == name]
    print('\n' + '=' * 70)
    print('{}  |  {}'.format(
        name,
        'NEW — never in the sheet' if seen.empty else
        'seen {} time(s), last {}'.format(len(seen), seen['_d'].max().date())))
    if not seen.empty:
        last = seen.sort_values('_d').iloc[-1]
        for c in ('location', 'physical_condition', 'outage_name', 'wind', 'reserve_zone'):
            if pd.notna(last.get(c)) and str(last.get(c)).strip():
                print('  {:20s}: {}'.format(c, last[c]))
    zone = 4
    if not seen.empty and pd.notna(last.get('reserve_zone')):
        try:
            zone = int(str(last['reserve_zone']).split(',')[0])
        except ValueError:
            pass
    try:
        analyze_constraint_by_zone(fundamentals, hourly_mvalue, name, zone, 95)
        table = get_constraints_node_price(hourly_mvalue, dfax, [name])
        get_metrics_for_nodes(table)
    except Exception as e:
        print('  analysis failed: {}: {}'.format(type(e).__name__, e))
```

## Step 2 — report

Read the output and write a briefing, not a transcript.

**Before anything else — which side of the market is real.** `dt` is the day whose DA has
cleared; its RT has not settled yet. So for `dt`, report DA only. Any RT figure for `dt` is an
artefact of the zero-fill, not a result — never rank or reason from it. RT is usable only as
history, from days strictly before `dt`.

1. **Bid date** — state `bid_dt` and `dt`, and how many constraints cleared the 500 threshold
   on the DA side.

2. **Zonal setup** — the `tomorrow_pct` summary: `ow`/`fw` are the on/off-peak **wind**
   percentiles, `ol`/`fl` the on/off-peak **load** percentiles, per reserve zone, measured
   against the trailing month at the same hour. Flag any zone above ~85 or below ~15 — those
   are where tomorrow departs from the recent norm, and they drive everything below.

3. **Outages** — from `linked_outages` (what the Daily Constraints page shows) and
   `sched_outages` (everything scheduled to be in effect on `bid_dt`, as known now). Say for
   each serious candidate whether a linked outage explains it, and call out any scheduled
   outage on `bid_dt` that has **no** constraint attached yet — a new outage on a path that
   has not bound recently is the most common source of a surprise.

4. **Classify every candidate** into one of these, and say which bucket each falls in:
   - **New** — did not bind in the trailing week but bound on the **RT** side yesterday.
     For these, look back through `hourly_mvalue` (it carries 104 weeks) for the same
     calendar window last year: did it bind then, and at what magnitude? A constraint with
     a seasonal precedent is a different proposition from one with none.
   - **Momentum** — bound on several of the recent days, and tomorrow's zonal setup resembles
     the days it bound on. Expect continuation; say which zone and which direction carries it.
   - **Condition mismatch** — binding persistently on the **DA** side, but tomorrow's setup is
     the opposite of the conditions it normally binds under (per `analyze_constraint_by_zone`
     lift). These are the ones where DA is likely to be wrong, so name them explicitly.

5. **Sheet cross-reference** — for candidates already in the sheet, carry your recorded
   `physical_condition` / `outage_name` forward and judge whether tomorrow's zonal setup
   matches the condition those notes describe. For first-time constraints, say so plainly.

6. **Node picks** — for each serious candidate, the best nodes from `get_metrics_for_nodes`,
   with their dfax and recent price behaviour.

7. **Gaps** — any constraint whose analysis raised, any date with no forecast, any empty
   result. State them; do not quietly drop them.

Do not edit the sheet. `SPP/Virtual/spp_update_daily_bidding_sheet.py` owns those writes;
this command is read-only analysis.
