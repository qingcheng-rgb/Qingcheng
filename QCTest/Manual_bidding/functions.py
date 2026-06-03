# Get daily constraint ranked by the abs RT - DA 
import sys
sys.path.append('/var/www/python/Prod/nighthawk/')

import pandas as pd
from nighthawk.data import Constraint
from nighthawk.data.network.node import Node
import numpy as np
from nighthawk.data.pipeline.common_functions.wind import Wind
from nighthawk.data.pipeline.common_functions.load import Load
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def get_recent_constraint_mvalue(start_dt=(pd.Timestamp(pd.Timestamp.now(tz='US/Central').strftime('%Y-%m-%d')) - pd.Timedelta(days=3)).strftime('%Y-%m-%d'), \
                                 end_dt=pd.Timestamp.now(tz='US/Central').strftime('%Y-%m-%d'), \
                                    threshold=1000):

    opex = 'SPP'
    # end_dt = pd.Timestamp.now(tz='US/Central').strftime('%Y-%m-%d')
    # start_dt = (pd.Timestamp(end_dt) - pd.Timedelta(days=3)).strftime('%Y-%m-%d')


    print(f"start_dt: {start_dt}, end_dt: {end_dt}, Market: {opex}")

    # Get RT and DA hourly mvalues for all constraints over the date range
    print("Fetching RT mvalues...")
    mv_rt = Constraint(oops_constraint_num_df=None, market=opex).get_mvalues(
        start_dt=start_dt, end_dt=end_dt, type='RT', granularity='hourly')

    print("Fetching DA mvalues...")
    mv_da = Constraint(oops_constraint_num_df=None, market=opex).get_mvalues(
        start_dt=start_dt, end_dt=end_dt, type='DA', granularity='hourly')

    print(f"RT constraints: {mv_rt['oops_constraint_num'].nunique()}, DA constraints: {mv_da['oops_constraint_num'].nunique()}")

    # Sum mvalues per constraint
    rt_total = mv_rt.groupby('oops_constraint_num')['mvalue'].sum().rename('rt_total')
    da_total = mv_da.groupby('oops_constraint_num')['mvalue'].sum().rename('da_total')

    # Merge and compute abs(RT - DA), ranked descending
    merged = pd.merge(rt_total, da_total, on='oops_constraint_num', how='outer').fillna(0)
    merged['rt_da'] = (merged['rt_total'] - merged['da_total'])
    merged['abs_rt_da_diff'] = (merged['rt_total'] - merged['da_total']).abs()
    merged = merged.sort_values('abs_rt_da_diff', ascending=False).reset_index()

    # Get constraint details
    all_cons = merged[['oops_constraint_num']].copy()
    print("Fetching constraint details...")
    details_rt = Constraint(oops_constraint_num_df=all_cons, market=opex).get_constraint_details(da_or_rt='RT')
    details_da = Constraint(oops_constraint_num_df=all_cons, market=opex).get_constraint_details(da_or_rt='DA')
    details = pd.concat([details_rt, details_da]).drop_duplicates('oops_constraint_num')

    # Merge everything
    result = pd.merge(merged, details[['oops_constraint_num', 'monitored_clean', 'contingency_clean']], on='oops_constraint_num', how='left')
    result = result.rename(columns={'monitored_clean': 'monitored', 'contingency_clean': 'contingency'})
    result_final = result.groupby('monitored')[['rt_total','da_total','rt_da']].sum().reset_index()
    result_final['abs_rt_da_diff'] = abs(result_final['rt_da'])
    # result_final = result_final[result_final['abs_rt_da_diff']>threshold]
    result_final = result_final[(result_final['da_total']<-threshold) | (result_final['rt_total']<-threshold)]
    result = result[result['monitored'].isin(result_final.monitored.tolist())]


    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_colwidth', 80)
    display(result[['oops_constraint_num', 'rt_total', 'da_total', 'rt_da', 'abs_rt_da_diff', 'monitored', 'contingency']])

    return result 

def get_node_dfax_from_constraint_num(result):
    opex= 'SPP'
    con_df = result[['oops_constraint_num']].drop_duplicates()

    if con_df.empty:
        print("No constraints found — mvalue result is empty.")
        return pd.DataFrame(columns=['oops_constraint_num', 'node_num', 'dfax'])

    dfax = Constraint(oops_constraint_num_df=con_df, market=opex).get_dfax_on_all_nodes(dfax_cutoff=0.05, dfax_type='RT')

    node_dfax = (result
        .merge(dfax, on='oops_constraint_num')
        .sort_values(['abs_rt_da_diff', 'dfax'], ascending=[False, False])
    )

    columns = ['oops_constraint_num','node_num','dfax']
    node_dfax = node_dfax[columns]
    return node_dfax

def get_hourly_mvalue_for_constraint_num(result, weeks=104):

    opex='SPP'
    end_dt2   = pd.Timestamp.now(tz='US/Central').strftime('%Y-%m-%d')
    start_dt2 = (pd.Timestamp(end_dt2) - pd.Timedelta(weeks=weeks)).strftime('%Y-%m-%d')

    con_df = result[['oops_constraint_num']].drop_duplicates()

    # Step 1: daily mvalues to find active days
    mv_rt_daily = Constraint(oops_constraint_num_df=con_df, market=opex).get_mvalues(
        start_dt=start_dt2, end_dt=end_dt2, type='RT', granularity='daily')
    mv_da_daily = Constraint(oops_constraint_num_df=con_df, market=opex).get_mvalues(
        start_dt=start_dt2, end_dt=end_dt2, type='DA', granularity='daily')

    # Step 2: keep only days where mvalue != 0
    rt_active = mv_rt_daily[mv_rt_daily['mvalue'] != 0][['oops_constraint_num', 'dt']].drop_duplicates()
    da_active = mv_da_daily[mv_da_daily['mvalue'] != 0][['oops_constraint_num', 'dt']].drop_duplicates()

    print(f"RT active constraint-days: {len(rt_active)}")
    print(f"DA active constraint-days: {len(da_active)}")

    # Step 3: fetch hourly mvalues only for those active days
    def get_hourly_for_active_days(active_df, ctype):
        if active_df.empty:
            return pd.DataFrame()
        results = []
        for con_num, grp in active_df.groupby('oops_constraint_num'):
            dates = grp['dt'].astype(str).tolist()
            mv_h = Constraint(
                oops_constraint_num_df=pd.DataFrame({'oops_constraint_num': [con_num]}),
                market=opex
            ).get_mvalues(start_dt=min(dates), end_dt=max(dates), type=ctype, granularity='hourly')
            mv_h = mv_h[mv_h['dt'].astype(str).isin(dates)]
            results.append(mv_h)
        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    mv_rt_2w = get_hourly_for_active_days(rt_active, 'RT')
    mv_da_2w = get_hourly_for_active_days(da_active, 'DA')

    cols = ['oops_constraint_num', 'dt', 'hr', 'mvalue']
    if mv_rt_2w.empty:
        mv_rt_2w = pd.DataFrame(columns=cols)
    if mv_da_2w.empty:
        mv_da_2w = pd.DataFrame(columns=cols)

    # Step 3: outer join on constraint, dt, hr — compute rt - da per hour
    mv_merged = (
        mv_rt_2w.rename(columns={'mvalue': 'rt_mvalue'})
        .merge(mv_da_2w.rename(columns={'mvalue': 'da_mvalue'}),
            on=['oops_constraint_num', 'dt', 'hr'], how='outer')
        .fillna(0)
    )
    mv_merged['rt_da'] = mv_merged['rt_mvalue'] - mv_merged['da_mvalue']
    print(mv_merged.head())

    names = result[['oops_constraint_num', 'monitored', 'contingency']].drop_duplicates()
    mv_merged = (mv_merged
        .merge(names, on='oops_constraint_num', how='left')
        .sort_values(['oops_constraint_num', 'dt', 'hr'])
        .reset_index(drop=True)
    )
    mv_merged['dt'] = pd.to_datetime(mv_merged['dt']).dt.strftime('%Y-%m-%d')
    print(f"\nMerged hourly rows: {len(mv_merged)}")
    display(mv_merged)
    return mv_merged

def get_constraint_node_prices_from_constraint_num(mv_merged, node_dfax, monitored_name):
    con_list = mv_merged.loc[mv_merged['monitored'] == monitored_name, 'oops_constraint_num'].unique().tolist()
    not_found= []
    if not con_list:
        print(f"No constraints found for monitored name: '{monitored_name}'")
        return pd.DataFrame()

    results = []
    for con_num in con_list:
        node_map = node_dfax[node_dfax['oops_constraint_num'] == con_num][['node_num', 'dfax']].drop_duplicates()
        mv_con   = mv_merged[mv_merged['oops_constraint_num'] == con_num].copy()

        if node_map.empty or mv_con.empty:
            continue

        mv_con['oops_constraint_num'] = con_num
        mv_con['monitored']   = monitored_name
        mv_con['contingency'] = mv_merged.loc[mv_merged['oops_constraint_num'] == con_num, 'contingency'].iloc[0]

        mv_nodes  = mv_con.merge(node_map, how='cross')
        node_list = node_map['node_num'].astype(int).tolist()

        prices_list = []
        for dt in mv_con.dt.unique():
            p = Node(node_nums=node_list, market='SPP').get_price(
                start_dt=dt, end_dt=dt,
                component=['LMP'], type=['DA', 'RT'], granularity='hourly'
            )[['dt', 'hr', 'node_num', 'da_total', 'rt_total']].rename(
                columns={'da_total': 'node_da_lmp', 'rt_total': 'node_rt_lmp'}
            )
            prices_list.append(p)

        prices = pd.concat(prices_list, ignore_index=True)
        results.append(mv_nodes.merge(prices, on=['dt', 'hr', 'node_num'], how='left'))
    if not results:
        not_found.append(monitored_name) 
        print(monitored_name, 'could not find the dfax value')
        return pd.DataFrame()
    else: 
        result = pd.concat(results, ignore_index=True).sort_values(['dt', 'hr', 'node_num']).reset_index(drop=True)
    return result

def get_constraints_node_price(mvalue, nodedfax, listofnames=[]):
    
    if not listofnames:
        # default to be the all monitored names
        all_monitored = mvalue['monitored'].unique().tolist()
    else: all_monitored = listofnames

    mv_final = pd.concat([get_constraint_node_prices_from_constraint_num(mvalue, nodedfax, name) \
                          for name in all_monitored],ignore_index=True)
    
    if mv_final.empty:
        return 
    mv_grouped = (mv_final
    .groupby(['monitored', 'dt', 'hr', 'node_num'], as_index=False)
    .agg(rt_mvalue=('rt_mvalue', 'sum'),
         da_mvalue=('da_mvalue', 'sum'),
         rt_da=('rt_da', 'sum'),
         dfax=('dfax', 'first'),
         node_da_lmp=('node_da_lmp', 'first'),
         node_rt_lmp=('node_rt_lmp', 'first')))
    
    print(f"\nTotal rows: {len(mv_final)}")
    
    return mv_grouped

def get_metrics_for_nodes(mv_grouped):
    if mv_grouped is None:
        return 'No nodes dfax'
    mv_grouped['node_rt_da']          = mv_grouped['node_rt_lmp'] - mv_grouped['node_da_lmp']
    mv_grouped['predicted_node_rt_da'] = -mv_grouped['dfax'] * mv_grouped['rt_da']

    # only evaluate hours where the constraint is actually binding
    active = mv_grouped[mv_grouped['rt_da'] != 0].copy()

    def node_constraint_stats(grp):
        x = grp['predicted_node_rt_da']
        y = grp['node_rt_da']
        dfx= grp['dfax'].iloc[0]
        mask = x.notna() & y.notna()
        x, y = x[mask], y[mask]
        n = len(x)
        if n < 3:
            return pd.Series({'dfax': dfx, 'corr': np.nan, 'sign_acc': np.nan,
                            'mean_pred': np.nan, 'mean_actual': np.nan, 'n': n})

        corr     = x.corr(y)
        sign_acc = (np.sign(x) == np.sign(y)).mean()
        return pd.Series({'dfax':dfx, 'corr': corr, 'sign_acc': sign_acc,
                        'mean_pred': x.mean(), 'mean_actual': y.mean(), 'n': n})

    stats_df = (active
        .groupby(['monitored', 'node_num'])
        .apply(node_constraint_stats)
        .reset_index()
        .sort_values(['monitored', 'corr'], ascending=[True, False])
        .reset_index(drop=True)
    )

    for monitored, grp in stats_df.groupby('monitored', sort=True):
        print(f"\n{'='*60}")
        print(f"  {monitored}")
        print(f"{'='*60}")
        display(grp.drop(columns='monitored').reset_index(drop=True))

def get_recent_direction_accuracy(mv_grouped, bid_dt, days=10):
    if mv_grouped is None:
        return

    end_dt   = (pd.Timestamp(bid_dt) - pd.Timedelta(days=2)).strftime('%Y-%m-%d')
    start_dt = (pd.Timestamp(bid_dt) - pd.Timedelta(days=days+2)).strftime('%Y-%m-%d')

    df = mv_grouped.copy()
    df['node_rt_da']           = df['node_rt_lmp'] - df['node_da_lmp']
    df['predicted_node_rt_da'] = -df['dfax'] * df['rt_da']

    df = df[(df['dt'] >= start_dt) & (df['dt'] <= end_dt) & (df['rt_da'] != 0)]

    if df.empty:
        print(f"No binding hours found between {start_dt} and {end_dt}")
        return

    def dir_stats(grp):
        same = (np.sign(grp['predicted_node_rt_da']) == np.sign(grp['node_rt_da']))
        return pd.Series({
            'n_hours':    len(grp),
            'same_dir':   same.sum(),
            'diff_dir':   (~same).sum(),
            'same_dir_pct': round(same.mean() * 100, 1),
        })

    result = (df
        .groupby(['monitored', 'node_num', 'hr'])
        .apply(dir_stats)
        .reset_index()
        .sort_values(['monitored', 'node_num', 'hr'])
        .reset_index(drop=True)
    )

    print(f"Period: {start_dt} → {end_dt}")
    for monitored, grp in result.groupby('monitored'):
        print(f"\n{'='*60}\n  {monitored}\n{'='*60}")
        display(grp.drop(columns='monitored').reset_index(drop=True))

    return result

def get_weather_date(mv_merged, bid_dt):
    start_dt_f = mv_merged['dt'].min()
    end_dt_f   = bid_dt

    rz_wind_df = Wind('SPP').get_res_zonal_wind(start_dt_f, end_dt_f, var_spec=['f'], pivot=True)
    rz_load_df = Load('SPP').get_res_zonal_load(start_dt_f, end_dt_f, var_spec=['f'], pivot=True)

    rz_wind_f_cols = [c for c in rz_wind_df.columns if 'forecast' in c]
    rz_load_f_cols = [c for c in rz_load_df.columns if 'forecast' in c]

    for df in [rz_wind_df, rz_load_df]:
        df['dt'] = df['dt'].astype(str)
        df['hr'] = df['hr'].astype(int)

    fundamentals = (rz_wind_df[['dt', 'hr'] + rz_wind_f_cols]
        .merge(rz_load_df[['dt', 'hr'] + rz_load_f_cols], on=['dt', 'hr'], how='outer'))

    available_zones = sorted(set(int(c.split('_')[0][2:]) for c in rz_wind_f_cols))
    print(f"Reserve zones available: {available_zones}")
    print(f"Wind cols: {rz_wind_f_cols}")
    print(f"Load cols: {rz_load_f_cols}")
    return fundamentals

def predict_tomorrow_percentile(fundamentals, bid_dt):
    tmr = fundamentals[fundamentals['dt'] == bid_dt].copy()
    if tmr.empty:
        print(f"No forecast data for {bid_dt}")
        return pd.DataFrame()

    cutoff = (pd.Timestamp(bid_dt) - pd.DateOffset(months=3)).strftime('%Y-%m-%d')
    fundamentals = fundamentals[(fundamentals['dt'] >= cutoff) & (fundamentals['dt'] < bid_dt)]

    zones = sorted(set(
        int(c.split('_')[0][2:])
        for c in fundamentals.columns
        if c.startswith('rz') and 'forecast' in c
    ))

    rows = []
    for zone in zones:
        wind_col = f'rz{zone}_spp_res_zonal_wind_forecast_f'
        load_col = f'rz{zone}_spp_res_zonal_load_forecast_f'

        for _, row in tmr.iterrows():
            hr = int(row['hr'])
            wind_hist = fundamentals[fundamentals['hr'] == hr][wind_col].dropna()
            load_hist = fundamentals[fundamentals['hr'] == hr][load_col].dropna()
            wind_val = row.get(wind_col)
            load_val = row.get(load_col)
            rows.append({
                'zone': zone,
                'hr': hr,
                'wind_value': round(wind_val, 1) if pd.notna(wind_val) else None,
                'wind_pct': round((wind_hist < wind_val).mean() * 100, 1) if pd.notna(wind_val) else None,
                'load_value': round(load_val, 1) if pd.notna(load_val) else None,
                'load_pct': round((load_hist < load_val).mean() * 100, 1) if pd.notna(load_val) else None,
            })

    result = pd.DataFrame(rows).sort_values(['zone', 'hr']).reset_index(drop=True)
    for zone, grp in result.groupby('zone'):
        print(f"\n--- Zone {zone} ---")
        display(grp.drop(columns='zone').reset_index(drop=True))

    result['peak'] = result['hr'].apply(lambda h: 'off' if h <= 7 or h == 24 else 'on')
    print("\n--- Summary (median pct by zone / peak) ---")
    for zone, grp in result.groupby('zone'):
        on  = grp[grp['peak'] == 'on']
        off = grp[grp['peak'] == 'off']
        print(f"{zone}: ow: {int(on['wind_pct'].median())}, fw: {int(off['wind_pct'].median())}, "
              f"ol: {int(on['load_pct'].median())}, fl: {int(off['load_pct'].median())}")

    return result

def analyze_constraint_by_zone(fundamentals,mv_merged, monitored_name, reserve_zone, top):
    """
    Conditional binding-probability analysis for one constraint vs a reserve zone's forecasts.

    Parameters
    ----------
    monitored_name : str  – value from mv_merged['monitored'], e.g. 'lnrussett-sbrown'
    reserve_zone   : int  – SPP reserve zone number (1, 2, 3, 4, 5, or 21)

    Returns
    -------
    pd.DataFrame  – one row per forecast column with baseline/conditional prob and lift
    """
    grp = mv_merged[mv_merged['monitored'] == monitored_name].copy()
    if grp.empty:
        print(f"No data for monitored='{monitored_name}'")
        return None

    binding_hours = grp[grp['rt_da'] != 0][['dt', 'hr']].drop_duplicates().copy()
    binding_hours['dt'] = binding_hours['dt'].astype(str)
    binding_hours['hr'] = binding_hours['hr'].astype(int)
    binding_hours['binding'] = 1

    all_dates = pd.date_range(start=mv_merged['dt'].min(), end=mv_merged['dt'].max(), freq='D')
    full_grid = pd.DataFrame([
        {'dt': d.strftime('%Y-%m-%d'), 'hr': h}
        for d in all_dates
        for h in range(1, 25)
    ])

    tmp = (full_grid
           .merge(fundamentals, on=['dt', 'hr'], how='left')
           .merge(binding_hours, on=['dt', 'hr'], how='left'))
    tmp['binding'] = tmp['binding'].fillna(0)

    rz_cols = [c for c in tmp.columns if c.startswith(f'rz{reserve_zone}_') and 'forecast' in c]
    if not rz_cols:
        print(f"No forecast columns found for reserve zone {reserve_zone}.")
        print(f"Available rz columns: {[c for c in tmp.columns if c.startswith('rz')]}")
        return None

    baseline = tmp['binding'].mean()
    n_total  = len(tmp)

    rows = []
    for col in rz_cols:
        valid = tmp[[col, 'binding']].dropna()
        if len(valid) < 20:
            continue
        threshold = valid[col].quantile(top/100)
        above = valid[valid[col] > threshold]
        if len(above) < 5:
            continue
        cond_prob = above['binding'].mean()
        rows.append({
            'monitored':          monitored_name,
            'reserve_zone':       reserve_zone,
            'fundamental':        col,
            'baseline_prob':      round(baseline, 4),
            f'cond_prob_top{top}pct': round(cond_prob, 4),
            'lift_pct':           round(cond_prob / baseline - 1, 2) if baseline > 0 else None
        })

    result_df = pd.DataFrame(rows)
    print(f"Constraint  : '{monitored_name}'")
    print(f"Reserve zone: {reserve_zone}  |  Columns: {rz_cols}")
    print(f"Baseline binding rate: {baseline:.4f}  |  Total hours: {n_total}")
    display(result_df.sort_values('lift_pct', ascending=False))

def plot_constraint_seasonality(mv_merged, monitored_name):
    con_nums = mv_merged.loc[mv_merged['monitored'] == monitored_name, 'oops_constraint_num'].tolist()
    if not con_nums:
        print(f"No constraint found for '{monitored_name}'")
        return

    month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    season_colors = {
        'Dec': '#5b9bd5', 'Jan': '#5b9bd5', 'Feb': '#5b9bd5',
        'Mar': '#70ad47', 'Apr': '#70ad47', 'May': '#70ad47',
        'Jun': '#d62728', 'Jul': '#d62728', 'Aug': '#d62728',
        'Sep': '#ed7d31', 'Oct': '#ed7d31', 'Nov': '#ed7d31',
    }
    season_map = {
        'Jun': 'Summer', 'Jul': 'Summer', 'Aug': 'Summer',
        'Sep': 'Fall',   'Oct': 'Fall',   'Nov': 'Fall',
        'Dec': 'Winter', 'Jan': 'Winter', 'Feb': 'Winter',
        'Mar': 'Spring', 'Apr': 'Spring', 'May': 'Spring',
    }

    def build_monthly(df_daily, rt_da):
        d = df_daily[df_daily['oops_constraint_num'].isin(con_nums)].copy()
        d['dt']         = pd.to_datetime(d['dt'])
        d['month']      = d['dt'].dt.month
        d['month_name'] = d['dt'].dt.strftime('%b')
        s   = d.groupby([ 'month', 'month_name'])[rt_da].sum().reset_index()
        return (pd.DataFrame({'month': range(1, 13), 'month_name': month_labels})
                .merge(s, on=['month', 'month_name'], how='left').fillna(0))

    def build_yearly(df_daily, rt_da):
        d = df_daily[df_daily['oops_constraint_num'].isin(con_nums)].copy()
        d['dt']   = pd.to_datetime(d['dt'])
        d['year'] = d['dt'].dt.year
        s   = d.groupby([ 'year'])[rt_da].sum().reset_index()
        avg = s.groupby('year')[rt_da].mean().reset_index()
        return avg

    rt_monthly = build_monthly(mv_merged, "rt_mvalue")
    da_monthly = build_monthly(mv_merged, "da_mvalue")
    rt_yearly  = build_yearly(mv_merged, "rt_mvalue")
    da_yearly  = build_yearly(mv_merged, "da_mvalue")

    # dominant season and month (based on RT abs mvalue)
    rt_monthly['abs_mv'] = rt_monthly['rt_mvalue'].abs()
    rt_monthly['season'] = rt_monthly['month_name'].map(season_map)
    season_totals   = rt_monthly.groupby('season')['abs_mv'].sum()
    dominant_season = season_totals.idxmax()
    peak_month_row  = rt_monthly.loc[rt_monthly['abs_mv'].idxmax()]

    # hourly averages
    hourly   = mv_merged[mv_merged['monitored'] == monitored_name].copy()
    rt_by_hr = (hourly.groupby('hr')['rt_mvalue']
                .sum().reindex(range(1, 25), fill_value=0).reset_index())
    da_by_hr = (hourly.groupby('hr')['da_mvalue']
                .sum().reindex(range(1, 25), fill_value=0).reset_index())
    rt_hr_colors = ['#aec7e8' if h <= 8 else '#ff7f0e' for h in rt_by_hr['hr']]
    da_hr_colors = ['#c5e0b4' if h <= 8 else '#9b59b6' for h in da_by_hr['hr']]

    rt_monthly.rename(columns={'rt_mvalue':'mvalue'},inplace=True)
    rt_yearly.rename(columns={'rt_mvalue':'mvalue'},inplace=True)
    da_monthly.rename(columns={'da_mvalue':'mvalue'},inplace=True)
    da_yearly.rename(columns={'da_mvalue':'mvalue'},inplace=True)
    peak_month_row.rename({'rt_mvalue':'mvalue'},inplace=True)

    fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    fig.suptitle(
        f"Constraint: '{monitored_name}'\n"
        f"Dominant season: {dominant_season}  |  Peak month: {peak_month_row['month_name']} "
        f"(RT sum = {peak_month_row['mvalue']:.0f})",
        fontsize=13, fontweight='bold'
    )

    season_legend = [
        mpatches.Patch(color='#d62728', label='Summer (Jun-Aug)'),
        mpatches.Patch(color='#ed7d31', label='Fall (Sep-Nov)'),
        mpatches.Patch(color='#5b9bd5', label='Winter (Dec-Feb)'),
        mpatches.Patch(color='#70ad47', label='Spring (Mar-May)'),
    ]

    # row 1 — monthly
    for ax, monthly, title in [
        (axes[0, 0], rt_monthly, 'RT mvalue — monthly sums'),
        (axes[0, 1], da_monthly, 'DA mvalue — monthly sums'),
    ]:
        colors = [season_colors[m] for m in monthly['month_name']]
        ax.bar(monthly['month_name'], monthly['mvalue'], color=colors, edgecolor='white')
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_title(title)
        ax.set_ylabel('Avg mvalue')
        ax.legend(handles=season_legend, fontsize=8)

    # row 2 — hourly
    for ax, hr_df, col, colors, title in [
        (axes[1, 0], rt_by_hr, 'rt_mvalue', rt_hr_colors, 'RT mvalue — sum hour'),
        (axes[1, 1], da_by_hr, 'da_mvalue', da_hr_colors, 'DA mvalue — sum hour'),
    ]:
        ax.bar(hr_df['hr'], hr_df[col], color=colors, edgecolor='white')
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xticks(range(1, 25))
        ax.set_xticklabels(range(1, 25), fontsize=8)
        ax.set_xlabel('Hour (CPT)')
        ax.set_ylabel('Avg mvalue')
        ax.set_title(title)

    axes[1, 0].legend(handles=[
        mpatches.Patch(color='#aec7e8', label='Off-Peak (hr 1-8)'),
        mpatches.Patch(color='#ff7f0e', label='Peak (hr 9-24)'),
    ], fontsize=9)
    axes[1, 1].legend(handles=[
        mpatches.Patch(color='#c5e0b4', label='Off-Peak (hr 1-8)'),
        mpatches.Patch(color='#914cac', label='Peak (hr 9-24)'),
    ], fontsize=9)

    # row 3 — yearly
    for ax, yr_df, color, title in [
        (axes[2, 0], rt_yearly, '#5b9bd5', 'RT mvalue — sum year'),
        (axes[2, 1], da_yearly, '#70ad47', 'DA mvalue — sum year'),
    ]:
        bars = ax.bar(yr_df['year'].astype(str), yr_df['mvalue'], color=color, edgecolor='white')
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xlabel('Year')
        ax.set_ylabel('Avg daily mvalue')
        ax.set_title(title)
        for bar, val in zip(bars, yr_df['mvalue']):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (abs(bar.get_height()) * 0.02),
                    f'{val:.0f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()
    print(f"Dominant season : {dominant_season}")
    print(f"\nRT  peak sum   (hr  9-24): {hourly[hourly['hr'] > 8]['rt_mvalue'].sum():>10.2f}  |  "
          f"off-peak (hr 1-8): {hourly[hourly['hr'] <= 8]['rt_mvalue'].sum():>10.2f}")
    print(f"DA  peak sum   (hr  9-24): {hourly[hourly['hr'] > 8]['da_mvalue'].sum():>10.2f}  |  "
          f"off-peak (hr 1-8): {hourly[hourly['hr'] <= 8]['da_mvalue'].sum():>10.2f}")

def plot_recent_hourly_distribution(mv_merged, monitored_name, bid_dt, days=30):
    end_dt   = (pd.Timestamp(bid_dt) - pd.Timedelta(days=2)).strftime('%Y-%m-%d')
    start_dt = (pd.Timestamp(bid_dt) - pd.Timedelta(days=days+2)).strftime('%Y-%m-%d')

    grp = mv_merged[
        (mv_merged['monitored'] == monitored_name) &
        (mv_merged['dt'] >= start_dt) &
        (mv_merged['dt'] <= end_dt)
    ].copy()

    if grp.empty:
        print(f"No data for '{monitored_name}' between {start_dt} and {end_dt}")
        return

    print(f"'{monitored_name}'  |  {start_dt} → {end_dt}  |  {grp['dt'].nunique()} days")

    # aggregate to (dt, hr) level first, then summarise by hr
    daily = (grp.groupby(['dt', 'hr'], as_index=False)
             .agg(rt_mvalue=('rt_mvalue', 'sum'),
                  da_mvalue=('da_mvalue', 'sum'),
                  rt_da=('rt_da', 'sum')))

    by_hr = (daily.groupby('hr')
             .agg(rt_sum=('rt_mvalue', 'sum'),
                  da_sum=('da_mvalue', 'sum'))
             .reindex(range(1, 25), fill_value=0).reset_index())
    by_hr['rt_da'] = by_hr['rt_sum'] - by_hr['da_sum']

    dir_hr = (daily.groupby('hr')
              .apply(lambda x: pd.Series({
                  'pos_count': (x['rt_da'] > 0).sum(),
                  'neg_count': (x['rt_da'] < 0).sum(),
              }))
              .reindex(range(1, 25), fill_value=0).reset_index())
    
    fig, axes = plt.subplots(2, 2, figsize=(28, 14))

    fig.suptitle(f"{monitored_name}  —  last {days} days by hour", fontsize=20, fontweight='bold', y=1.02)

    for ax in axes.flat:
        ax.title.set_fontsize(16)
        ax.xaxis.label.set_fontsize(13)
        ax.yaxis.label.set_fontsize(13)
        ax.tick_params(labelsize=11)

    axes[0, 0].bar(by_hr['hr'], by_hr['rt_sum'], color='#ff7f0e')
    axes[0, 0].axhline(0, color='black', linewidth=0.8)
    axes[0, 0].set_title('RT mvalue sum by hour', fontsize=16)
    axes[0, 0].set_xlabel('Hour'); axes[0, 0].set_xticks(range(1, 25))

    axes[0, 1].bar(by_hr['hr'], by_hr['da_sum'], color='#5b9bd5')
    axes[0, 1].axhline(0, color='black', linewidth=0.8)
    axes[0, 1].set_title('DA mvalue sum by hour', fontsize=16)
    axes[0, 1].set_xlabel('Hour'); axes[0, 1].set_xticks(range(1, 25))

    axes[1, 0].bar(by_hr['hr'], by_hr['rt_da'],
                color=['#d62728' if v > 0 else '#2ca02c' for v in by_hr['rt_da']])
    axes[1, 0].axhline(0, color='black', linewidth=0.8)
    axes[1, 0].set_title('RT − DA sum by hour', fontsize=16)
    axes[1, 0].set_xlabel('Hour'); axes[1, 0].set_xticks(range(1, 25))

    axes[1, 1].bar(dir_hr['hr'], dir_hr['pos_count'], label='RT−DA > 0', color='#d62728')
    axes[1, 1].bar(dir_hr['hr'], -dir_hr['neg_count'], label='RT−DA < 0', color='#2ca02c')
    axes[1, 1].axhline(0, color='black', linewidth=0.8)
    axes[1, 1].set_title('Direction count by hour', fontsize=16)
    axes[1, 1].set_xlabel('Hour'); axes[1, 1].set_xticks(range(1, 25))
    axes[1, 1].legend()


    plt.tight_layout()
    plt.show()

