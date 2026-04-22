import sys
sys.path.insert(0, '/var/www/python/Qingcheng/Darwin')
sys.path.append('/var/www/python/Prod/nighthawk/')
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from nighthawk.models.valuation import node_price_predictor
import time
import pickle
import dill
from google.cloud import bigquery, storage
warnings.filterwarnings("ignore")
from datetime import datetime
print("Here the job starts!")
print(str(datetime.now()))
from nighthawk.models.valuation import ve_model_functions
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys, os, inspect, shutil, importlib
sys.path.insert(0, '/var/www/python/Qingcheng/Fourier')
sys.path.append('/var/www/python/Prod/nighthawk/')
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from nighthawk.util import bigquery_functions, connections, sql_functions, dataframe_functions
from nighthawk.data.pipeline.var_handler import loadwindgen_vh, wind_vh
from nighthawk.data.pipeline.common_functions import wind
from google.cloud import bigquery
import utils_fourier.ve_portfolio_constructor_fourier as ve_portfolio_constructor_fourier
from common_functions import get_scale_factor
from nighthawk.data.network.node import Node
from nighthawk.data.product.ve import DailyBidsManager
from nighthawk.data.network.node import Node
import math
import matplotlib.pyplot as plt


def eval_valuation_model(df):
    idx      = ['dt', 'hr', 'node_num']
    col_list = ['da_total_mean', 'rt_total_mean']
    df_new = (df[idx + [c for c in col_list if c in df.columns]]
                .groupby(idx, as_index=False)
                .first())
    df_new['dt'] = df_new['dt'].astype(str)

    #Get unique nodes and date range from predictions df
    node_nums = sorted(df['node_num'].dropna().unique().tolist())
    start_dt  = str(df['dt'].min())
    end_dt    = str(df['dt'].max())

    #Pull actuals from nighthawk Node (MCC=congestion, LMP=total)
    actuals_df = Node(node_nums, 'SPP').get_price(
        start_dt, end_dt,
        component=['MCC', 'LMP'],
        type=['DA', 'RT'],
        granularity='hourly'
    )[['dt', 'hr', 'node_num',  'da_total', 'rt_total']]

    actuals_df['dt']       = actuals_df['dt'].astype(str)
    actuals_df['hr']       = actuals_df['hr'].astype(int)
    actuals_df['node_num'] = actuals_df['node_num'].astype(int)

    merged = df_new.merge(actuals_df, on=idx)
    return merged

def get_metrics(df):
    # Compare with_dayzer vs without_dayzer predictions
    metrics = {}
    for target in ['da_total', 'rt_total']:
        pred = df[f'{target}_mean']
        actual = df[target]
        diff = pred - actual
        metrics[( f'{target}_ME')]  = np.mean(diff).round(2),
        metrics[( f'{target}_MSE')]  = np.mean(diff**2).round(2),
        metrics[( f'{target}_MAE')]  = np.mean(np.abs(diff)).round(2),
        metrics[( f'{target}_RMSE')] = np.sqrt(np.mean(diff ** 2)).round(2)

    met = pd.DataFrame(metrics).T
    met.columns = ['Prod']
    return met

def run_in_kubernetes(node_time_list, training_fn):
    fn_source = inspect.getsource(training_fn)
    fn_name   = training_fn.__name__

    work_str = f'''
import pandas as pd
import numpy as np
import time
import sys
import pickle
import dill
import warnings
from google.cloud import bigquery, storage
warnings.filterwarnings("ignore")
from datetime import datetime
print("Here the job starts!")
print(str(datetime.now()))

bucket_name      = sys.argv[1]
save_file_folder = sys.argv[2]
job_table_id     = sys.argv[3]
with open("job_info.pickle", "rb") as file:
    job_info = dill.load(file)

job_id    = job_info["jobId"]
record_id = job_info["recordId"]
node_num  = job_info["node_num"]
dt        = job_info["dt"]

try:
    sys.path.append("/var/www/python/Prod/nighthawk/")
except:
    print("oh..this is in kubernetes")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from nighthawk.models.valuation import ve_model_functions
import math
import torch.nn.functional as F

{fn_source}

df = {fn_name}(node_num, dt)

testing_data_file_name = "gs://" + bucket_name + "/" + save_file_folder + "/record_" + str(record_id) + "/prediction/jobId_" + str(job_id) + ".csv"
df.to_csv(testing_data_file_name, index=False)
print("Here the job ends!")
print(str(datetime.now()))
'''

    opexchange    = 'SPP'
    daily_node_df = pd.DataFrame(node_time_list, columns=['dt','node_num'])
    daily_node_df['dt'] = daily_node_df['dt'].astype(str)

    npp = node_price_predictor.NodePricePredictor(opexchange, daily_node_df[['dt', 'node_num']])
    print('  record_id', npp.get_record_id())

    yaml_dict = {'spec': {'parallelism': 800, 'template': {'spec': {
        'containers': [{'resources': {
            'limits':   {'cpu': '1.5', 'memory': '11Gi'},
            'requests': {'cpu': '1.5', 'memory': '11Gi'}
        }}],
        'nodeSelector': {'cloud.google.com/gke-nodepool': 've-pool-e2hm4-spot'}}}}}

    predict_table = npp.run(work_str, envir='KUBERNETES', scale='FULL',
                            initialization='us-central1-docker.pkg.dev/movetocloud-999/fourier/ve_2024_nn',
                            yaml_dict_original=yaml_dict,
                            install_basic_pkgs=False)
    print('  predict_table:', predict_table)

    df = bigquery_functions.download_df_from_bq(f"SELECT * FROM `{predict_table[0]}`")
    return df

def fourier_port(df, start_date='2020-01-01', end_date='2026-09-01', saved='temp_file'):
    def hourly_lwg_cut(df, bid_date):
        total_lwg_df = loadwindgen_vh.get_data_and_mapping_for_total_lwg(
            [636], "SPP", start_dt=bid_date, end_dt=bid_date, n_day_pctl_flag=False)[0]
        high_lwg_hr_lst = list(total_lwg_df.loc[total_lwg_df['spp_loadwindgen_forecast_f'] >= 55000, 'hr'].unique())
        print(f"lwg cut hours: {high_lwg_hr_lst}")
        if len(high_lwg_hr_lst) > 0:
            df['bid_mw'] = np.where(df['hr'].isin(high_lwg_hr_lst), 0, df['bid_mw'])
        return df

    def apply_price_filter(df):
        df['bid_price'] = np.where((df['incdec'] == 'Increment') & (df['bid_price'] < -100), -100, df['bid_price'])
        df['bid_price'] = np.where((df['incdec'] == 'Decrement') & (df['bid_price'] > 300), 300, df['bid_price'])
        return df[df['bid_price'] < 2000]

    def apply_mw_filter(df):
        return df[df['bid_mw'] >= 0.1]

    def portfolio_cut(portfolio, bid_date):
        portfolio = apply_price_filter(portfolio)
        portfolio = hourly_lwg_cut(portfolio, bid_date)
        portfolio = apply_mw_filter(portfolio)
        return portfolio

    # --- config ---
    opexchange    = 'SPP'
    run_number    = 1
    condition_label = 2  # 0=test, 1=holdout, 2=production
    valuationModel_label = ''
    nodeSelection_label  = 'Fourier'
    file_location_cloudserver = '/var/www/python/Qingcheng/temp_test_fourier/'
    os.makedirs(file_location_cloudserver + 'return_and_risk', exist_ok=True)
    os.makedirs(file_location_cloudserver + 'nodeSelection', exist_ok=True)
    conn = connections.get_sql_connection(database='temp')

    # --- segments (matches production) ---
    segments = pd.DataFrame({
        'quantile_DA': ['da_total_q95', 'da_total_q90', 'da_total_q80', 'da_total_q70',
                        'da_total_q60', 'da_total_q50', 'da_total_q40', 'da_total_q30',
                        'da_total_q20', 'da_total_q10', 'da_total_q5'],
        'decSegment': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'incSegment': [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]})

    # --- load data ---
    all_portfolios = []
    dates = sorted(df['dt'].unique())
    print(f'Running portfolio construction for {len(dates)} dates')

    try:
        for bid_date in dates:
            print(f'\n--- {bid_date} ---')
            try:
                # 1. Valuation model from CSV
                df_date = df[df['dt'] == bid_date].copy()
                df_date['node_num'] = df_date['node_num'].astype(int)
                df_date['dt']       = df_date['dt'].astype(str)
                df_date['hr']       = df_date['hr'].astype(int)
                total_col = [col for col in df_date.columns if
                            col.startswith('da_total_') or col.startswith('rt_total_')]
                valuationModel = df_date[['dt', 'hr', 'node_num'] + total_col].groupby(
                    ['dt', 'hr', 'node_num']).max().reset_index()

                # 2. Node selection
                nodeSelection = pd.read_sql(
                    f"SELECT * FROM Fourier_{opexchange}.nodeSelection WHERE dt = '{bid_date}' AND source = 'PCA'", conn)
                nodeSelection['dt']       = nodeSelection['dt'].astype(str)
                nodeSelection['node_num'] = nodeSelection['node_num'].astype(int)
                nodeSelection['nodeSelection'] = nodeSelection_label
                nodeSelection.to_csv(file_location_cloudserver + 'nodeSelection/' + bid_date + '.csv', index=False)

                # 3. Op_rate and ref price
                ve_port = ve_portfolio_constructor_fourier.VEPortfolioConstructorFourier(opexchange)
                valuationModel = ve_port.get_oprate_lmp_price_and_ref_value(valuationModel)
                for col in ['op_rate_inc_a', 'op_rate_dec_a', 'op_rate_inc_f', 'op_rate_dec_f', 'bid_ref_price', 'offer_ref_price']:
                    if col in valuationModel.columns:
                        valuationModel[col].fillna(value=valuationModel[col].mean(), inplace=True)

                # 4. Cumulative return and risk
                ve_port.calculate_cumulative_return_and_risk_for_one_day(
                    valuationModel,
                    save_cloudserver_location=file_location_cloudserver + 'return_and_risk/',
                    prediction='total', segments=segments, date=bid_date)

                # 5. Wind / LWG for constraints
                total_wind_df = wind_vh.get_data_and_mapping_for_baa_zonal_wind(
                    node_list=[636], opexchange='SPP',
                    start_dt=(pd.to_datetime(bid_date) - pd.to_timedelta('1D')).strftime('%Y-%m-%d'),
                    end_dt=(pd.to_datetime(bid_date) + pd.to_timedelta('1D')).strftime('%Y-%m-%d'),
                    var_spec=['f'], impute=True, ramp_flag=True, ramp_periods=[2])[0].rename(columns={
                        'e_spp_baa_zonal_wind_forecast_f': 'spp_wind_total_forecast_f',
                        'e_spp_baa_zonal_wind_actual_a': 'spp_wind_total_actual_a',
                        'BackwardRampNoSlope2_e_spp_baa_zonal_wind_forecast_f': 'BackwardRampNoSlope2_spp_wind_total_forecast_f',
                        'BackwardRampNoSlope2_e_spp_baa_zonal_wind_actual_a': 'BackwardRampNoSlope2_spp_wind_total_actual_a'})
                
                total_lwg_df = loadwindgen_vh.get_data_and_mapping_for_baa_zonal_lwg(
                    [636], "SPP",
                    start_dt=(pd.to_datetime(bid_date) - pd.to_timedelta('35D')).strftime('%Y-%m-%d'),
                    end_dt=(pd.to_datetime(bid_date) + pd.to_timedelta('1D')).strftime('%Y-%m-%d'),
                    n_day_pctl_flag=True)[0].rename(columns={
                        'e_spp_baa_zonal_loadwindgen_forecast_f': 'spp_loadwindgen_forecast_f',
                        'e_spp_baa_zonal_loadwindgen_actual_a': 'spp_loadwindgen_actual_a',
                        'Perc_30D_e_spp_baa_zonal_loadwindgen_forecast_f': 'Perc_30D_spp_loadwindgen_forecast_f',
                        'Perc_30D_e_spp_baa_zonal_loadwindgen_actual_a': 'Perc_30D_spp_loadwindgen_actual_a'})

                constraints_option = ['v2', {
                    'max_mw_per_segment': 5, 'totalRiskAllowed': 10000, 'total_mw_limit': 20000,
                    'nodal_mw_limit': 700, 'nodal_hrly_mw_limit': 60, 'nodal_hrly_incdec_mw_limit': 40,
                    'totalCollateralAllowed': 700000, 'inc_upper_perc_limit': 0.7, 'dec_upper_perc_limit': 0.7,
                    'hrly_inc_mw_limit': 450,
                    'hrly_inc_upper_perc_limit_extreme_physical_condition': {
                        'BackwardRampNoSlope2_spp_wind_total_forecast_f': -3000,
                        'hrly_inc_upper_perc_limit': 0.1, 'physical_var_df': total_wind_df},
                    'hrly_dec_upper_perc_limit_extreme_physical_condition': {
                        'Perc_30D_spp_loadwindgen_forecast_f': 0.02, 'hrly_dec_upper_perc_limit': 0.2,
                        'physical_var_df': total_lwg_df},
                }]

                # 6. Portfolio construction (production params)
                para = {
                    'maxDecPrice': 300, 'minIncPrice': -75, 'PowerROC': 1,
                    'minExpectedProfit': 0.7, 'maximumROR': 3000, 'minimumROR': 250, 'maximumROC': 100,
                    'MinExpectedReturnOnCollateral': 0, 'maxSegmentNum': 8,
                    'objectiveFunction': 'expectedProfit',
                    'constraints_option': constraints_option, 'PortionOfFullRisk': 1,
                    'segments_clear_prob': pd.DataFrame(),
                    'data_location': file_location_cloudserver,
                    'nodeSelection_label': nodeSelection_label,
                    'valuationModel_label': valuationModel_label,
                    'condition_label': condition_label, 'cumulativeRisk_ceil': -0.1
                }
                portfolio = ve_port.get_daily_terence_portfolio(bid_date, **para)

                # 7. Cut
                portfolio = portfolio[portfolio['bid_mw'] > 0]
                portfolio = portfolio_cut(portfolio, bid_date)

                # 8. Scale (fixed scale only — no wind scaling for Fourier backtest)
                scale_factor = get_scale_factor(opexchange).values[0][0]
                portfolio['bid_mw'] = portfolio['bid_mw'] * scale_factor
                portfolio['run_number'] = run_number

                print(f'  {len(portfolio)} bids, {round(portfolio["bid_mw"].sum())} total MW')
                all_portfolios.append(portfolio)

            except Exception as e:
                import traceback
                print(f'  [SKIP] {bid_date}: {e}')
                traceback.print_exc()
                continue

    finally:
        shutil.rmtree(file_location_cloudserver, ignore_errors=True)
        print(f'Cleaned up {file_location_cloudserver}')

    # --- save ---
    if all_portfolios:
        final_portfolio = pd.concat(all_portfolios, ignore_index=True)
        save_path = f'/var/www/python/Qingcheng/WFiles/{saved}.csv'
        final_portfolio.to_csv(save_path, index=False)
        print(f'\nSaved {final_portfolio.shape} to {save_path}')
    else:
        print('No portfolios generated.')
        final_portfolio = pd.DataFrame()
    return final_portfolio




def simulate_total_ftp(table):

    MARKET_CONFIG = {
    'SPP': {
        'load_table': 'spp_physical.spp_latest_load_forecast',
        'wind_table': 'spp_physical.spp_latest_wind_forecast',
        'inc_op_rate': 2.0,
        'dec_op_rate': 0.1
    },
    'PJM': {
        'load_table': '',
        'wind_table': '',
        'inc_op_rate': 0.0,
        'dec_op_rate': 0.0
    },
    'MISO': {'load_table': '', 'wind_table': '', 'inc_op_rate': 0.0, 'dec_op_rate': 0.0},
    'ERCOT': {'load_table': '', 'wind_table': '', 'inc_op_rate': 0.0, 'dec_op_rate': 0.0},
    'NYISO': {'load_table': '', 'wind_table': '', 'inc_op_rate': 0.0, 'dec_op_rate': 0.0},
}
    
    m_cfg = MARKET_CONFIG['SPP']
    df_bids = table

    # 3. Prices
    node_list = df_bids['node_num'].unique().tolist()

    node_obj = Node(node_nums=node_list, market='SPP')
    date_start = table.dt.min()
    date_end = table.dt.max()
    print(date_start,date_end)

    df_prices = node_obj.get_price(
        start_dt=date_start,
        end_dt=date_end,
        component=['LMP','MCC'],
        type=['DA', 'RT'],
        granularity='hourly'
    )

    if not df_prices.empty:
        df_prices.rename(columns={'da_total': 'dalmp', 'rt_total': 'rtlmp'}, inplace=True)
        df_prices['hr'] = df_prices['hr'].astype(int)
        df_prices['node_num'] = df_prices['node_num'].astype(int)
        df_sim = pd.merge(df_bids, df_prices, left_on=['dt', 'hr', 'node_num'],
                            right_on=['dt', 'hr', 'node_num'], how='left')
    else:
        df_sim['dalmp'] = np.nan
        df_sim['rtlmp'] = np.nan

    # 4. Calculations (Preserved as per instruction)
    inc_op_rate = m_cfg['inc_op_rate']
    dec_op_rate = m_cfg['dec_op_rate']

    conditions_clear = [
        (df_sim['incdec'] == 'Decrement') & (df_sim['bid_price'] >= df_sim['dalmp']),
        (df_sim['incdec'] == 'Increment') & (df_sim['bid_price'] <= df_sim['dalmp'])
    ]
    df_sim['is_cleared'] = np.select(conditions_clear, [True, True], default=False)
    df_sim.loc[df_sim['dalmp'].isna(), 'is_cleared'] = False
    df_sim['clear_mw'] = np.where(df_sim['is_cleared'], df_sim['bid_mw'], 0.0)

    # DA & RT Cash
    df_sim['total_da_val'] = np.where(df_sim['incdec'] == 'Decrement', -1 * df_sim['clear_mw'] * df_sim['dalmp'], df_sim['clear_mw'] * df_sim['dalmp'])
    df_sim['total_da_elapsed'] = np.where(df_sim['rtlmp'].isna(), 0, df_sim['total_da_val'])
    df_sim['total_da_slack'] = np.where(df_sim['incdec'] == 'Decrement', -1 * df_sim['clear_mw'] * df_sim['da_slack'], df_sim['clear_mw'] * df_sim['da_slack'])
    df_sim['total_da_slack_elapsed'] = np.where(df_sim['rtlmp'].isna(), 0, df_sim['total_da_slack'])
    df_sim['total_da_congestional'] = np.where(df_sim['incdec'] == 'Decrement', -1 * df_sim['clear_mw'] * df_sim['da_congestion_x'], df_sim['clear_mw'] * df_sim['da_congestion_x'])
    df_sim['total_da_cong_elapsed'] = np.where(df_sim['rtlmp'].isna(), 0, df_sim['total_da_congestional'])

    rt_calc = np.where(df_sim['incdec'] == 'Decrement', df_sim['clear_mw'] * df_sim['rtlmp'], -1 * df_sim['clear_mw'] * df_sim['rtlmp'])
    df_sim['total_rt_elapsed'] = np.where(df_sim['rtlmp'].isna(), 0, rt_calc)
    rt_calc = np.where(df_sim['incdec'] == 'Decrement', df_sim['clear_mw'] * df_sim['rt_slack'], -1 * df_sim['clear_mw'] * df_sim['rt_slack'])
    df_sim['total_rt_slack_elapsed'] = np.where(df_sim['rtlmp'].isna(), 0, rt_calc)
    rt_calc = np.where(df_sim['incdec'] == 'Decrement', df_sim['clear_mw'] * df_sim['rt_congestion_y'], -1 * df_sim['clear_mw'] * df_sim['rt_congestion_y'])
    df_sim['total_rt_cong_elapsed'] = np.where(df_sim['rtlmp'].isna(), 0, rt_calc)

    # RSG
    rsg_cost = np.where(df_sim['incdec'] == 'Increment', df_sim['clear_mw'] * inc_op_rate, df_sim['clear_mw'] * dec_op_rate)
    df_sim['op_rate_val'] = np.where(df_sim['rtlmp'].isna(), 0, -1 * rsg_cost)

    # Totals
    df_sim['gross_pnl'] = df_sim['total_da_elapsed'] + df_sim['total_rt_elapsed']
    df_sim['net_pnl'] = df_sim['gross_pnl'] + df_sim['op_rate_val']
    df_sim['slack_pnl'] = df_sim['total_da_slack_elapsed']+df_sim['total_rt_slack_elapsed']
    df_sim['cong_pnl'] = df_sim['total_da_cong_elapsed']+df_sim['total_rt_cong_elapsed']
    
    df_sim.fillna(0, inplace=True)
    df_sim.sort_values(by=['gross_pnl'],ascending=True,inplace=True)
    # 5. Aggregation & Data Return

    agg_dict = {
        'bid_mw': 'sum', 'clear_mw': 'sum',
        'total_da_val': 'sum', 'total_da_elapsed': 'sum', 'total_rt_elapsed': 'sum', 'op_rate_val': 'sum',
        'gross_pnl': 'sum', 'net_pnl': 'sum'
    }
    renamer = {
        'bid_mw': 'BidMW', 'clear_mw': 'ClearMW',
        'total_da_val': 'TotalDA$', 'total_da_elapsed': 'TotalDAElapsed$',
        'total_rt_elapsed': 'TotalRTElapsed$', 'op_rate_val': 'OpRate$',
        'gross_pnl': 'Gross$', 'net_pnl': 'Net$'
    }

    # A. Overall Summary
    summ_data = df_sim.agg(agg_dict).to_frame().T.rename(columns=renamer).to_dict('records')
    return df_sim


def pnl_metrics(df):
    # nogo full days
    nogo_days = pd.date_range(start='2026-01-23', end='2026-01-28').strftime('%Y-%m-%d').tolist()
    df = df[~df['dt'].isin(nogo_days)]

    # nogo specific (dt, hr) pairs
    nogo_dthr = (
        [('2026-01-30', hr) for hr in range(8, 13)] +   # hr 8–12
        [('2026-01-31', hr) for hr in range(2, 12)]      # hr 2–11
    )
    df['_key'] = list(zip(pd.to_datetime(df['dt']).dt.strftime('%Y-%m-%d'), df['hr'].astype(int)))
    df = df[~df['_key'].isin(nogo_dthr)].drop(columns=['_key'])


    # Daily PnL
    # multiply by the risk factor which is 0.65
    df['net_pnl'] = df['net_pnl']
    daily_pnl = df.groupby('dt')['net_pnl'].sum().reset_index()
    daily_pnl.columns = ['dt', 'daily_pnl']
    daily_pnl['cumulative_pnl'] = daily_pnl['daily_pnl'].cumsum()

    # Cumulative PnL graph
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(daily_pnl['dt'], daily_pnl['cumulative_pnl'], linewidth=1.5)
    ax.set_title('Cumulative PnL (with dayzer)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Net PnL ($)')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.xticks(rotation=45, ha='right')
    ax.xaxis.set_major_locator(plt.MaxNLocator(20))
    plt.tight_layout()
    plt.show()

    # Summary stats
    total_pnl = daily_pnl['daily_pnl'].sum()
    profit_days = daily_pnl[daily_pnl['daily_pnl'] > 0]['daily_pnl'].sum()
    loss_days = daily_pnl[daily_pnl['daily_pnl'] < 0]['daily_pnl'].sum()
    max_loss = daily_pnl['daily_pnl'].min()
    profit_to_loss = abs(profit_days / loss_days) if loss_days != 0 else float('inf')

    print(f'Total PnL:          ${total_pnl:,.2f}')
    print(f'Max Daily Loss:     ${max_loss:,.2f}')
    print(f'Profit-to-Loss:     {profit_to_loss:.2f}')
    print(f'Profit Days Total:  ${profit_days:,.2f}')
    print(f'Loss Days Total:    ${loss_days:,.2f}')
    print(f'Win Rate:           {(daily_pnl["daily_pnl"] > 0).mean():.1%}')


def select_unique_nodes_across_dates(start_date,end_date, nodes_per_day=3, seed=42):
    daily_node_df = sql_functions.download_df_from_sql_db(
    f"SELECT DISTINCT dt, node_num FROM Fourier_SPP.nodeSelection WHERE dt >= '{start_date}' AND dt <= '{end_date}'")
    rng = np.random.default_rng(seed)
    selected_rows = []
    used_nodes = set()

    for dt in sorted(daily_node_df['dt'].unique()):
        available = daily_node_df[daily_node_df['dt'] == dt]['node_num'].tolist()
        candidates = [n for n in available if n not in used_nodes]

        if len(candidates) < nodes_per_day:
            chosen = candidates
        else:
            chosen = rng.choice(candidates, size=nodes_per_day, replace=False).tolist()

        for node in chosen:
            selected_rows.append({'dt': dt, 'node_num': node})
            used_nodes.add(node)
    
    result =  pd.DataFrame(selected_rows, columns=['dt', 'node_num'])
    result.to_csv(f'/var/www/python/Qingcheng/WFiles/Ultra/{start_date}_{end_date}_node_selection')
    return result

def run_valuation_backtest(function_call, num_of_nodes=1000):
    opexchange = 'SPP'
    daily_node_df = pd.read_csv('/var/www/python/Qingcheng/WFiles/Ultra/node_selection_42_0121_0421.csv')
    daily_node_df = daily_node_df[6:][:num_of_nodes]
    dataframe = run_in_kubernetes(daily_node_df, function_call)
    print('Valuation result is: ', get_metrics(dataframe))
    return dataframe, get_metrics(dataframe)

def run_portfolio_backtest(df):
    dataframe = fourier_port(df)
    print(dataframe.head())
    pnl = simulate_total_ftp(dataframe)
    pnl_metrics(pnl)
    return dataframe, pnl