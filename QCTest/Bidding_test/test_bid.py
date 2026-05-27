import sys
sys.path.append("/var/www/python/Qingcheng/nighthawk/")
from nighthawk.data.product import ve
from nighthawk.data.product.ve import DailyBidsManager, BidsManager, Collateral
from nighthawk.risk.risk_manager_ve import RiskManagerAutomated


def check_allzero_groups(portfolio, label):
    grp = portfolio.groupby(['node_num', 'dt', 'hr', 'incdec'])['bid_mw'].sum()
    allzero = grp[grp == 0]
    if not allzero.empty:
        print(f"[ALLZERO after {label}] {len(allzero)} group(s) with all bid_mw=0:")
        print(allzero.reset_index().to_string(index=False))


bid_date = '2026-05-12'
opexchange = 'SPP'

bid_manager = DailyBidsManager(opexchange, bid_date)
portfolio = bid_manager.get_bids_from_table(label='preautomated_cuts')
print(portfolio.groupby(['strategy', 'incdec'])['bid_mw'].sum().reset_index().pivot(
    index='strategy', columns='incdec', values='bid_mw').round(1))

risk_manager = RiskManagerAutomated(opexchange=opexchange, bid_date=bid_date)
assert risk_manager.check_portfolio_df_quality(portfolio, production=True) == True, \
    "portfolio failed check_portfolio_df_quality"

cuts_list = [
    'go_nogo', 'constraint_family_exposure', 'hub_node_predictions', 'weather_variables',
    'condition_change', 'rep_node', 'pricing_signals', 'last_hr_price_change',
    'dayzer_nodal_price', 'tx_outage', 'agg_volume', 'consolidation', 'investment_limit',
]

check_allzero_groups(portfolio, 'no_cut')

for cut in cuts_list:
    print(f"\n=== CUT: {cut} ===")

    if 'go_nogo' in cut:
        portfolio, portfolio_diff, reason_str = risk_manager.run_no_go_days_cut(portfolio)
        print(portfolio_diff, reason_str)
        check_allzero_groups(portfolio, cut)

    if 'constraint_family_exposure' in cut:
        pass  # PJM/MISO only

    if 'hub_node_predictions' in cut:
        pass  # PJM only

    if 'weather_variables' in cut:
        pass  # PJM only

    if 'condition_change' in cut:
        pass  # PJM only

    if 'rep_node' in cut:
        pass  # SPP skipped

    if 'pricing_signals' in cut:
        portfolio, portfolio_diff = risk_manager.run_pricing_signals_risk_cut(portfolio)
        print(portfolio_diff)
        check_allzero_groups(portfolio, cut)

    if 'last_hr_price_change' in cut:
        pass  # SPP skipped

    if 'dayzer_nodal_price' in cut:
        pass  # MISO/PJM only

    if 'tx_outage' in cut:
        portfolio, portfolio_diff = risk_manager.run_tx_outage_risk_cut(
            portfolio,
            cut_off_rank=20,
            plan_otg_start_dt_limit=7,
            plan_otg_end_dt_limit=7,
            actual_otg_start_dt_limit=7,
            actual_otg_end_dt_limit=7,
            min_outage_kv=0,
            dfax_cutoff=0.05,
        )
        print(portfolio_diff)
        check_allzero_groups(portfolio, cut)

    if 'agg_volume' in cut:
        portfolio, portfolio_diff = risk_manager.run_daily_portfolio_limits_cut(portfolio)
        print(portfolio_diff)
        check_allzero_groups(portfolio, cut)

    if 'consolidation' in cut:
        portfolio = bid_manager.set_portfolio(portfolio)
        _nan_before = portfolio['bid_price'].isna().sum()
        _mw_nan_before = portfolio['bid_mw'].isna().sum()
        _mw_zero_before = (portfolio['bid_mw'] <= 0).sum()
        if _nan_before or _mw_nan_before or _mw_zero_before:
            print(f"[CONSOLIDATION] WARNING before consolidate_all_bids: bid_price NaN={_nan_before}, bid_mw NaN={_mw_nan_before}, bid_mw<=0 count={_mw_zero_before}")
        portfolio = bid_manager.consolidate_all_bids(portfolio, method='kmeans')
        _nan_after = portfolio['bid_price'].isna().sum()
        if _nan_after:
            print(f"[CONSOLIDATION] WARNING after consolidate_all_bids: bid_price NaN={_nan_after}")
        portfolio = bid_manager.reset_segments(portfolio)
        _nan_final = portfolio['bid_price'].isna().sum()
        if _nan_final:
            print(f"[CONSOLIDATION] WARNING after reset_segments: bid_price NaN={_nan_final}")

    if 'investment_limit' in cut:
        portfolio, portfolio_diff = risk_manager.run_investment_risk_control_cut(portfolio)
        print(portfolio_diff)

print("\n=== FINAL PORTFOLIO ===")
print(portfolio.columns.tolist())
print(portfolio.groupby(['strategy', 'incdec'])['bid_mw'].sum().reset_index().pivot(
    index='strategy', columns='incdec', values='bid_mw').round(1))
print(f"bid_price NaN total: {portfolio['bid_price'].isna().sum()}")
