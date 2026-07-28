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


bid_date = '2026-07-18'
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
    
    if 'investment_limit' in cut:
        portfolio = risk_manager.run_investment_risk_control_cut(portfolio)
        print(portfolio)

print("\n=== FINAL PORTFOLIO ===")
print(portfolio.columns.tolist())
print(portfolio.groupby(['strategy', 'incdec'])['bid_mw'].sum().reset_index().pivot(
    index='strategy', columns='incdec', values='bid_mw').round(1))
print(f"bid_price NaN total: {portfolio['bid_price'].isna().sum()}")
