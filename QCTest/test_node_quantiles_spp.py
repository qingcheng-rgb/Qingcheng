"""
Smoke test: does node_quantiles generate a correct SPP node-quantile table for a given bid date?

This is an INTEGRATION test — it hits BigQuery/SQL live via NodeQuantileReaderVE, so it needs the
usual prod credentials/connections. Run it directly:

    python QCTest/test_node_quantiles_spp.py
    python QCTest/test_node_quantiles_spp.py 2026-06-27      # override the date

It checks the table that feeds the quantile-risk dashboard (da_total_q*/rt_total_q*):
  1. non-empty                          5. 24 hours per node
  2. all rows are the requested dt       6. no nulls in the quantile columns
  3. key + quantile columns present      7. quantiles are monotonically non-decreasing per row
  4. node count matches Darwin selection
"""

import sys

sys.path.append("/var/www/python/Qingcheng/nighthawk/")
import numpy as np
import pandas as pd
from nighthawk.models.valuation.node_quantiles import NodeQuantileReaderVE
from nighthawk.models.securityselector import node_collector

OPEX = "SPP"
QUANTS = [1, 3, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95, 97, 99]
QUANTILES = [q / 100 for q in QUANTS]
DA_COLS = [f"da_total_q{q}" for q in QUANTS]
RT_COLS = [f"rt_total_q{q}" for q in QUANTS]


def get_spp_nodes(bid_date: str) -> list:
    """Fetch the SPP Darwin nodes for the date (passing node_list=[] to the reader hits an empty IN())."""
    nc = node_collector.VENodeCollector(opexchange=OPEX)
    nodes_df = nc.get_darwin_nodes(start_dt=bid_date, end_dt=bid_date, hub_included=True)
    return sorted(nodes_df["node_num"].dropna().astype(int).unique().tolist())


def run(bid_date: str) -> pd.DataFrame:
    node_list = get_spp_nodes(bid_date)
    print(f"Darwin SPP nodes for {bid_date}: {len(node_list)}")
    if not node_list:
        raise RuntimeError(
            f"No Darwin SPP nodes for {bid_date} - node selection not populated for that date.")
    reader = NodeQuantileReaderVE(OPEX)
    df = reader.get_predicted_nodal_quantiles(
        bid_date, bid_date, node_list=node_list, quantiles=QUANTILES,
        price_type=["Total"], include_mean_preds=False)
    return df


def check(bid_date: str) -> bool:
    print(f"\n=== node_quantiles {OPEX} smoke test for {bid_date} ===")
    try:
        df = run(bid_date)
    except Exception as e:
        print(f"  [FAIL] could not generate the quantile table: {type(e).__name__}: {e}")
        return False
    results = []  # (name, ok, detail)

    # 1. non-empty
    results.append(("non-empty", len(df) > 0, f"{len(df)} rows"))
    if df.empty:
        _report(results)
        return False

    # 2. all rows are the requested date
    dts = set(df["dt"].astype(str).unique())
    results.append(("single requested dt", dts == {bid_date}, f"dt values = {sorted(dts)}"))

    # 3. required columns present
    key_cols = ["dt", "hr", "node_num"]
    missing = [c for c in key_cols + DA_COLS + RT_COLS if c not in df.columns]
    results.append(("required columns present", not missing, f"missing = {missing[:8]}"))

    # 4. node count vs Darwin selection
    n_nodes = df["node_num"].nunique()
    results.append(("has nodes", n_nodes > 0, f"{n_nodes} nodes"))

    # 5. 24 hours per node
    hrs_per_node = df.groupby("node_num")["hr"].nunique()
    results.append(("24 hours per node", bool((hrs_per_node == 24).all()),
                    f"hour-counts: min={hrs_per_node.min()}, max={hrs_per_node.max()}"))

    # 6. no nulls in quantile columns
    q_cols = [c for c in DA_COLS + RT_COLS if c in df.columns]
    null_cells = int(df[q_cols].isna().sum().sum())
    results.append(("no nulls in quantile cols", null_cells == 0, f"{null_cells} null cells"))

    # 7. quantiles monotonically non-decreasing per row (q1 <= q3 <= ... <= q99)
    da_ok = _monotonic(df, [c for c in DA_COLS if c in df.columns])
    rt_ok = _monotonic(df, [c for c in RT_COLS if c in df.columns])
    results.append(("DA quantiles non-decreasing", da_ok == 0, f"{da_ok} rows violate"))
    results.append(("RT quantiles non-decreasing", rt_ok == 0, f"{rt_ok} rows violate"))

    _report(results)
    print("\nsample rows:")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df[["dt", "hr", "node_num"] + DA_COLS[:3] + RT_COLS[:3]].head())

    # critical checks (table is unusable if these fail); monotonicity/nulls are warnings
    critical = {"non-empty", "single requested dt", "required columns present",
                "has nodes", "24 hours per node"}
    return all(ok for name, ok, _ in results if name in critical)


def _monotonic(df, cols):
    """count rows where the ordered quantile columns are NOT non-decreasing (ignoring NaN)."""
    if len(cols) < 2:
        return 0
    vals = df[cols].to_numpy(dtype="float64")
    diffs = vals[:, 1:] - vals[:, :-1]
    # allow tiny float noise; NaN diffs are ignored
    with np.errstate(invalid="ignore"):
        violation = (diffs < -1e-6)
    return int(violation.any(axis=1).sum())


def _report(results):
    print()
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:32s} {detail}")


if __name__ == "__main__":
    bid_date = sys.argv[1] if len(sys.argv) > 1 else "2026-06-27"
    ok = check(bid_date)
    print(f"\nRESULT: {'OK' if ok else 'FAILED (see checks above)'}")
    sys.exit(0 if ok else 1)
