import pandas as pd
import time

def all_tables(node_selection):
    nodes = pd.read_csv(f'/var/www/python/Qingcheng/WFiles/Ultra/{node_selection}')

    gs_loc    = "gs://ve_fourier/production/SPP/training"
    max_retries = 1
    retry_delay = 5

    full_table = None

    for _, row in nodes.iterrows():
        node_num = int(row['node_num'])
        dt       = pd.to_datetime(row['dt']).strftime('%Y-%m-%d')

        for attempt in range(max_retries):
            try:
                df = pd.read_csv(f"{gs_loc}/{node_num}_{dt}.csv")
                df["dt"] = pd.to_datetime(df["dt"]).dt.strftime("%Y-%m-%d")
                break
            except Exception as e:
                print(f"[{node_num} {dt}] Attempt {attempt+1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    df = None
                    break

        if df is None:
            continue

        if full_table is None:
            full_table = df
        else:
            common_cols = full_table.columns.intersection(df.columns)
            full_table  = pd.concat([full_table[common_cols], df[common_cols]], ignore_index=True)

    print(f"Final shape: {full_table.shape}")
    return full_table


