"""
Converts the raw Excel file into a compressed Parquet file (same rows and columns, no cleaning).
The Excel file (41 MB) takes ~30 s to read; the Parquet file loads in under a second on Streamlit Cloud.
All cleaning is done inside streamlit_app.py so that every step is visible in the dashboard.

Usage: python convert_data.py B2B_Transaction_Data_.xlsx data/b2b_transactions.parquet
"""
import sys

import pandas as pd

src = sys.argv[1] if len(sys.argv) > 1 else "B2B_Transaction_Data_.xlsx"
dst = sys.argv[2] if len(sys.argv) > 2 else "data/b2b_transactions.parquet"

raw = pd.read_excel(src, dtype={"StockCode": str, "Description": str})
raw.to_parquet(dst, index=False, compression="zstd")
print(f"{len(raw):,} rows x {raw.shape[1]} columns written to {dst}")
