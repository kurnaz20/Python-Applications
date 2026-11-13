# B2B Book Distributor: Decision Support Dashboard

ISL467E Management Engineering Applications with Python · Homework I

Interactive Streamlit dashboard built on the B2B book distributor's 2020 transaction data
(664,154 invoice lines). It supports decisions of the **Sales**, **Marketing** and **Supply Chain**
departments.

## Dashboard structure

| Tab | What it answers | Methods |
| --- | --- | --- |
| Overview | How is the business doing? | KPIs with month-over-month change, monthly revenue with 3-month average, executive summary |
| Sales | Where, when and what do we sell? | Daily/weekly/monthly trend with anomaly detection (z-score), weekday × month heatmap, city and category growth (H2 vs H1), discount analysis, best-selling titles |
| Marketing | Who are our customers and whom should we contact? | RFM + K-Means segmentation (k by silhouette score), churn early warning, new vs. returning customers, market basket analysis (support, confidence, lift) |
| Supply Chain | What should we stock and when should we reorder? | ABC–XYZ classification with policies, safety stock and reorder points (service level and lead time inputs), slow-moving titles, shipment load by city |
| Forecasting | How much will we sell in the coming weeks? | Naive, 4-week moving average, Holt's damped trend and Random Forest compared on a hold-out period (MAE, RMSE, MAPE), 80% interval |
| Data & Methods | How was the data prepared? | Preprocessing log, missing values, method summary, Excel download |

Sidebar filters (period, city, category, invoice series, bulk lines) apply to every tab.

## Files

```
streamlit_app.py              main application (data cleaning + analysis + dashboard)
convert_data.py               converts the raw Excel file to Parquet (no cleaning)
requirements.txt              Python libraries
.streamlit/config.toml        theme
data/b2b_transactions.parquet raw dataset in Parquet format
```

The original `B2B_Transaction_Data_.xlsx` (41 MB) is stored as Parquet (11 MB) with identical rows and
columns so the app loads quickly on Streamlit Community Cloud. All cleaning steps are in
`streamlit_app.py` and are listed in the *Data & Methods* tab.

## Run locally

```
pip install -r requirements.txt
streamlit run streamlit_app.py
```
