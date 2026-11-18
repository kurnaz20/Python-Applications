# B2B Book Distributor - Decision Support Dashboard

ISL467E Management Engineering Applications with Python, Homework I

A Streamlit dashboard for the sales, marketing and supply chain departments of a B2B book distributor,
built on the 2020 transaction data (664,154 invoice lines).

## Tabs

| Tab | Content |
| --- | --- |
| Summary | Executive summary and recommendations for each department |
| Sales | Monthly sales, top categories, cities and titles, weekday sales, order statistics (describe, CV, skewness), correlation heatmap, sales by discount level |
| Marketing | RFM customer segments, customer concentration (Pareto), list of at-risk customers |
| Supply Chain | ABC and XYZ analysis of titles, ABC-XYZ matrix, outlier analysis of order size (IQR rule), units by city |
| Forecast | Weekly forecast with linear regression, polynomial regression and moving average; models compared with MAE, RMSE and MAPE |
| Data | Preprocessing steps, missing values, Excel download |

The sidebar filters (date range, city, category) apply to all tabs.

## Files

- `streamlit_app.py`: main source code
- `requirements.txt`: Python libraries
- `data/b2b_transactions.parquet`: dataset (B2B_Transaction_Data_.xlsx saved as Parquet with the same rows and columns, so it loads faster online)
- `convert_data.py`: script used to convert the Excel file to Parquet
- `.streamlit/config.toml`: purple color theme

## Run locally

```
pip install -r requirements.txt
streamlit run streamlit_app.py
```
