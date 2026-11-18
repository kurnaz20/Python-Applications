# ISL467E Management Engineering Applications with Python - Homework I
# Intelligent Dashboard Design using Python Streamlit
# Decision support dashboard for a B2B book distributor (Sales, Marketing, Supply Chain)
#
# Run locally:  streamlit run streamlit_app.py

from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(page_title="B2B Book Distributor Dashboard", layout="wide")

COLORS = ["#5B2A86", "#9D6BD8", "#3E1C5E", "#B794F4", "#7B3FA0", "#D6BCFA"]   # purple theme
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = COLORS

DATA_DIR = Path(__file__).parent

# Purple background for the insight boxes
st.markdown("""
<style>
div[data-testid="stAlert"] > div, div[data-testid="stAlertContainer"] {background-color: #F4EEFA !important;}
div[data-testid="stAlert"] p {color: #3E1C5E !important;}
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 1. Data loading and preprocessing
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    # The Excel file was saved as Parquet (same rows and columns) so it loads fast online
    path = DATA_DIR / "data" / "b2b_transactions.parquet"
    if not path.exists():
        path = DATA_DIR / "b2b_transactions.parquet"
    raw = pd.read_parquet(path)
    df = raw.copy()
    steps = [("Raw data", len(df))]

    # Data types
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["StockCode"] = df["StockCode"].astype(str)

    # Missing values
    df = df.dropna(subset=["Quantity"])
    steps.append(("After removing rows with missing Quantity", len(df)))
    df["City"] = df["City"].fillna("UNKNOWN")
    df["Category"] = df["Category"].fillna("Uncategorized")

    # Wrong records: zero/negative quantity or price
    df = df[(df["Quantity"] > 0) & (df["NetPrice"] > 0) & (df["UnitPrice"] > 0)]
    steps.append(("After removing zero/negative quantity or price", len(df)))

    # Wrong records: net price higher than list price
    df = df[df["NetPrice"] <= df["UnitPrice"]]
    steps.append(("After removing net price above list price", len(df)))

    # New columns
    df["Sales"] = df["Quantity"] * df["NetPrice"]
    df["Discount"] = 1 - df["NetPrice"] / df["UnitPrice"]
    df["Month"] = df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
    df["Week"] = df["InvoiceDate"].dt.to_period("W").dt.start_time
    df["Weekday"] = df["InvoiceDate"].dt.day_name()

    missing = raw.isna().sum()
    return df, pd.DataFrame(steps, columns=["Step", "Rows"]), missing


def outliers_bp(series):
    # Outlier analysis with the box-plot (IQR) rule
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return outliers, lower_bound, upper_bound


def abc_class(p):
    if p <= 80:
        return "A"
    elif p <= 95:
        return "B"
    else:
        return "C"


def xyz_class(cv):
    if cv < 0.5:
        return "X"
    elif cv <= 1.0:
        return "Y"
    else:
        return "Z"


def to_excel(tables):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, table in tables.items():
            table.to_excel(writer, sheet_name=name, index=False)
    return output.getvalue()


data, prep_steps, missing_values = load_data()

# -----------------------------------------------------------------------------
# 2. Sidebar filters
# -----------------------------------------------------------------------------
st.sidebar.header("Filters")
date_range = st.sidebar.date_input(
    "Date range",
    value=(data["InvoiceDate"].min(), data["InvoiceDate"].max()),
    min_value=data["InvoiceDate"].min(),
    max_value=data["InvoiceDate"].max(),
)
top_cities = data.groupby("City")["Sales"].sum().sort_values(ascending=False).index
top_categories = data.groupby("Category")["Sales"].sum().sort_values(ascending=False).index
selected_cities = st.sidebar.multiselect("City", top_cities)
selected_categories = st.sidebar.multiselect("Category", top_categories)

if len(date_range) != 2:
    st.stop()

df = data[(data["InvoiceDate"] >= pd.to_datetime(date_range[0])) &
          (data["InvoiceDate"] <= pd.to_datetime(date_range[1]))]
if selected_cities:
    df = df[df["City"].isin(selected_cities)]
if selected_categories:
    df = df[df["Category"].isin(selected_categories)]

if df.empty:
    st.warning("No data for the selected filters.")
    st.stop()

st.sidebar.write(f"Selected rows: {len(df):,}")

# -----------------------------------------------------------------------------
# 3. Title and KPIs
# -----------------------------------------------------------------------------
st.title("B2B Book Distributor - Decision Support Dashboard")
st.caption("Sales, marketing and supply chain analysis of 2020 transaction data")

monthly = df.groupby("Month").agg(Sales=("Sales", "sum"), Quantity=("Quantity", "sum"),
                                  Orders=("InvoiceNo", "nunique"),
                                  Customers=("CustomerID", "nunique")).reset_index()


def change(column):
    # Last month vs. previous month (%)
    if len(monthly) < 2:
        return None
    return f"{(monthly[column].iloc[-1] / monthly[column].iloc[-2] - 1) * 100:.1f}%"


total_sales = df["Sales"].sum()
total_orders = df["InvoiceNo"].nunique()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total sales", f"₺{total_sales:,.0f}", change("Sales"))
col2.metric("Orders", f"{total_orders:,}", change("Orders"))
col3.metric("Units sold", f"{df['Quantity'].sum():,.0f}", change("Quantity"))
col4.metric("Customers", f"{df['CustomerID'].nunique():,}", change("Customers"))
col5.metric("Average order value", f"₺{total_sales / total_orders:,.0f}")
st.caption("Change = last month compared with the previous month")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Summary", "Sales", "Marketing", "Supply Chain", "Forecast", "Data"])

# -----------------------------------------------------------------------------
# 4. Sales
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Monthly sales")
    fig = px.line(monthly, x="Month", y="Sales", markers=True, title="Monthly sales (₺)")
    st.plotly_chart(fig, width="stretch")

    col1, col2 = st.columns(2)
    category_sales = df.groupby("Category")["Sales"].sum().sort_values(ascending=False).head(10).reset_index()
    fig = px.bar(category_sales, x="Sales", y="Category", orientation="h", title="Top 10 categories")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    col1.plotly_chart(fig, width="stretch")

    city_sales = df.groupby("City")["Sales"].sum().sort_values(ascending=False).head(10).reset_index()
    fig = px.bar(city_sales, x="Sales", y="City", orientation="h", title="Top 10 cities")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    col2.plotly_chart(fig, width="stretch")

    col1, col2 = st.columns(2)
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_sales = df.groupby("Weekday")["Sales"].sum().reindex(days).reset_index()
    fig = px.bar(weekday_sales, x="Weekday", y="Sales", title="Sales by weekday")
    col1.plotly_chart(fig, width="stretch")

    product_sales = df.groupby("Description")[["Sales", "Quantity"]].sum() \
                      .sort_values("Sales", ascending=False).head(10).reset_index()
    col2.markdown("**Top 10 titles**")
    col2.dataframe(product_sales.round(0), hide_index=True, width="stretch")

    st.subheader("Order statistics")
    orders = df.groupby("InvoiceNo").agg(OrderValue=("Sales", "sum"), Units=("Quantity", "sum"),
                                         Discount=("Discount", "mean"))
    stats = orders.describe()
    stats.loc["CV"] = stats.loc["std"] / stats.loc["mean"]
    stats.loc["skewness"] = orders.skew()
    st.dataframe(stats.round(2), width="stretch")

    col1, col2 = st.columns(2)
    corr = monthly[["Sales", "Quantity", "Orders", "Customers"]].corr()
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Purples", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlation of monthly KPIs")
    col1.pyplot(fig)

    discount_sales = df.assign(Band=pd.cut(df["Discount"], [0, 0.3, 0.35, 0.4, 0.45, 1],
                                           labels=["<30%", "30-35%", "35-40%", "40-45%", ">45%"]))
    discount_sales = discount_sales.groupby("Band", observed=True)["Sales"].sum().reset_index()
    fig = px.bar(discount_sales, x="Band", y="Sales", title="Sales by discount level")
    col2.plotly_chart(fig, width="stretch")

    best_month = monthly.loc[monthly["Sales"].idxmax()]
    worst_month = monthly.loc[monthly["Sales"].idxmin()]
    st.info(f"Highest sales: {best_month['Month']:%B} (₺{best_month['Sales']:,.0f}). "
            f"Lowest sales: {worst_month['Month']:%B} (₺{worst_month['Sales']:,.0f}). "
            f"Top city {city_sales['City'].iloc[0].title()} has {city_sales['Sales'].iloc[0] / total_sales * 100:.0f}% "
            f"of sales.")

# -----------------------------------------------------------------------------
# 5. Marketing
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Customer analysis (RFM)")
    last_date = df["InvoiceDate"].max()
    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (last_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Sales", "sum")).reset_index()

    # Scores from 1 to 5 (5 = best)
    rfm["R"] = pd.qcut(rfm["Recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["F"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M"] = pd.qcut(rfm["Monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)

    def segment(row):
        if row["R"] >= 4 and row["F"] >= 4:
            return "Loyal customers"
        elif row["R"] >= 4:
            return "New / active customers"
        elif row["R"] <= 2 and row["F"] >= 3:
            return "At risk"
        elif row["R"] <= 2:
            return "Lost customers"
        else:
            return "Need attention"

    rfm["Segment"] = rfm.apply(segment, axis=1)
    segments = rfm.groupby("Segment").agg(Customers=("CustomerID", "count"),
                                          Sales=("Monetary", "sum")).reset_index()
    segments["Sales share %"] = segments["Sales"] / segments["Sales"].sum() * 100

    col1, col2 = st.columns(2)
    fig = px.pie(segments, names="Segment", values="Customers", title="Number of customers by segment",
                 color_discrete_sequence=COLORS)
    col1.plotly_chart(fig, width="stretch")
    fig = px.bar(segments.sort_values("Sales"), x="Sales", y="Segment", orientation="h",
                 title="Sales by segment")
    col2.plotly_chart(fig, width="stretch")
    st.dataframe(segments.round({"Sales": 0, "Sales share %": 1}), hide_index=True, width="stretch")

    st.subheader("Customer concentration")
    customer_sales = rfm.sort_values("Monetary", ascending=False).reset_index(drop=True)
    customer_sales["Cumulative %"] = customer_sales["Monetary"].cumsum() / customer_sales["Monetary"].sum() * 100
    customer_sales["Customer %"] = (customer_sales.index + 1) / len(customer_sales) * 100
    fig = px.line(customer_sales, x="Customer %", y="Cumulative %", title="Cumulative sales share of customers")
    fig.add_hline(y=80, line_dash="dash")
    st.plotly_chart(fig, width="stretch")
    top20_share = customer_sales.loc[customer_sales["Customer %"] <= 20, "Monetary"].sum() / total_sales * 100

    st.subheader("Customers to contact")
    at_risk = rfm[rfm["Segment"] == "At risk"].sort_values("Monetary", ascending=False)
    st.dataframe(at_risk.head(15)[["CustomerID", "Recency", "Frequency", "Monetary"]],
                 hide_index=True, width="stretch")

    st.info(f"The top 20% of customers make {top20_share:.0f}% of sales. "
            f"{len(at_risk)} customers bought often before but have not ordered recently "
            f"(₺{at_risk['Monetary'].sum():,.0f} past sales). These customers should be contacted first.")

# -----------------------------------------------------------------------------
# 6. Supply chain
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("ABC - XYZ analysis of titles")
    products = df.groupby("StockCode").agg(Description=("Description", "first"),
                                           Sales=("Sales", "sum"), Quantity=("Quantity", "sum")).reset_index()
    products = products.sort_values("Sales", ascending=False)
    products["% Cumulative"] = products["Sales"].cumsum() / products["Sales"].sum() * 100
    products["ABC"] = products["% Cumulative"].apply(abc_class)

    monthly_units = df.groupby(["StockCode", "Month"])["Quantity"].sum().unstack(fill_value=0)
    cv = monthly_units.std(axis=1) / monthly_units.mean(axis=1)
    products["CV"] = products["StockCode"].map(cv)
    products["XYZ"] = products["CV"].apply(xyz_class)

    col1, col2 = st.columns(2)
    abc_summary = products.groupby("ABC").agg(Titles=("StockCode", "count"), Sales=("Sales", "sum")).reset_index()
    abc_summary["Sales share %"] = abc_summary["Sales"] / abc_summary["Sales"].sum() * 100
    fig = px.bar(abc_summary, x="ABC", y="Titles", text="Titles", title="Number of titles in each ABC class")
    col1.plotly_chart(fig, width="stretch")

    matrix = pd.crosstab(products["ABC"], products["XYZ"])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Purples", ax=ax)
    ax.set_title("ABC - XYZ matrix (number of titles)")
    col2.pyplot(fig)
    st.caption("ABC: A = first 80% of sales, B = next 15%, C = last 5%.  "
               "XYZ (CV of monthly demand): X < 0.5 stable, Y 0.5-1.0 variable, Z > 1.0 irregular.")

    st.markdown("**Top A-class titles**")
    st.dataframe(products[products["ABC"] == "A"].head(15)[["Description", "Sales", "Quantity", "CV", "XYZ"]]
                 .round({"Sales": 0, "CV": 2}), hide_index=True, width="stretch")

    st.subheader("Order size and outliers")
    order_units = df.groupby("InvoiceNo")["Quantity"].sum()
    outliers, lower_bound, upper_bound = outliers_bp(order_units)
    col1, col2 = st.columns(2)
    fig = px.box(order_units.reset_index(), y="Quantity", log_y=True, title="Units per order (box plot, log scale)")
    col1.plotly_chart(fig, width="stretch")
    city_orders = df.groupby("City").agg(Units=("Quantity", "sum"), Orders=("InvoiceNo", "nunique")).reset_index()
    city_orders["Units per order"] = city_orders["Units"] / city_orders["Orders"]
    city_orders = city_orders.sort_values("Units", ascending=False).head(10)
    fig = px.bar(city_orders, x="City", y="Units", title="Units shipped to top 10 cities")
    col2.plotly_chart(fig, width="stretch")

    a_titles = (products["ABC"] == "A").sum()
    st.info(f"Only {a_titles:,} of {len(products):,} titles ({a_titles / len(products) * 100:.1f}%) are in class A "
            f"and make 80% of sales. {len(outliers):,} orders are larger than {upper_bound:.0f} units (outliers); "
            "these big orders should be planned with the customer in advance.")

# -----------------------------------------------------------------------------
# 7. Forecast (linear and polynomial regression)
# -----------------------------------------------------------------------------
with tab5:
    st.subheader("Weekly sales forecast")
    col1, col2 = st.columns(2)
    target = col1.selectbox("Forecast", ["Sales", "Quantity"])
    weeks_ahead = col2.slider("Weeks to forecast", 4, 12, 8)

    weekly = df.groupby("Week")[target].sum().reset_index()
    weekly = weekly.iloc[1:-1]                      # first and last week of the year are incomplete
    weekly["t"] = np.arange(1, len(weekly) + 1)     # time index

    if len(weekly) < 15:
        st.warning("Not enough weeks for a forecast. Please select a longer date range.")
    else:
        X = weekly[["t"]]
        y = weekly[target]
        test_size = 8
        X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
        y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

        # Model 1: linear regression (trend)
        model1 = LinearRegression()
        model1.fit(X_train, y_train)
        pred1 = model1.predict(X_test)

        # Model 2: polynomial regression (degree 2)
        poly = PolynomialFeatures(degree=2)
        model2 = LinearRegression()
        model2.fit(poly.fit_transform(X_train), y_train)
        pred2 = model2.predict(poly.transform(X_test))

        # Model 3: 4-week moving average (benchmark)
        pred3 = np.repeat(y_train.iloc[-4:].mean(), test_size)

        def scores(pred):
            return [metrics.mean_absolute_error(y_test, pred),
                    np.sqrt(metrics.mean_squared_error(y_test, pred)),
                    metrics.mean_absolute_percentage_error(y_test, pred) * 100]

        perf = pd.DataFrame({"Metric": ["MAE", "RMSE", "MAPE (%)"],
                             "Linear regression": scores(pred1),
                             "Polynomial regression": scores(pred2),
                             "Moving average": scores(pred3)})
        best = perf.set_index("Metric").loc["MAPE (%)"].idxmin()

        # Retrain the best model on all weeks and forecast the future
        future_t = pd.DataFrame({"t": np.arange(len(weekly) + 1, len(weekly) + weeks_ahead + 1)})
        if best == "Linear regression":
            forecast = LinearRegression().fit(X, y).predict(future_t)
        elif best == "Polynomial regression":
            forecast = LinearRegression().fit(poly.fit_transform(X), y).predict(poly.transform(future_t))
        else:
            forecast = np.repeat(y.iloc[-4:].mean(), weeks_ahead)
        future_weeks = pd.date_range(weekly["Week"].iloc[-1] + pd.Timedelta(weeks=1), periods=weeks_ahead, freq="7D")
        forecast_df = pd.DataFrame({"Week": future_weeks.date, "Forecast": forecast.round(0)})

        chart = pd.concat([
            weekly[["Week", target]].rename(columns={target: "Value"}).assign(Type="Actual"),
            pd.DataFrame({"Week": weekly["Week"].iloc[-test_size:], "Value": pred1, "Type": "Linear (test)"}),
            pd.DataFrame({"Week": weekly["Week"].iloc[-test_size:], "Value": pred2, "Type": "Polynomial (test)"}),
            pd.DataFrame({"Week": future_weeks, "Value": forecast, "Type": f"Forecast ({best})"})])
        fig = px.line(chart, x="Week", y="Value", color="Type", markers=True, title=f"Weekly {target.lower()}",
                      color_discrete_sequence=["#3E1C5E", "#B794F4", "#E0B0FF", "#9D6BD8"])
        st.plotly_chart(fig, width="stretch")

        col1, col2 = st.columns(2)
        col1.markdown("**Model performance on the last 8 weeks**")
        col1.dataframe(perf.round(2), hide_index=True, width="stretch")
        col2.markdown("**Forecast**")
        col2.dataframe(forecast_df, hide_index=True, width="stretch")

        st.info(f"The best model is {best} (lowest MAPE: {perf.set_index('Metric').loc['MAPE (%)', best]:.1f}%). "
                f"Expected total {target.lower()} for the next {weeks_ahead} weeks: {forecast.sum():,.0f}. "
                "Only one year of data is available, so the forecast is for short-term planning.")

# -----------------------------------------------------------------------------
# 8. Summary
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Executive summary")
    st.markdown(f"""
- Sales are highest in **{best_month['Month']:%B}** and lowest in **{worst_month['Month']:%B}**.
- **{city_sales['City'].iloc[0].title()}** brings {city_sales['Sales'].iloc[0] / total_sales * 100:.0f}% of sales.
- The top 20% of customers make **{top20_share:.0f}%** of sales.
- **{len(at_risk)}** regular customers have not ordered recently.
- **{a_titles:,}** titles ({a_titles / len(products) * 100:.1f}%) make 80% of sales.
""")
    st.subheader("Recommendations")
    st.markdown("""
- **Sales:** Focus on the top cities and categories, and prepare campaigns before the peak months.
- **Marketing:** Contact the at-risk customers in the Marketing tab and keep the top customers loyal.
- **Supply chain:** Keep enough stock of A-class titles and order more before the autumn peak.
""")
    fig = px.bar(monthly, x="Month", y="Sales", title="Monthly sales (₺)")
    st.plotly_chart(fig, width="stretch")

# -----------------------------------------------------------------------------
# 9. Data
# -----------------------------------------------------------------------------
with tab6:
    st.subheader("Data preprocessing")
    col1, col2 = st.columns(2)
    col1.dataframe(prep_steps, hide_index=True, width="stretch")
    missing_table = missing_values[missing_values > 0].reset_index()
    missing_table.columns = ["Column", "Missing values"]
    col2.markdown("**Missing values in the raw data**")
    col2.dataframe(missing_table, hide_index=True, width="stretch")
    st.markdown("Missing City and Category values were filled as UNKNOWN / Uncategorized. "
                "New columns: Sales = Quantity x NetPrice, Discount = 1 - NetPrice / UnitPrice, Month, Week, Weekday.")

    st.download_button("Download results (Excel)",
                       data=to_excel({"Monthly": monthly, "RFM": rfm, "ABC_XYZ": products.head(5000)}),
                       file_name="dashboard_results.xlsx")
    st.markdown("**First 100 rows of the cleaned data**")
    st.dataframe(df.head(100), width="stretch")
