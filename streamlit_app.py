# ISL467E Management Engineering Applications with Python - Homework I
# Intelligent Dashboard Design using Python Streamlit
#
# Decision support dashboard for the sales, marketing and supply chain
# departments of a B2B book distributor (2020 transaction data).
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


# =============================================================================
# 1. PAGE SETUP AND STYLE
# =============================================================================
st.set_page_config(page_title="B2B Book Distributor Dashboard", layout="wide")

PURPLE = "#5B2A86"
COLORS = ["#5B2A86", "#9D6BD8", "#3E1C5E", "#B794F4", "#7B3FA0", "#D6BCFA"]
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = COLORS

st.markdown("""
<style>
div[data-testid="stMetric"] {background-color: #F4EEFA; border-radius: 10px; padding: 12px 16px;}
div[data-testid="stAlert"] > div {background-color: #F4EEFA !important;}
div[data-testid="stAlert"] p {color: #3E1C5E !important;}
</style>
""", unsafe_allow_html=True)


def style(fig, height=380):
    """Same look for every plotly chart."""
    fig.update_layout(height=height, margin=dict(t=50, b=20, l=10, r=10), title_font_size=16)
    return fig


def tl(value):
    """Format a number as Turkish lira."""
    return f"₺{value:,.0f}"


def tl_short(value):
    """Short format for KPI cards, e.g. ₺29.58M."""
    if value >= 1e6:
        return f"₺{value / 1e6:.2f}M"
    return tl(value)


# =============================================================================
# 2. DATA LOADING AND PREPROCESSING
# =============================================================================
@st.cache_data
def load_data():
    # The Excel file was saved as Parquet (same rows and columns) so it loads fast online
    folder = Path(__file__).parent
    path = folder / "data" / "b2b_transactions.parquet"
    if not path.exists():
        path = folder / "b2b_transactions.parquet"
    raw = pd.read_parquet(path)

    df = raw.copy()
    steps = [("Raw data", len(df))]

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["StockCode"] = df["StockCode"].astype(str)

    # Missing values
    df = df.dropna(subset=["Quantity"])
    steps.append(("Rows with missing Quantity removed", len(df)))
    df["City"] = df["City"].fillna("UNKNOWN").str.title()
    df["Category"] = df["Category"].fillna("Uncategorized")

    # Wrong records
    df = df[(df["Quantity"] > 0) & (df["NetPrice"] > 0) & (df["UnitPrice"] > 0)]
    steps.append(("Zero or negative quantity / price removed", len(df)))
    df = df[df["NetPrice"] <= df["UnitPrice"]]
    steps.append(("Net price above list price removed", len(df)))

    # New columns
    df["Sales"] = df["Quantity"] * df["NetPrice"]
    df["Discount"] = 1 - df["NetPrice"] / df["UnitPrice"]
    df["Month"] = df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
    df["Week"] = df["InvoiceDate"].dt.to_period("W").dt.start_time
    df["Weekday"] = df["InvoiceDate"].dt.day_name()

    return df, pd.DataFrame(steps, columns=["Step", "Rows"]), raw.isna().sum()


# =============================================================================
# 3. ANALYSIS FUNCTIONS
# =============================================================================
def outliers_bp(series):
    """Outlier analysis with the box-plot (IQR) rule."""
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


def rfm_segment(row):
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


@st.cache_data
def monthly_summary(df):
    return df.groupby("Month").agg(Sales=("Sales", "sum"), Quantity=("Quantity", "sum"),
                                   Orders=("InvoiceNo", "nunique"),
                                   Customers=("CustomerID", "nunique")).reset_index()


@st.cache_data
def rfm_analysis(df):
    last_date = df["InvoiceDate"].max()
    rfm = df.groupby("CustomerID").agg(
        City=("City", "first"),
        Recency=("InvoiceDate", lambda x: (last_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Sales", "sum")).reset_index()

    # Scores from 1 to 5 (5 = best)
    rfm["R"] = pd.qcut(rfm["Recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["F"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M"] = pd.qcut(rfm["Monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["Segment"] = rfm.apply(rfm_segment, axis=1)
    return rfm


@st.cache_data
def abc_xyz_analysis(df):
    products = df.groupby("StockCode").agg(Title=("Description", "first"), Category=("Category", "first"),
                                           Sales=("Sales", "sum"), Quantity=("Quantity", "sum")).reset_index()
    products = products.sort_values("Sales", ascending=False)
    products["Cumulative %"] = products["Sales"].cumsum() / products["Sales"].sum() * 100
    products["ABC"] = products["Cumulative %"].apply(abc_class)

    monthly_units = df.groupby(["StockCode", "Month"])["Quantity"].sum().unstack(fill_value=0)
    cv = monthly_units.std(axis=1) / monthly_units.mean(axis=1)
    products["CV"] = products["StockCode"].map(cv)
    products["XYZ"] = products["CV"].apply(xyz_class)
    return products


@st.cache_data
def forecast_analysis(df, target, weeks_ahead, test_size=8):
    weekly = df.groupby("Week")[target].sum().reset_index()
    weekly = weekly.iloc[1:-1].reset_index(drop=True)      # first and last week are incomplete
    weekly["t"] = np.arange(1, len(weekly) + 1)            # time index
    if len(weekly) < 15:
        return None

    X, y = weekly[["t"]], weekly[target]
    X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]
    poly = PolynomialFeatures(degree=2)

    # Train the three models on the training weeks
    predictions = {
        "Linear regression": LinearRegression().fit(X_train, y_train).predict(X_test),
        "Polynomial regression": LinearRegression().fit(poly.fit_transform(X_train), y_train)
                                                   .predict(poly.transform(X_test)),
        "Moving average (4 weeks)": np.repeat(y_train.iloc[-4:].mean(), test_size),
    }

    # Compare them on the test weeks
    perf = pd.DataFrame({name: [metrics.mean_absolute_error(y_test, p),
                                np.sqrt(metrics.mean_squared_error(y_test, p)),
                                metrics.mean_absolute_percentage_error(y_test, p) * 100]
                         for name, p in predictions.items()},
                        index=["MAE", "RMSE", "MAPE (%)"])
    best = perf.loc["MAPE (%)"].idxmin()

    # Retrain the best model on all weeks and forecast the future
    future_t = pd.DataFrame({"t": np.arange(len(weekly) + 1, len(weekly) + weeks_ahead + 1)})
    if best == "Linear regression":
        forecast = LinearRegression().fit(X, y).predict(future_t)
    elif best == "Polynomial regression":
        forecast = LinearRegression().fit(poly.fit_transform(X), y).predict(poly.transform(future_t))
    else:
        forecast = np.repeat(y.iloc[-4:].mean(), weeks_ahead)

    future_weeks = pd.date_range(weekly["Week"].iloc[-1] + pd.Timedelta(weeks=1), periods=weeks_ahead, freq="7D")
    test_weeks = weekly["Week"].iloc[-test_size:]

    chart = pd.concat(
        [pd.DataFrame({"Week": weekly["Week"], "Value": y, "Line": "Actual"})]
        + [pd.DataFrame({"Week": test_weeks, "Value": p, "Line": f"{name} (test)"})
           for name, p in predictions.items()]
        + [pd.DataFrame({"Week": future_weeks, "Value": forecast, "Line": f"Forecast: {best}"})])
    table = pd.DataFrame({"Week": future_weeks.date, "Forecast": forecast.round(0)})
    return perf, best, chart, table


def to_excel(tables):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, table in tables.items():
            table.to_excel(writer, sheet_name=name, index=False)
    return output.getvalue()


# =============================================================================
# 4. SIDEBAR FILTERS
# =============================================================================
data, prep_steps, missing_values = load_data()

st.sidebar.header("Filters")
date_range = st.sidebar.date_input(
    "Date range",
    value=(data["InvoiceDate"].min(), data["InvoiceDate"].max()),
    min_value=data["InvoiceDate"].min(),
    max_value=data["InvoiceDate"].max(),
)
city_list = data.groupby("City")["Sales"].sum().sort_values(ascending=False).index
category_list = data.groupby("Category")["Sales"].sum().sort_values(ascending=False).index
selected_cities = st.sidebar.multiselect("City", city_list, placeholder="All cities")
selected_categories = st.sidebar.multiselect("Category", category_list, placeholder="All categories")

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

st.sidebar.caption(f"{len(df):,} of {len(data):,} rows selected")


# =============================================================================
# 5. CALCULATIONS USED IN SEVERAL TABS
# =============================================================================
monthly = monthly_summary(df)
rfm = rfm_analysis(df)
products = abc_xyz_analysis(df)

total_sales = df["Sales"].sum()
total_orders = df["InvoiceNo"].nunique()
city_sales = df.groupby("City")["Sales"].sum().sort_values(ascending=False).reset_index()
category_sales = df.groupby("Category")["Sales"].sum().sort_values(ascending=False).reset_index()

best_month = monthly.loc[monthly["Sales"].idxmax()]
worst_month = monthly.loc[monthly["Sales"].idxmin()]
top_city_share = city_sales["Sales"].iloc[0] / total_sales * 100

customer_sales = rfm.sort_values("Monetary", ascending=False).reset_index(drop=True)
customer_sales["Customer %"] = (customer_sales.index + 1) / len(customer_sales) * 100
customer_sales["Cumulative %"] = customer_sales["Monetary"].cumsum() / customer_sales["Monetary"].sum() * 100
top20_share = customer_sales.loc[customer_sales["Customer %"] <= 20, "Monetary"].sum() / total_sales * 100
at_risk = rfm[rfm["Segment"] == "At risk"].sort_values("Monetary", ascending=False)

a_titles = (products["ABC"] == "A").sum()
a_share = a_titles / len(products) * 100


# =============================================================================
# 6. HEADER AND KPIs
# =============================================================================
st.title("B2B Book Distributor - Decision Support Dashboard")
st.caption(f"{date_range[0]:%d %b %Y} to {date_range[1]:%d %b %Y}  |  "
           "Sales, marketing and supply chain analysis")


def change(column):
    """Last month compared with the previous month (%)."""
    if len(monthly) < 2:
        return None
    return f"{(monthly[column].iloc[-1] / monthly[column].iloc[-2] - 1) * 100:.1f}%"


col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total sales", tl_short(total_sales), change("Sales"))
col2.metric("Orders", f"{total_orders:,}", change("Orders"))
col3.metric("Units sold", f"{df['Quantity'].sum():,.0f}", change("Quantity"))
col4.metric("Customers", f"{df['CustomerID'].nunique():,}", change("Customers"))
col5.metric("Average order value", tl(total_sales / total_orders))
st.caption("Change = last month compared with the previous month")

tab_summary, tab_sales, tab_marketing, tab_supply, tab_forecast, tab_data = st.tabs(
    ["Summary", "Sales", "Marketing", "Supply Chain", "Forecast", "Data"])


# =============================================================================
# 7. SUMMARY TAB
# =============================================================================
with tab_summary:
    left, right = st.columns([3, 2])
    with left:
        fig = px.bar(monthly, x="Month", y="Sales", title="Monthly sales (₺)")
        st.plotly_chart(style(fig, 420), width="stretch")
    with right:
        st.subheader("Executive summary")
        st.markdown(f"""
- Sales are highest in **{best_month['Month']:%B}** and lowest in **{worst_month['Month']:%B}**.
- **{city_sales['City'].iloc[0]}** brings {top_city_share:.0f}% of sales.
- The top 20% of customers make **{top20_share:.0f}%** of sales.
- **{len(at_risk)}** regular customers have not ordered recently.
- **{a_titles:,}** titles ({a_share:.1f}%) make 80% of sales.
""")
        st.subheader("Recommendations")
        st.markdown("""
- **Sales:** Focus on the top cities and categories, and prepare campaigns before the peak months.
- **Marketing:** Contact the at-risk customers and keep the top customers loyal.
- **Supply chain:** Keep enough stock of A-class titles and order more before the autumn peak.
""")


# =============================================================================
# 8. SALES TAB
# =============================================================================
with tab_sales:
    st.subheader("Sales trend")
    fig = px.line(monthly, x="Month", y="Sales", markers=True, title="Monthly sales (₺)")
    st.plotly_chart(style(fig), width="stretch")

    col1, col2 = st.columns(2)
    fig = px.bar(category_sales.head(10), x="Sales", y="Category", orientation="h", title="Top 10 categories")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    col1.plotly_chart(style(fig, 420), width="stretch")
    fig = px.bar(city_sales.head(10), x="Sales", y="City", orientation="h", title="Top 10 cities")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    col2.plotly_chart(style(fig, 420), width="stretch")

    st.subheader("Dynamic data grouping")
    col1, col2, col3 = st.columns(3)
    group_col = col1.selectbox("Group by", ["Category", "City", "Month", "Weekday"])
    metric_col = col2.selectbox("Metric", ["Sales", "Quantity", "Orders"])
    top_n = col3.slider("Number of groups", 5, 20, 10)

    if metric_col == "Orders":
        grouped = df.groupby(group_col)["InvoiceNo"].nunique().rename("Orders").reset_index()
    else:
        grouped = df.groupby(group_col)[metric_col].sum().reset_index()
    if group_col in ["Month", "Weekday"]:
        if group_col == "Weekday":
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            grouped = grouped.set_index("Weekday").reindex(days).dropna().reset_index()
        fig = px.bar(grouped, x=group_col, y=metric_col, title=f"{metric_col} by {group_col.lower()}")
    else:
        grouped = grouped.sort_values(metric_col, ascending=False).head(top_n)
        fig = px.bar(grouped, x=metric_col, y=group_col, orientation="h",
                     title=f"{metric_col} by {group_col.lower()} (top {top_n})")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(style(fig, 420), width="stretch")

    st.subheader("Order statistics")
    orders = df.groupby("InvoiceNo").agg(OrderValue=("Sales", "sum"), Units=("Quantity", "sum"),
                                         Discount=("Discount", "mean"))
    stats = orders.describe()
    stats.loc["CV"] = stats.loc["std"] / stats.loc["mean"]
    stats.loc["skewness"] = orders.skew()

    col1, col2 = st.columns(2)
    col1.dataframe(stats.round(2), width="stretch")
    discount_bands = pd.cut(df["Discount"], [0, 0.3, 0.35, 0.4, 0.45, 1], include_lowest=True,
                            labels=["<30%", "30-35%", "35-40%", "40-45%", ">45%"])
    discount_sales = df.groupby(discount_bands, observed=True)["Sales"].sum().reset_index()
    fig = px.bar(discount_sales, x="Discount", y="Sales", title="Sales by discount level")
    col2.plotly_chart(style(fig, 360), width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        corr = monthly[["Sales", "Quantity", "Orders", "Customers"]].corr()
        fig, ax = plt.subplots(figsize=(5, 3.6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="Purples", vmin=-1, vmax=1, ax=ax)
        ax.set_title("Correlation of monthly KPIs")
        st.pyplot(fig)
    with col2:
        top_titles = df.groupby("Description")[["Sales", "Quantity"]].sum() \
                       .sort_values("Sales", ascending=False).head(10).reset_index()
        st.markdown("**Top 10 titles**")
        st.dataframe(top_titles.rename(columns={"Description": "Title"}).round(0),
                     hide_index=True, width="stretch")

    st.info(f"Highest sales are in {best_month['Month']:%B} ({tl(best_month['Sales'])}) and lowest in "
            f"{worst_month['Month']:%B} ({tl(worst_month['Sales'])}). {city_sales['City'].iloc[0]} brings "
            f"{top_city_share:.0f}% of sales and {category_sales['Category'].iloc[0]} is the best-selling category.")


# =============================================================================
# 9. MARKETING TAB
# =============================================================================
with tab_marketing:
    st.subheader("Customer segments (RFM)")
    st.caption("Recency = days since last order, Frequency = number of orders, Monetary = total sales. "
               "Each is scored from 1 to 5.")
    segments = rfm.groupby("Segment").agg(Customers=("CustomerID", "count"),
                                          Sales=("Monetary", "sum")).reset_index()
    segments["Sales share %"] = segments["Sales"] / segments["Sales"].sum() * 100

    col1, col2 = st.columns(2)
    fig = px.pie(segments, names="Segment", values="Customers", hole=0.4,
                 title="Customers by segment", color_discrete_sequence=COLORS)
    col1.plotly_chart(style(fig), width="stretch")
    fig = px.bar(segments.sort_values("Sales"), x="Sales", y="Segment", orientation="h", title="Sales by segment")
    col2.plotly_chart(style(fig), width="stretch")
    st.dataframe(segments.round({"Sales": 0, "Sales share %": 1}), hide_index=True, width="stretch")

    st.subheader("Customer concentration")
    fig = px.line(customer_sales, x="Customer %", y="Cumulative %",
                  title="Cumulative share of sales (customers sorted by sales)")
    fig.add_hline(y=80, line_dash="dash", line_color="#9D6BD8")
    st.plotly_chart(style(fig, 340), width="stretch")

    st.subheader("At-risk customers to contact")
    st.dataframe(at_risk.head(15)[["CustomerID", "City", "Recency", "Frequency", "Monetary"]].round(0),
                 hide_index=True, width="stretch")

    st.info(f"The top 20% of customers make {top20_share:.0f}% of sales. {len(at_risk)} customers bought often "
            f"before but have not ordered recently ({tl(at_risk['Monetary'].sum())} past sales). "
            "They should be contacted first.")


# =============================================================================
# 10. SUPPLY CHAIN TAB
# =============================================================================
with tab_supply:
    st.subheader("ABC - XYZ analysis of titles")
    st.caption("ABC: A = first 80% of sales, B = next 15%, C = last 5%.  "
               "XYZ (CV of monthly demand): X < 0.5 stable, Y 0.5-1.0 variable, Z > 1.0 irregular.")

    col1, col2 = st.columns(2)
    abc_summary = products.groupby("ABC").agg(Titles=("StockCode", "count")).reset_index()
    fig = px.bar(abc_summary, x="ABC", y="Titles", text="Titles", title="Number of titles in each ABC class")
    col1.plotly_chart(style(fig), width="stretch")
    with col2:
        matrix = pd.crosstab(products["ABC"], products["XYZ"])
        fig, ax = plt.subplots(figsize=(5, 3.6))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Purples", ax=ax)
        ax.set_title("ABC - XYZ matrix (number of titles)")
        st.pyplot(fig)

    st.markdown("**Top A-class titles**")
    st.dataframe(products[products["ABC"] == "A"].head(15)[["Title", "Category", "Sales", "Quantity", "CV", "XYZ"]]
                 .round({"Sales": 0, "CV": 2}), hide_index=True, width="stretch")

    st.subheader("Order size and outliers")
    order_units = df.groupby("InvoiceNo")["Quantity"].sum()
    outliers, lower_bound, upper_bound = outliers_bp(order_units)
    city_units = df.groupby("City")["Quantity"].sum().sort_values(ascending=False).head(10).reset_index()

    col1, col2 = st.columns(2)
    fig = px.box(order_units.reset_index(), y="Quantity", log_y=True, title="Units per order (log scale)")
    col1.plotly_chart(style(fig), width="stretch")
    fig = px.bar(city_units, x="City", y="Quantity", title="Units shipped to the top 10 cities")
    col2.plotly_chart(style(fig), width="stretch")

    st.info(f"Only {a_titles:,} of {len(products):,} titles ({a_share:.1f}%) are in class A and make 80% of sales. "
            f"{len(outliers):,} orders are larger than {upper_bound:.0f} units (outliers); these big orders "
            "should be planned with the customer in advance.")


# =============================================================================
# 11. FORECAST TAB
# =============================================================================
with tab_forecast:
    st.subheader("Weekly forecast")
    col1, col2 = st.columns(2)
    target = col1.selectbox("Forecast", ["Sales", "Quantity"])
    weeks_ahead = col2.slider("Weeks to forecast", 4, 12, 8)

    result = forecast_analysis(df, target, weeks_ahead)
    if result is None:
        st.warning("Not enough weeks for a forecast. Please select a longer date range.")
    else:
        perf, best, chart, forecast_table = result
        fig = px.line(chart, x="Week", y="Value", color="Line", markers=True, title=f"Weekly {target.lower()}",
                      color_discrete_sequence=["#3E1C5E", "#B794F4", "#E0B0FF", "#D6BCFA", "#A855F7"])
        fig.update_layout(legend=dict(orientation="h", y=-0.2), legend_title_text="")
        st.plotly_chart(style(fig, 460), width="stretch")

        col1, col2 = st.columns(2)
        col1.markdown("**Model performance on the last 8 weeks**")
        col1.dataframe(perf.round(2), width="stretch")
        col2.markdown("**Forecast**")
        col2.dataframe(forecast_table, hide_index=True, width="stretch")

        st.info(f"The best model is {best} (lowest MAPE: {perf.loc['MAPE (%)', best]:.1f}%). "
                f"Expected total {target.lower()} for the next {weeks_ahead} weeks: "
                f"{forecast_table['Forecast'].sum():,.0f}. Only one year of data is available, "
                "so the forecast is for short-term planning.")


# =============================================================================
# 12. DATA TAB
# =============================================================================
with tab_data:
    st.subheader("Data preprocessing")
    missing_table = missing_values[missing_values > 0].rename("Missing values").reset_index()
    missing_table.columns = ["Column", "Missing values"]

    col1, col2 = st.columns(2)
    col1.markdown("**Cleaning steps**")
    col1.dataframe(prep_steps, hide_index=True, width="stretch")
    col2.markdown("**Missing values in the raw data**")
    col2.dataframe(missing_table, hide_index=True, width="stretch")
    st.markdown("Missing City and Category values were filled as Unknown / Uncategorized. "
                "New columns: Sales = Quantity x NetPrice, Discount = 1 - NetPrice / UnitPrice, "
                "Month, Week and Weekday.")

    st.download_button("Download results (Excel)",
                       data=to_excel({"Monthly": monthly, "RFM": rfm, "ABC_XYZ": products.head(5000)}),
                       file_name="dashboard_results.xlsx")
    st.markdown("**First 100 rows of the cleaned data**")
    st.dataframe(df.head(100), hide_index=True, width="stretch")
