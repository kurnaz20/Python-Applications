"""
B2B Book Distributor: Intelligent Decision Support Dashboard
ISL467E Management Engineering Applications with Python - Homework I

Departments served: Sales, Marketing, Supply Chain Management
Run locally:  streamlit run streamlit_app.py
"""
import itertools
import warnings
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

# =============================================================================
# 1. PAGE SETUP
# =============================================================================
st.set_page_config(page_title="B2B Book Distributor | Decision Support", page_icon="📘", layout="wide")

APP_DIR = Path(__file__).parent
# The data file may be in the repository root or in a "data" folder
_CANDIDATES = [APP_DIR / "data" / "b2b_transactions.parquet", APP_DIR / "b2b_transactions.parquet"]
DATA_FILE = next((p for p in _CANDIDATES if p.exists()), None)
if DATA_FILE is None:
    st.error("Data file `b2b_transactions.parquet` was not found. Upload it to the repository root or to a "
             "folder named `data` next to streamlit_app.py.")
    st.stop()

NAVY, ORANGE, GREEN, RED, GREY = "#16365C", "#F28C28", "#2A9D8F", "#D1495B", "#8D99AE"
PALETTE = [NAVY, ORANGE, GREEN, RED, "#6D597A", "#E9C46A", GREY, "#264653"]
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = PALETTE

st.markdown("""
<style>
.block-container {padding-top: 1.4rem;}
div[data-testid="stMetric"] {background:#F5F7FA; border:1px solid #E1E6EE; border-radius:10px; padding:10px 14px;}
.note  {background:#FFF6EC; border-left:4px solid #F28C28; padding:9px 13px; border-radius:6px; margin:6px 0 10px 0;}
.do    {background:#EAF6F4; border-left:4px solid #2A9D8F; padding:9px 13px; border-radius:6px; margin:6px 0 10px 0;}
.small {color:#6B7280; font-size:0.85rem;}
</style>
""", unsafe_allow_html=True)


def money(x: float) -> str:
    """Turkish lira with K / M suffix."""
    if abs(x) >= 1e6:
        return f"₺{x / 1e6:,.2f}M"
    if abs(x) >= 1e3:
        return f"₺{x / 1e3:,.1f}K"
    return f"₺{x:,.0f}"


def num(x: float) -> str:
    return f"{x / 1e6:,.2f}M" if abs(x) >= 1e6 else f"{x:,.0f}"


def insight(text: str):
    st.markdown(f'<div class="note">💡 <b>Insight:</b> {text}</div>', unsafe_allow_html=True)


def action(text: str):
    st.markdown(f'<div class="do">✅ <b>Action:</b> {text}</div>', unsafe_allow_html=True)


def show(fig, height=380):
    fig.update_layout(height=height, margin=dict(t=70, b=10, l=10, r=10), title_font_size=15)
    if fig.layout.legend.orientation == "h" and (fig.layout.legend.y or 0) > 1:
        fig.update_layout(legend=dict(y=1.02, x=1, xanchor="right", yanchor="bottom"))
    st.plotly_chart(fig, width="stretch")


# =============================================================================
# 2. DATA LOADING AND PREPROCESSING
# =============================================================================
@st.cache_data(show_spinner="Loading and cleaning data...")
def load_and_clean() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    raw = pd.read_parquet(DATA_FILE)
    missing = raw.isna().sum()
    steps = [("Raw transaction lines", len(raw), "Original file B2B_Transaction_Data_.xlsx")]
    df = raw.copy()

    # 2.1 Data types
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    for col in ["Quantity", "NetPrice", "UnitPrice"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2.2 Missing values
    n = len(df)
    df = df.dropna(subset=["InvoiceDate", "Quantity", "NetPrice"])
    steps.append(("Removed: missing date / quantity / price", n - len(df), "Cannot be used in any calculation"))
    steps.append(("Filled: missing City → 'UNKNOWN'", int(df["City"].isna().sum()), "Kept for sales totals"))
    steps.append(("Filled: missing Category → 'Uncategorized'", int(df["Category"].isna().sum()),
                  "Kept for sales totals"))
    df["City"] = df["City"].fillna("UNKNOWN").str.strip().str.upper()
    df["Category"] = df["Category"].fillna("Uncategorized").str.strip()
    df["Description"] = df["Description"].fillna("").str.strip()

    # 2.3 Invalid lines (zero / negative quantity or price = price corrections, not sales)
    n = len(df)
    df = df[(df["Quantity"] > 0) & (df["NetPrice"] > 0) & (df["UnitPrice"] > 0.01)]
    steps.append(("Removed: zero/negative quantity or price", n - len(df), "Price corrections, not real sales"))

    # 2.4 Net price above list price is a data entry error
    n = len(df)
    df = df[df["NetPrice"] <= df["UnitPrice"] * 1.01]
    steps.append(("Removed: net price above list price", n - len(df), "Inconsistent price entry"))

    # 2.5 Derived variables
    df["Revenue"] = df["Quantity"] * df["NetPrice"]
    df["ListValue"] = df["Quantity"] * df["UnitPrice"]
    df["Discount"] = 1 - df["NetPrice"] / df["UnitPrice"]
    df["Series"] = df["InvoiceNo"].str[:3]                      # invoice series prefix
    df["Month"] = df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
    df["Week"] = df["InvoiceDate"].dt.to_period("W-SUN").dt.start_time
    df["Weekday"] = df["InvoiceDate"].dt.day_name()

    # 2.6 Outliers (IQR rule on line quantity) - flagged, not removed: bulk B2B orders are real
    q1, q3 = df["Quantity"].quantile([0.25, 0.75])
    limit = q3 + 1.5 * (q3 - q1)
    df["BulkLine"] = df["Quantity"] > limit
    steps.append((f"Flagged: bulk lines (quantity > {limit:.0f}, IQR rule)", int(df["BulkLine"].sum()),
                  "Kept; can be excluded with the sidebar filter"))

    # 2.7 Memory optimisation for Streamlit Cloud
    for col in ["InvoiceNo", "CustomerID", "StockCode", "Description", "Category", "City", "Series", "Weekday"]:
        df[col] = df[col].astype("category")
    df["Quantity"] = df["Quantity"].astype("int32")
    for col in ["NetPrice", "UnitPrice", "Revenue", "ListValue", "Discount"]:
        df[col] = df[col].astype("float32")

    steps.append(("Clean transaction lines", len(df), "Used in all analyses"))
    report = pd.DataFrame(steps, columns=["Step", "Lines", "Reason"])
    return df.reset_index(drop=True), report, missing


data, quality_report, raw_missing = load_and_clean()

# =============================================================================
# 3. SIDEBAR FILTERS
# =============================================================================
st.sidebar.header("Filters")
d_min, d_max = data["InvoiceDate"].min().date(), data["InvoiceDate"].max().date()
period = st.sidebar.date_input("Period", (d_min, d_max), min_value=d_min, max_value=d_max)
if not isinstance(period, (tuple, list)) or len(period) != 2:
    st.sidebar.info("Select a start and end date.")
    st.stop()

city_order = data.groupby("City", observed=True)["Revenue"].sum().sort_values(ascending=False).index.tolist()
cat_order = data.groupby("Category", observed=True)["Revenue"].sum().sort_values(ascending=False).index.tolist()
sel_city = st.sidebar.multiselect("City", city_order, placeholder="All cities")
sel_cat = st.sidebar.multiselect("Category", cat_order, placeholder="All categories")
sel_series = st.sidebar.multiselect("Invoice series", sorted(data["Series"].cat.categories), placeholder="All series")
keep_bulk = st.sidebar.checkbox("Include bulk order lines", True)


@st.cache_data(show_spinner=False)
def apply_filters(start, end, cities, cats, series, bulk) -> pd.DataFrame:
    d = data[(data["InvoiceDate"] >= pd.Timestamp(start)) & (data["InvoiceDate"] <= pd.Timestamp(end))]
    if cities:
        d = d[d["City"].isin(cities)]
    if cats:
        d = d[d["Category"].isin(cats)]
    if series:
        d = d[d["Series"].isin(series)]
    if not bulk:
        d = d[~d["BulkLine"]]
    return d


df = apply_filters(period[0], period[1], tuple(sel_city), tuple(sel_cat), tuple(sel_series), keep_bulk)
if len(df) < 100:
    st.warning("Too few transactions for the selected filters. Please widen the selection.")
    st.stop()
st.sidebar.markdown(f'<p class="small">{len(df):,} of {len(data):,} lines selected</p>', unsafe_allow_html=True)


# =============================================================================
# 4. ANALYTICAL FUNCTIONS
# =============================================================================
@st.cache_data(show_spinner=False)
def monthly_kpis(d: pd.DataFrame) -> pd.DataFrame:
    return d.groupby("Month").agg(Revenue=("Revenue", "sum"), Units=("Quantity", "sum"),
                                  Orders=("InvoiceNo", "nunique"),
                                  Customers=("CustomerID", "nunique")).reset_index()


def half_growth(d: pd.DataFrame, key: str) -> pd.DataFrame:
    """Revenue in H1 vs H2 of the selected period and growth %."""
    mid = d["InvoiceDate"].min() + (d["InvoiceDate"].max() - d["InvoiceDate"].min()) / 2
    g = d.assign(Half=np.where(d["InvoiceDate"] <= mid, "H1", "H2")) \
         .pivot_table(index=key, columns="Half", values="Revenue", aggfunc="sum", observed=True, fill_value=0)
    for h in ["H1", "H2"]:
        if h not in g:
            g[h] = 0.0
    g["Revenue"] = g["H1"] + g["H2"]
    g["Growth %"] = np.where(g["H1"] > 0, (g["H2"] / g["H1"] - 1) * 100, np.nan)
    return g.reset_index()


@st.cache_data(show_spinner="Segmenting customers...")
def customer_segments(d: pd.DataFrame):
    ref = d["InvoiceDate"].max() + pd.Timedelta(days=1)
    orders = d.groupby(["CustomerID", "InvoiceNo"], observed=True).agg(
        Date=("InvoiceDate", "min"), Revenue=("Revenue", "sum")).reset_index()
    rfm = orders.groupby("CustomerID", observed=True).agg(
        Recency=("Date", lambda s: (ref - s.max()).days), Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum"), First=("Date", "min"), Last=("Date", "max")).reset_index()
    city = d.groupby(["CustomerID", "City"], observed=True)["Revenue"].sum().reset_index() \
            .sort_values("Revenue").drop_duplicates("CustomerID", keep="last").set_index("CustomerID")["City"]
    rfm["City"] = rfm["CustomerID"].map(city).astype(str)

    # K-Means on log-transformed, standardised R-F-M; k chosen by silhouette score
    X = StandardScaler().fit_transform(np.log1p(rfm[["Recency", "Frequency", "Monetary"]]))
    scores = {}
    for k in range(3, 7):
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
        scores[k] = silhouette_score(X, labels)
    best_k = max(scores, key=scores.get)
    rfm["Cluster"] = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit_predict(X)

    # Name clusters from their profiles
    prof = rfm.groupby("Cluster").agg(R=("Recency", "median"), F=("Frequency", "median"), M=("Monetary", "median"))
    names, used = {}, set()
    key_acc = prof["M"].idxmax(); names[key_acc] = "Key Accounts"; used.add(key_acc)
    rest = prof.drop(index=list(used))
    dormant = rest["R"].idxmax(); names[dormant] = "Dormant"; used.add(dormant)
    rest = prof.drop(index=list(used)).sort_values("M", ascending=False)
    for c, label in zip(rest.index, ["Regular Buyers", "Occasional Buyers", "Small Accounts", "Low Activity"]):
        names[c] = label
    rfm["Segment"] = rfm["Cluster"].map(names)

    # Churn risk: silent for more than 2x the customer's own average gap between orders
    gaps = orders.sort_values("Date").groupby("CustomerID", observed=True)["Date"] \
                 .apply(lambda s: s.diff().dt.days.mean())
    rfm["AvgGap"] = rfm["CustomerID"].map(gaps)
    rfm["ChurnRisk"] = (rfm["Frequency"] >= 3) & (rfm["Recency"] > 2 * rfm["AvgGap"]) & (rfm["Recency"] > 30)
    return rfm, scores, best_k


@st.cache_data(show_spinner="Mining category associations...")
def basket_rules(d: pd.DataFrame, top_n: int = 40, min_support: float = 0.02) -> pd.DataFrame:
    top = d.groupby("Category", observed=True)["Revenue"].sum().nlargest(top_n).index
    baskets = d[d["Category"].isin(top)].groupby("InvoiceNo", observed=True)["Category"] \
        .apply(lambda s: sorted(set(s.astype(str))))
    baskets = baskets[baskets.str.len() >= 1]
    n = len(baskets)
    single, pair = {}, {}
    for items in baskets:
        for a in items:
            single[a] = single.get(a, 0) + 1
        for a, b in itertools.combinations(items, 2):
            pair[(a, b)] = pair.get((a, b), 0) + 1
    rows = []
    for (a, b), c in pair.items():
        sup = c / n
        if sup < min_support:
            continue
        for x, y in [(a, b), (b, a)]:
            conf = c / single[x]
            lift = conf / (single[y] / n)
            rows.append((x, y, sup * 100, conf * 100, lift))
    rules = pd.DataFrame(rows, columns=["If customer buys", "Recommend", "Support %", "Confidence %", "Lift"])
    return rules.sort_values("Lift", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def abc_xyz(d: pd.DataFrame, level: str) -> pd.DataFrame:
    g = d.groupby(level, observed=True).agg(Revenue=("Revenue", "sum"), Units=("Quantity", "sum")).reset_index()
    g = g.sort_values("Revenue", ascending=False)
    g["Cum %"] = g["Revenue"].cumsum() / g["Revenue"].sum() * 100
    g["ABC"] = np.select([g["Cum %"] <= 80, g["Cum %"] <= 95], ["A", "B"], "C")
    weekly = d.groupby([level, "Week"], observed=True)["Quantity"].sum().unstack(fill_value=0)
    g["CV"] = g[level].map(weekly.std(axis=1) / weekly.mean(axis=1))
    g["XYZ"] = np.select([g["CV"] <= 1.0, g["CV"] <= 2.0], ["X", "Y"], "Z")
    g["Class"] = g["ABC"] + g["XYZ"]
    return g


@st.cache_data(show_spinner=False)
def inventory_plan(d: pd.DataFrame, service: float, lead_weeks: float, top: int) -> pd.DataFrame:
    weeks = pd.Index(sorted(d["Week"].unique()))
    top_items = d.groupby("StockCode", observed=True)["Revenue"].sum().nlargest(top).index
    w = d[d["StockCode"].isin(top_items)].groupby(["StockCode", "Week"], observed=True)["Quantity"].sum() \
         .unstack(fill_value=0).reindex(columns=weeks, fill_value=0)
    recent = w.iloc[:, -12:]                                       # last 12 weeks reflect current demand
    z = norm.ppf(service)
    plan = pd.DataFrame({"StockCode": w.index, "Avg weekly demand": recent.mean(axis=1).values,
                         "Std weekly demand": recent.std(axis=1).values})
    plan["Safety stock"] = z * plan["Std weekly demand"] * np.sqrt(lead_weeks)
    plan["Reorder point"] = plan["Avg weekly demand"] * lead_weeks + plan["Safety stock"]
    info = d.drop_duplicates("StockCode").set_index("StockCode")[["Description", "Category"]]
    plan = plan.join(info, on="StockCode")
    return plan[["StockCode", "Description", "Category", "Avg weekly demand", "Std weekly demand",
                 "Safety stock", "Reorder point"]].round(0)


def weekly(d: pd.DataFrame, value: str) -> pd.Series:
    s = d.groupby("Week")[value].sum()
    days = d.groupby("Week")["InvoiceDate"].nunique()
    full = days.median()
    while len(s) > 2 and days[s.index[0]] < full - 1:     # drop incomplete first/last weeks
        s = s.iloc[1:]
    while len(s) > 2 and days[s.index[-1]] < full - 1:
        s = s.iloc[:-1]
    return s


def _lag_frame(values: np.ndarray) -> pd.DataFrame:
    f = pd.DataFrame({"y": values})
    for lag in (1, 2, 3, 4):
        f[f"lag{lag}"] = f["y"].shift(lag)
    f["ma4"] = f["y"].shift(1).rolling(4).mean()
    return f.dropna()


def _predict(name: str, hist: np.ndarray, steps: int) -> np.ndarray:
    if name == "Naive":
        return np.repeat(hist[-1], steps)
    if name == "Moving average (4 weeks)":
        return np.repeat(hist[-4:].mean(), steps)
    if name == "Holt (damped trend)":
        return ExponentialSmoothing(hist, trend="add", damped_trend=True,
                                    initialization_method="estimated").fit().forecast(steps)
    if name == "Random Forest (lags)":
        f = _lag_frame(hist)
        model = RandomForestRegressor(n_estimators=200, max_depth=4, random_state=42)
        model.fit(f.drop(columns="y"), f["y"])
        h, out = list(hist), []
        for _ in range(steps):
            row = pd.DataFrame([{"lag1": h[-1], "lag2": h[-2], "lag3": h[-3], "lag4": h[-4],
                                 "ma4": np.mean(h[-4:])}])
            p = float(model.predict(row)[0])
            out.append(p)
            h.append(p)
        return np.array(out)
    raise ValueError(name)


MODELS = ["Naive", "Moving average (4 weeks)", "Holt (damped trend)", "Random Forest (lags)"]


@st.cache_data(show_spinner="Training forecasting models...")
def run_forecast(s: pd.Series, horizon: int, holdout: int):
    train, test = s.values[:-holdout], s.values[-holdout:]
    preds, scores = {}, {}
    for m in MODELS:
        p = _predict(m, train, holdout)
        preds[m] = p
        e = test - p
        scores[m] = {"MAE": np.mean(np.abs(e)), "RMSE": np.sqrt(np.mean(e ** 2)),
                     "MAPE %": np.mean(np.abs(e) / test) * 100}
    table = pd.DataFrame(scores).T.sort_values("MAPE %")
    best = table.index[0]
    fc = _predict(best, s.values, horizon)
    err = test - preds[best]
    lo, hi = np.percentile(err, 10), np.percentile(err, 90)      # empirical 80% interval
    idx = pd.date_range(s.index[-1] + pd.Timedelta(weeks=1), periods=horizon, freq="W-MON")
    out = pd.DataFrame({"Week": idx, "Forecast": fc, "Low": np.maximum(fc + lo, 0), "High": fc + hi})
    return table, best, preds, s.index[-holdout:], out


def excel_file(sheets: dict) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for name, frame in sheets.items():
            frame.to_excel(w, sheet_name=name[:31], index=False)
    return buf.getvalue()


# =============================================================================
# 5. HEADER AND KPI ROW
# =============================================================================
st.title("📘 B2B Book Distributor — Decision Support Dashboard")
st.markdown(f'<p class="small">{period[0]:%d %b %Y} – {period[1]:%d %b %Y} · Sales, Marketing and '
            f'Supply Chain views · all figures react to the sidebar filters</p>', unsafe_allow_html=True)

mk = monthly_kpis(df)
revenue, units = float(df["Revenue"].sum()), int(df["Quantity"].sum())
orders, customers = df["InvoiceNo"].nunique(), df["CustomerID"].nunique()
discount = 1 - revenue / float(df["ListValue"].sum())


def mom(col):
    if len(mk) < 2 or mk[col].iloc[-2] == 0:
        return None
    return f"{(mk[col].iloc[-1] / mk[col].iloc[-2] - 1) * 100:+.1f}% MoM"


c = st.columns(6)
c[0].metric("Net revenue", money(revenue), mom("Revenue"))
c[1].metric("Orders", f"{orders:,}", mom("Orders"))
c[2].metric("Units shipped", num(units), mom("Units"))
c[3].metric("Active customers", f"{customers:,}", mom("Customers"))
c[4].metric("Avg. order value", money(revenue / orders))
c[5].metric("Avg. discount off list", f"{discount * 100:.1f}%")

t_over, t_sales, t_mkt, t_sc, t_fc, t_data = st.tabs(
    ["Overview", "Sales", "Marketing", "Supply Chain", "Forecasting", "Data & Methods"])

# =============================================================================
# 6. OVERVIEW
# =============================================================================
with t_over:
    left, right = st.columns([3, 2])
    with left:
        fig = go.Figure()
        fig.add_bar(x=mk["Month"], y=mk["Revenue"], name="Net revenue", marker_color=NAVY)
        fig.add_scatter(x=mk["Month"], y=mk["Revenue"].rolling(3, min_periods=1).mean(), name="3-month average",
                        line=dict(color=ORANGE, width=3))
        fig.update_layout(title="Monthly net revenue", yaxis_title="₺", legend=dict(orientation="h", y=1.12))
        show(fig, 400)
    with right:
        seg_share = df.groupby("Series", observed=True)["Revenue"].sum().reset_index()
        fig = px.pie(seg_share, names="Series", values="Revenue", hole=0.55, title="Revenue by invoice series",
                     color_discrete_sequence=PALETTE)
        fig.update_traces(textinfo="percent+label")
        show(fig, 400)

    st.subheader("Executive summary")
    rfm, sil_scores, k_best = customer_segments(df)
    cust_rev = rfm.sort_values("Monetary", ascending=False)
    top20 = cust_rev["Monetary"].head(max(1, len(cust_rev) // 5)).sum() / cust_rev["Monetary"].sum() * 100
    city_rev = df.groupby("City", observed=True)["Revenue"].sum().sort_values(ascending=False)
    churn = rfm[rfm["ChurnRisk"]]
    abc_p = abc_xyz(df, "StockCode")
    a_items = (abc_p["ABC"] == "A").sum()

    col1, col2 = st.columns(2)
    with col1:
        if len(mk) >= 2:
            pk, lw = mk.loc[mk["Revenue"].idxmax()], mk.loc[mk["Revenue"].idxmin()]
            insight(f"Peak month is <b>{pk['Month']:%B}</b> ({money(pk['Revenue'])}); weakest is "
                    f"<b>{lw['Month']:%B}</b> ({money(lw['Revenue'])}). The peak is "
                    f"{pk['Revenue'] / lw['Revenue']:.1f}× the low month, so capacity and stock must be planned "
                    "for strong seasonality.")
        insight(f"<b>{city_rev.index[0].title()}</b> accounts for {city_rev.iloc[0] / revenue * 100:.0f}% of revenue; "
                f"the top 5 cities for {city_rev.head(5).sum() / revenue * 100:.0f}%.")
    with col2:
        insight(f"The top 20% of customers generate <b>{top20:.0f}%</b> of revenue. "
                f"<b>{len(churn)}</b> regular customers show churn signals, with "
                f"{money(churn['Monetary'].sum())} past revenue at stake.")
        insight(f"Only <b>{a_items:,}</b> of {len(abc_p):,} titles ({a_items / len(abc_p) * 100:.1f}%) make up "
                "80% of revenue; these deserve tight stock control.")
    action("Sales: prioritise key cities and peak months · Marketing: contact churn-risk accounts and use the "
           "cross-sell rules · Supply chain: apply the reorder points for A titles before the peak.")

# =============================================================================
# 7. SALES
# =============================================================================
with t_sales:
    st.subheader("Revenue trend")
    gran = st.radio("Granularity", ["Daily", "Weekly", "Monthly"], index=1, horizontal=True)
    key = {"Daily": "InvoiceDate", "Weekly": "Week", "Monthly": "Month"}[gran]
    ts = df.groupby(key)["Revenue"].sum()
    fig = px.line(x=ts.index, y=ts.values, markers=gran != "Daily", title=f"{gran} net revenue")
    fig.update_traces(line_color=NAVY)
    if gran == "Daily":
        z = (ts - ts.mean()) / ts.std()
        peaks = ts[z > 3]
        fig.add_scatter(x=peaks.index, y=peaks.values, mode="markers", name="Unusual day (z > 3)",
                        marker=dict(color=RED, size=10))
    fig.update_layout(xaxis_title=None, yaxis_title="₺")
    show(fig)

    hm = df.groupby(["Weekday", "Month"], observed=True)["Revenue"].sum().unstack(fill_value=0)
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    hm = hm.reindex([d for d in days_order if d in hm.index])
    hm.columns = [c.strftime("%b") for c in hm.columns]
    fig = px.imshow(hm, aspect="auto", color_continuous_scale="Blues", title="Revenue by weekday and month")
    show(fig, 320)

    st.subheader("Cities and categories")
    left, right = st.columns(2)
    with left:
        cg = half_growth(df, "City").sort_values("Revenue", ascending=False).head(15)
        cg["City"] = cg["City"].astype(str).str.title()
        fig = px.bar(cg.sort_values("Revenue"), x="Revenue", y="City", orientation="h", color="Growth %",
                     color_continuous_scale="RdYlGn", range_color=(-50, 50),
                     title="Top 15 cities (color = H2 vs H1 growth %)")
        fig.update_layout(yaxis_title=None, xaxis_title="₺")
        show(fig, 460)
    with right:
        kg = half_growth(df, "Category").sort_values("Revenue", ascending=False).head(15)
        kg["Category"] = kg["Category"].astype(str)
        kg["Label"] = kg["Category"].map(lambda t: t if len(t) <= 30 else t[:28] + "…")
        fig = px.bar(kg.sort_values("Revenue"), x="Revenue", y="Label", orientation="h", color="Growth %",
                     color_continuous_scale="RdYlGn", range_color=(-50, 50),
                     title="Top 15 categories (color = H2 vs H1 growth %)")
        fig.update_layout(yaxis_title=None, xaxis_title="₺")
        show(fig, 460)
    grow = kg.dropna(subset=["Growth %"]).sort_values("Growth %")
    if len(grow) >= 2:
        insight(f"Among the top categories, <b>{grow.iloc[-1]['Category']}</b> grew fastest "
                f"({grow.iloc[-1]['Growth %']:+.0f}% H2 vs H1) and <b>{grow.iloc[0]['Category']}</b> declined most "
                f"({grow.iloc[0]['Growth %']:+.0f}%).")

    st.subheader("Pricing and discounts")
    left, right = st.columns(2)
    with left:
        bands = pd.cut(df["Discount"], [-0.001, 0.30, 0.35, 0.40, 0.45, 1.0],
                       labels=["≤30%", "30–35%", "35–40%", "40–45%", ">45%"])
        db = df.groupby(bands, observed=True)["Revenue"].sum().reset_index()
        db.columns = ["Discount band", "Revenue"]
        fig = px.bar(db, x="Discount band", y="Revenue", text_auto=".2s", title="Revenue by discount band",
                     color_discrete_sequence=[NAVY])
        show(fig, 340)
    with right:
        cust_disc = df.groupby("CustomerID", observed=True).agg(Revenue=("Revenue", "sum"),
                                                                ListValue=("ListValue", "sum"))
        cust_disc["Discount %"] = (1 - cust_disc["Revenue"] / cust_disc["ListValue"]) * 100
        fig = px.scatter(cust_disc.reset_index(), x="Revenue", y="Discount %", log_x=True, opacity=0.6,
                         hover_name="CustomerID", title="Customer revenue vs. average discount",
                         color_discrete_sequence=[ORANGE])
        show(fig, 340)
    corr = np.corrcoef(np.log1p(cust_disc["Revenue"]), cust_disc["Discount %"])[0, 1]
    insight(f"The correlation between customer size (log revenue) and discount is {corr:.2f}. "
            + ("Larger customers receive higher discounts, as expected in B2B pricing."
               if corr > 0.2 else "Discounts are only weakly tied to customer size — review the discount policy "
                                  "for small accounts with high discounts."))

    st.subheader("Best-selling titles")
    prod = df.groupby(["StockCode", "Description", "Category"], observed=True) \
             .agg(Revenue=("Revenue", "sum"), Units=("Quantity", "sum"), Customers=("CustomerID", "nunique")) \
             .reset_index().nlargest(15, "Revenue")
    st.dataframe(prod.style.format({"Revenue": "₺{:,.0f}", "Units": "{:,}"}), width="stretch", hide_index=True)

# =============================================================================
# 8. MARKETING
# =============================================================================
with t_mkt:
    st.subheader("Customer segmentation (RFM + K-Means)")
    st.markdown(f'<p class="small">Recency (days since last order), Frequency (orders) and Monetary (net revenue) '
                f'are log-transformed and standardised; K-Means with k = {k_best} was selected by the highest '
                f'silhouette score ({sil_scores[k_best]:.2f}).</p>', unsafe_allow_html=True)
    prof = rfm.groupby("Segment").agg(Customers=("CustomerID", "count"), Revenue=("Monetary", "sum"),
                                      Median_recency=("Recency", "median"), Median_orders=("Frequency", "median"),
                                      Median_revenue=("Monetary", "median")).reset_index()
    prof["Revenue share %"] = prof["Revenue"] / prof["Revenue"].sum() * 100
    prof = prof.sort_values("Revenue", ascending=False)

    left, right = st.columns(2)
    with left:
        fig = px.scatter(rfm, x="Recency", y="Frequency", size="Monetary", color="Segment", log_y=True,
                         hover_data={"CustomerID": True, "City": True, "Monetary": ":,.0f"}, size_max=40,
                         title="Customers by recency and frequency (bubble = revenue)")
        fig.update_layout(xaxis_title="Days since last order", yaxis_title="Orders (log scale)")
        show(fig, 420)
    with right:
        fig = go.Figure()
        fig.add_bar(x=prof["Segment"], y=prof["Customers"] / prof["Customers"].sum() * 100, name="% of customers",
                    marker_color=GREY)
        fig.add_bar(x=prof["Segment"], y=prof["Revenue share %"], name="% of revenue", marker_color=NAVY)
        fig.update_layout(barmode="group", title="Share of customers vs. share of revenue", yaxis_title="%",
                          legend=dict(orientation="h", y=1.12))
        show(fig, 420)
    st.dataframe(prof.style.format({"Revenue": "₺{:,.0f}", "Median_recency": "{:.0f} days",
                                    "Median_orders": "{:.0f}", "Median_revenue": "₺{:,.0f}",
                                    "Revenue share %": "{:.1f}%"}), width="stretch", hide_index=True)
    ka = prof[prof["Segment"] == "Key Accounts"]
    if len(ka):
        insight(f"<b>Key Accounts</b> are {int(ka['Customers'].iloc[0])} customers "
                f"({ka['Customers'].iloc[0] / len(rfm) * 100:.0f}%) producing "
                f"{ka['Revenue share %'].iloc[0]:.0f}% of revenue.")

    st.subheader("Churn early warning")
    st.markdown('<p class="small">A customer with at least 3 orders is flagged when the days since the last order '
                'exceed twice the customer\'s own average gap between orders.</p>', unsafe_allow_html=True)
    risk = churn.sort_values("Monetary", ascending=False)
    m1, m2, m3 = st.columns(3)
    m1.metric("Customers at risk", f"{len(risk):,}")
    m2.metric("Past revenue at risk", money(risk["Monetary"].sum()))
    m3.metric("Share of total revenue", f"{risk['Monetary'].sum() / revenue * 100:.1f}%")
    st.dataframe(risk.head(20)[["CustomerID", "City", "Segment", "Frequency", "AvgGap", "Recency", "Monetary"]]
                 .rename(columns={"AvgGap": "Avg gap (days)", "Recency": "Days silent", "Monetary": "Revenue"})
                 .style.format({"Avg gap (days)": "{:.0f}", "Revenue": "₺{:,.0f}"}),
                 width="stretch", hide_index=True)
    action("Call the listed accounts in order of past revenue and offer a re-order incentive.")

    st.subheader("Customer activity over time")
    first_month = rfm.set_index("CustomerID")["First"].dt.to_period("M").dt.to_timestamp()
    active = df.groupby("Month")["CustomerID"].nunique()
    new = first_month.value_counts().reindex(active.index, fill_value=0)
    fig = go.Figure()
    fig.add_bar(x=active.index, y=active - new, name="Returning customers", marker_color=NAVY)
    fig.add_bar(x=active.index, y=new, name="New customers", marker_color=ORANGE)
    fig.update_layout(barmode="stack", title="Active customers per month", legend=dict(orientation="h", y=1.12))
    show(fig, 340)

    st.subheader("Cross-selling opportunities (market basket analysis)")
    st.markdown('<p class="small">Association rules between the 40 largest categories at invoice level. '
                'Lift > 1 means the two categories are bought together more often than by chance.</p>',
                unsafe_allow_html=True)
    rules = basket_rules(df)
    if rules.empty:
        st.info("No frequent category pairs for this selection.")
    else:
        min_lift = st.slider("Minimum lift", 1.0, 3.0, 1.5, 0.1)
        shown = rules[rules["Lift"] >= min_lift].head(15)
        st.dataframe(shown.style.format({"Support %": "{:.1f}", "Confidence %": "{:.0f}", "Lift": "{:.2f}"}),
                     width="stretch", hide_index=True)
        if len(shown):
            r0 = shown.iloc[0]
            insight(f"Invoices with <b>{r0['If customer buys']}</b> contain <b>{r0['Recommend']}</b> in "
                    f"{r0['Confidence %']:.0f}% of cases — {r0['Lift']:.1f}× more often than average.")
            action("Bundle the strongest pairs in catalogues and suggest them to sales reps at order entry.")

# =============================================================================
# 9. SUPPLY CHAIN
# =============================================================================
with t_sc:
    st.subheader("ABC–XYZ inventory classification")
    st.markdown('<p class="small">ABC: cumulative revenue share (A ≤ 80%, B ≤ 95%, C rest). XYZ: coefficient of '
                'variation of weekly units (X ≤ 1.0 regular, Y ≤ 2.0 variable, Z > 2.0 sporadic; thresholds '
                'suited to weekly book demand).</p>', unsafe_allow_html=True)
    level = st.radio("Level", ["StockCode", "Category"], horizontal=True,
                     format_func=lambda x: "Title" if x == "StockCode" else "Category")
    ax = abc_p if level == "StockCode" else abc_xyz(df, "Category")
    count_m = pd.crosstab(ax["ABC"], ax["XYZ"]).reindex(index=list("ABC"), columns=list("XYZ"), fill_value=0)
    rev_m = ax.pivot_table(index="ABC", columns="XYZ", values="Revenue", aggfunc="sum", fill_value=0) \
              .reindex(index=list("ABC"), columns=list("XYZ"), fill_value=0)
    left, right = st.columns(2)
    with left:
        fig = px.imshow(count_m, color_continuous_scale="Blues", aspect="auto", title="Number of items per class")
        fig.update_traces(texttemplate="%{z:,}")
        show(fig, 330)
    with right:
        show(px.imshow((rev_m / rev_m.values.sum() * 100).round(1), text_auto=True, color_continuous_scale="Blues",
                       aspect="auto", title="Share of revenue per class (%)"), 330)
    policy = pd.DataFrame({
        "Class": ["AX", "AY", "AZ", "BX", "BY", "BZ", "CX", "CY", "CZ"],
        "Recommended policy": [
            "Continuous review, automatic replenishment, low safety stock",
            "Forecast-driven replenishment with seasonal safety stock",
            "Order close to demand; high safety stock or supplier agreement",
            "Periodic review, standard safety stock",
            "Periodic review with seasonal adjustment",
            "Order on demand, minimal stock",
            "Simple min–max, long review interval",
            "Min–max, consolidate orders",
            "Print-on-demand or delist; free warehouse space"]})
    policy["Items"] = policy["Class"].map(ax["Class"].value_counts()).fillna(0).astype(int)
    policy["Revenue share %"] = policy["Class"].map(ax.groupby("Class")["Revenue"].sum() / ax["Revenue"].sum() * 100) \
                                     .fillna(0).round(1)
    st.dataframe(policy, width="stretch", hide_index=True)

    st.subheader("Safety stock and reorder points")
    s1, s2, s3 = st.columns(3)
    service = s1.slider("Target service level", 0.80, 0.99, 0.95, 0.01)
    lead = s2.slider("Supplier lead time (weeks)", 0.5, 6.0, 2.0, 0.5)
    top_n = s3.slider("Number of top titles", 10, 100, 30, 10)
    plan = inventory_plan(df, service, lead, top_n)
    st.dataframe(plan.style.format({c: "{:,.0f}" for c in plan.columns[3:]}), width="stretch", hide_index=True,
                 height=360)
    st.markdown(f'<p class="small">Safety stock = z × σ<sub>weekly</sub> × √L (z = {norm.ppf(service):.2f}); '
                'Reorder point = average weekly demand × L + safety stock. Demand statistics use the last 12 '
                'weeks.</p>', unsafe_allow_html=True)
    action(f"Place a replenishment order when stock falls below the reorder point; at {service:.0%} service "
           f"the top {top_n} titles need {plan['Safety stock'].sum():,.0f} units of safety stock in total.")

    st.subheader("Slow-moving titles")
    last_sale = df.groupby("StockCode", observed=True)["InvoiceDate"].max()
    total_units = df.groupby("StockCode", observed=True)["Quantity"].sum()
    cutoff = df["InvoiceDate"].max() - pd.Timedelta(days=90)
    slow = pd.DataFrame({"Last sale": last_sale, "Units": total_units})
    slow = slow[(slow["Last sale"] < cutoff) & (slow["Units"] >= 20)].sort_values("Units", ascending=False)
    slow = slow.join(df.drop_duplicates("StockCode").set_index("StockCode")[["Description", "Category"]])
    st.metric("Titles with ≥ 20 units sold but no sale in the last 90 days", f"{len(slow):,}")
    st.dataframe(slow.head(15).reset_index()[["StockCode", "Description", "Category", "Units", "Last sale"]]
                 .assign(**{"Last sale": lambda t: t["Last sale"].dt.strftime("%d %b %Y")}),
                 width="stretch", hide_index=True)
    action("Check remaining stock of these titles; return to publisher or discount to free capacity.")

    st.subheader("Shipment load by city")
    load = df.groupby("City", observed=True).agg(Units=("Quantity", "sum"), Orders=("InvoiceNo", "nunique")) \
             .reset_index()
    load["Units per order"] = load["Units"] / load["Orders"]
    load = load.sort_values("Units", ascending=False).head(15)
    load["City"] = load["City"].astype(str).str.title()
    fig = px.scatter(load, x="Orders", y="Units per order", size="Units", text="City", log_x=True,
                     title="Top 15 cities: number of orders vs. average order size (bubble = units)",
                     color_discrete_sequence=[GREEN])
    fig.update_traces(textposition="top center")
    fig.update_layout(xaxis_title="Orders (log scale)")
    show(fig, 420)
    insight("Cities with many small orders are candidates for consolidated weekly shipments; cities with large "
            "average orders can be served by direct full-truck deliveries.")

# =============================================================================
# 10. FORECASTING
# =============================================================================
with t_fc:
    st.subheader("Weekly demand forecast")
    f1, f2, f3, f4 = st.columns(4)
    measure = f1.selectbox("Measure", ["Quantity", "Revenue"],
                           format_func=lambda x: "Units" if x == "Quantity" else "Net revenue (₺)")
    scope = f2.selectbox("Scope", ["All categories"] + cat_order[:15])
    horizon = f3.slider("Horizon (weeks)", 4, 16, 8)
    holdout = f4.slider("Test period (weeks)", 4, 12, 8)
    base = df if scope == "All categories" else df[df["Category"] == scope]
    series = weekly(base, measure)
    if len(series) < holdout + 12:
        st.info("Not enough weekly history for this selection.")
    else:
        table, best, preds, test_idx, fc = run_forecast(series, horizon, holdout)
        fig = go.Figure()
        fig.add_scatter(x=series.index, y=series.values, name="Actual", line=dict(color=NAVY, width=2))
        for (m, p), col in zip(preds.items(), [GREY, "#6D597A", GREEN, ORANGE]):
            fig.add_scatter(x=test_idx, y=p, name=f"{m} (test)", line=dict(color=col, dash="dot"))
        fig.add_scatter(x=fc["Week"], y=fc["High"], mode="lines", line=dict(width=0), showlegend=False,
                        hoverinfo="skip")
        fig.add_scatter(x=fc["Week"], y=fc["Low"], mode="lines", fill="tonexty", line=dict(width=0),
                        fillcolor="rgba(209,73,91,0.15)", name="80% interval")
        fig.add_scatter(x=fc["Week"], y=fc["Forecast"], mode="lines+markers", name=f"Forecast ({best})",
                        line=dict(color=RED, width=3))
        fig.update_layout(title=f"{scope}: weekly {'units' if measure == 'Quantity' else 'net revenue'}",
                          legend=dict(orientation="h", y=-0.18))
        show(fig, 460)

        left, right = st.columns(2)
        fmt = money if measure == "Revenue" else num
        with left:
            st.markdown("**Model accuracy on the test period**")
            st.dataframe(table.style.format({"MAE": "{:,.0f}", "RMSE": "{:,.0f}", "MAPE %": "{:.1f}"})
                         .highlight_min(subset=["MAPE %"], color="#D9F2EE"), width="stretch")
        with right:
            st.markdown("**Forecast**")
            st.dataframe(fc.assign(Week=fc["Week"].dt.strftime("%d %b %Y"), Forecast=fc["Forecast"].map(fmt),
                                   Low=fc["Low"].map(fmt), High=fc["High"].map(fmt)),
                         width="stretch", hide_index=True)
        naive_mape = table.loc["Naive", "MAPE %"]
        if best == "Naive":
            lead_txt = (f"No model beats the <b>naive benchmark</b> (MAPE {naive_mape:.1f}%), so the latest week "
                        "is the best short-term estimate; demand currently moves without a stable trend.")
        else:
            lead_txt = (f"<b>{best}</b> has the lowest test error (MAPE {table.loc[best, 'MAPE %']:.1f}% vs. "
                        f"{naive_mape:.1f}% for the naive benchmark).")
        insight(f"{lead_txt} Expected total for the next {horizon} weeks: <b>{fmt(fc['Forecast'].sum())}</b> "
                f"(80% range {fmt(fc['Low'].sum())} – {fmt(fc['High'].sum())}).")
        st.markdown('<p class="small">Models are trained on all weeks before the test period, evaluated on the test '
                    'weeks, and the best model is retrained on the full history. With one year of data, yearly '
                    'seasonality cannot be estimated; use the forecast for short-term planning.</p>',
                    unsafe_allow_html=True)

# =============================================================================
# 11. DATA & METHODS
# =============================================================================
with t_data:
    st.subheader("Data preprocessing")
    left, right = st.columns([3, 2])
    with left:
        st.dataframe(quality_report.style.format({"Lines": "{:,}"}), width="stretch", hide_index=True)
    with right:
        mv = raw_missing[raw_missing > 0].reset_index()
        mv.columns = ["Column", "Missing values"]
        show(px.bar(mv, x="Column", y="Missing values", text_auto=True, title="Missing values in the raw data",
                    color_discrete_sequence=[ORANGE]), 300)
    st.markdown("""
**Derived variables:** Revenue = Quantity × NetPrice · List value = Quantity × UnitPrice ·
Discount = 1 − NetPrice / UnitPrice · Invoice series = first 3 characters of InvoiceNo · Month, Week, Weekday.

| Module | Method |
|---|---|
| KPIs, trends | Aggregation, month-over-month change, 3-month moving average, H2 vs H1 growth |
| Anomalies | z-score of daily revenue (> 3) |
| Customer segmentation | RFM, log transform + standardisation, K-Means (k by silhouette score) |
| Churn warning | Days since last order > 2 × customer's average gap between orders |
| Cross-selling | Market basket analysis (support, confidence, lift) on category level |
| Inventory | ABC–XYZ classification, safety stock = z·σ·√L, reorder point |
| Forecasting | Naive, 4-week moving average, Holt's damped trend, Random Forest with lags; compared by MAE, RMSE, MAPE on a hold-out period |
""")
    st.subheader("Downloads")
    st.download_button("Download analysis tables (Excel)",
                       data=excel_file({"Monthly KPIs": mk, "Customer segments": rfm.drop(columns=["Cluster"]),
                                        "Churn risk": risk, "ABC-XYZ titles": abc_p.head(5000),
                                        "Reorder points": plan, "Cross-sell rules": rules}),
                       file_name="b2b_decision_support.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.markdown("**Sample of the cleaned data**")
    st.dataframe(df.head(100), width="stretch", hide_index=True)
