"""Streamlit demo app for the Retail Sales Forecasting project.

Run with:  streamlit run app.py
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA = "data"
PRIMARY = "#2563eb"
COLORS = {"Actual": "#111827", "Holt-Winters": "#2563eb", "SARIMA": "#f59e0b", "Random Forest": "#10b981"}

st.set_page_config(page_title="Retail Sales Forecasting", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
.hero {background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%); color: white;
       padding: 1.6rem 2rem; border-radius: 14px; margin-bottom: 1.5rem;}
.hero h1 {color: white; font-size: 2rem; margin: 0 0 .3rem 0; padding: 0;}
.hero p {color: #dbeafe; margin: 0; font-size: 1rem;}
.card {background: white; border: 1px solid #e5e7eb; border-radius: 12px;
       padding: 1rem 1.2rem; box-shadow: 0 1px 2px rgba(0,0,0,.04);}
.card .label {color: #6b7280; font-size: .8rem; text-transform: uppercase; letter-spacing: .05em;}
.card .value {color: #111827; font-size: 1.6rem; font-weight: 700; margin-top: .2rem;}
.card .sub {color: #6b7280; font-size: .8rem; margin-top: .1rem;}
.card.best {border: 2px solid #10b981;}
.note {background: #f1f5f9; border-left: 4px solid #2563eb; padding: .8rem 1rem;
       border-radius: 6px; color: #334155; font-size: .9rem;}
.stTabs [data-baseweb="tab"] {font-size: 1rem; font-weight: 600; padding: .6rem 1.2rem;}
</style>
""", unsafe_allow_html=True)


def card(label, value, sub="", best=False):
    cls = "card best" if best else "card"
    return f'<div class="{cls}"><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div></div>'


def style_fig(fig, height=380):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=1.08, x=0), font=dict(size=13),
                      hoverlabel=dict(bgcolor="white"))
    return fig


# ---------- Data (same cleaning as the notebook) ----------
@st.cache_data(show_spinner="Loading data...")
def load_data():
    train = pd.read_csv(f"{DATA}/train.csv", parse_dates=["date"])
    train = train.drop_duplicates()
    family = train.groupby("family")["sales"].sum().sort_values(ascending=False)

    daily = train.groupby("date").agg(sales=("sales", "sum"), onpromotion=("onpromotion", "sum"))
    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_range).interpolate()

    oil = pd.read_csv(f"{DATA}/oil.csv", parse_dates=["date"]).set_index("date").reindex(full_range)
    daily["oil"] = oil["dcoilwtico"].interpolate().bfill().values
    daily.index.name = "date"
    return daily, family, train["store_nbr"].nunique()


# ---------- Models (same as the notebook) ----------
def holt_winters(train_s, n):
    model = ExponentialSmoothing(train_s, trend="add", seasonal="add", seasonal_periods=7).fit()
    return model.forecast(n).values


def sarima(train_s, n):
    model = SARIMAX(train_s[-365:], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)).fit(disp=False)
    return model.forecast(n).values


def random_forest(series, n):
    df = series.to_frame("sales")
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["day"] = df.index.day
    df["lag_7"] = df["sales"].shift(7)
    df["rolling_mean_7"] = df["sales"].shift(1).rolling(7).mean()
    df = df.dropna()
    features = ["dayofweek", "month", "day", "lag_7", "rolling_mean_7"]
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(df[features][:-n], df["sales"][:-n])
    return rf.predict(df[features][-n:])


MODELS = {"Holt-Winters": holt_winters, "SARIMA": sarima, "Random Forest": random_forest}
MODEL_INFO = {
    "Holt-Winters": "Exponential smoothing with additive trend and weekly seasonality (period 7).",
    "SARIMA": "SARIMA (1,1,1)x(1,1,1,7), trained on the last 365 days for speed.",
    "Random Forest": "200 trees on day of week, month, day, lag_7 and rolling_mean_7.",
}


@st.cache_data(show_spinner=False)
def run_model(name, series, n):
    if name == "Random Forest":
        return MODELS[name](series, n)
    return MODELS[name](series[:-n], n)


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### Retail Sales Forecasting")
    st.markdown("College ML project on the Kaggle "
                "[Store Sales](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) dataset "
                "(Corporacion Favorita, Ecuador).")
    st.divider()
    st.markdown("**Pipeline**")
    st.markdown("1. Load and clean data\n2. Aggregate to daily total sales\n3. Exploratory analysis\n"
                "4. Train 3 forecasting models\n5. Compare with MAE and RMSE")
    st.divider()
    st.markdown("**Models**")
    for name, info in MODEL_INFO.items():
        st.markdown(f"<span style='color:{COLORS[name]};font-weight:700'>{name}</span><br>"
                    f"<span style='font-size:.85rem;color:#6b7280'>{info}</span>", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown('<div class="hero"><h1>Retail Sales Analysis and Forecasting</h1>'
            '<p>Explore five years of daily store sales and compare three time series forecasting models.</p></div>',
            unsafe_allow_html=True)

try:
    daily, family, n_stores = load_data()
except FileNotFoundError:
    st.error("Data not found. Put train.csv and oil.csv in the data/ folder.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Overview", "Exploratory Analysis", "Forecast"])

# ---------- Overview ----------
with tab1:
    c = st.columns(4)
    c[0].markdown(card("Date range", f"{len(daily):,} days",
                       f"{daily.index.min():%b %Y} to {daily.index.max():%b %Y}"), unsafe_allow_html=True)
    c[1].markdown(card("Total sales", f"{daily['sales'].sum() / 1e6:,.1f} M", "units sold"), unsafe_allow_html=True)
    c[2].markdown(card("Average per day", f"{daily['sales'].mean() / 1e3:,.0f} K", "units sold"), unsafe_allow_html=True)
    c[3].markdown(card("Stores / families", f"{n_stores} / {len(family)}", "stores and product families"),
                  unsafe_allow_html=True)
    st.write("")

    view = st.radio("View", ["Daily", "7-day average", "30-day average"], horizontal=True)
    window = {"Daily": 1, "7-day average": 7, "30-day average": 30}[view]
    trend = daily["sales"].rolling(window).mean()
    fig = go.Figure(go.Scatter(x=trend.index, y=trend, line=dict(color=PRIMARY, width=1.5),
                               fill="tozeroy", fillcolor="rgba(37,99,235,0.08)", name="Sales"))
    fig.update_layout(title="Total daily sales", xaxis=dict(rangeslider=dict(visible=True)))
    st.plotly_chart(style_fig(fig, 450), use_container_width=True)

# ---------- EDA ----------
with tab2:
    col1, col2 = st.columns(2)
    with col1:
        monthly = daily["sales"].resample("ME").sum()
        fig = px.bar(x=monthly.index, y=monthly.values, labels={"x": "Month", "y": "Sales"},
                     title="Monthly total sales", color_discrete_sequence=[PRIMARY])
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with col2:
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow = daily.groupby(daily.index.day_name())["sales"].mean().reindex(days)
        fig = px.bar(x=dow.index, y=dow.values, labels={"x": "", "y": "Average sales"},
                     title="Average sales by day of week",
                     color=["Weekend" if d in ("Saturday", "Sunday") else "Weekday" for d in days],
                     color_discrete_map={"Weekday": "#93c5fd", "Weekend": PRIMARY})
        st.plotly_chart(style_fig(fig), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        top = family.head(10)[::-1]
        fig = px.bar(x=top.values, y=top.index, orientation="h", labels={"x": "Total sales", "y": ""},
                     title="Top 10 product families", color_discrete_sequence=["#10b981"])
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with col4:
        fig = px.scatter(daily, x="onpromotion", y="sales", opacity=0.4, trendline="ols",
                         labels={"onpromotion": "Items on promotion", "sales": "Sales"},
                         title=f"Promotion vs sales (corr = {daily['onpromotion'].corr(daily['sales']):.2f})",
                         color_discrete_sequence=["#8b5cf6"])
        st.plotly_chart(style_fig(fig), use_container_width=True)

    col5, col6 = st.columns(2)
    with col5:
        fig = px.scatter(daily, x="oil", y="sales", opacity=0.4, trendline="ols",
                         labels={"oil": "Oil price (USD)", "sales": "Sales"},
                         title=f"Oil price vs sales (corr = {daily['oil'].corr(daily['sales']):.2f})",
                         color_discrete_sequence=["#f59e0b"])
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with col6:
        st.markdown("#### Key findings")
        st.markdown(
            "- Sales show a steady **upward trend** from 2013 to 2017.\n"
            "- There is a strong **weekly pattern**: weekends are the busiest days.\n"
            "- **Grocery, beverages and produce** make up most of the sales.\n"
            "- More items on **promotion** go with higher sales.\n"
            "- **Oil price** is negatively related to sales. Prices fell after 2014 while sales kept growing.\n"
            "- Sales drop to almost zero on **1 January**, when most stores are closed.")

# ---------- Forecast ----------
with tab3:
    st.markdown("#### What are we predicting?")
    st.markdown("The **total number of units sold per day across all 54 stores and all product families** "
                "of the Favorita chain. The models learn from past daily sales and forecast the next N days. "
                "We hide the last N days, predict them, and compare the predictions with the real sales "
                "to see which model is most accurate.")
    c1, c2, c3 = st.columns([1, 2, 1])
    n_days = c1.slider("Test period (days)", 7, 60, 30)
    chosen = c2.multiselect("Models", list(MODELS), default=list(MODELS))
    c3.write("")
    c3.write("")
    run = c3.button("Run forecast", type="primary", use_container_width=True)

    if run and chosen:
        series = daily["sales"]
        test_s = series[-n_days:]
        preds, rows = {}, []
        for name in chosen:
            with st.spinner(f"Training {name}..."):
                preds[name] = run_model(name, series, n_days)
            rows.append({"Model": name,
                         "MAE": mean_absolute_error(test_s, preds[name]),
                         "RMSE": np.sqrt(mean_squared_error(test_s, preds[name]))})
        results = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
        best = results.iloc[0]["Model"]

        cols = st.columns(len(results))
        for col, (_, r) in zip(cols, results.iterrows()):
            sub = f"MAE {r['MAE']:,.0f}" + ("  |  best model" if r["Model"] == best else "")
            col.markdown(card(r["Model"], f"RMSE {r['RMSE']:,.0f}", sub, best=r["Model"] == best),
                         unsafe_allow_html=True)
        st.write("")

        history = series[-(n_days + 60):-n_days]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history.index, y=history, name="History",
                                 line=dict(color="#9ca3af", width=1.5)))
        fig.add_trace(go.Scatter(x=test_s.index, y=test_s, name="Actual",
                                 line=dict(color=COLORS["Actual"], width=2.5)))
        for name, pred in preds.items():
            fig.add_trace(go.Scatter(x=test_s.index, y=pred, name=name,
                                     line=dict(color=COLORS[name], width=2, dash="dash")))
        fig.add_vrect(x0=test_s.index[0], x1=test_s.index[-1], fillcolor="#e0e7ff", opacity=0.35,
                      line_width=0, annotation_text="Test period", annotation_position="top left")
        fig.update_layout(title="Actual vs predicted daily sales", yaxis_title="Sales")
        st.plotly_chart(style_fig(fig, 480), use_container_width=True)

        st.dataframe(results.style.format({"MAE": "{:,.0f}", "RMSE": "{:,.0f}"}),
                     hide_index=True, use_container_width=True)
        st.markdown('<div class="note">Random Forest uses recent actual sales (lag_7 and rolling_mean_7), '
                    'so it predicts one day ahead. Holt-Winters and SARIMA forecast the whole test period at once.</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="note">Choose a test period and the models, then click <b>Run forecast</b>. '
                    'The last N days are held out as the test set and every model is trained on the data before it.</div>',
                    unsafe_allow_html=True)
