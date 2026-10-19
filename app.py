"""Streamlit demo app for the Retail Sales Forecasting project.

Run with:  streamlit run app.py
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA = "data"
st.set_page_config(page_title="Retail Sales Forecasting", page_icon="📈", layout="wide")


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
    return daily, family


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


@st.cache_data(show_spinner=False)
def run_model(name, series, n):
    if name == "Random Forest":
        return MODELS[name](series, n)
    return MODELS[name](series[:-n], n)


# ---------- UI ----------
st.title("📈 Retail Sales Analysis & Forecasting")
st.caption("Kaggle: Store Sales - Time Series Forecasting (Corporación Favorita, Ecuador)")

try:
    daily, family = load_data()
except FileNotFoundError:
    st.error("Data not found. Put train.csv and oil.csv in the data/ folder.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 EDA", "🤖 Forecast"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Days of data", f"{len(daily):,}")
    c2.metric("Total sales", f"{daily['sales'].sum() / 1e6:,.1f} M")
    c3.metric("Avg daily sales", f"{daily['sales'].mean():,.0f}")
    c4.metric("Product families", len(family))

    st.subheader("Daily total sales")
    start, end = st.slider("Date range", daily.index.min().date(), daily.index.max().date(),
                           (daily.index.min().date(), daily.index.max().date()))
    st.line_chart(daily.loc[str(start):str(end), "sales"])

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Monthly sales")
        st.bar_chart(daily["sales"].resample("ME").sum())
    with col2:
        st.subheader("Average sales by day of week")
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow = daily.groupby(daily.index.day_name())["sales"].mean().reindex(days)
        dow.index = [f"{i + 1}. {d}" for i, d in enumerate(days)]  # keep weekday order
        st.bar_chart(dow)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Top 10 product families")
        st.bar_chart(family.head(10), horizontal=True)
    with col4:
        st.subheader("Promotion vs sales")
        st.scatter_chart(daily, x="onpromotion", y="sales")
        st.caption(f"Correlation: {daily['onpromotion'].corr(daily['sales']):.2f}")

    st.subheader("Oil price vs sales")
    st.scatter_chart(daily, x="oil", y="sales", color="#ff8c00")
    st.caption(f"Correlation: {daily['oil'].corr(daily['sales']):.2f}")

with tab3:
    st.subheader("Forecast the last N days and compare models")
    c1, c2 = st.columns([1, 2])
    n_days = c1.slider("Test period (days)", 7, 60, 30)
    chosen = c2.multiselect("Models", list(MODELS), default=list(MODELS))

    if st.button("Run forecast", type="primary") and chosen:
        series = daily["sales"]
        test_s = series[-n_days:]
        chart = pd.DataFrame({"Actual": test_s})
        rows = []
        for name in chosen:
            with st.spinner(f"Training {name}..."):
                pred = run_model(name, series, n_days)
            chart[name] = pred
            rows.append({"Model": name,
                         "MAE": mean_absolute_error(test_s, pred),
                         "RMSE": np.sqrt(mean_squared_error(test_s, pred))})

        results = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
        best = results.iloc[0]
        st.success(f"🏆 Best model: **{best['Model']}** (RMSE = {best['RMSE']:,.0f})")

        st.line_chart(chart)
        st.dataframe(results.style.format({"MAE": "{:,.0f}", "RMSE": "{:,.0f}"}),
                     hide_index=True, use_container_width=True)
        st.caption("Note: Random Forest uses recent actual sales (lag_7, rolling_mean_7), "
                   "so it predicts one day ahead, while Holt-Winters and SARIMA forecast all days at once.")
