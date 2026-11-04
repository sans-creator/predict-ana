"""The four forecasting models, recursive forecasting and evaluation.

Every model has the same interface:

    forecast_fn(history, horizon) -> pd.Series

It is trained only on `history` (a daily sales Series) and returns
`horizon` predictions for the days right after the last date in `history`.
No model ever sees a value after the end of its history, so the same
function is used for backtesting (history = data before the test period)
and for real future forecasting (history = all data).
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

SEASON = 7  # weekly seasonality in daily data

# Settings chosen with rolling validation on data before the test period
# (see notebooks/sales_forecasting.ipynb, section "Model selection").
SARIMA_ORDER = (1, 0, 1)
SARIMA_SEASONAL_ORDER = (1, 1, 1, SEASON)
SARIMA_WINDOW = 730  # train on the most recent two years
HW_TREND = "add"
HW_DAMPED = True
HW_SEASONAL = "add"

CALENDAR = ["dayofweek", "day", "month", "year"]
LAGS = ["lag_7", "lag_14", "rolling_mean_28"]
FEATURES = CALENDAR + LAGS
MAX_LOOKBACK = 28  # longest window used by any feature


def make_features(sales):
    """Calendar and lag features. Every lag uses only earlier days."""
    df = sales.to_frame("sales")
    df["dayofweek"] = df.index.dayofweek
    df["day"] = df.index.day
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["lag_7"] = df["sales"].shift(7)
    df["lag_14"] = df["sales"].shift(14)
    # shift(1) first so the average never includes the day being predicted
    df["rolling_mean_28"] = df["sales"].shift(1).rolling(28).mean()
    return df


def recursive_forecast(model, history, horizon):
    """Forecast one day at a time, feeding each prediction back as history.

    The lag features for future days are built from earlier predictions,
    never from actual future sales.
    """
    extended = history.copy()
    preds = []
    for _ in range(horizon):
        next_day = extended.index[-1] + pd.Timedelta(days=1)
        window = pd.concat([extended.iloc[-MAX_LOOKBACK:], pd.Series([np.nan], index=[next_day])])
        x = make_features(window)[FEATURES].iloc[[-1]]
        y = float(model.predict(x)[0])
        preds.append(y)
        extended[next_day] = y
    return pd.Series(preds, index=future_index(history, horizon), name="forecast")


def future_index(history, horizon):
    return pd.date_range(history.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D", name="date")


def _training_data(history):
    df = make_features(history).dropna()
    return df[FEATURES], df["sales"]


def linear_regression(history, horizon):
    """Linear Regression on calendar and lag features (recursive)."""
    X, y = _training_data(history)
    encode = ColumnTransformer(
        # one-hot encoding: weekday and month are categories, not quantities;
        # drop="first" avoids the dummy variable trap
        [("calendar", OneHotEncoder(drop="first", handle_unknown="ignore"), ["dayofweek", "month"])],
        remainder="passthrough",
    )
    model = make_pipeline(encode, LinearRegression()).fit(X, y)
    return recursive_forecast(model, history, horizon)


def random_forest(history, horizon):
    """Random Forest on the same features (recursive)."""
    X, y = _training_data(history)
    model = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return recursive_forecast(model, history, horizon)


def sarima(history, horizon):
    """Seasonal ARIMA with a weekly season, trained on the most recent year."""
    y = history.iloc[-SARIMA_WINDOW:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        fit = SARIMAX(y, order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL_ORDER).fit(disp=False)
    return pd.Series(fit.forecast(horizon).values, index=future_index(history, horizon), name="forecast")


def exponential_smoothing(history, horizon):
    """Holt-Winters: level + damped trend + weekly seasonality."""
    fit = ExponentialSmoothing(history, trend=HW_TREND, damped_trend=HW_DAMPED,
                               seasonal=HW_SEASONAL, seasonal_periods=SEASON).fit()
    return pd.Series(fit.forecast(horizon).values, index=future_index(history, horizon), name="forecast")


def seasonal_naive(history, horizon):
    """Reference baseline: repeat the last observed week."""
    values = np.resize(history.iloc[-SEASON:].values, horizon)
    return pd.Series(values, index=future_index(history, horizon), name="forecast")


MODELS = {
    "SARIMA": sarima,
    "Exponential Smoothing": exponential_smoothing,
    "Linear Regression": linear_regression,
    "Random Forest": random_forest,
}
BASELINE = "Seasonal Naive"

MODEL_INFO = {
    "SARIMA": {
        "family": "Statistical time series",
        "config": f"SARIMA{SARIMA_ORDER}x{SARIMA_SEASONAL_ORDER}, trained on the last {SARIMA_WINDOW} days",
        "summary": "Predicts each day from the previous days and from the same weekday in previous weeks, "
                   "after seasonal differencing (today minus the same day last week) removes the weekly pattern.",
    },
    "Exponential Smoothing": {
        "family": "Statistical time series",
        "config": "Holt-Winters, additive damped trend, additive weekly seasonality (period 7)",
        "summary": "Keeps a running level, trend and weekday effect, giving recent days more weight. "
                   "The trend is damped so it flattens out instead of growing forever.",
    },
    "Linear Regression": {
        "family": "Supervised machine learning",
        "config": "One-hot weekday and month, day, year, lag_7, lag_14, 28-day rolling mean",
        "summary": "Learns a weighted sum of the features. Future days are predicted one at a time and "
                   "each prediction is fed back as the lag for the next day.",
    },
    "Random Forest": {
        "family": "Supervised machine learning",
        "config": "300 trees, min_samples_leaf=2, same features, recursive forecasting",
        "summary": "Averages 300 decision trees, each answering yes/no questions about the features. "
                   "Captures non-linear patterns but cannot predict outside the range it was trained on.",
    },
}


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAPE (%)": float(np.mean(np.abs((actual - predicted) / actual)) * 100),
        "R²": r2_score(actual, predicted),
    }


def backtest(series, test_days=30, include_baseline=True):
    """Train every model on data before the last `test_days` days and
    forecast those days in one go (multi-step, same horizon for all)."""
    history, test = series.iloc[:-test_days], series.iloc[-test_days:]
    models = dict(MODELS)
    if include_baseline:
        models[BASELINE] = seasonal_naive
    preds, rows = {}, []
    for name, fn in models.items():
        preds[name] = fn(history, test_days)
        rows.append({"Model": name, **metrics(test, preds[name])})
    table = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    return table, pd.DataFrame(preds), test
