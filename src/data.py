"""Loading and preprocessing of the Kaggle Store Sales data.

Preprocessing decisions (documented here and in the README):

1. All dates are parsed as datetimes. The raw files are never modified.
2. train.csv has no rows for 25 December (2013-2016) because every store is
   closed. For analysis these days are recorded as 0 sales with closed=True
   instead of interpolating sales that never happened.
3. 1 January is also a closure day: a few stores open, so total sales are
   tiny (about 1-2% of a normal day).
4. Forecasting models need a continuous daily series without these one-off
   shocks, so model_series() replaces each closure day with the sales of the
   same weekday one week earlier. This is documented and only affects the
   model input; the evaluation period (July-August 2017) contains no
   closure days, so all metrics are computed on real observed sales.
5. Oil prices are only published on trading days. Oil analysis uses those
   real trading days only (no filling) when joining to daily sales.
6. Holidays: only national holidays apply to the chain-wide daily total.
   Transferred holidays (moved to another date) and "Work Day" rows are
   excluded because they are normal working days.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FILES = {
    "train": "train.csv",
    "test": "test.csv",
    "stores": "stores.csv",
    "oil": "oil.csv",
    "holidays": "holidays_events.csv",
    "transactions": "transactions.csv",
}

# (month, day) pairs on which the chain is closed
CLOSURE_DAYS = {(12, 25), (1, 1)}

# Holiday types that are real days off (Work Day = compensating working day)
HOLIDAY_TYPES = {"Holiday", "Transfer", "Additional", "Bridge"}

HOLIDAY_NAMES = {
    "Navidad": "Christmas season",
    "Primer dia del ano": "New Year",
    "Carnaval": "Carnival",
    "Viernes Santo": "Good Friday",
    "Dia del Trabajo": "Labour Day",
    "Dia de la Madre": "Mother's Day",
    "Batalla de Pichincha": "Battle of Pichincha",
    "Primer Grito de Independencia": "Independence Day",
    "Independencia de Guayaquil": "Guayaquil Independence",
    "Dia de Difuntos": "Day of the Dead",
    "Independencia de Cuenca": "Cuenca Independence",
    "Puente": "Bridge days",
}


def load_raw(data_dir=DATA_DIR):
    """Read all six CSV files and parse the date columns."""
    data_dir = Path(data_dir)
    raw = {}
    for name, file in FILES.items():
        path = data_dir / file
        has_date = "date" in pd.read_csv(path, nrows=0).columns
        raw[name] = pd.read_csv(path, parse_dates=["date"] if has_date else None)
    return raw


def dataset_summary(raw):
    """One row per CSV file: rows, columns and date range."""
    rows = []
    for name, df in raw.items():
        dates = df["date"] if "date" in df else None
        rows.append({
            "File": FILES[name],
            "Rows": len(df),
            "Columns": df.shape[1],
            "From": dates.min().date() if dates is not None else None,
            "To": dates.max().date() if dates is not None else None,
        })
    return pd.DataFrame(rows)


def daily_sales(train):
    """Total sales and items on promotion per day, on a full daily calendar.

    Missing dates (25 December) get 0 sales and closed=True.
    """
    daily = train.groupby("date").agg(sales=("sales", "sum"), onpromotion=("onpromotion", "sum"))
    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D", name="date")
    daily = daily.reindex(full_range)
    daily["observed"] = daily["sales"].notna()
    daily = daily.fillna({"sales": 0.0, "onpromotion": 0})
    daily["closed"] = [(d.month, d.day) in CLOSURE_DAYS for d in daily.index]
    return daily


def model_series(daily):
    """Daily sales series used to train the forecasting models.

    Closure days are replaced by the value of the same weekday one week
    earlier (one week later for the very first day, which has no history).
    """
    s = daily["sales"].copy()
    for d in daily.index[daily["closed"]]:
        prev = d - pd.Timedelta(days=7)
        s[d] = s[prev] if prev in s.index else s[d + pd.Timedelta(days=7)]
    s.index.freq = "D"
    return s.rename("sales")


def national_holidays(holidays):
    """National days off, one row per date."""
    h = holidays[(holidays["locale"] == "National")
                 & (~holidays["transferred"])
                 & (holidays["type"].isin(HOLIDAY_TYPES))]
    return h.sort_values("date").drop_duplicates("date")[["date", "type", "description"]]


def holiday_effects(daily, holidays):
    """Sales on each national holiday compared with a normal day.

    The baseline for a date is the average sales of the same weekday in the
    four previous weeks, so the comparison is not distorted by the long-term
    trend or by the weekly pattern. Closure days are excluded.
    """
    s = daily["sales"]
    baseline = sum(s.shift(7 * k) for k in range(1, 5)) / 4
    h = national_holidays(holidays)
    h = h[h["date"].isin(s.index) & ~h["date"].isin(s.index[daily["closed"]])].copy()
    h["sales"] = s.loc[h["date"]].values
    h["baseline"] = baseline.loc[h["date"]].values
    h = h.dropna(subset=["baseline"])
    h["uplift_pct"] = (h["sales"] / h["baseline"] - 1) * 100
    # "Navidad-1" -> "Navidad", "Traslado X" (holiday moved to this day) -> "X"
    name = (h["description"].str.replace(r"[-+]\d+$", "", regex=True)
            .str.replace(r"^Traslado ", "", regex=True).str.strip())
    name = name.where(~name.str.startswith("Puente "), "Puente")
    h["holiday"] = name.map(HOLIDAY_NAMES).fillna(name)
    return h


def family_sales(train):
    """Total sales per product family, largest first."""
    return train.groupby("family")["sales"].sum().sort_values(ascending=False)


def family_monthly(train):
    """Average daily sales per product family for each month (long format).

    Averages are used instead of totals so that months of different length
    and the incomplete last month (August 2017) are comparable.
    """
    per_day = train.groupby(["date", "family"])["sales"].sum().reset_index()
    out = (per_day.groupby([pd.Grouper(key="date", freq="MS"), "family"])["sales"].mean()
           .reset_index())
    return out


def store_sales(train, stores):
    """Total and average daily sales per store, with store attributes."""
    store_day = train.groupby(["store_nbr", "date"])["sales"].sum()
    per_store = store_day.groupby("store_nbr").sum().to_frame("sales")
    # days with any sales: stores that opened after 2013 are not penalised
    per_store["days_open"] = (store_day > 0).groupby("store_nbr").sum()
    per_store["avg_daily_sales"] = per_store["sales"] / per_store["days_open"]
    # stores.csv has exactly one row per store, so this join keeps every row once
    return per_store.reset_index().merge(stores, on="store_nbr", how="left", validate="1:1")


def daily_transactions(transactions, train):
    """Daily transactions and units sold per transaction.

    transactions.csv has no row for store-days when a store was closed or not
    yet open, so units per transaction only uses store-days present in both
    files.
    """
    tx_daily = transactions.groupby("date")["transactions"].sum()
    store_day = train.groupby(["date", "store_nbr"])["sales"].sum().reset_index()
    both = store_day.merge(transactions, on=["date", "store_nbr"], how="inner", validate="1:1")
    both = both.groupby("date")[["sales", "transactions"]].sum()
    out = pd.DataFrame({"transactions": tx_daily})
    out["units_per_transaction"] = both["sales"] / both["transactions"]
    return out


def oil_vs_sales(oil, daily):
    """Oil price and daily sales on real oil trading days only (no filling)."""
    o = oil.dropna(subset=["dcoilwtico"]).set_index("date")["dcoilwtico"]
    open_days = daily.loc[~daily["closed"], "sales"]
    joined = pd.concat([o.rename("oil"), open_days], axis=1, join="inner")
    return o, joined


def promotion_vs_sales(daily):
    """Daily promotions and sales from April 2014.

    onpromotion is 0 for every row before 2014-04-01, which means it was not
    recorded, so earlier days are left out of the promotion analysis.
    """
    d = daily[(daily.index >= "2014-04-01") & ~daily["closed"]]
    return d[["onpromotion", "sales"]]


def build_dataset(data_dir=DATA_DIR):
    """Load everything and prepare the small tables used by the app."""
    raw = load_raw(data_dir)
    train = raw["train"]
    daily = daily_sales(train)
    oil_price, oil_join = oil_vs_sales(raw["oil"], daily)
    return {
        "summary": dataset_summary(raw),
        "daily": daily,
        "series": model_series(daily),
        "family": family_sales(train),
        "family_monthly": family_monthly(train),
        "stores": store_sales(train, raw["stores"]),
        "holidays": holiday_effects(daily, raw["holidays"]),
        "transactions": daily_transactions(raw["transactions"], train),
        "oil_price": oil_price,
        "oil_join": oil_join,
        "promo": promotion_vs_sales(daily),
        "test_dates": (raw["test"]["date"].min(), raw["test"]["date"].max()),
        "n_stores": train["store_nbr"].nunique(),
        "n_families": train["family"].nunique(),
    }
