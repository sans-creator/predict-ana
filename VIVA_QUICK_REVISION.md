# Viva Quick Revision
**Retail Sales Analysis and Time Series Forecasting Using Machine Learning**
Full details: `PROJECT_VIVA_GUIDE.md`. All numbers below are verified against the code, data and `results/`.

## In one sentence
A Streamlit dashboard that analyses Favorita's 2013–2017 sales and forecasts **total units sold per day across all
54 stores and 33 families** using **SARIMA, Holt-Winters, Linear Regression and Random Forest**, compared with MAE and RMSE.

## Key numbers
| | |
|---|---|
| train.csv | 3,000,888 rows = 54 stores × 33 families × 1,684 dates |
| Daily series | 1,688 days, 1 Jan 2013 – 15 Aug 2017 |
| Total sales | 1.07 billion units |
| Average per trading day | 639.4K units (1,679 days) |
| Busiest day | Sat 1 Apr 2017, 1.46M |
| Top family | Grocery I, 32% (then Beverages 20.2%, Produce 11.4%) |
| Weekly pattern | Sunday +29.6%, Saturday +20.8%, Thursday −20.7% vs average |
| Growth | 386.8K/day (2013) → 859.3K/day (2017) |
| National holidays | +22.1% on average (74 dates); Good Friday −5.5% |
| Correlations | promotions +0.43 (from Apr 2014), oil −0.71 (trend-driven, not causal) |
| Store types | A ≈ 25.7K/day per store, about 2× the others |

## The six files
| File | Rows | Used for |
|---|---|---|
| train.csv | 3,000,888 | **model training** + all EDA |
| test.csv | 28,512 | listed only (16–31 Aug 2017, no sales) |
| stores.csv | 54 | store performance chart |
| oil.csv | 1,218 (43 missing prices) | oil charts |
| holidays_events.csv | 350 | holiday effect chart (national only) |
| transactions.csv | 83,488 | transactions chart |

## Preprocessing (src/data.py)
1. `load_raw` — read CSVs, parse dates.
2. `daily_sales` — `groupby("date").sum()`, full calendar, **25 Dec = 0 sales + closed flag** (not interpolated).
3. 1 Jan also marked closed (0.6–2.1% of a normal day).
4. `model_series` — **for training only**, closure days ← same weekday one week earlier.
5. Oil: only real trading days (dropna + inner join). Promotions: from 2014-04-01. Holidays: national, not transferred, not Work Day.
6. Not done: scaling, interpolation, duplicate removal (none existed), outlier removal.

## Features (Linear Regression + Random Forest, src/models.py → make_features)
`dayofweek, day, month, year, lag_7, lag_14, rolling_mean_28` (rolling mean uses `shift(1)` → no leakage).
Linear Regression one-hot encodes weekday and month (`drop="first"`) → 22 columns.
Random Forest importance: lag_7 0.64, lag_14 0.20.

## Models
| Model | Settings | Forecast style |
|---|---|---|
| SARIMA | (1,0,1)(1,1,1,7), last 730 days, statsmodels `SARIMAX` | fitted model forecast |
| Exponential Smoothing | Holt-Winters, additive damped trend, additive seasonality, period 7 | fitted model forecast |
| Linear Regression | OLS, sklearn | recursive |
| Random Forest | 300 trees, min_samples_leaf 2, random_state 42 | recursive |
| Seasonal naive (baseline) | repeat the last week | — |

Settings chosen on **six 30-day validation windows (Jan–Jul 2017)**, before the test period.

## Evaluation
- Train: **1 Jan 2013 – 16 Jul 2017**. Test: **17 Jul – 15 Aug 2017** (30 days). No shuffling.
- Every model forecasts all 30 days in one go from the same point; supervised models use their own predictions as lags.

| Model | MAE | RMSE | MAPE | R² |
|---|---|---|---|---|
| **Random Forest** | **49,569** | **68,483** | **5.88%** | **0.689** |
| Linear Regression | 57,777 | 75,944 | 6.73% | 0.618 |
| Exponential Smoothing | 57,897 | 87,234 | 6.71% | 0.495 |
| Seasonal naive | 60,672 | 91,696 | 7.03% | 0.443 |
| SARIMA | 61,381 | 92,511 | 7.17% | 0.433 |

- **MAE** = average |error|. **RMSE** = √(average error²), punishes big misses. **MAPE** = average % error.
  **R²** = share of variation explained (1 perfect, 0 = predicting the mean).
- Errors grow with horizon (e.g. Random Forest MAE 27K in days 1–10 → 65K in days 21–30).
- Linear Regression was last on validation but 2nd on test → rankings vary; never claim "always best".

## Future forecast (Forecasting page)
Model (pills) + horizon 7/14/30 → **Generate** → chart (last 90 days + forecast, ± test RMSE band), KPIs, table,
**CSV download** (`date, weekday, model, predicted_sales, typical_low, typical_high`). Starts **16 Aug 2017**.
Cannot be scored: no actual sales exist.

## Dashboard pages
1. **Overview** — 4 KPIs, sales trend (Daily/Weekly/Monthly), about card, datasets table.
2. **Sales Analytics** — tabs: Trends (month × year, year, weekday) · Products & stores · Holidays & promotions · Oil & transactions.
3. **Forecasting** — real future forecast + download.
4. **Model Comparison** — KPI cards, MAE/RMSE bars vs baseline, results table, actual vs predicted, error by day.

## Streamlit in 3 lines
- Every interaction **reruns the script**.
- `@st.cache_data` → data load, backtest and forecasts computed once.
- `st.session_state["forecast"]` → forecast stays visible after other clicks or page changes.

## Leakage protection
Chronological split · shifted lags · recursive forecasting · tuning before the test period · notebook check: forecasts
unchanged when test sales are set to 0.

## Limitations (say these honestly)
Chain-wide total only · holidays/promotions/oil not used as model inputs · recording gaps in 2014–15 (e.g. Produce) ·
one 30-day test window · ± RMSE band is not a confidence interval · future forecasts cannot be scored · no saved
models, no deployment.

## Never say
"per-store forecasts" · "we used holidays/oil in the models" · "we interpolated" · "we scaled / removed duplicates" ·
"GridSearchCV / cross-validation" · "95% confidence interval" · "test.csv was our test set" · "deep learning" ·
"Random Forest is always best".
