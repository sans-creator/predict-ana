# Project Viva Guide
## Retail Sales Analysis and Time Series Forecasting Using Machine Learning

This guide explains the project **as it is actually implemented**. Every number in it was taken from the
code, the data in `data/`, or the saved results in `results/`. Where something is *not* implemented, it says so.

> **How to read this guide.** Each topic starts with a plain-language explanation, followed by the technical
> details. Code references use the form `file → function`. Numbers marked **(example)** are made up to
> explain a concept; every other number is real.

---

## Table of contents
1. [Project overview](#part-1-project-overview)
2. [Complete website walkthrough](#part-2-complete-website-walkthrough)
3. [Dataset explanation](#part-3-dataset-explanation)
4. [Data cleaning and preprocessing](#part-4-data-cleaning-and-preprocessing)
5. [Exploratory data analysis](#part-5-exploratory-data-analysis)
6. [Feature engineering](#part-6-feature-engineering)
7. [The four models](#part-7-the-four-models)
8. [Training and evaluation](#part-8-model-training-and-evaluation)
9. [Future forecasting](#part-9-future-forecasting)
10. [Technologies and libraries](#part-10-technologies-and-libraries)
11. [Codebase walkthrough](#part-11-complete-codebase-walkthrough)
12. [Viva questions and answers](#part-12-viva-questions-and-answers)
13. [Presentation scripts](#part-13-presentation-scripts)
14. [Things we must not misrepresent](#part-14-things-we-must-not-misrepresent)

---

# PART 1: PROJECT OVERVIEW

### 1.1 What exactly is our project?
**Simple:** A web dashboard that studies five years of sales from an Ecuadorian supermarket chain and predicts
how many items the whole chain will sell on each of the coming days.

**Technical:** A Python project with two parts that share the same code:
- a **Streamlit dashboard** (`app.py`) with four pages: Overview, Sales Analytics, Forecasting, Model Comparison;
- a **Jupyter notebook** (`notebooks/sales_forecasting.ipynb`) that documents the full analysis, model selection,
  evaluation and checks, and saves plots to `results/`.

It forecasts **one time series**: total units sold per day across all 54 stores and all 33 product families of
Corporación Favorita, using four models: **SARIMA**, **Holt-Winters Exponential Smoothing**, **Linear Regression**
and **Random Forest Regressor**.

### 1.2 What problem are we solving?
A retailer has to decide in advance how much stock to order, how many staff to schedule and how much cash to plan
for. These decisions depend on how much will be sold. Guessing leads to empty shelves (lost sales) or excess stock
(waste, especially for perishable food). Our project predicts future daily sales from past sales so these decisions
can be based on data.

### 1.3 Why is retail sales forecasting important?
- **Inventory:** order the right quantity; perishables (Produce is 11.4% of units sold in our data) spoil.
- **Staffing:** our data shows Sundays sell about 30% more than an average day; stores need more staff then.
- **Promotions and holidays:** our data shows national holidays sell about 22% more than a normal day on average.
- **Finance:** revenue planning and supplier contracts.

### 1.4 What is time-series forecasting?
**Simple:** A time series is a list of values recorded at regular time steps, such as sales every day. Forecasting
means using the past values to predict the next ones.

**Technical:** Given observations y₁, y₂, …, y_T at equal intervals, a forecasting model estimates
ŷ_{T+1}, …, ŷ_{T+h} for a horizon h. Time series usually contain:
- **trend** (long-term growth: our sales rose from ~387K/day in 2013 to ~859K/day in 2017),
- **seasonality** (repeating cycles: our sales repeat every 7 days, peaking at weekends),
- **noise** (random variation).

Unlike ordinary machine learning, the **order of the rows matters**, so the data must never be shuffled.

### 1.5 Objective of the application
1. Analyse sales patterns (trend, weekly cycle, products, stores, holidays, promotions, oil, transactions).
2. Train four forecasting models and compare them **fairly** on the same 30-day test period with MAE, RMSE, MAPE and R².
3. Produce **real future forecasts** (7, 14 or 30 days after the last date in the data) that the user can download.

### 1.6 Inputs and outputs
| Inputs | Outputs |
|---|---|
| The six Kaggle CSV files in `data/` | KPI cards and interactive charts (analysis) |
| User choices: model, horizon (7/14/30 days), chart filters | Comparison table: MAE, RMSE, MAPE, R² per model |
| | Future forecast chart and table |
| | Downloadable forecast CSV |
| | Notebook plots and CSVs in `results/` |

### 1.7 Technologies used
Python 3.13 (tested), Streamlit 1.66.1, pandas 3.1.0, NumPy 2.5.4, scikit-learn 1.9.1, statsmodels 0.15.0,
Plotly 7.1.0, Matplotlib 3.11.2 (notebook only), Jupyter 1.1.1. Details in [Part 10](#part-10-technologies-and-libraries).

### 1.8 Architecture
There is **no separate backend, database or API**. Streamlit runs a Python server that executes `app.py` and sends
the result to the browser.

```
data/*.csv
   │
   ▼
src/data.py      load_raw → daily_sales → model_series, plus analysis tables (build_dataset)
   │
   ├──────────────► src/ui.py (styling)  ──►  app.py pages: Overview, Sales Analytics
   ▼
src/models.py    make_features, 4 models, recursive_forecast, metrics, backtest
   │
   └──────────────► app.py pages: Forecasting, Model Comparison
                    (results cached with st.cache_data)

notebooks/sales_forecasting.ipynb imports the same src/ code → writes results/*.png, results/*.csv
```

### 1.9 How data flows from CSV files to the dashboard
1. `app.py → get_data()` calls `src/data.py → build_dataset()`.
2. `load_raw()` reads all six CSVs with pandas and converts the `date` columns to datetimes.
3. `daily_sales()` sums the 3,000,888 rows of `train.csv` into 1,688 daily totals and marks closure days.
4. `model_series()` makes the version of the series used by the models (closure days replaced).
5. Other functions build small analysis tables (families, stores, holidays, transactions, oil, promotions).
6. `get_backtest()` trains all models on data up to 16 Jul 2017 and scores them on 17 Jul – 15 Aug 2017.
7. `get_forecast(model)` trains the chosen model on all data and predicts 30 days from 16 Aug 2017.
8. Page functions turn these results into Plotly charts, KPI cards and tables.
All three `get_*` functions are cached, so this work happens once per server run, not on every click.

### 1.10 Why is it useful for businesses?
It turns raw transaction data into: (a) clear patterns a manager can act on (weekend peaks, December peak, holiday
effects, best store types), (b) a forecast for the next weeks, and (c) an honest error estimate for each model, so
the business knows how much to trust the forecast (e.g. Random Forest's typical error on the test period was about
±50K units on days averaging ~833K, roughly 6%).

---

# PART 2: COMPLETE WEBSITE WALKTHROUGH

### General layout (all pages)
- **Sidebar** (left): section title "Retail Sales Forecasting" and four links: Overview (`/`), Sales Analytics
  (`/analytics`), Forecasting (`/forecasting`), Model Comparison (`/comparison`). Below them: "Kaggle Store Sales,
  Corporación Favorita (Ecuador). Data up to 15 Aug 2017." The date is computed from the data.
  *Code:* bottom of `app.py` (`st.navigation`, `st.Page`).
- **Page header**: a small coloured label, a title and a one-line description (`src/ui.py → page_header`).
- **Cards**: dark "glass" boxes with a title and caption (`src/ui.py → card`, `kpi`).
- **Charts**: Plotly, interactive (hover to see values, drag to zoom, double-click to reset).
  Styled by `src/ui.py → style` and shown by `src/ui.py → chart`.
- **Missing data**: if a CSV is missing, the app shows an error naming the missing file and listing the six required files, then stops
  (`app.py`, `try: get_data() except FileNotFoundError`).
- **Theme**: dark, defined in `.streamlit/config.toml` (Inter font, indigo accent `#818cf8`).
- **Mobile**: Streamlit stacks columns on narrow screens; the sidebar collapses into a menu. (Only the Overview page
  was checked at phone width.)

---

## PAGE 1: OVERVIEW  (`app.py → overview()`)

**Purpose:** a one-screen summary of the business and the project.

**What the user sees:** header "Retail Sales Analysis & Forecasting" with the subtitle
"Daily unit sales of Corporación Favorita … 54 stores and 33 product families, January 2013 to August 2017"
(store and family counts are computed: `n_stores`, `n_families` in `build_dataset`), then 4 KPI cards, the sales
trend chart, the "About the project" card and the "Datasets" table.

### KPI cards
| Card | Value shown | How it is calculated | Source |
|---|---|---|---|
| **Total sales** | 1.07B, "units over 1,688 days" | Sum of daily total sales over all 1,688 days (closure days count as 0) | train.csv |
| **Average daily sales** | 639.4K, "across 1,679 trading days" | Mean of daily sales over days that are not closure days (1,688 − 9 closure days) | train.csv |
| **Highest sales day** | 1.46M, "Saturday, 01 Apr 2017" | `idxmax()` of daily sales | train.csv |
| **Top product family** | Grocery I, "32% of all units sold" | Family with the largest total; share = its total / all sales | train.csv |

The `human()` function in `src/ui.py` formats numbers (1,073,644,952 → "1.07B").

**How to explain (pointing):** "Over 4.6 years the chain sold about 1.07 billion units. A normal trading day
sells about 640 thousand units. The busiest day ever was Saturday 1 April 2017 with 1.46 million. Grocery I alone
is almost a third of everything sold."

### Sales trend chart
- **Control:** a segmented button **Daily / Weekly / Monthly** (key `ov_view`; remembered when you change page).
- **Daily view:**
  - grey thin line = each day's total sales;
  - indigo line = 30-day rolling average (average of the last 30 days, which smooths out the weekly zig-zag);
  - red ✕ markers = the 9 closure days (25 Dec 2013–2016 at 0, 1 Jan 2013–2017 near 0).
- **Weekly / Monthly views:** one point per week or per month = **average daily sales** in that week/month
  (closure days excluded). Averages are used so a short week/month is not penalised.
  *Code:* `open_days["sales"].resample("W" or "MS").mean()`.
- **X-axis:** date (2013 → Aug 2017). **Y-axis:** units sold per day.
- **Data:** `daily` table from `src/data.py → daily_sales` (train.csv).
- **Conclusions:** steady upward trend; strong weekly oscillation; sharp drops on closure days; a few dips in
  2014 – early 2015 that come from recording gaps in the source data (see Part 4.16).
- **Why useful:** the first thing a forecaster must know is whether there is trend and seasonality — there are both.

**How to explain:** "This is total sales per day. The thin grey line is every single day and you can see it
zig-zags — that is the weekly cycle. The indigo line is a 30-day average showing the trend going up from about
400 thousand to over 800 thousand per day. The red crosses are Christmas and New Year, when the stores are closed."

### About the project card
Five bullet points: goal, data, models, evaluation, forecasting. The row count (3,000,888) and the first forecast
date (16 Aug 2017) are computed from the data; the rest is fixed text.
Note: it says "7 to 30 days into the future"; the app actually offers exactly **7, 14 or 30** days.

### Datasets table
One row per CSV: File, Rows, Columns, From, To (`src/data.py → dataset_summary`). stores.csv has no dates, so it shows "—".

| File | Rows | Columns | From | To |
|---|---|---|---|---|
| train.csv | 3,000,888 | 6 | 01 Jan 2013 | 15 Aug 2017 |
| test.csv | 28,512 | 5 | 16 Aug 2017 | 31 Aug 2017 |
| stores.csv | 54 | 5 | — | — |
| oil.csv | 1,218 | 2 | 01 Jan 2013 | 31 Aug 2017 |
| holidays_events.csv | 350 | 6 | 02 Mar 2012 | 26 Dec 2017 |
| transactions.csv | 83,488 | 3 | 01 Jan 2013 | 15 Aug 2017 |

---

## PAGE 2: SALES ANALYTICS  (`app.py → analytics()`)

**Purpose:** explain *what drives sales* before modelling. Four tabs: **Trends**, **Products & stores**,
**Holidays & promotions**, **Oil & transactions**. Clicking a tab only switches the view (all tabs are computed
on load).

### Tab 1: Trends

#### Chart 1 — Average daily sales by month
- **Control:** year "pills" 2013–2017 (multi-select, key `an_years`). Click a year to hide/show its line. If all are
  deselected, all years are shown (`chosen or years`).
- **X-axis:** month (Jan–Dec). **Y-axis:** average units sold per day in that month. One line per year.
- **Calculation:** open (non-closure) days grouped by year and month, mean of sales.
- **Data:** train.csv via `daily`.
- **Conclusions:** each line is higher than the previous year (growth). December is the strongest month
  (all-year average of open days: Dec ≈ 808.6K/day, then Nov ≈ 669.5K, Jul ≈ 666.9K); February is the weakest.
  2017 stops at August (data ends 15 Aug).
- **How to explain:** "Each line is one year. Every year sits above the previous one, so the business is growing,
  and every year peaks in December because of Christmas shopping."

#### Chart 2 — Average daily sales by year
- **X-axis:** year. **Y-axis:** average units per day. Labels above bars; hover shows % change vs previous year.
- **Values:** 2013 386.8K · 2014 577.0K (+49.2%) · 2015 663.5K (+15.0%) · 2016 793.0K (+19.5%) · 2017 859.3K (+8.4%).
- **Why averages, not totals:** 2017 only has 7.5 months; totals would make 2017 look like a bad year.
- **How to explain:** "We use average per day so that 2017, which is incomplete, can be compared fairly. Sales per day more than doubled from 2013 to 2017."

#### Chart 3 — Average sales by weekday
- **X-axis:** weekday (Mon–Sun). **Y-axis:** average units per day. Saturday and Sunday are highlighted in indigo.
- **Values:** Mon 617.5K · Tue 572.3K · Wed 595.7K · **Thu 507.3K (lowest, −20.7% vs average)** · Fri 581.9K ·
  Sat 772.2K (+20.8%) · **Sun 828.6K (highest, +29.6%)**.
- **Conclusion:** strong weekly seasonality — the reason all our models use a 7-day season or weekday features.
- **How to explain:** "This chart proves weekly seasonality. Sunday sells about 60% more than Thursday. That is
  why SARIMA uses a season of 7 and the regression models use day-of-week and 7-day lags."

### Tab 2: Products & stores

#### Chart 4 — Top product families
- Horizontal bars for the 12 largest families; label = share of all units sold.
- **X-axis:** total units sold 2013–2017. **Y-axis:** family name.
- **Values (top 5):** Grocery I 32.0%, Beverages 20.2%, Produce 11.4%, Cleaning 9.1%, Dairy 6.0%.
- **Calculation:** `src/data.py → family_sales` (groupby family, sum).
- **How to explain:** "Three families — Grocery I, Beverages and Produce — make up almost two thirds of units sold."

#### Chart 5 — Product family trends
- **Control:** multi-select dropdown of families (default: Grocery I, Beverages, Produce; maximum 6; key `an_families`).
- **X-axis:** month. **Y-axis:** average daily units for that family in that month.
- **Calculation:** `src/data.py → family_monthly` (sum per family per day, then mean per month).
- **Data-quality note:** if a selected family has a month below 20% of its own median month, a note appears
  saying the drop comes from the source data. With the defaults it names **Produce**, which falls to a few hundred
  units/day in several months of 2014 and early 2015 (normal: ~100K+). Beverages roughly halves in the same months
  but stays above the 20% threshold, so it is not named.
- **How to explain:** "Produce drops almost to zero in some months of 2014 — that is not real; most stores simply
  did not record Produce sales then. We show it honestly instead of hiding it."

#### Chart 6 — Store performance
- **Control:** segmented button **Store type / City / Cluster / Store** (key `an_store_group`).
- **X-axis:** the group (Type A…E, city, cluster, or store number). **Y-axis:** average daily units **per store**.
  Hover shows how many stores are in the group. "Store" view shows the top 20 stores.
- **Calculation:** `src/data.py → store_sales`: for each store, total sales ÷ number of days on which that store had
  sales (so stores that opened later are not penalised). Then the group value is the mean over its stores.
  stores.csv is joined one-to-one on `store_nbr` (`validate="1:1"`).
- **Values (store type):** A 25.7K (9 stores) · B 12.5K (8) · D 12.2K (18) · E 10.4K (4) · C 6.8K (15).
  Best stores: #44 (36,979/day), #45 (32,459), #47 (30,344), all type A in Quito. 22 cities, 16 states, 17 clusters.
- **Note:** Kaggle does not explain what types A–E mean; we can only say type A stores sell the most.
- **How to explain:** "Type A stores sell about twice as much per day as the other types, and the three best stores are all type A stores in Quito, the capital."

### Tab 3: Holidays & promotions

#### Chart 7 — National holiday effect
- Horizontal bars, green = more sales than normal, red = fewer.
- **X-axis:** % change versus a normal day. **Y-axis:** holiday name (translated to English).
- **Calculation:** `src/data.py → holiday_effects`:
  1. keep only national holidays that are real days off (`national_holidays`): locale = National, transferred = False,
     type in {Holiday, Transfer, Additional, Bridge};
  2. baseline for each date = average sales of the same weekday in the 4 previous weeks;
  3. uplift % = (sales ÷ baseline − 1) × 100; closure days are excluded;
  4. averaged per holiday name ("Navidad-3" → "Christmas season", "Traslado X" → X, "Puente …" → "Bridge days").
- **Values:** Labour Day +44.9% · Cuenca Independence +44.5% · Christmas season +38.4% (19 days around Christmas) ·
  Carnival +18.2% · Guayaquil Independence +17.4% · Bridge days +16.2% · New Year +15.9% · Battle of Pichincha
  +13.2% · Day of the Dead +10.2% · Mother's Day +10.2% · Independence Day −1.5% · Good Friday −5.5%.
  Average over all 74 holiday dates: **+22.1%** (shown in the note below the chart).
- **Why compare with the same weekday:** a holiday on a Sunday would look "good" just because Sundays are busy.
- **How to explain:** "For each national holiday we compare sales with the same weekday in the previous four weeks.
  Most holidays increase sales — on average by 22% — while Good Friday slightly decreases them."

#### Chart 8 — Promotions vs sales
- Scatter plot, one dot per day from 1 April 2014, coloured by year; white line = overall linear trend (OLS fitted by
  Plotly using statsmodels).
- **X-axis:** total items on promotion that day. **Y-axis:** total units sold that day.
- **Calculation:** `src/data.py → promotion_vs_sales` (days from 2014-04-01, closure days excluded; 1,227 days).
  Correlation **0.43** (pandas `.corr()`, Pearson).
- **Why from April 2014:** `onpromotion` is 0 on 100% of days before 2014-04-01, i.e. not recorded.
- **Conclusion:** moderate positive relationship, but both promotions and sales grew over time (colours move right
  and up), so part of the correlation is the shared trend — not proof that promotions cause sales.
- **How to explain:** "More items on promotion go with higher sales, correlation 0.43. But notice the colours: later
  years are top-right, so both just grew together. Correlation is not causation."

### Tab 4: Oil & transactions

#### Chart 9 — Oil price and sales over time
- Two lines with two y-axes: orange = oil price (left axis, USD); indigo = sales 30-day average (right axis).
- **Calculation:** oil from `oil_vs_sales` (trading days only, no filling); sales = rolling 30-row mean over open days.
- **Conclusion:** oil was ~$90–110 until mid-2014, crashed to a low of $26.19 (11 Feb 2016); sales kept rising.
- **How to explain:** "Ecuador's economy depends on oil. Oil collapsed in late 2014, but sales kept growing."

#### Chart 10 — Oil price vs daily sales
- Scatter: **X** = oil price, **Y** = units sold, coloured by year. 1,163 days where both values exist.
- **Correlation: −0.71.** Strong negative, but caused by two opposite trends over time (oil down, sales up), not by
  cheap oil making people shop more.
- **How to explain:** "The correlation is −0.71, but it is a time effect: early years are on the right with expensive
  oil and low sales; later years are on the left. We do not use oil to predict sales."

#### Chart 11 — Transactions
- **Control:** segmented button **Transactions per day / Units per transaction** (key `an_tx`).
- **X-axis:** date. **Y-axis:** weekly average of the chosen metric.
- **Calculation:** `src/data.py → daily_transactions`: transactions summed per day; units per transaction = sales ÷
  transactions using only store-days present in both files (inner join on date + store, `validate="1:1"`).
- **Values:** about 84K transactions per day on average; highest day 24 Dec 2015 (171,169); about 7.45 units per
  transaction on average.
- **Conclusion:** transactions are fairly flat with big Christmas spikes; growth in units comes more from bigger
  baskets than from more visits.
- **How to explain:** "A transaction is one customer purchase. Visits stay around 80–90 thousand a day except at
  Christmas; units per transaction is basically basket size."

---

## PAGE 3: FORECASTING  (`app.py → forecasting()`)

**Purpose:** produce **real future forecasts** — for days after 15 Aug 2017 that are not in the data.

**What the user sees:** header "Forecast future sales" with "Models are trained on all 1,688 days of history and
predict the days after the last observed date, 15 Aug 2017…", a settings card, a model information card, and —
after clicking Generate — KPIs, chart, table and download button.

### Controls
| Control | Options | Default | What happens |
|---|---|---|---|
| **Model** (pills, key `fc_model`) | SARIMA, Exponential Smoothing, Linear Regression, Random Forest | The model with the lowest test RMSE (computed: Random Forest) | Updates the information card immediately. Does not change the forecast until Generate is clicked. |
| **Forecast horizon** (segmented, key `fc_horizon`) | 7, 14, 30 days | 14 days | Same: only takes effect after Generate. |
| **Generate forecast** (button) | — | — | Trains/loads the forecast and stores `{"model", "horizon"}` in `st.session_state["forecast"]`. |

If the selection differs from the forecast on screen, a note says "Showing the N-day X forecast. Click Generate
forecast to update." The forecast stays visible when other widgets change or when you switch pages and come back.

### Model information card
Shows the model's category badge (Statistical time series / Supervised machine learning), a plain description,
its settings, and its **test-period MAE and RMSE** (from the Model Comparison evaluation). Text comes from
`src/models.py → MODEL_INFO`.

### KPI cards (after Generate)
| Card | Meaning | Example (Random Forest, 14 days) |
|---|---|---|
| Forecast period | horizon and date range | 14 days, 16 Aug to 29 Aug 2017 |
| Total predicted | sum of forecasts; % vs the sum of the last N actual days | 11.14M, −2.6% vs last 14 days |
| Average per day | mean forecast | 795.9K |
| Peak day | highest forecast and its date | 946.5K, Sunday 20 Aug |

### Forecast chart
- **X-axis:** date (last 90 days of history + the forecast days). **Y-axis:** units sold per day.
- White solid line = actual sales (last 90 days). Dashed coloured line with dots = forecast (colour per model),
  connected to the last actual point. Grey band = forecast ± the model's test RMSE ("typical error", **not** a
  statistical confidence interval). Shaded region = future. Dotted vertical line = "Last observed day" (15 Aug 2017).
- **How to explain:** "Everything to the left of the dotted line is real data; everything to the right is our
  prediction. The forecast keeps the weekly pattern — peaks on Saturday and Sunday. The grey band shows the
  typical error we measured for this model on the test period."

### Predicted values table and download
- Table columns: Date (with weekday), Predicted units, Typical low (−RMSE), Typical high (+RMSE).
- **Download forecast (CSV)** gives `forecast_<model>_<N>d.csv` with columns
  `date, weekday, model, predicted_sales, typical_low, typical_high` (values rounded to whole units).
  The download does not rerun the app (`on_click="ignore"`).

### Behind the scenes
`get_forecast(name)` → `models.MODELS[name](series, 30)`: trains the model on **all 1,688 days** and predicts 30
days. Shorter horizons are the first 7 or 14 of those 30 days (this gives exactly the same numbers as forecasting 7
or 14 directly, because every model forecasts one step after another). Cached per model.
Files involved: train.csv only (the models use only past sales and the calendar).

---

## PAGE 4: MODEL COMPARISON  (`app.py → comparison()`)

**Purpose:** show which model forecasts best, measured fairly on data the models never saw.

**Header text (computed):** "Each model was trained on data up to 16 Jul 2017 and then forecast the next 30 days
(17 Jul to 15 Aug 2017) in one go, without seeing any of those days."

### KPI cards
One card per model, sorted by RMSE; value = RMSE, subtitle = MAE and MAPE. The best (lowest RMSE) is highlighted in
green with "Best model".
- Random Forest — RMSE 68.5K · MAE 49.6K · MAPE 5.9% (**Best model**)
- Linear Regression — RMSE 75.9K · MAE 57.8K · MAPE 6.7%
- Exponential Smoothing — RMSE 87.2K · MAE 57.9K · MAPE 6.7%
- SARIMA — RMSE 92.5K · MAE 61.4K · MAPE 7.2%

Below: a note naming the best model, noting if a different model had the lowest MAE (not the case now), and giving
the seasonal naive baseline (MAE 60,672, RMSE 91,696).

### Chart 12 — MAE and RMSE (two bar charts side by side)
- **X-axis:** model. **Y-axis:** error in units. Dotted horizontal line = seasonal naive baseline.
- **Conclusion:** bars below the dotted line beat the simple "repeat last week" rule. SARIMA is just above it.
- **How to explain:** "Lower is better. The dotted line is the error of simply copying last week. Random Forest,
  Linear Regression and Exponential Smoothing beat it; SARIMA does not on this test period."

### Results table
Columns: Model, Type, MAE, RMSE, MAPE (%), R². Includes the baseline row. The best row is highlighted.
Caption explains MAPE and R².

### Chart 13 — Actual vs predicted sales
- **Control:** pills to show/hide each model and the seasonal naive baseline (key `cmp_models`; default: the four models).
- **X-axis:** date (60 days before the test + the 30 test days). **Y-axis:** units sold.
- Grey = history, white with dots = actual test sales, dashed = each model's forecast, dotted = baseline. Shaded =
  test period.
- **How to explain:** "The white line is what really happened from 17 July to 15 August. The dashed lines are what
  each model predicted on 16 July without seeing any of these days."

### Chart 14 — Error by forecast day
- **X-axis:** days ahead (1–30). **Y-axis:** absolute error |actual − predicted| on that day.
- **Conclusion:** errors grow with the horizon for every model. Mean absolute error in days 1–10 vs days 21–30:
  SARIMA 38.5K → 79.0K; Exponential Smoothing 33.2K → 64.2K; Linear Regression 19.0K → 68.1K;
  Random Forest 27.1K → 65.2K. The largest single errors happen around 1 Aug and 12–13 Aug.
- **How to explain:** "The further ahead we predict, the larger the error. Linear Regression is actually the most
  accurate for the first ten days, but its errors grow faster."

### "How the evaluation works" expander
Bullet points: chronological split, same task for every model, no future information (recursive lags), settings
chosen before testing (six validation windows), closure-day handling, metric definitions.

**Behind the scenes:** `get_backtest()` → `src/models.py → backtest(series, 30)`; cached.

---

# PART 3: DATASET EXPLANATION

Source: Kaggle competition **"Store Sales – Time Series Forecasting"** (Corporación Favorita, Ecuador).
Files are downloaded manually into `data/`; nothing is generated or downloaded by the code.

### train.csv — 3,000,888 rows × 6 columns (121.8 MB)
One row per **store × product family × day**: 54 stores × 33 families × 1,684 dates = 3,000,888 (complete grid).
| Column | Type | Meaning | Used? |
|---|---|---|---|
| id | int | row number | no |
| date | date | the day (2013-01-01 → 2017-08-15) | yes |
| store_nbr | int | store number 1–54 | yes (store analysis, transactions join) |
| family | text | product category (33, e.g. GROCERY I, BEVERAGES) | yes (family analysis) |
| sales | float | units sold for that store/family/day; can be fractional because some products are sold by weight (Kaggle's description) | **yes — the target** |
| onpromotion | int | number of items in that family on promotion that day | yes (promotion analysis only) |
**Role:** main data and **the only file used to train the models.** Also used for EDA. No missing values, no duplicates. 31.3% of rows have sales = 0 (stores not open yet, families not sold in a store, etc.).

### test.csv — 28,512 rows × 5 columns
Same columns as train **without `sales`**: id, date, store_nbr, family, onpromotion. Covers 16–31 Aug 2017
(54 × 33 × 16 days). It is Kaggle's "future" period whose answers are hidden.
**Role in our project:** shown in the Datasets table; the notebook confirms it has no sales column. It is **not used
for training, evaluation or forecasting**, and the app does not use its dates for anything.

### stores.csv — 54 rows × 5 columns
| Column | Meaning |
|---|---|
| store_nbr | store number (key to join with train/transactions) |
| city | city (22 distinct) |
| state | state/province (16 distinct) |
| type | store type A–E (Kaggle does not define them) |
| cluster | group number of similar stores (17 distinct), as defined by Kaggle |
**Role:** EDA only — store performance chart.

### oil.csv — 1,218 rows × 2 columns
| Column | Meaning |
|---|---|
| date | trading day (2013-01-01 → 2017-08-31; weekdays only) |
| dcoilwtico | daily crude oil price, WTI benchmark, in USD; **43 values are missing** |
**Role:** EDA only (oil charts, correlation). Ecuador's economy depends on oil, which is why Kaggle includes it.

### holidays_events.csv — 350 rows × 6 columns
| Column | Meaning |
|---|---|
| date | date of the event (2012-03-02 → 2017-12-26) |
| type | Holiday, Transfer (the day a moved holiday is celebrated), Additional (extra days around a holiday), Bridge (extra day off joining a holiday to a weekend), Work Day (a weekend day worked to compensate a bridge), Event |
| locale | National, Regional or Local |
| locale_name | Ecuador (national), a state (regional) or a city (local) |
| description | name, e.g. "Navidad-1", "Traslado Batalla de Pichincha" |
| transferred | True if the holiday was officially moved to another day (so this date was a normal day) |
**Role:** EDA only (holiday effect chart). 38 dates have more than one row.

### transactions.csv — 83,488 rows × 3 columns
| Column | Meaning |
|---|---|
| date | day (2013-01-01 → 2017-08-15) |
| store_nbr | store |
| transactions | number of customer purchases (receipts) in that store that day |
**Role:** EDA only (transaction chart). It covers 91.8% of store-days in train; missing store-days are almost all days with zero sales (store closed or not yet open).

### How the files connect
```
train.csv ──store_nbr──► stores.csv          (many-to-one)
train.csv ──date + store_nbr──► transactions.csv
train.csv ──date──► oil.csv                  (chain-wide, by date)
train.csv ──date──► holidays_events.csv      (by date; locale_name could match stores.city/state — not used)
test.csv  = same structure as train, next 16 days, no sales
```

### Key terms
- **date** — calendar day; parsed to a pandas datetime.
- **store_nbr** — store ID (1–54).
- **family** — product category (33).
- **sales** — units sold (target after aggregation).
- **onpromotion** — items on promotion.
- **transactions** — number of receipts per store per day.
- **dcoilwtico** — WTI crude oil price per barrel (USD).
- **cluster** — Kaggle's grouping of similar stores.
- **locale** — whether a holiday is national, regional or local.
- **transferred** — the holiday was moved to another date.

### Target, dependent and independent variables
- **Target / dependent variable:** total units sold per day across all stores and families (`series` in the code).
- **Independent variables (inputs):**
  - for SARIMA and Exponential Smoothing: **only the past values of the target itself** (univariate models);
  - for Linear Regression and Random Forest: calendar features (dayofweek, day, month, year) and lag features
    (lag_7, lag_14, rolling_mean_28) derived from past sales.
- Promotions, oil, holidays, stores and transactions are **not** model inputs.

### Why aggregate to daily totals?
- **Simple:** 3 million rows become one clear line of 1,688 days — easier to model, explain and visualise.
- **Technical:** the aggregate series is smooth, has clear trend and weekly seasonality, and has no zero-inflation
  problems (individual store-family series are 31% zeros). Classical models like SARIMA and Holt-Winters work on one
  series at a time; fitting 1,782 separate series would be far beyond a college project's scope.
- **What is lost:** which store or family sells what; local effects (regional holidays, store openings); promotions
  per family. The forecast cannot tell a store manager how much Produce to order for store 44.
- **Why total daily sales:** it answers a chain-level planning question (overall demand) and is the most reliable
  series to forecast.

---

# PART 4: DATA CLEANING AND PREPROCESSING

All code is in `src/data.py`. The module docstring at the top lists every preprocessing decision.

### 4.1 Loading CSV files — `load_raw()`
- **Simple:** read each file into a table in memory.
- **Technical:** `pd.read_csv(path, parse_dates=["date"])` for each of the six files in the `FILES` dictionary.
  The path is built from the location of `src/data.py` (`DATA_DIR`), so it works no matter which folder you run from.
- **Example:** train.csv becomes a DataFrame of 3,000,888 rows × 6 columns.

### 4.2 Converting dates — `load_raw()`
- **Simple:** text like "2013-01-01" becomes a real date the computer understands.
- **Technical:** `load_raw` first reads only the header (`nrows=0`); if a `date` column exists it passes
  `parse_dates=["date"]`, producing `datetime64` values. stores.csv has no date column, so it is read as is.
- **Why needed:** grouping by day, resampling by week/month, extracting weekday/month, and time-ordered splitting all need real dates.

### 4.3 Missing values
- **Detection:** done in the **notebook** (`isnull().sum()` per file). Result: only oil.csv has missing values
  (43 prices). The app does not run a missing-value report.
- **Handling:**
  - train.csv has none;
  - missing *dates* in train (Christmas) → see 4.5;
  - missing oil prices → rows dropped for oil analysis (4.8);
  - NaNs created by reindexing the daily table → filled with 0 sales / 0 promotions, flagged as not observed.

### 4.4 Duplicates
- **Checked in the notebook** with `df.duplicated().sum()` → 0 duplicate rows in every file.
- **No duplicate removal is performed in the code** (there is nothing to remove). Do not claim we used `drop_duplicates()` — the current code does not.

### 4.5 The four missing Christmas dates — `daily_sales()` and `model_series()`
- **Simple:** stores are closed on 25 December, so there are no rows for those days. We record them as "closed,
  0 sales" instead of inventing numbers.
- **Technical:**
  1. `daily_sales` builds a full calendar with `pd.date_range(min, max, freq="D")` and `reindex`es the daily table.
     The 4 new rows (25 Dec 2013, 2014, 2015, 2016) get `sales = 0`, `observed = False`.
  2. A column `closed` is True when (month, day) is in `CLOSURE_DAYS = {(12, 25), (1, 1)}`.
  3. `model_series` creates the **model input**: each closure day is replaced by the value of the same weekday one
     week earlier (`s[d - 7 days]`), or one week later for the very first day (1 Jan 2013 has no history).
     Days are processed in date order, so 1 Jan 2014 takes the already-replaced 25 Dec 2013 value.
- **Real example (from the notebook):**
  | Date | Raw sales | Model input |
  |---|---|---|
  | 2013-01-01 | 2,512 | 318,348 (= 8 Jan 2013) |
  | 2013-12-25 | 0 (missing) | 527,481 (= 18 Dec 2013) |
  | 2016-12-25 | 0 (missing) | 1,260,472 (= 18 Dec 2016) |
- **Why:** a 0 in the training data is a huge one-off shock that would distort SARIMA, Holt-Winters and the lag
  features (e.g. lag_7 on 1 Jan would be 0). Since closure days are known in advance, a business would plan them
  separately.
- **Important:** the analysis charts use the raw values (with zeros); only the models use the replaced series.
  The test period (Jul–Aug 2017) contains no closure days, so **all reported scores are on real sales**.

### 4.6 Why Christmas and New Year are closures
- 25 December: no data at all for any store → all closed.
- 1 January: sales are 0.6%–2.1% of that year's average day (e.g. 2,512 units on 1 Jan 2013 vs a 2013 average of
  ~387K) → almost all stores closed.

### 4.7 Daily aggregation and groupby — `daily_sales()`
- **Simple:** add up all store-family sales for each day.
- **Technical:** `train.groupby("date").agg(sales=("sales","sum"), onpromotion=("onpromotion","sum"))`.
  `groupby` splits the rows into groups with the same date, applies `sum` to each group and combines the results
  into one row per date: 3,000,888 rows → 1,684 dates → 1,688 after adding the 4 Christmas days.
- **Example (example numbers):** if store 1 sold 10 units and store 2 sold 5 units of all families on a day, the daily total is 15.

### 4.8 Resampling (used)
- **Simple:** change the time step, e.g. from days to weeks.
- **Where:** Overview weekly/monthly views (`resample("W")`, `resample("MS")` then `.mean()`), transactions chart
  (`resample("W").mean()`), and `family_monthly` (`pd.Grouper(freq="MS")`). `model_series` also sets the daily
  frequency (`s.index.freq = "D"`) required by statsmodels.

### 4.9 Oil missing values — `oil_vs_sales()`
- **Simple:** oil is not traded on weekends and holidays, so we only compare oil and sales on days that have a price.
- **Technical:** `oil.dropna(subset=["dcoilwtico"])`, then an **inner** join with open-day sales (`pd.concat(..., join="inner")`) → 1,163 days. No interpolation or filling is used.
- **Why:** filled-in prices would be invented values.

### 4.10 Holidays — `national_holidays()`, `holiday_effects()`
- **Simple:** we only look at national holidays because our forecast is for the whole country.
- **Technical:** filter locale == "National", transferred == False, type in {Holiday, Transfer, Additional, Bridge}
  (excludes Work Day and Event); keep one row per date (`drop_duplicates("date")`).
- **Holidays are NOT matched to individual stores.** Regional/local holidays (which would match `locale_name` to
  `stores.state`/`stores.city`) are deliberately not used, because they affect only a few stores, not the
  chain-wide total. (A separate dataset check confirmed that every local/regional `locale_name` exists in
  stores.csv as a city or state, so store-level matching would be possible, but the code does not do it.)

### 4.11 Transactions — `daily_transactions()`
- Total transactions per day (groupby date, sum).
- Units per transaction: per store-day sales joined to transactions on `date` + `store_nbr` with an **inner** join and
  `validate="1:1"` (pandas raises an error if a key repeats), then summed per day and divided.

### 4.12 Promotions — `promotion_vs_sales()`
- Keeps days from 2014-04-01 (first day with any recorded promotion) that are not closure days → 1,227 days.

### 4.13 Store data — `store_sales()`
- Total sales per store ÷ days with sales > 0; then a `merge` with stores.csv on `store_nbr`, `validate="1:1"`.

### 4.14 Data types
- Dates → `datetime64`. sales → float. onpromotion, store_nbr, transactions, cluster → integers. family, city,
  state, type, locale, description → text. transferred → boolean. No manual type conversion is needed beyond date parsing.

### 4.15 Not done (do not claim these)
- No normalisation / standardisation / scaling (not needed for these models).
- No outlier removal (apart from the closure-day replacement for models).
- No interpolation of sales or oil.
- No duplicate removal (none existed).

### 4.16 Known data-quality issue: recording gaps
In several months of 2014 and early 2015, most stores recorded almost no Produce sales (e.g. Produce averaged ~235
units/day in Feb 2014 vs ~104,633 in Jan 2014); Beverages roughly halved in the same months. This causes the dips in
total sales visible in 2014–early 2015. The data is **not** corrected. A validation test of starting training in
June 2015 gave mixed results (better for Random Forest, worse for Linear Regression and Holt-Winters), so all models
keep the full history. This test was run during development; its code is not in the notebook.

### 4.17 Data leakage and how we prevent it
- **Simple:** leakage means the model accidentally "sees the answers" during training, so its test score looks
  better than it really is.
- **How we prevent it:**
  1. **Chronological split** — training ends 16 Jul 2017, testing starts 17 Jul 2017.
  2. **Shifted features** — lag_7 and lag_14 look 7 and 14 days back; the 28-day rolling mean is computed on
     `shift(1)`, so it never includes the day being predicted (`src/models.py → make_features`).
  3. **Recursive forecasting** — during the 30-day test forecast, lag features use the model's own earlier
     predictions, never actual test sales (`recursive_forecast`).
  4. **Model settings chosen on validation windows before the test period** — the test period was never used for tuning.
  5. **Proof in the notebook (section 6):** replacing all test-period sales with 0 leaves every model's forecast
     exactly unchanged.

---

# PART 5: EXPLORATORY DATA ANALYSIS

### 5.1 What is EDA and why do it first?
**EDA (Exploratory Data Analysis)** is studying the data with summaries and charts before modelling. It tells us
what the models must capture (trend, weekly seasonality), what data problems exist (missing Christmas days, oil gaps,
promotion recording starting in 2014, recording gaps in 2014–15), and which variables might matter.

### 5.2 Analyses implemented (app and notebook)
| Analysis | Where | Key finding |
|---|---|---|
| Daily sales | Overview chart; notebook `01_daily_sales.png` | Upward trend, strong weekly zig-zag, closure-day drops |
| Monthly | Analytics/Trends; `02_monthly_sales.png` | December peak every year; February weakest |
| Yearly | Analytics/Trends | 386.8K/day (2013) → 859.3K/day (2017) |
| Weekday | Analytics/Trends; `03_sales_by_weekday.png` | Sunday +29.6%, Saturday +20.8%, Thursday −20.7% vs average |
| Product families | Products & stores; `04_top_families.png` | Grocery I 32%, Beverages 20%, Produce 11% |
| Stores | Products & stores; `10_store_types.png` | Type A ≈ 2× other types; top stores in Quito |
| Holidays | Holidays & promotions; `09_holiday_effect.png` | Average +22% on national holidays; Good Friday −5.5% |
| Promotions | Holidays & promotions; `05_promotion_vs_sales.png` | Correlation 0.43 (from Apr 2014) |
| Oil | Oil & transactions; `06_oil_vs_sales.png` | Correlation −0.71 (trend-driven) |
| Transactions | Oil & transactions; `11_transactions.png` | ~84K per day, Christmas spikes, ~7.45 units each |
| Seasonal decomposition | Notebook only; `07_seasonal_decomposition.png` | Rising trend + stable weekly seasonal component |

The seasonal decomposition (statsmodels `seasonal_decompose`, additive, period 7) splits the series into trend,
weekly seasonal and residual parts. It is **only in the notebook**, not in the app.

### 5.3 Correlation
- **Simple:** a number from −1 to +1 saying how strongly two things move together in a straight-line way.
- **Formula (Pearson):** r = Σ(xᵢ − x̄)(yᵢ − ȳ) / √(Σ(xᵢ − x̄)² · Σ(yᵢ − ȳ)²). Computed with pandas `.corr()`.
- **Positive** (e.g. promotions 0.43): when one is higher, the other tends to be higher.
- **Negative** (e.g. oil −0.71): when one is higher, the other tends to be lower.
- **Near 0:** no linear relationship.

### 5.4 Why correlation is not causation
Two variables can move together because a third factor drives both. Here the third factor is **time**: from 2014 to
2017, sales rose for many reasons (more stores, growth), promotions were recorded more and more, and oil prices fell
for global reasons. So the oil correlation (−0.71) does not mean cheap oil makes people buy more, and the promotion
correlation (0.43) overstates the effect of promotions. Proving cause would need a controlled comparison.

### 5.5 Business insights
- Staff and stock up for weekends; Thursday is the quietest day.
- Plan for December and national holidays (Labour Day, Christmas season ~+38–45%); Good Friday is slightly below normal.
- Grocery I, Beverages and Produce drive two thirds of volume — forecasting errors there matter most.
- Type A stores (and Quito) are the highest-volume locations.
- Data quality must be checked: some family records are incomplete in 2014–15.

---

# PART 6: FEATURE ENGINEERING

Code: `src/models.py → make_features(sales)`; feature list `FEATURES = CALENDAR + LAGS`.

### 6.1 What is feature engineering?
Creating input columns from raw data that help a model learn. Linear Regression and Random Forest cannot read a
time series directly; they need a table where each row is one day and the columns describe that day.

### 6.2 Features actually used (7 features, shared by both supervised models)
| Feature | Meaning | Code |
|---|---|---|
| dayofweek | 0 = Monday … 6 = Sunday | `df.index.dayofweek` |
| day | day of month 1–31 | `df.index.day` |
| month | 1–12 | `df.index.month` |
| year | 2013–2017 (lets the model follow the trend) | `df.index.year` |
| lag_7 | sales exactly 7 days earlier (same weekday last week) | `sales.shift(7)` |
| lag_14 | sales 14 days earlier (same weekday two weeks ago) | `sales.shift(14)` |
| rolling_mean_28 | average of the previous 28 days (recent level) | `sales.shift(1).rolling(28).mean()` |

**Not used:** lag_1, rolling_mean_7, holidays, promotions, oil, store features. (A rolling average of 7 days is not in the final code.)

### 6.3 Encoding
- **Linear Regression:** `dayofweek` and `month` are **one-hot encoded** with `OneHotEncoder(drop="first")` inside a
  `ColumnTransformer`; the other 5 features pass through unchanged. Result: 6 weekday columns + 11 month columns + 5
  = **22 input columns**. `drop="first"` avoids the "dummy variable trap" (perfect redundancy with the intercept).
  Why: Sunday = 6 is not "6 times" Monday = 0; they are categories.
- **Random Forest:** uses the 7 features as plain numbers; trees split on thresholds, so no encoding is needed.

### 6.4 Lag features
- **Simple:** "what were sales on the same weekday last week?"
- **Why 7 and 14:** the strongest pattern is weekly. Last Sunday is the best simple guide to next Sunday.
  In our trained Random Forest, lag_7 carries **64%** of the feature importance and lag_14 **20%**.
- **Real example (15 Aug 2017, Tuesday):** actual sales 762,662; lag_7 = 717,766 (Tue 8 Aug); lag_14 = 988,528
  (Tue 1 Aug); rolling_mean_28 = 835,869 (average of 18 Jul – 14 Aug).

### 6.5 Rolling average and why it is shifted
- **Rolling average:** the mean of the last N values, moving forward one day at a time; it captures the current level.
- **Why `shift(1)` first:** without it, the 28-day window ending today would include today's sales — the value we are
  trying to predict. That is leakage. `shift(1)` moves the series one day back so the window covers yesterday and
  the 27 days before.
- **Example (example numbers):** sales for 4 days = 10, 20, 30, 40. A 3-day rolling mean *without* shift at day 4 =
  (20+30+40)/3 = 30 — it includes day 4's 40. *With* shift(1), day 4's value = (10+20+30)/3 = 20 — only past days.

### 6.6 Rows lost
`dropna()` removes the first 28 days (no full 28-day history). Training for the backtest therefore uses 1,630 rows
(29 Jan 2013 – 16 Jul 2017); training for the future forecast uses 1,660 rows.

### 6.7 How the features help
Calendar features tell the model *which kind of day* it is (weekend, December, which year). Lags tell it *what
similar days recently looked like*. The rolling mean tells it *the current level*. Together they turn forecasting
into ordinary regression.

---

# PART 7: THE FOUR MODELS

All in `src/models.py`. Every model function has the same signature: `model(history, horizon) → Series` — it trains
on `history` only and returns predictions for the `horizon` days after it. The same function is used for the
backtest and for future forecasts. Registry: `MODELS` dictionary; descriptions: `MODEL_INFO`.

## 7.1 SARIMA — `sarima(history, horizon)`

**A. Simple.** SARIMA predicts today's sales from (1) yesterday's sales, (2) the same weekday last week, and
(3) the mistakes it made recently, after first removing the weekly pattern by subtracting last week's value.

**B. Technical.** Seasonal AutoRegressive Integrated Moving Average, written SARIMA(p,d,q)(P,D,Q,s). Implemented with
`statsmodels.tsa.statespace.sarimax.SARIMAX` (with no external variables, so it is a pure SARIMA), fitted by maximum
likelihood.

**ARIMA vs SARIMA.** ARIMA(p,d,q) models short-term dependence. SARIMA adds the same three ideas at the seasonal lag
s, so it can model "this Sunday depends on last Sunday".

| Symbol | Meaning | Ours |
|---|---|---|
| p | number of autoregressive terms (past values) | 1 |
| d | number of ordinary differences (yₜ − yₜ₋₁) to remove trend | 0 |
| q | number of moving-average terms (past errors) | 1 |
| P | seasonal autoregressive terms (value s steps back) | 1 |
| D | seasonal differences (yₜ − yₜ₋ₛ) | 1 |
| Q | seasonal moving-average terms (error s steps back) | 1 |
| s | season length | 7 (days in a week) |

**Seasonal period 7** means the pattern repeats every 7 observations (days). Our weekday chart shows a strong,
stable weekly cycle, so s = 7.

**Differencing.** Replace each value by a difference to remove patterns. With D = 1 and s = 7 the model works on
zₜ = yₜ − yₜ₋₇ (today minus the same day last week). This removes the weekly pattern and most of the slow trend at the
same time, which is why d = 0 was best in validation.

**E. Formula (intuition).** On zₜ:
zₜ = φ₁·zₜ₋₁ + Φ₁·zₜ₋₇ + εₜ + θ₁·εₜ₋₁ + Θ₁·εₜ₋₇ (plus the cross term from multiplying the seasonal and non-seasonal parts).
Fitted values on the backtest training window: φ₁ (ar.L1) = 0.781, θ₁ (ma.L1) = −0.114, Φ₁ (ar.S.L7) = 0.016,
Θ₁ (ma.S.L7) = −0.930. The strong seasonal MA term means "correct heavily for last week's surprise".

**F. Parameters in our code.** `SARIMA_ORDER = (1, 0, 1)`, `SARIMA_SEASONAL_ORDER = (1, 1, 1, 7)`,
`SARIMA_WINDOW = 730` — trained only on the **last 730 days** of history (backtest: 18 Jul 2015 – 16 Jul 2017;
future: 17 Aug 2015 – 15 Aug 2017). Convergence warnings are suppressed.

**G/H. Training and prediction.** `SARIMAX(y, order, seasonal_order).fit(disp=False)` estimates the coefficients;
`fit.forecast(horizon)` produces all future days at once (multi-step forecast, internally each step builds on the
previous step's forecast).

**C. Why selected.** It is the standard statistical model for seasonal series and was required by our project description.

**Why these orders.** Chosen in the notebook (section 4) from 4 standard order combinations × 2 windows (365/730 days)
by average RMSE over six validation windows: (1,0,1)(1,1,1,7) with 730 days was best (validation MAE 68,538, RMSE 96,745).

**J. Advantages:** well-established theory; few parameters; handles trend and seasonality; interpretable.
**Disadvantages:** linear; assumes a stable pattern; uses no outside information (holidays); slower to fit; only one season length.

**K. On our data:** it came last on the test period (RMSE 92,511), slightly worse than repeating last week. Its
largest error was on 13 Aug 2017 (249K) — it predicted a normal weekend peak, but that weekend was lower than usual.
It also uses only 2 years of history.

**L/M.** statsmodels `SARIMAX`; `src/models.py → sarima`.

## 7.2 Exponential Smoothing (Holt-Winters) — `exponential_smoothing(history, horizon)`

**A. Simple.** The model keeps three running numbers that it updates every day: the current **level** (how much a
typical day sells now), the **trend** (how fast that level is changing) and a **weekday effect** for each of the 7
days. Recent days count more than old days. To forecast, it adds them up.

**B. Technical.** Triple exponential smoothing (Holt-Winters), `statsmodels.tsa.holtwinters.ExponentialSmoothing`,
parameters estimated by minimising squared one-step errors.

- **Level (ℓ):** smoothed base value.
- **Trend (b):** smoothed change of the level per day.
- **Seasonality (s):** smoothed weekday effect, 7 values.
- **Additive seasonality:** weekday effect is a fixed number of units ("Sunday +200K").
  **Multiplicative:** a percentage ("Sunday +30%").
- **Damped trend:** the trend is multiplied by φ (< 1) at each step ahead, so it fades out instead of growing forever.

**E. Formulas (additive seasonality, damped additive trend, season m = 7, α, β, γ, φ smoothing/damping parameters):**
- Level: ℓₜ = α(yₜ − sₜ₋ₘ) + (1 − α)(ℓₜ₋₁ + φbₜ₋₁)
- Trend: bₜ = β(ℓₜ − ℓₜ₋₁) + (1 − β)φbₜ₋₁
- Season: sₜ = γ(yₜ − ℓₜ₋₁ − φbₜ₋₁) + (1 − γ)sₜ₋ₘ
- Forecast h days ahead: ŷₜ₊ₕ = ℓₜ + (φ + φ² + … + φʰ)bₜ + sₜ₊ₕ₋ₘ (using the latest matching weekday effect)

**F. Parameters in our code:** `trend="add"`, `damped_trend=True`, `seasonal="add"`, `seasonal_periods=7`, trained on
the **full history**. Fitted values on the backtest training data: α (smoothing_level) = 0.649, β (smoothing_trend) =
0.0, γ (smoothing_seasonal) = 0.031, φ (damping_trend) = 0.80. φ = 0.80 is the lower limit statsmodels allows, so the
trend fades away within a few days: the forecast is essentially "current level + weekday effect".

**Additive vs multiplicative decision (notebook section 4):** the weekly swing is ~50% of the yearly level in every
year (2013: 0.515 … 2017: 0.497), which in theory suggests multiplicative seasonality. On the six validation windows,
damped additive/additive had the lowest MAE (67,759) and an RMSE practically tied with multiplicative (95,773 vs
95,730). Over a 30-day horizon the level hardly changes, so both behave alike; the simpler additive version was kept.

**G/H.** `.fit()` estimates α, β, γ, φ and the initial states; `.forecast(horizon)` gives all days at once.

**J. Advantages:** fast, simple, intuitive, adapts quickly to recent level changes. **Disadvantages:** one fixed
season; no outside variables; linear structure.
**K. On our data:** RMSE 87,234, third; almost the same MAE as Linear Regression (57,897 vs 57,777) but larger misses.

## 7.3 Linear Regression — `linear_regression(history, horizon)`

**A. Simple.** It learns a formula: predicted sales = a starting number + a weight × each feature. For example
"sales ≈ base + 0.6 × last week's sales + a Sunday bonus + …". It finds the weights that fit past days best.

**B. Technical.** Ordinary least squares, `sklearn.linear_model.LinearRegression` inside a `make_pipeline` with the
one-hot `ColumnTransformer` (Part 6.3).

- **Dependent variable:** daily total sales. **Independent variables:** the 22 encoded feature columns.
- **Prediction equation:** ŷ = b₀ + b₁·day + b₂·year + b₃·lag_7 + b₄·lag_14 + b₅·rolling_mean_28
  + Σ (weekday weights × weekday dummies) + Σ (month weights × month dummies).
- **Intercept b₀:** the prediction when all inputs are 0 (the baseline for the dropped Monday/January categories).
- **Coefficients bᵢ:** how much the prediction changes when that input increases by 1 (holding others fixed).
- **How it fits:** chooses b to minimise the sum of squared errors Σ(yᵢ − ŷᵢ)² (least squares; solved directly with linear algebra, no iterations).

**F. Parameters:** library defaults (with intercept). **I. Forecasting:** recursive (Part 9).
**C. Why selected:** the simplest supervised baseline; easy to explain.
**J. Advantages:** fast, transparent, can extrapolate a trend through `year`. **Disadvantages:** only straight-line
relationships; sensitive to its own recursive errors.
**K. On our data:** second on the test period (RMSE 75,944) and the most accurate model in the first 10 days
(MAE 19.0K), but its errors grow quickly. On the six validation windows it was the **worst** model (MAE 71,191) — a
sign that rankings change between periods.

## 7.4 Random Forest Regressor — `random_forest(history, horizon)`

**A. Simple.** A decision tree is a flowchart of yes/no questions ("Is lag_7 above 900K? Is it a Sunday?") ending in a
prediction. One tree is unstable, so we grow 300 trees, each on a different random sample of the past days, and
average their answers.

**B. Technical.** `sklearn.ensemble.RandomForestRegressor`, an ensemble of regression trees using **bagging**.
- **Decision tree:** splits the data on the feature/threshold that most reduces squared error, recursively; a leaf
  predicts the mean of its training rows.
- **Bootstrap sampling:** each tree trains on a random sample of rows drawn *with replacement* (`bootstrap=True`, default).
- **Combining:** the forest's prediction is the **average** of the 300 trees' predictions. Averaging reduces variance (overfitting) of individual trees.
- **n_estimators:** number of trees = 300.

**F. Parameters in our code:** `n_estimators=300`, `min_samples_leaf=2` (each leaf must contain at least 2 training
days — limits overfitting), `random_state=42` (reproducible), `n_jobs=-1` (use all CPU cores). Defaults: `max_depth=None`,
`max_features=1.0` (every split considers all 7 features — so the randomness comes from bootstrap sampling only),
`criterion="squared_error"`.

**Feature importance (backtest model):** lag_7 0.641, lag_14 0.200, day 0.070, rolling_mean_28 0.034, dayofweek 0.025,
month 0.023, year 0.006.

**J. Advantages:** captures non-linear patterns and interactions; robust; little tuning. **Disadvantages:** cannot
predict values outside the range seen in training (cannot extrapolate a trend); slower; less interpretable.
**K. On our data:** best on the test period on all metrics and also best on the validation windows (MAE 62,200).

## 7.5 Seasonal Naive baseline — `seasonal_naive(history, horizon)`
Not one of the four models; a reference. It repeats the last observed 7 days (`np.resize`). For the test period it
predicted the week 10–16 Jul 2017 repeated (804,159; 730,535; 748,800; 629,651; 765,490; 969,077; 1,100,804; …).

---

# PART 8: MODEL TRAINING AND EVALUATION

### 8.1 Training data, testing data, chronological split
- **Training data:** days the model learns from — **1 Jan 2013 to 16 Jul 2017** (1,658 days).
- **Testing data:** days held back to measure accuracy — **17 Jul 2017 to 15 Aug 2017** (last 30 days).
- **Chronological split:** in `backtest()`, `history = series.iloc[:-30]`, `test = series.iloc[-30:]`.
- **Why not shuffle:** shuffling would put future days into training, letting the model learn from the future
  (leakage) and making scores unrealistically good. In real use we only ever have the past.

### 8.2 Backtesting
Pretending we are standing on 16 Jul 2017, forecasting the next 30 days, and comparing with what really happened.

### 8.3 Fair comparison — `backtest(series, test_days=30)`
- same training cut-off for all models;
- same horizon (30 days);
- all models forecast **all 30 days in one go** (multi-step); supervised models use their own predictions as lags;
- same metric code (`metrics()`);
- table sorted by RMSE.

### 8.4 Six validation windows (model selection, notebook section 4)
Before testing, settings were chosen on six 30-day windows inside the training period:
18 Jan–16 Feb, 17 Feb–18 Mar, 19 Mar–17 Apr, 18 Apr–17 May, 18 May–16 Jun, 17 Jun–16 Jul 2017.
For each window the model is trained on all data before it and forecasts the whole window; errors are averaged.
This is "rolling-origin" validation. The test period is never touched.

Validation results of the final settings (average of 6 windows): Random Forest MAE 62,200 / RMSE 84,061 ·
Exponential Smoothing 67,759 / 95,773 · SARIMA 68,538 / 96,745 · Linear Regression 71,191 / 103,822.

### 8.5 Baseline and why seasonal naive
A baseline is the simplest reasonable forecast. "Repeat last week" already captures the weekly pattern for free. A
model is only worth using if it beats it. On the test period the baseline scored MAE 60,672 / RMSE 91,696.

### 8.6 Metrics — `src/models.py → metrics()`
Small example used below (example numbers): actual = [100, 120, 80], predicted = [110, 100, 80].
Errors (actual − predicted) = −10, 20, 0.

| Metric | Formula | Example | Better | Meaning |
|---|---|---|---|---|
| **MAE** | (1/n) Σ \|yᵢ − ŷᵢ\| | (10 + 20 + 0)/3 = **10** | lower | average size of a miss, in units |
| **RMSE** | √[(1/n) Σ (yᵢ − ŷᵢ)²] | √((100 + 400 + 0)/3) = √166.7 = **12.9** | lower | like MAE but punishes large misses more |
| **MAPE** | (100/n) Σ \|yᵢ − ŷᵢ\| / \|yᵢ\| | (10% + 16.7% + 0%)/3 = **8.9%** | lower | average error as % of actual |
| **R²** | 1 − Σ(yᵢ − ŷᵢ)² / Σ(yᵢ − ȳ)² | ȳ = 100; 1 − 500/800 = **0.375** | higher (max 1) | share of day-to-day variation explained; 0 = no better than predicting the average; can be negative |

MAE, RMSE and R² use scikit-learn (`mean_absolute_error`, `mean_squared_error` then `np.sqrt`, `r2_score`); MAPE is
computed with NumPy. R² is computed for **all** models, including the statistical ones.

### 8.7 Final results — verified
Values in `results/model_comparison.csv` and recomputed during this documentation (identical):

| Model | MAE | RMSE | MAPE | R² |
|---|---|---|---|---|
| Random Forest | 49,569 | 68,483 | 5.88% | 0.689 |
| Linear Regression | 57,777 | 75,944 | 6.73% | 0.618 |
| Exponential Smoothing | 57,897 | 87,234 | 6.71% | 0.495 |
| Seasonal Naive (baseline) | 60,672 | 91,696 | 7.03% | 0.443 |
| SARIMA | 61,381 | 92,511 | 7.17% | 0.433 |

The values given in the request (RF 49,569 / 68,483 / 0.689; LR 57,777 / 75,944 / 0.618; ES 57,897 / 87,234 / 0.495;
SARIMA 61,381 / 92,511 / 0.433) **match exactly**. Test-period actual sales averaged 832,844 per day (min 651,387,
max 1,123,752), so Random Forest's MAE is about 6% of a typical day.

### 8.8 Interpreting the results
- **Why Random Forest did best:** it learned non-linear relationships between lag_7/lag_14 and the next day, and it
  is robust to unusual weeks because it averages many trees. It was also best on the validation windows, so this
  is not just luck on one window. It still has large misses (up to 210K on 12 Aug).
- **Why SARIMA did worse:** it is linear, uses only 2 years of data, and its strong seasonal correction follows last
  week closely — so it behaves much like the seasonal naive baseline and also misses unusual weekends (13 Aug: −249K).
  It has no knowledge of holidays; the test period contains a national holiday (Independence Day, officially 10 Aug,
  moved to Fri 11 Aug in 2017). Note: we did **not** investigate whether the holiday caused the misses.
- **Why the baseline matters:** it shows that SARIMA added no value over "repeat last week" in this period, while the
  others did.
- **What the results mean:** on this 30-day window, typical daily errors are 6–7% of sales; Random Forest is the most
  accurate.
- **Not universally best:** one 30-day window; Linear Regression was 2nd on test but last on validation; results
  could change with another period, horizon, or with holiday features.
- **MAE vs RMSE gap:** Exponential Smoothing and Linear Regression have almost the same MAE, but ES has a much larger
  RMSE → ES makes fewer but bigger misses.

---

# PART 9: FUTURE FORECASTING

### 9.1 Backtesting vs future forecasting
| | Backtesting (Model Comparison page) | Future forecasting (Forecasting page) |
|---|---|---|
| Training data | up to 16 Jul 2017 | all data, up to 15 Aug 2017 |
| Predicted days | 17 Jul – 15 Aug 2017 (actual sales known) | 16 Aug – 14 Sep 2017 (no actual sales exist) |
| Can be scored? | Yes — MAE, RMSE, MAPE, R² | No |
| Code | `backtest()` / `get_backtest()` | `MODELS[name](series, 30)` / `get_forecast()` |

Kaggle's test.csv covers 16–31 Aug 2017 but has no sales, so even those days cannot be scored.

### 9.2 Selecting model and horizon
Pills for the model, segmented control for 7/14/30 days, then **Generate forecast** (Part 2, Page 3).

### 9.3 Future dates — `future_index()`
`pd.date_range(last_date + 1 day, periods=horizon, freq="D")` → starting 16 Aug 2017.

### 9.4 How each model forecasts
- **SARIMA / Exponential Smoothing:** the fitted model's `.forecast(horizon)`.
- **Linear Regression / Random Forest:** `recursive_forecast()`:
  1. take the last 28 days of history;
  2. add an empty row for the next date and compute its features with `make_features` (same code as training);
  3. predict that day;
  4. append the prediction to the history as if it were real;
  5. repeat for the next day.

### 9.5 Small recursive example (example numbers, using only lag_7 for simplicity)
Suppose the last 7 actual days (Mon–Sun) were 600, 550, 580, 500, 570, 760, 820 (thousands), and the model's rule is
"prediction = 1.02 × lag_7".
- Day 1 (Mon): lag_7 = last Monday's actual 600 → prediction 612.
- …
- Day 7 (Sun): lag_7 = last Sunday's actual 820 → prediction 836.4.
- **Day 8 (next Mon): lag_7 = Day 1, which is our prediction 612 (no actual exists)** → prediction 624.2.
So from day 8 onward predictions are built on predictions; with lag_14 this starts at day 15, and the 28-day rolling
mean gets one predicted value more each day. Any error made early is carried forward — which is why errors grow with
the horizon (Chart 14).

### 9.6 CSV download
`out` DataFrame (date, weekday, model, predicted_sales, typical_low, typical_high) → `to_csv` → `st.download_button`.
Typical low/high = forecast ∓ the model's test RMSE.

### 9.7 Display
Plotly chart with last 90 days of actual sales, the forecast, the ±RMSE band and the "Last observed day" marker;
table below. Results persist in `st.session_state["forecast"]`.

Example outputs (computed by the app): Random Forest 14 days total 11.14M (average 795.9K/day, peak 946.5K on Sunday 20
Aug); SARIMA 30 days total 25.22M (average 840.6K/day, peak about 1.1M on Sunday 10 Sep).

---

# PART 10: TECHNOLOGIES AND LIBRARIES

| Technology | What it does | Where used |
|---|---|---|
| **Python** | the programming language | everything |
| **Streamlit 1.66.1** | turns a Python script into a web app | `app.py` (pages, widgets, caching, session state), `src/ui.py` |
| **pandas 3.1.0** | tables (DataFrames), CSV reading, groupby, resample, joins | `src/data.py`, `src/models.py`, `app.py` |
| **NumPy 2.5.4** | fast numeric arrays and maths | metrics, recursion, notebook |
| **scikit-learn 1.9.1** | ML models and metrics | `LinearRegression`, `RandomForestRegressor`, `OneHotEncoder`, `ColumnTransformer`, `make_pipeline`, `mean_absolute_error`, `mean_squared_error`, `r2_score` |
| **statsmodels 0.15.0** | statistical models | `SARIMAX`, `ExponentialSmoothing`, `seasonal_decompose` (notebook); OLS trendline behind Plotly's `trendline="ols"` |
| **Plotly 7.1.0** | interactive charts | all dashboard charts (`plotly.express`, `graph_objects`, `make_subplots`) |
| **Matplotlib 3.11.2** | static charts | notebook only (PNG files in `results/`) |
| **Jupyter 1.1.1** | runs the notebook | `notebooks/sales_forecasting.ipynb` |
| Inter font | typography | loaded from Google Fonts via `.streamlit/config.toml` |

Seaborn was used by the original project but is no longer used or listed.

### Streamlit concepts
- **How it creates the UI:** each `st.something()` call adds an element (text, chart, button) to the page. When the
  user interacts with any widget, Streamlit **re-runs the script from top to bottom** and redraws the page.
- **Pages:** `st.navigation` + `st.Page(function)`; each page is a Python function.
- **Caching:** `@st.cache_data` stores a function's result keyed by its arguments. `get_data()` (≈3 s CSV loading),
  `get_backtest()` (training five models) and `get_forecast(name)` run once per server process; later reruns reuse
  the stored result. Without caching, every click would reload 122 MB of CSV and retrain models.
- **Session state:** `st.session_state` is a per-user dictionary that survives reruns. We store
  `st.session_state["forecast"] = {"model": …, "horizon": …}` so the forecast stays on screen. Widgets with
  `persist_state="session"` (keys `ov_view`, `an_years`, `an_families`, `an_store_group`, `an_tx`, `fc_model`,
  `fc_horizon`, `cmp_models`) keep their values when you switch pages.
- **requirements.txt:** lists exact tested versions so `pip install -r requirements.txt` recreates the same
  environment.
- **Running locally:** `streamlit run app.py` starts a local web server (default http://localhost:8501).

---

# PART 11: COMPLETE CODEBASE WALKTHROUGH

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI. Cached loaders (`get_data`, `get_backtest`, `get_forecast`), four page functions (`overview`, `analytics`, `forecasting`, `comparison`), error message for missing data, navigation. Contains no data-processing or model logic. |
| `src/__init__.py` | Marks `src` as a package (one-line docstring). |
| `src/data.py` | Loading and preprocessing: `load_raw`, `dataset_summary`, `daily_sales`, `model_series`, `national_holidays`, `holiday_effects`, `family_sales`, `family_monthly`, `store_sales`, `daily_transactions`, `oil_vs_sales`, `promotion_vs_sales`, `build_dataset`. Constants `DATA_DIR`, `FILES`, `CLOSURE_DAYS`, `HOLIDAY_TYPES`, `HOLIDAY_NAMES`. |
| `src/models.py` | Model settings constants; `make_features`, `recursive_forecast`, `future_index`, `_training_data`; models `linear_regression`, `random_forest`, `sarima`, `exponential_smoothing`, `seasonal_naive`; `MODELS`, `BASELINE`, `MODEL_INFO`; `metrics`, `backtest`. |
| `src/ui.py` | Design: colour constants, `MODEL_COLORS`, `PALETTE`, CSS; `inject_css`, `page_header`, `kpi`, `card`, `note`, `human`, `style`, `chart`. |
| `notebooks/sales_forecasting.ipynb` | Analysis report using the same `src` code: loading checks, preprocessing, EDA (11 plots), model selection on 6 validation windows, final evaluation, leakage checks, future forecast, conclusion. |
| `requirements.txt` | Pinned versions of the 8 libraries. |
| `README.md` | Project summary, structure, preprocessing, models, results, run instructions, limitations. |
| `.streamlit/config.toml` | Dark theme, font, radius, minimal toolbar, usage statistics off. |
| `.gitignore` | Excludes `data/`, CSVs (except `results/*.csv`), virtual environments, caches. |
| `results/` | Written by the notebook: 13 PNG plots, `model_comparison.csv`, `future_forecasts.csv`. The app does **not** read these files; it recomputes everything. |
| `data/` | The six Kaggle CSVs (not in git). |

### End-to-end trace
1. **User runs** `streamlit run app.py` and opens the browser.
2. `app.py` runs: `st.set_page_config`, `inject_css()`, then `get_data()` inside `try`.
3. **First run:** `get_data()` → `build_dataset()` → `load_raw()` reads six CSVs → `daily_sales()`, `model_series()`,
   `family_sales()`, `family_monthly()`, `store_sales()`, `holiday_effects()`, `daily_transactions()`,
   `oil_vs_sales()`, `promotion_vs_sales()` → a dictionary of small tables. Cached.
4. `st.navigation(...)` picks the page from the URL; the sidebar is drawn; `nav.run()` calls e.g. `overview()`.
5. **User clicks Forecasting:** Streamlit reruns the script; `forecasting()` calls `get_backtest()` (first time:
   `backtest(series, 30)` trains all five models on data to 16 Jul and scores them; cached) to find the default model
   and show test errors.
6. **User picks "SARIMA" and "30 days" and clicks Generate:** `get_forecast("SARIMA")` →
   `models.sarima(series, 30)` → fits SARIMAX on the last 730 days → 30 predictions from 16 Aug 2017 (cached).
   `st.session_state["forecast"] = {"model": "SARIMA", "horizon": 30}`.
7. The page slices the first 30 days, computes KPIs, builds the Plotly figure (`style()`, `chart()`), the table and
   the CSV download.
8. **User changes another widget or page:** the script reruns; cached results and session state keep the forecast.

---

# PART 12: VIVA QUESTIONS AND ANSWERS

### 1. Project introduction
1. **What is your project?** A Streamlit dashboard that analyses Favorita's 2013–2017 sales and forecasts total daily units sold with SARIMA, Holt-Winters, Linear Regression and Random Forest.
2. **What problem does it solve?** It predicts future demand so a retailer can plan stock, staff and finances.
3. **What exactly do you predict?** Total units sold per day across all 54 stores and 33 product families.
4. **Is it classification or regression?** Regression / time-series forecasting — the target is a continuous number.
5. **What are the outputs?** Charts and KPIs, a comparison table of MAE/RMSE/MAPE/R², and 7/14/30-day future forecasts downloadable as CSV.
6. **Who would use it?** Planners and managers at chain level (not individual stores).
7. **What is the architecture?** Streamlit app (`app.py`) + `src/data.py` (data) + `src/models.py` (models) + `src/ui.py` (style); no separate backend or database.
8. **Why Streamlit?** Pure Python, fast to build interactive data apps, built-in caching and widgets.

### 2. Dataset
9. **Which dataset?** Kaggle "Store Sales – Time Series Forecasting", Corporación Favorita, Ecuador, six CSV files.
10. **Size of train.csv?** 3,000,888 rows × 6 columns: 54 stores × 33 families × 1,684 dates.
11. **What is the target variable?** `sales`, summed per day.
12. **Which files train the models?** Only train.csv.
13. **What are the other files used for?** EDA: stores (store performance), oil (oil analysis), holidays (holiday effect), transactions (transaction trends); test.csv is listed but not used in modelling.
14. **Why is test.csv not used to evaluate?** It has no sales column; Kaggle hides the answers.
15. **Date range?** 1 Jan 2013 – 15 Aug 2017 (1,688 days including the 4 missing Christmas days).
16. **What is `onpromotion`?** Number of items in a family on promotion; recorded only from April 2014.
17. **What is `dcoilwtico`?** Daily WTI crude oil price in USD.
18. **What does aggregating lose?** Store- and family-level detail; we cannot forecast per store or product.

### 3. Data cleaning
19. **Missing values?** None in train; 43 missing oil prices; 4 missing dates in train (25 Dec).
20. **How did you treat the missing Christmas days?** Recorded as 0 sales and flagged closed — not interpolated. For model training only, replaced by the same weekday a week earlier.
21. **Why not interpolate?** Interpolation would invent sales on days the stores were closed.
22. **Why is 1 January also treated as a closure?** Sales are only 0.6–2.1% of a normal day.
23. **Doesn't replacing closure days change the data?** Only the model input; charts use raw values, and the test period contains no closure days, so scores use real sales.
24. **How did you handle oil gaps?** Used only real trading days (dropped missing, inner join); no filling.
25. **Duplicates?** Checked in the notebook — zero in every file; nothing removed.
26. **Did you scale the data?** No — tree models don't need it, statistical models work on raw values, and Linear Regression's predictions don't change with scaling.
27. **How are holidays matched to stores?** They are not; we only use national holidays because the forecast is chain-wide.
28. **What is groupby?** Split rows into groups by a key (date), apply a function (sum), combine into one row per group.

### 4. EDA
29. **Main patterns?** Upward trend, weekly seasonality (weekends high, Thursday low), December peak, holiday uplift.
30. **Busiest day of week?** Sunday (~828.6K average), +29.6% vs average.
31. **Top family?** Grocery I, 32% of units.
32. **Effect of national holidays?** +22% on average vs the same weekday in the previous 4 weeks; Good Friday −5.5%.
33. **Promotion correlation?** 0.43 from April 2014.
34. **Oil correlation?** −0.71.
35. **Does cheap oil cause more sales?** No evidence; both follow opposite time trends — correlation is not causation.
36. **Why compare holidays with the same weekday?** Otherwise weekend holidays would look good just because weekends are busy.
37. **What is seasonal decomposition?** Splitting the series into trend + seasonal + residual; done in the notebook (additive, period 7).

### 5. Feature engineering
38. **Which features?** dayofweek, day, month, year, lag_7, lag_14, rolling_mean_28.
39. **What is a lag feature?** A past value of the target used as input, e.g. sales 7 days ago.
40. **Why lag 7?** Strong weekly seasonality; it is the most important feature (64% importance in Random Forest).
41. **Why shift before rolling?** So the average never includes the day being predicted — prevents leakage.
42. **Why one-hot encode weekday and month for Linear Regression?** They are categories; numbers 0–6 would imply an order and equal steps.
43. **Why drop="first"?** To avoid the dummy-variable trap (redundancy with the intercept).
44. **Why no lag_1?** In recursive forecasting, a 1-day lag feeds the model's own errors straight back; it made Linear Regression much worse during development.
45. **How many training rows?** 1,630 for the backtest (first 28 days lost to the rolling mean).

### 6. Time-series forecasting
46. **What is a time series?** Values recorded at regular time intervals.
47. **Trend vs seasonality?** Trend = long-term direction; seasonality = regularly repeating pattern (weekly here).
48. **What is a forecast horizon?** How many steps ahead we predict: 7, 14 or 30 days.
49. **Why not shuffle?** It would leak future information into training.
50. **What is multi-step forecasting?** Predicting several future days from one starting point.
51. **What is recursive forecasting?** Predicting one day at a time and feeding each prediction back as an input.
52. **What is stationarity?** Constant mean/variance over time; differencing helps make a series stationary for ARIMA.

### 7. SARIMA
53. **What does SARIMA stand for?** Seasonal AutoRegressive Integrated Moving Average.
54. **Your orders?** (1,0,1)(1,1,1,7), last 730 days.
55. **p, d, q?** AR terms, ordinary differences, MA terms.
56. **P, D, Q, s?** Their seasonal versions at lag s; s = 7 for weekly.
57. **What does D = 1 do?** Works on today minus the same day last week, removing the weekly pattern.
58. **Why d = 0?** Seasonal differencing already removed most of the trend; d = 0 had the lowest validation error.
59. **How did you choose the orders?** Compared 4 standard combinations × 2 windows on six validation windows; chose the lowest average RMSE.
60. **Why did SARIMA perform worst?** Linear, univariate, no holiday information, follows last week closely; it missed unusual days such as 13 Aug and did slightly worse than the seasonal naive baseline.
61. **Library?** statsmodels `SARIMAX` (no exogenous variables).

### 8. Exponential Smoothing
62. **What is exponential smoothing?** A weighted average where recent observations get exponentially more weight.
63. **Holt-Winters components?** Level, trend, seasonality.
64. **Additive vs multiplicative seasonality?** Fixed amount vs percentage of the level.
65. **Which did you use and why?** Additive; it had the lowest validation MAE and an RMSE practically tied with multiplicative.
66. **What is a damped trend?** The trend is multiplied by φ < 1 each step so it flattens out; ours fitted φ = 0.80.
67. **Smoothing parameters?** Fitted automatically: α = 0.649 (level), β = 0.0 (trend), γ = 0.031 (season).

### 9. Linear Regression
68. **Equation?** ŷ = b₀ + Σ bᵢ·xᵢ.
69. **How are coefficients found?** Ordinary least squares — minimise the sum of squared errors.
70. **What is the intercept?** The prediction when all inputs are zero (baseline category).
71. **Limitations?** Only linear relationships; errors compound in recursive forecasting.
72. **How did it rank?** Second on test (RMSE 75,944), but last on validation — rankings vary between periods.

### 10. Random Forest
73. **What is a decision tree?** A sequence of yes/no splits on features ending in a predicted value.
74. **What is a random forest?** Many trees trained on bootstrap samples, predictions averaged.
75. **What is bootstrap sampling?** Sampling rows with replacement to create a different training set per tree.
76. **n_estimators?** Number of trees = 300.
77. **min_samples_leaf = 2?** Each leaf needs ≥ 2 training days, which limits overfitting.
78. **Why did it perform best?** Captures non-linear relationships with lag_7/lag_14, robust averaging; best on validation and test.
79. **Main weakness?** Cannot predict values outside the training range (no extrapolation of trends).
80. **Most important feature?** lag_7 (0.641), then lag_14 (0.200).

### 11. Model evaluation
81. **Which metrics?** MAE, RMSE, MAPE, R².
82. **MAE vs RMSE?** RMSE squares errors, so large misses count more; RMSE ≥ MAE always.
83. **Why rank by RMSE?** Large forecasting misses are costly (stock-outs or waste).
84. **What does R² = 0.689 mean?** Random Forest explains about 69% of the day-to-day variation in the test period.
85. **What is a baseline?** The simplest reasonable forecast; ours repeats the last week.
86. **Test period?** 17 Jul – 15 Aug 2017 (30 days); training up to 16 Jul 2017.
87. **How do you know there is no leakage?** Chronological split, shifted lags, recursive prediction, tuning before the test; the notebook shows forecasts are unchanged if test sales are set to 0.
88. **Is Random Forest always the best?** No — tested on one 30-day window; rankings change (LR was last on validation, second on test).

### 12. Streamlit and Python
89. **What does `st.cache_data` do?** Stores function results so data loading and training run once.
90. **What is session state?** A per-user dictionary that survives reruns; keeps the generated forecast visible.
91. **What happens when a widget changes?** Streamlit reruns the whole script; cached functions return stored results.
92. **Why requirements.txt?** Reproducible installation of exact library versions.
93. **How do you run it?** `pip install -r requirements.txt` then `streamlit run app.py`.

### 13. Limitations and future scope
94. **Main limitations?** One aggregate series; no holiday/promotion/oil features in models; recording gaps in 2014–15; single 30-day test window; no prediction intervals; forecasts beyond 15 Aug cannot be scored.
95. **Future scope?** Holiday and promotion features, per-store/family forecasting, rolling-origin evaluation for the final comparison, gradient boosting (e.g. LightGBM), proper prediction intervals, deployment.
96. **Is the ± band a confidence interval?** No — it is ± the test RMSE, a rough guide.

### Difficult follow-up questions
- **"If Random Forest can't extrapolate, how does it handle growth?"** Its inputs are lags, so it predicts relative to recent sales levels; but it cannot predict a value higher than anything in its training targets — a limitation for long horizons.
- **"Isn't replacing closure days also making up data?"** Yes, for the model input only, by a documented rule (same weekday last week); it affects 9 of 1,688 days, none in the test period, and the analysis keeps the real zeros.
- **"Why did SARIMA lose to repeating last week?"** Its strongest learned term (seasonal MA −0.93) makes it close to a seasonal naive model; on this window the extra terms added noise. On validation it did beat the baseline.
- **"Why not use promotions as a feature, they're known in advance?"** True, it would be a good improvement; we kept the models univariate + calendar for simplicity and listed it as future work.
- **"Why 30 days for testing?"** It matches the longest forecast horizon offered (30 days) and is a realistic planning horizon.
- **"Your R² is only 0.69 — is that bad?"** R² on a 30-day window measures how well daily ups and downs are captured; MAPE of 5.9% shows predictions are typically within about 6% of actual sales.
- **"How did you pick 300 trees?"** It is a sensible default-sized ensemble with stable results and acceptable training time (~2 s); it was not tuned.
- **"Did you tune Random Forest's hyperparameters?"** No grid search; only the feature set was compared on validation windows.

---

# PART 13: PRESENTATION SCRIPTS

### 13.1 One-minute introduction
"Our project is Retail Sales Analysis and Time Series Forecasting. We used real data from Corporación Favorita, a
supermarket chain in Ecuador — 3 million sales records from 2013 to 2017 across 54 stores. We built a dashboard that
analyses what drives sales and forecasts total daily sales for the next 7, 14 or 30 days. We compared two statistical
models, SARIMA and Holt-Winters exponential smoothing, with two machine-learning models, Linear Regression and Random
Forest, on the same 30-day test period using MAE and RMSE. Random Forest was the most accurate, with an average error
of about 6% of daily sales."

### 13.2 Three-minute explanation
1. **Problem (30 s):** retailers must plan stock and staff before demand is known.
2. **Data (40 s):** six Kaggle files; we summed 3 million rows into 1,688 daily totals. Christmas is missing because
   stores are closed — we recorded it as closed instead of inventing sales.
3. **Patterns (40 s):** growth from ~387K to ~859K units per day; strong weekly cycle with Sunday +30%; December
   peak; national holidays +22%.
4. **Models (40 s):** SARIMA with a 7-day season, Holt-Winters with level/trend/weekday; Linear Regression and
   Random Forest using calendar and lag features, forecasting recursively.
5. **Results (30 s):** trained to 16 July, tested on 17 July–15 August: Random Forest RMSE 68K, Linear Regression 76K,
   Holt-Winters 87K, SARIMA 93K; a repeat-last-week baseline scored 92K.

### 13.3 Five-minute technical explanation
Add to the 3-minute version:
- `src/data.py` pipeline: `load_raw` (parse dates) → `daily_sales` (groupby + full calendar + closed flag) →
  `model_series` (closure replacement for training only).
- Features: `make_features` with lag_7, lag_14 and a 28-day rolling mean computed after `shift(1)`; one-hot weekday
  and month for Linear Regression.
- Recursive forecasting: each prediction becomes a lag for later days — no actual future sales are used.
- Model selection: six 30-day rolling validation windows from January to July 2017; additive damped Holt-Winters,
  SARIMA(1,0,1)(1,1,1,7) on 730 days, shared feature set.
- Leakage proof: forecasts unchanged when test sales are set to zero.
- Streamlit: cached data and models, session state for persistent forecasts.

### 13.4 Live demonstration script
1. **Overview:** "Here are four headline numbers…" (Part 2). Click **Weekly** and **Monthly** to show smoothing.
   Point at the red crosses: "closure days".
2. **Sales Analytics → Trends:** deselect years with the pills to show a single year. Point at December and Sunday.
3. **Products & stores:** add DAIRY to the family dropdown; show the Produce data-gap note. Switch **Group by** to City.
4. **Holidays & promotions:** explain the same-weekday baseline and the colours in the promotion scatter.
5. **Oil & transactions:** "correlation −0.71, but it's two opposite trends". Toggle **Units per transaction**.
6. **Model Comparison:** cards → bars with dotted baseline → table → actual vs predicted (toggle models off/on) →
   error by day ("errors grow with the horizon") → open "How the evaluation works".
7. **Forecasting:** choose Random Forest, 14 days, **Generate** → explain dotted line, band, KPIs → **Download CSV** →
   switch to SARIMA 30 days → Generate → compare.

### 13.5 Explaining why Random Forest performed best
"Random Forest combines 300 decision trees, so it can learn non-linear relationships such as 'if last Sunday was
very high, this Sunday is high but slightly less'. Its main inputs are the same weekday one and two weeks ago. It was
best both on the six validation windows and on the final test, so it is consistently strong on our data — but we only
tested one 30-day period, so we don't claim it is always the best."

### 13.6 Answering "why did SARIMA perform poorly?"
"SARIMA is linear and uses only past sales, with no holiday information. Its strongest learned term makes it follow
last week's pattern closely, so on this test period it was essentially as good as just repeating last week, and it
over-predicted an unusually quiet weekend on 12–13 August. It did beat that baseline on the earlier validation windows,
so it is a reasonable model, just not the best on this data."

### 13.7 Stating limitations honestly
"We forecast only the chain-wide total, not per store. Holidays and promotions are analysed but not used as model
inputs. Some product families have recording gaps in 2014 and 2015. Our final comparison uses one 30-day test window,
and forecasts after 15 August 2017 can't be scored because the real sales are not public. The band on the forecast
chart is a rough error guide, not a statistical confidence interval."

---

# PART 14: THINGS WE MUST NOT MISREPRESENT

### What the application actually does
- Loads six Kaggle CSVs; analyses sales by time, weekday, family, store, national holiday, promotion, oil and transactions.
- Trains four models (+ a seasonal naive baseline) on total daily sales.
- Evaluates them on the last 30 days (17 Jul – 15 Aug 2017) with MAE, RMSE, MAPE and R².
- Forecasts 7, 14 or 30 days from 16 Aug 2017 and lets you download the forecast.

### What it does NOT do
- No per-store or per-family forecasts.
- No database, user login, API or cloud deployment.
- No model files are saved; models are retrained when the server starts and cached in memory.
- No hyperparameter grid search (only a small comparison of a few configurations on validation windows).
- No cross-validation in the scikit-learn sense; the final comparison uses one test window (rolling windows were used only for selection, in the notebook).
- No prediction/confidence intervals (the band is ± test RMSE).
- No forecasts scored on the future period, and no Kaggle submission.
- No deep learning (no LSTM, no neural networks).
- No data scaling, outlier removal (other than closure replacement for models) or duplicate removal.
- test.csv is not used for training or evaluation.

### CSVs used for analysis
train (everything), stores (store chart), oil (oil charts), holidays_events (holiday chart), transactions (transaction
chart), test (listed in the Datasets table only).

### Features used for model training
Linear Regression and Random Forest: dayofweek, day, month, year, lag_7, lag_14, rolling_mean_28 (weekday and month
one-hot encoded for Linear Regression). SARIMA and Holt-Winters: past daily sales only.

### Models actually trained
SARIMA(1,0,1)(1,1,1,7) on the last 730 days; Holt-Winters (additive damped trend, additive weekly seasonality) on the
full history; Linear Regression (OLS); Random Forest (300 trees, min_samples_leaf 2). Seasonal naive is a reference rule, not a trained model.

### Metrics actually calculated
MAE, RMSE, MAPE (%), R² — for all models including the baseline.

### Incorrect claims to avoid in the viva
| ❌ Don't say | ✅ Say instead |
|---|---|
| "We predict sales for each store / product." | We predict the chain-wide daily total. |
| "We used holidays/promotions/oil to train the models." | We analysed them; the models use only past sales and the calendar. |
| "We interpolated missing values." | Closure days are recorded as 0; for training only they are replaced by the same weekday last week. Oil gaps are dropped. |
| "We removed duplicates." | We checked: there were none. |
| "We normalised/scaled the data." | No scaling was needed or used. |
| "We used cross-validation / GridSearchCV." | We used six rolling validation windows to pick a few settings, then one 30-day test. |
| "Random Forest is the best model for retail forecasting." | It was the best on our 30-day test period (and our validation windows). |
| "The shaded band is a 95% confidence interval." | It is ± the model's test RMSE, a rough guide. |
| "Our future forecasts are 94% accurate." | Future forecasts cannot be scored; the test-period MAPE was 5.9% for Random Forest. |
| "SARIMA failed." | SARIMA was slightly worse than the seasonal naive baseline on this test period but beat it on validation. |
| "test.csv was used as our test set." | Our test set is the last 30 days of train.csv; Kaggle's test.csv has no sales. |
| "Oil prices cause sales changes." | They are correlated (−0.71) because of opposite time trends; no causation shown. |
| "We used deep learning / LSTM." | No neural networks are used. |
| "The app saves trained models." | Models are retrained when the server starts and cached in memory. |
