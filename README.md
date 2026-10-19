# Retail Sales Analysis and Time Series Forecasting

A simple college ML project that analyses daily retail sales and forecasts the next 30 days.

## Dataset
Kaggle: [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)

Files used (put them in `data/`): `train.csv`, `oil.csv`, `holidays_events.csv`.

## Steps
1. **Load data**: read `train.csv`, check shape, info, missing values and duplicates.
2. **Cleaning**: drop duplicates, aggregate to daily total sales, fill missing dates by interpolation, merge oil price.
3. **EDA**: daily trend, monthly sales, day of week, top 10 product families, promotion vs sales, oil vs sales, seasonal decomposition. Plots are saved in `results/`.
4. **Models**: the last 30 days are the test set.
5. **Evaluation**: MAE and RMSE for each model, plus a plot of actual vs predicted sales.

## Models
We predict the **total units sold per day** across all stores and product families.

- **Linear Regression** with features: day of week, month, day, `lag_7`, `rolling_mean_7`
- **SARIMA** `(1,1,1)x(1,1,1,7)`, trained on the last 1 year
- **Random Forest** (200 trees) on the same features as Linear Regression

## Results (last 30 days)

| Model             | MAE    | RMSE    |
|-------------------|--------|---------|
| Random Forest     | 54,050 | 63,279  |
| SARIMA            | 60,118 | 90,266  |
| Linear Regression | 80,990 | 101,264 |

Random Forest is the best model. Linear Regression and Random Forest use recent actual sales as features (one-step-ahead), while SARIMA forecasts all 30 days at once.

## How to run
```bash
pip install -r requirements.txt
jupyter notebook notebooks/sales_forecasting.ipynb
```
Then run all cells.

### Demo UI
A simple Streamlit web app with 3 tabs (Overview, EDA, Forecast). In the Forecast tab you can pick the test period and the models, then compare their predictions and errors.
```bash
streamlit run app.py
```
