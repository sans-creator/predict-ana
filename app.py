"""Retail Sales Analysis and Time Series Forecasting - Streamlit dashboard.

Run from the project root:  streamlit run app.py

Data processing lives in src/data.py, the models in src/models.py and the
visual helpers in src/ui.py. This file only builds the pages.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src import models
from src.data import FILES, build_dataset
from src.ui import (ACCENT, HISTORY, MODEL_COLORS, MUTED, NEGATIVE, PALETTE, POSITIVE, TEXT,
                    card, chart, human, inject_css, kpi, note, page_header, style)

st.set_page_config(page_title="Retail Sales Forecasting", page_icon=":material/insights:", layout="wide")
inject_css()

TEST_DAYS = 30           # evaluation period: the last 30 observed days
HORIZONS = [7, 14, 30]   # future forecast lengths offered to the user
FOUR = list(models.MODELS)


# ---------- Cached computations ----------
@st.cache_data(show_spinner="Loading and preparing the Kaggle data...")
def get_data():
    return build_dataset()


@st.cache_data(show_spinner="Training all models and forecasting the test period...")
def get_backtest():
    return models.backtest(get_data()["series"], TEST_DAYS)


@st.cache_data(show_spinner=False)
def get_forecast(name):
    """Forecast of the longest horizon from the last observed day.
    Shorter horizons are its first days, so each model is trained only once."""
    return models.MODELS[name](get_data()["series"], max(HORIZONS))


def short_date(d):
    return f"{d:%d %b %Y}"


# ---------- Page: Overview ----------
def overview():
    d = get_data()
    daily = d["daily"]
    start, end = daily.index[0], daily.index[-1]
    page_header("Overview", "Retail Sales Analysis &amp; Forecasting",
                f"Daily unit sales of Corporación Favorita, an Ecuadorian supermarket chain: "
                f"{d['n_stores']} stores and {d['n_families']} product families, "
                f"{start:%B %Y} to {end:%B %Y}.")

    open_days = daily[~daily["closed"]]
    peak = daily["sales"].idxmax()
    family = d["family"]
    c = st.columns(4)
    kpi(c[0], "Total sales", human(daily["sales"].sum(), 2), f"units over {len(daily):,} days")
    kpi(c[1], "Average daily sales", human(open_days["sales"].mean()), f"across {len(open_days):,} trading days")
    kpi(c[2], "Highest sales day", human(daily.loc[peak, "sales"], 2), f"{peak:%A, %d %b %Y}")
    kpi(c[3], "Top product family", family.index[0].title(),
        f"{family.iloc[0] / family.sum():.0%} of all units sold")
    st.write("")

    with card("trend", "Sales trend",
              "Total units sold per day across all stores. Weekly and monthly views show the average day."):
        view = st.segmented_control("View", ["Daily", "Weekly", "Monthly"], default="Daily", required=True,
                                    key="ov_view", label_visibility="collapsed", persist_state="session")
        s = daily["sales"]
        fig = go.Figure()
        if view == "Daily":
            fig.add_scatter(x=s.index, y=s, name="Daily sales", line=dict(color=HISTORY, width=1), opacity=0.6)
            fig.add_scatter(x=s.index, y=s.rolling(30).mean(), name="30-day average",
                            line=dict(color=ACCENT, width=2.5))
            closed = daily[daily["closed"]]
            fig.add_scatter(x=closed.index, y=closed["sales"], mode="markers", name="Stores closed (25 Dec, 1 Jan)",
                            marker=dict(color=NEGATIVE, size=8, symbol="x"))
        else:
            rule = "W" if view == "Weekly" else "MS"
            r = open_days["sales"].resample(rule).mean()
            fig.add_scatter(x=r.index, y=r, name=f"Average daily sales per {view[:-2].lower()}",
                            line=dict(color=ACCENT, width=2.5, shape="spline"),
                            fill="tozeroy", fillcolor="rgba(129,140,248,0.10)")
        fig.update_yaxes(title="Units sold")
        chart(style(fig, 400))

    left, right = st.columns([1, 1.25])
    with left, card("about", "About the project"):
        st.markdown(f"""
<ul class="clean">
<li><b>Goal:</b> forecast the total number of units sold per day across the whole chain.</li>
<li><b>Data:</b> six Kaggle CSV files, including {d['summary'].loc[0, 'Rows']:,} store-family-day sales rows.</li>
<li><b>Models:</b> SARIMA and Holt-Winters exponential smoothing (statistical), Linear Regression
and Random Forest (supervised machine learning).</li>
<li><b>Evaluation:</b> every model forecasts the same last {TEST_DAYS} days from the same starting point,
scored with MAE and RMSE.</li>
<li><b>Forecasting:</b> 7 to 30 days into the future, starting {short_date(end + pd.Timedelta(days=1))}.</li>
</ul>""", unsafe_allow_html=True)
    with right, card("datasets", "Datasets", "All six files from the Kaggle Store Sales competition."):
        summary = d["summary"].copy()
        for col in ["From", "To"]:  # stores.csv has no date column
            summary[col] = [f"{v:%d %b %Y}" if v is not None else "—" for v in summary[col]]
        st.dataframe(summary, hide_index=True, column_config={
            "Rows": st.column_config.NumberColumn(format="localized")})


# ---------- Page: Sales analytics ----------
def analytics():
    d = get_data()
    daily = d["daily"]
    open_days = daily[~daily["closed"]]
    page_header("Sales analytics", "What drives sales?",
                "Trends, product families, stores, holidays, promotions, oil prices and transactions.")
    trends, products, holidays, external = st.tabs(
        ["Trends", "Products & stores", "Holidays & promotions", "Oil & transactions"])

    with trends:
        years = sorted(open_days.index.year.unique())
        with card("seasonality", "Average daily sales by month",
                  "Each line is one year. 2017 ends on 15 August."):
            chosen = st.pills("Years", years, selection_mode="multi", default=years, key="an_years",
                              label_visibility="collapsed", persist_state="session")
            m = open_days.groupby([open_days.index.year, open_days.index.month])["sales"].mean()
            m.index.names = ["year", "month"]
            m = m.reset_index()
            m = m[m["year"].isin(chosen or years)]
            m["year"] = m["year"].astype(str)
            fig = px.line(m, x="month", y="sales", color="year", markers=True,
                          color_discrete_sequence=PALETTE)
            fig.update_xaxes(tickvals=list(range(1, 13)),
                             ticktext=pd.date_range("2000-01-01", periods=12, freq="MS").strftime("%b"), title="")
            fig.update_yaxes(title="Avg units per day")
            fig.update_traces(hovertemplate="%{y:,.0f}")
            chart(style(fig, 360))

        c1, c2 = st.columns(2)
        with c1, card("yearly", "Average daily sales by year", "Averages keep the partial year 2017 comparable."):
            y = open_days.groupby(open_days.index.year)["sales"].mean()
            growth = y.pct_change() * 100
            fig = go.Figure(go.Bar(x=y.index.astype(str), y=y, marker_color=ACCENT,
                                   text=[human(v) for v in y], textposition="outside",
                                   customdata=growth, hovertemplate="%{y:,.0f} units/day<br>%{customdata:+.1f}% vs previous year<extra></extra>"))
            fig.update_yaxes(title="Avg units per day", range=[0, y.max() * 1.15])
            chart(style(fig, 320, legend=False, hover="closest"))
        with c2, card("weekday", "Average sales by weekday", "Weekends are the busiest days."):
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            w = open_days.groupby(open_days.index.day_name())["sales"].mean().reindex(days)
            fig = go.Figure(go.Bar(x=[x[:3] for x in days], y=w, text=[human(v) for v in w], textposition="outside",
                                   marker_color=[ACCENT if x in ("Saturday", "Sunday") else "#475569" for x in days],
                                   hovertemplate="%{y:,.0f} units/day<extra></extra>"))
            fig.update_yaxes(title="Avg units per day", range=[0, w.max() * 1.15])
            chart(style(fig, 320, legend=False, hover="closest"))

    with products:
        family = d["family"]
        c1, c2 = st.columns([1, 1.2])
        with c1, card("families", "Top product families", "Share of all units sold, 2013 to 2017."):
            top = family.head(12)[::-1]
            share = top / family.sum() * 100
            fig = go.Figure(go.Bar(x=top, y=[f.title() for f in top.index], orientation="h", marker_color=ACCENT,
                                   text=[f"{v:.1f}%" for v in share], textposition="outside",
                                   hovertemplate="%{y}: %{x:,.0f} units<extra></extra>"))
            fig.update_xaxes(title="Units sold", range=[0, top.max() * 1.18])
            chart(style(fig, 420, legend=False, hover="closest", left=150))
        with c2, card("family_trend", "Product family trends", "Average daily units sold per month."):
            options = list(family.index)
            picked = st.multiselect("Product families", options, default=options[:3], key="an_families",
                                    format_func=str.title, max_selections=6, label_visibility="collapsed",
                                    persist_state="session")
            fm = d["family_monthly"]
            fm = fm[fm["family"].isin(picked)].assign(family=lambda x: x["family"].str.title())
            fig = px.line(fm, x="date", y="sales", color="family", color_discrete_sequence=PALETTE)
            fig.update_traces(hovertemplate="%{y:,.0f}")
            fig.update_yaxes(title="Avg units per day")
            fig.update_xaxes(title="")
            chart(style(fig, 358))
            # months far below a family's usual level are recording gaps in train.csv
            typical = fm.groupby("family")["sales"].transform("median")
            gaps = fm[fm["sales"] < 0.2 * typical]
            if not gaps.empty:
                names = sorted(gaps["family"].unique())
                note(f"Sharp drops for {', '.join(names)} come from the source data: in some months most stores "
                     f"recorded little or no sales for {'this family' if len(names) == 1 else 'these families'}.")

        with card("stores", "Store performance",
                  "Average daily units sold per store, counting only days on which the store had sales, "
                  "so stores that opened after 2013 are not penalised."):
            stores = d["stores"]
            group = st.segmented_control("Group by", ["Store type", "City", "Cluster", "Store"], default="Store type",
                                         required=True, key="an_store_group", persist_state="session")
            column = {"Store type": "type", "City": "city", "Cluster": "cluster", "Store": "store_nbr"}[group]
            g = (stores.groupby(column).agg(avg=("avg_daily_sales", "mean"), stores=("store_nbr", "size"))
                 .sort_values("avg", ascending=False).head(20))
            labels = [f"Store {i}" if group == "Store" else
                      (f"Cluster {i}" if group == "Cluster" else
                       (f"Type {i}" if group == "Store type" else str(i))) for i in g.index]
            fig = go.Figure(go.Bar(x=labels, y=g["avg"], marker_color=ACCENT, customdata=g["stores"],
                                   hovertemplate="%{x}<br>%{y:,.0f} units/day per store<br>%{customdata} store(s)<extra></extra>"))
            fig.update_yaxes(title="Avg units per day per store")
            chart(style(fig, 340, legend=False, hover="closest"))
            if group == "Store":
                note("Showing the 20 stores with the highest average daily sales.")

    with holidays:
        c1, c2 = st.columns(2)
        h = d["holidays"]
        with c1, card("holiday", "National holiday effect",
                      "Sales on each holiday compared with the same weekday in the four previous weeks. "
                      "Closure days (25 Dec, 1 Jan) are left out."):
            e = h.groupby("holiday").agg(uplift=("uplift_pct", "mean"), n=("uplift_pct", "size")).sort_values("uplift")
            fig = go.Figure(go.Bar(x=e["uplift"], y=e.index, orientation="h", customdata=e["n"],
                                   marker_color=[POSITIVE if v >= 0 else NEGATIVE for v in e["uplift"]],
                                   text=[f"{v:+.0f}%" for v in e["uplift"]], textposition="outside",
                                   hovertemplate="%{y}: %{x:+.1f}% vs a normal day<br>%{customdata} day(s)<extra></extra>"))
            fig.update_xaxes(title="Change vs a normal day (%)", ticksuffix="%",
                             range=[min(e["uplift"].min() * 1.6, -15), e["uplift"].max() * 1.25])
            chart(style(fig, 440, legend=False, hover="closest", left=160))
            note(f"On average, national holidays sell <b>{h['uplift_pct'].mean():+.0f}%</b> compared with a "
                 f"normal day ({len(h)} holiday dates). Only national holidays are used because regional and "
                 "local holidays affect a few stores, not the chain-wide total.")
        with c2, card("promo", "Promotions vs sales",
                      "Each dot is one day. Promotions are only recorded from April 2014."):
            p = d["promo"].copy()
            p["year"] = p.index.year.astype(str)
            corr = p["onpromotion"].corr(p["sales"])
            fig = px.scatter(p.reset_index(), x="onpromotion", y="sales", color="year", opacity=0.6,
                             trendline="ols", trendline_scope="overall", trendline_color_override=TEXT,
                             color_discrete_sequence=PALETTE,
                             labels={"onpromotion": "Items on promotion", "sales": "Units sold"})
            fig.update_traces(marker=dict(size=5))
            chart(style(fig, 440, hover="closest"))
            note(f"Correlation: <b>{corr:.2f}</b>. Promotions and sales both grew over the years "
                 "(see the colours), so part of this relationship is the shared upward trend, not cause and effect.")

    with external:
        oil, joined = d["oil_price"], d["oil_join"]
        with card("oil_time", "Oil price and sales over time",
                  "Ecuador's economy depends on oil. Oil prices are shown on trading days only."):
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_scatter(x=oil.index, y=oil, name="Oil price (USD)", line=dict(color="#f59e0b", width=1.8))
            sales30 = open_days["sales"].rolling(30, min_periods=20).mean()
            fig.add_scatter(x=sales30.index, y=sales30, name="Sales, 30-day average",
                            line=dict(color=ACCENT, width=2.2), secondary_y=True)
            fig.update_yaxes(title="Oil price (USD)", secondary_y=False)
            fig.update_yaxes(title="Units per day", secondary_y=True, showgrid=False, tickformat=".2s")
            chart(style(fig, 340, right=72))

        c1, c2 = st.columns(2)
        with c1, card("oil_scatter", "Oil price vs daily sales"):
            j = joined.copy()
            j["year"] = j.index.year.astype(str)
            corr = j["oil"].corr(j["sales"])
            fig = px.scatter(j.reset_index(), x="oil", y="sales", color="year", opacity=0.6,
                             color_discrete_sequence=PALETTE, labels={"oil": "Oil price (USD)", "sales": "Units sold"})
            fig.update_traces(marker=dict(size=5))
            chart(style(fig, 340, hover="closest"))
            note(f"Correlation: <b>{corr:.2f}</b>. Oil fell sharply in late 2014 while sales kept growing, "
                 "so the negative correlation mostly reflects two opposite trends over time.")
        with c2, card("tx", "Transactions"):
            tx = d["transactions"]
            metric = st.segmented_control("Metric", ["Transactions per day", "Units per transaction"],
                                          default="Transactions per day", required=True, key="an_tx",
                                          label_visibility="collapsed", persist_state="session")
            col = "transactions" if metric == "Transactions per day" else "units_per_transaction"
            weekly = tx[col].resample("W").mean()
            fig = go.Figure(go.Scatter(x=weekly.index, y=weekly, name=metric, line=dict(color="#38bdf8", width=2),
                                       fill="tozeroy", fillcolor="rgba(56,189,248,0.08)"))
            fig.update_yaxes(title=metric + " (weekly avg)")
            chart(style(fig, 300, legend=False))
            note("Transactions are store visits with a purchase. Units per transaction is roughly the basket size.")


# ---------- Page: Forecasting ----------
def forecasting():
    d = get_data()
    series = d["series"]
    last = series.index[-1]
    table, _, _ = get_backtest()
    scores = table.set_index("Model")
    best = table[table["Model"].isin(FOUR)].iloc[0]["Model"]

    page_header("Forecasting", "Forecast future sales",
                f"Models are trained on all {len(series):,} days of history and predict the days after the last "
                f"observed date, {short_date(last)}. These are true forecasts: no actual sales exist for these days.")

    left, right = st.columns([1.2, 1])
    with left, card("fc_controls", "Forecast settings"):
        model = st.pills("Model", FOUR, default=best, required=True, key="fc_model", persist_state="session")
        horizon = st.segmented_control("Forecast horizon", HORIZONS, default=14, required=True, key="fc_horizon",
                                       format_func=lambda h: f"{h} days", persist_state="session")
        if st.button("Generate forecast", type="primary", icon=":material/auto_graph:", width="stretch"):
            with st.spinner(f"Training {model} and forecasting {horizon} days..."):
                get_forecast(model)
            st.session_state["forecast"] = {"model": model, "horizon": horizon}
    with right, card("fc_info", model):
        info = models.MODEL_INFO[model]
        st.markdown(f'<span class="badge">{info["family"]}</span>', unsafe_allow_html=True)
        st.markdown(f'<p class="card-caption" style="margin-top:.6rem">{info["summary"]}</p>'
                    f'<p class="card-caption"><b>Settings:</b> {info["config"]}</p>'
                    f'<p class="card-caption"><b>Test-period error ({TEST_DAYS} days):</b> '
                    f'MAE {human(scores.loc[model, "MAE"])}, RMSE {human(scores.loc[model, "RMSE"])}</p>',
                    unsafe_allow_html=True)

    req = st.session_state.get("forecast")
    if req is None:
        note("Choose a model and a horizon, then click <b>Generate forecast</b>.")
        return
    if req != {"model": model, "horizon": horizon}:
        note(f"Showing the {req['horizon']}-day {req['model']} forecast. Click <b>Generate forecast</b> to update.")

    name, h = req["model"], req["horizon"]
    fc = get_forecast(name).iloc[:h]
    rmse = scores.loc[name, "RMSE"]
    recent = series.iloc[-h:]
    st.write("")
    c = st.columns(4)
    kpi(c[0], "Forecast period", f"{h} days", f"{fc.index[0]:%d %b} to {fc.index[-1]:%d %b %Y}")
    kpi(c[1], "Total predicted", human(fc.sum(), 2), f"{(fc.sum() / recent.sum() - 1):+.1%} vs last {h} days")
    kpi(c[2], "Average per day", human(fc.mean()), "units")
    kpi(c[3], "Peak day", human(fc.max()), f"{fc.idxmax():%A, %d %b}")
    st.write("")

    with card("fc_chart", f"{name}: {h}-day forecast",
              "Solid line: actual sales. Dashed line: forecast. The band shows ± the model's RMSE on the test "
              "period, a rough guide to typical error (not a statistical confidence interval)."):
        hist = series.iloc[-90:]
        color = MODEL_COLORS[name]
        x = [hist.index[-1]] + list(fc.index)          # join the forecast to the last actual point
        y = [hist.iloc[-1]] + list(fc.values)
        fig = go.Figure()
        fig.add_scatter(x=list(fc.index) + list(fc.index[::-1]),
                        y=list(fc.values + rmse) + list((fc.values - rmse)[::-1]),
                        fill="toself", fillcolor="rgba(148,163,184,0.12)", line=dict(width=0),
                        name="± test RMSE", hoverinfo="skip")
        fig.add_scatter(x=hist.index, y=hist, name="Actual sales (history)", line=dict(color=TEXT, width=2))
        fig.add_scatter(x=x, y=y, name=f"Forecast ({name})", line=dict(color=color, width=2.5, dash="dash"),
                        mode="lines+markers", marker=dict(size=5))
        fig.add_vrect(x0=last, x1=fc.index[-1], fillcolor=color, opacity=0.06, line_width=0)
        fig.add_vline(x=last, line=dict(color=MUTED, width=1, dash="dot"))
        fig.add_annotation(x=last, y=1.0, yref="paper", text="Last observed day", showarrow=False,
                           xanchor="right", xshift=-6, font=dict(color=MUTED, size=11))
        fig.update_yaxes(title="Units sold")
        fig.update_traces(hovertemplate="%{y:,.0f}")
        chart(style(fig, 430))

    out = pd.DataFrame({
        "date": fc.index.date,
        "weekday": fc.index.day_name(),
        "model": name,
        "predicted_sales": fc.values.round(0),
        "typical_low": (fc.values - rmse).round(0),
        "typical_high": (fc.values + rmse).round(0),
    })
    with card("fc_table", "Predicted values"):
        st.dataframe(out.drop(columns="model"), hide_index=True, height=min(38 + 35 * h, 420), column_config={
            "date": st.column_config.DateColumn("Date", format="ddd DD MMM YYYY"),
            "weekday": None,
            "predicted_sales": st.column_config.NumberColumn("Predicted units", format="localized"),
            "typical_low": st.column_config.NumberColumn("Typical low (−RMSE)", format="localized"),
            "typical_high": st.column_config.NumberColumn("Typical high (+RMSE)", format="localized"),
        })
        st.download_button("Download forecast (CSV)", out.to_csv(index=False).encode("utf-8"),
                           file_name=f"forecast_{name.lower().replace(' ', '_')}_{h}d.csv", mime="text/csv",
                           icon=":material/download:", on_click="ignore")


# ---------- Page: Model comparison ----------
def comparison():
    d = get_data()
    series = d["series"]
    table, preds, test = get_backtest()
    four = table[table["Model"].isin(FOUR)].reset_index(drop=True)
    best = four.iloc[0]
    best_mae = four.loc[four["MAE"].idxmin(), "Model"]
    baseline = table.set_index("Model").loc[models.BASELINE]
    train_end = test.index[0] - pd.Timedelta(days=1)

    page_header("Model comparison", "Which model forecasts best?",
                f"Each model was trained on data up to {short_date(train_end)} and then forecast the next "
                f"{TEST_DAYS} days ({test.index[0]:%d %b} to {short_date(test.index[-1])}) in one go, without "
                "seeing any of those days. Lower errors are better.")

    c = st.columns(4)
    for col, (_, r) in zip(c, four.iterrows()):
        tag = "Best model · " if r["Model"] == best["Model"] else ""
        kpi(col, r["Model"], f"RMSE {human(r['RMSE'])}",
            f"{tag}MAE {human(r['MAE'])} · MAPE {r['MAPE (%)']:.1f}%", highlight=r["Model"] == best["Model"])
    st.write("")
    msg = (f"<b>{best['Model']}</b> has the lowest RMSE ({best['RMSE']:,.0f}) and is ranked best. ")
    if best_mae != best["Model"]:
        msg += f"<b>{best_mae}</b> has the lowest MAE, so it is closer on a typical day but makes larger misses. "
    msg += (f"Reference: repeating the last observed week (seasonal naive) gives MAE {baseline['MAE']:,.0f} and "
            f"RMSE {baseline['RMSE']:,.0f}. A model is only useful if it beats this simple rule.")
    note(msg)

    c1, c2 = st.columns([1, 1.1])
    with c1, card("errors", "MAE and RMSE", "Dotted line: seasonal naive baseline."):
        fig = make_subplots(rows=1, cols=2, subplot_titles=("MAE", "RMSE"), horizontal_spacing=0.12)
        for i, metric in enumerate(["MAE", "RMSE"], start=1):
            fig.add_bar(x=[m.replace(" ", "<br>", 1) for m in four["Model"]], y=four[metric],
                        marker_color=[MODEL_COLORS[m] for m in four["Model"]], showlegend=False,
                        text=[human(v) for v in four[metric]], textposition="outside",
                        hovertemplate="%{y:,.0f}<extra></extra>", row=1, col=i)
            fig.add_hline(y=baseline[metric], line=dict(color=MUTED, dash="dot", width=1.2), row=1, col=i)
            fig.update_yaxes(range=[0, max(four[metric].max(), baseline[metric]) * 1.18], row=1, col=i)
        fig.update_annotations(font=dict(color=TEXT, size=13))
        style(fig, 360, legend=False, hover="closest").update_layout(margin_t=32)  # room for subplot titles
        chart(fig)
    with c2, card("table", "Results table",
                  "MAPE: average error as a % of actual sales. R²: share of the day-to-day variation in the test "
                  "period that the forecast explains (1 is perfect, 0 is no better than the average)."):
        shown = table.copy()
        shown["Type"] = [models.MODEL_INFO[m]["family"] if m in models.MODEL_INFO else "Reference baseline"
                         for m in shown["Model"]]
        shown = shown[["Model", "Type", "MAE", "RMSE", "MAPE (%)", "R²"]]

        def highlight(row):
            on = row["Model"] == best["Model"]
            return ["background-color: rgba(52,211,153,0.14)" if on else "" for _ in row]

        styled = shown.style.apply(highlight, axis=1).format(
            {"MAE": "{:,.0f}", "RMSE": "{:,.0f}", "MAPE (%)": "{:.2f}", "R²": "{:.3f}"})
        st.dataframe(styled, hide_index=True)

    with card("avp", "Actual vs predicted sales",
              f"Grey: the 60 days before the test period. White: actual sales during the test period."):
        options = FOUR + [models.BASELINE]
        shown_models = st.pills("Models", options, selection_mode="multi", default=FOUR, key="cmp_models",
                                label_visibility="collapsed", persist_state="session")
        hist = series.loc[:train_end].iloc[-60:]
        fig = go.Figure()
        fig.add_vrect(x0=test.index[0], x1=test.index[-1], fillcolor=ACCENT, opacity=0.06, line_width=0,
                      annotation_text="Test period", annotation_position="top left",
                      annotation_font=dict(color=MUTED, size=11))
        fig.add_scatter(x=hist.index, y=hist, name="History", line=dict(color=HISTORY, width=1.6))
        fig.add_scatter(x=test.index, y=test, name="Actual", line=dict(color=TEXT, width=2.6),
                        mode="lines+markers", marker=dict(size=5))
        for m in shown_models or []:
            fig.add_scatter(x=preds.index, y=preds[m], name=m,
                            line=dict(color=MODEL_COLORS[m], width=2, dash="dot" if m == models.BASELINE else "dash"))
        fig.update_traces(hovertemplate="%{y:,.0f}")
        fig.update_yaxes(title="Units sold")
        chart(style(fig, 440))

    with card("horizon", "Error by forecast day",
              "Absolute error on each day of the test period. Errors usually grow the further ahead a model forecasts."):
        err = preds[FOUR].sub(test, axis=0).abs()
        err.index = range(1, len(err) + 1)
        fig = go.Figure()
        for m in FOUR:
            fig.add_scatter(x=err.index, y=err[m], name=m, line=dict(color=MODEL_COLORS[m], width=2))
        fig.update_xaxes(title="Days ahead")
        fig.update_yaxes(title="Absolute error (units)")
        fig.update_traces(hovertemplate="%{y:,.0f}")
        chart(style(fig, 340))

    with st.expander("How the evaluation works", icon=":material/info:"):
        st.markdown(f"""
- **Chronological split.** Training data: {short_date(series.index[0])} to {short_date(train_end)}.
  Test data: the last {TEST_DAYS} days, {short_date(test.index[0])} to {short_date(test.index[-1])}. Nothing is shuffled.
- **Same task for every model.** All four models forecast all {TEST_DAYS} test days at once from the same
  starting point (multi-step forecasting).
- **No future information.** Linear Regression and Random Forest build their lag features from their own
  earlier predictions (recursive forecasting), never from actual test-period sales.
- **Settings chosen before testing.** Model settings were chosen on six earlier 30-day validation windows
  (January to July 2017), so the test period was not used for tuning.
- **Closure days.** 25 December and 1 January (stores closed) are replaced in the training series by the same
  weekday of the previous week. The test period contains no closure days, so all scores use real sales.
- **Metrics.** MAE is the average absolute error in units. RMSE squares errors before averaging, so large
  misses count more. The best model is the one with the lowest RMSE.
""")


# ---------- App ----------
try:
    get_data()
except FileNotFoundError as e:
    st.error(f"Dataset not found: {e.filename}. Place the six Kaggle CSV files "
             f"({', '.join(FILES.values())}) in the data/ folder and reload the page.")
    st.stop()

pages = [
    st.Page(overview, title="Overview", icon=":material/space_dashboard:", default=True),
    st.Page(analytics, title="Sales Analytics", icon=":material/bar_chart:"),
    st.Page(forecasting, title="Forecasting", icon=":material/trending_up:"),
    st.Page(comparison, title="Model Comparison", icon=":material/leaderboard:"),
]
nav = st.navigation({"Retail Sales Forecasting": pages})
with st.sidebar:
    data_end = get_data()["daily"].index[-1]
    st.markdown(f'<div class="brand-sub" style="margin-top:1rem">Kaggle Store Sales, Corporación Favorita '
                f'(Ecuador). Data up to {short_date(data_end)}.</div>', unsafe_allow_html=True)
nav.run()
