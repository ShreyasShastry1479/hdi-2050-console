"""Streamlit Dashboard for Global HDI Projections to 2050.

Run: streamlit run dashboard.py
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import config
from data.countries import COUNTRY_NAMES, INCOME_GROUP_LABELS


st.set_page_config(
    page_title="Global HDI Projections 2050",
    page_icon=":earth_americas:",
    layout="wide",
)

@st.cache_data
def load_projections() -> dict:
    scenarios = {}
    for name in ["baseline", "high_growth", "low_growth", "green_transition"]:
        path = config.OUTPUT_DIR / f"projections_{name}.parquet"
        if path.exists():
            scenarios[name] = pd.read_parquet(path)
    return scenarios


@st.cache_data
def load_cv_results() -> pd.DataFrame:
    path = config.OUTPUT_DIR / "cv_results.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def main():
    st.title(":earth_americas: Global HDI Projections to 2050")
    st.markdown(
        "Hybrid ML pipeline: Official UNDP HDI formula + Random Forest/Ridge ensemble "
        "+ Prophet/LSTM forecasting + Scenario engine"
    )

    scenarios = load_projections()
    cv_df = load_cv_results()

    if not scenarios:
        st.error(
            "No projection data found. Run `python main.py` first to generate projections."
        )
        return

    tab_overview, tab_country, tab_scenarios, tab_map, tab_methodology = st.tabs([
        "Overview", "Country Explorer", "Scenario Comparison",
        "World Map", "Methodology",
    ])

    with tab_overview:
        render_overview(scenarios, cv_df)

    with tab_country:
        render_country_explorer(scenarios)

    with tab_scenarios:
        render_scenario_comparison(scenarios)

    with tab_map:
        render_world_map(scenarios)

    with tab_methodology:
        render_methodology(cv_df)


def render_overview(scenarios: dict, cv_df: pd.DataFrame):
    st.header("Global HDI Projections Overview")

    baseline = scenarios.get("baseline", scenarios[list(scenarios.keys())[0]])
    latest = baseline[baseline["year"] == config.FORECAST_END]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        mean_hdi = latest["predicted_hdi"].mean()
        st.metric("Baseline HDI 2050", f"{mean_hdi:.4f}")
    with col2:
        median_hdi = latest["predicted_hdi"].median()
        st.metric("Median HDI 2050", f"{median_hdi:.4f}")
    with col3:
        n_countries = latest["country_id"].nunique()
        st.metric("Countries", f"{n_countries}")
    with col4:
        if "country_name" in latest.columns:
            top = latest.nlargest(1, "predicted_hdi")
            st.metric("Top Country", top["country_name"].values[0])

    st.subheader("HDI Trajectories by Scenario")
    fig = go.Figure()
    colors = {
        "baseline": "#2196F3",
        "high_growth": "#4CAF50",
        "low_growth": "#F44336",
        "green_transition": "#FF9800",
    }
    for name, df in scenarios.items():
        grouped = df.groupby("year")["predicted_hdi"].agg(["mean", "std"]).reset_index()
        fig.add_trace(go.Scatter(
            x=grouped["year"], y=grouped["mean"],
            name=name.replace("_", " ").title(),
            line=dict(color=colors.get(name, "gray"), width=2),
            hovertemplate="%{y:.4f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([grouped["year"], grouped["year"][::-1]]),
            y=pd.concat([grouped["mean"] + grouped["std"], (grouped["mean"] - grouped["std"])[::-1]]),
            fill="toself", fillcolor=colors.get(name, "gray"),
            opacity=0.1, showlegend=False, line=dict(width=0),
        ))
    fig.update_layout(
        xaxis_title="Year", yaxis_title="Mean HDI",
        hovermode="x unified", template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("HDI Distribution in 2050")
    dist_data = []
    for name, df in scenarios.items():
        vals = df[df["year"] == config.FORECAST_END]["predicted_hdi"]
        for v in vals:
            dist_data.append({"Scenario": name.replace("_", " ").title(), "HDI": v})
    dist_df = pd.DataFrame(dist_data)
    fig = px.box(dist_df, x="Scenario", y="HDI", color="Scenario",
                 color_discrete_map=colors, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    if not cv_df.empty:
        st.subheader("Cross-Validation Metrics")
        metric_cols = st.columns(4)
        targets = ["life_expectancy", "expected_years_schooling",
                    "mean_years_schooling", "gni_per_capita_ppp"]
        target_labels = ["Life Expectancy", "Expected Schooling",
                         "Mean Schooling", "GNI per Capita"]
        for col, target, label in zip(metric_cols, targets, target_labels):
            with col:
                r2_col = f"{target}_r2"
                mae_col = f"{target}_mae"
                if r2_col in cv_df.columns:
                    st.metric(f"{label} R²", f"{cv_df[r2_col].mean():.4f}")
                if mae_col in cv_df.columns:
                    st.metric(f"{label} MAE", f"{cv_df[mae_col].mean():.4f}")


def render_country_explorer(scenarios: dict):
    st.header("Country Explorer")

    baseline = scenarios.get("baseline", scenarios[list(scenarios.keys())[0]])
    if "country_name" in baseline.columns:
        countries = sorted(baseline["country_name"].unique())
        selected = st.selectbox("Select Country", countries)
    else:
        countries = sorted(baseline["country_id"].unique())
        selected = st.selectbox("Select Country ID", countries)

    col1, col2 = st.columns(2)
    with col1:
        scenario_filter = st.multiselect(
            "Scenarios",
            list(scenarios.keys()),
            default=["baseline"],
        )
    with col2:
        year_range = st.slider(
            "Year Range",
            config.HIST_END + 1, config.FORECAST_END,
            (config.HIST_END + 1, config.FORECAST_END),
        )

    country_data = []
    for sname in scenario_filter:
        df = scenarios[sname]
        if "country_name" in df.columns:
            cdf = df[df["country_name"] == selected]
        else:
            cdf = df[df["country_id"] == selected]
        cdf = cdf[(cdf["year"] >= year_range[0]) & (cdf["year"] <= year_range[1])]
        cdf = cdf.copy()
        cdf["scenario"] = sname.replace("_", " ").title()
        country_data.append(cdf)

    if not country_data:
        st.warning("No data found for selected country")
        return

    cdf_all = pd.concat(country_data)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("HDI Projection", "Life Expectancy",
                        "GNI per Capita (PPP)", "Internet Penetration"),
        vertical_spacing=0.12,
    )
    color_map = {"Baseline": "#2196F3", "High Growth": "#4CAF50",
                 "Low Growth": "#F44336", "Green Transition": "#FF9800"}

    for scenario in cdf_all["scenario"].unique():
        sdf = cdf_all[cdf_all["scenario"] == scenario]
        color = color_map.get(scenario, "gray")
        fig.add_trace(go.Scatter(
            x=sdf["year"], y=sdf["predicted_hdi"], name=scenario,
            line=dict(color=color, width=2), showlegend=True,
        ), row=1, col=1)
        if "life_exp" in sdf.columns:
            fig.add_trace(go.Scatter(
                x=sdf["year"], y=sdf["life_exp"], name=scenario,
                line=dict(color=color, width=2), showlegend=False,
            ), row=1, col=2)
        if "gni_ppp" in sdf.columns:
            fig.add_trace(go.Scatter(
                x=sdf["year"], y=sdf["gni_ppp"], name=scenario,
                line=dict(color=color, width=2), showlegend=False,
            ), row=2, col=1)
        if "internet" in sdf.columns:
            fig.add_trace(go.Scatter(
                x=sdf["year"], y=sdf["internet"], name=scenario,
                line=dict(color=color, width=2), showlegend=False,
            ), row=2, col=2)

    fig.update_layout(height=600, template="plotly_white",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    fig.update_xaxes(title_text="Year", row=2)
    fig.update_yaxes(title_text="HDI", row=1, col=1)
    fig.update_yaxes(title_text="Years", row=1, col=2)
    fig.update_yaxes(title_text="USD", row=2, col=1)
    fig.update_yaxes(title_text="%", row=2, col=2)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Key Indicators (2050)")
    latest = cdf_all[cdf_all["year"] == config.FORECAST_END]
    if not latest.empty:
        indicators = {}
        for col in ["predicted_hdi", "life_exp", "gni_ppp", "internet",
                     "expected_school", "mean_school", "fertility", "urbanization"]:
            if col in latest.columns:
                indicators[col] = latest[col].values[0]
        cols = st.columns(4)
        labels = {
            "predicted_hdi": "HDI", "life_exp": "Life Expectancy",
            "gni_ppp": "GNI per Capita", "internet": "Internet %",
            "expected_school": "Expected Schooling", "mean_school": "Mean Schooling",
            "fertility": "Fertility Rate", "urbanization": "Urbanization %",
        }
        for i, (key, val) in enumerate(indicators.items()):
            with cols[i % 4]:
                fmt = f"{val:.4f}" if key == "predicted_hdi" else f"{val:.2f}"
                st.metric(labels.get(key, key), fmt)


def render_scenario_comparison(scenarios: dict):
    st.header("Scenario Comparison")

    if "country_name" in scenarios["baseline"].columns:
        countries = sorted(scenarios["baseline"]["country_name"].unique())
        selected_countries = st.multiselect(
            "Select Countries", countries,
            default=countries[:5] if len(countries) >= 5 else countries,
        )
    else:
        countries = sorted(scenarios["baseline"]["country_id"].unique())
        selected_countries = st.multiselect(
            "Select Country IDs", countries,
            default=countries[:5] if len(countries) >= 5 else countries,
        )

    id_col = "country_name" if "country_name" in scenarios["baseline"].columns else "country_id"

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("HDI 2050 by Scenario", "Life Expectancy 2050",
                        "GNI per Capita 2050", "Internet 2050"),
        vertical_spacing=0.12,
    )

    for sname, df in scenarios.items():
        sdf = df[(df[id_col].isin(selected_countries)) & (df["year"] == config.FORECAST_END)]
        color = {"baseline": "#2196F3", "high_growth": "#4CAF50",
                 "low_growth": "#F44336", "green_transition": "#FF9800"}.get(sname, "gray")
        fig.add_trace(go.Bar(
            x=sdf[id_col], y=sdf["predicted_hdi"],
            name=sname.replace("_", " ").title(), marker_color=color,
        ), row=1, col=1)
        if "life_exp" in sdf.columns:
            fig.add_trace(go.Bar(
                x=sdf[id_col], y=sdf["life_exp"],
                name=sname.replace("_", " ").title(), marker_color=color,
                showlegend=False,
            ), row=1, col=2)
        if "gni_ppp" in sdf.columns:
            fig.add_trace(go.Bar(
                x=sdf[id_col], y=sdf["gni_ppp"],
                name=sname.replace("_", " ").title(), marker_color=color,
                showlegend=False,
            ), row=2, col=1)
        if "internet" in sdf.columns:
            fig.add_trace(go.Bar(
                x=sdf[id_col], y=sdf["internet"],
                name=sname.replace("_", " ").title(), marker_color=color,
                showlegend=False,
            ), row=2, col=2)

    fig.update_layout(height=700, barmode="group", template="plotly_white",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)


def render_world_map(scenarios: dict):
    st.header("World HDI Map")

    scenario = st.selectbox(
        "Scenario",
        list(scenarios.keys()),
        format_func=lambda x: x.replace("_", " ").title(),
    )
    year = st.slider("Year", config.HIST_END + 1, config.FORECAST_END, 2050)

    df = scenarios[scenario]
    df_year = df[df["year"] == year]

    if "iso3" in df_year.columns:
        geo_col = "iso3"
    elif "country_id" in df_year.columns:
        geo_col = "country_id"
    else:
        st.warning("No geographic data available")
        return

    fig = px.choropleth(
        df_year, locations=geo_col, color="predicted_hdi",
        color_continuous_scale="RdYlGn", range_color=[0.3, 0.95],
        hover_name="country_name" if "country_name" in df_year.columns else geo_col,
        hover_data={"predicted_hdi": ":.4f", "life_exp": ":.1f", "gni_ppp": ":,.0f"},
        title=f"Global HDI {year} ({scenario.replace('_', ' ').title()})",
    )
    fig.update_layout(
        geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth"),
        height=600,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_methodology(cv_df: pd.DataFrame):
    st.header("Methodology")

    st.markdown("""
    ### Pipeline Architecture

    ```
    Historical Data (1990-2025)  -->  Feature Engineering  -->  ML Ensemble Predictor
         |                                                           |
         v                                                           v
    Time-Series Forecasting (Prophet + LSTM + Ridge)          Scenario Engine
         |                                                           |
         v                                                           v
    Independent Variable Forecasts to 2050              HDI = (Health x Education x Income)^(1/3)
    ```

    ### Data Sources
    - **World Bank API** (wbgapi): Life expectancy, schooling, GNI, internet, fertility,
      urbanization, governance, trade, CO2, renewables, health expenditure, physicians
    - **UNDP HDI methodology**: Official geometric mean formula for HDI computation

    ### Model Architecture
    - **Ensemble**: 60% Random Forest + 40% Ridge Regression per HDI component
    - **Forecasting**: Weighted ensemble of Prophet, LSTM, and Ridge polynomial extrapolation
    - **Scenarios**: Baseline, High Growth, Low Growth, Green Transition

    ### HDI Formula (Official UNDP)
    ```python
    HDI = (I_health x I_education x I_income)^(1/3)

    I_health = (LE - 20) / (85 - 20)
    I_education = sqrt(EYI x MYI)
    I_income = (ln(GNI) - ln(100)) / (ln(75000) - ln(100))
    ```

    ### Feature Engineering
    - Lag features (1, 3, 5 years)
    - Rolling statistics (5-year mean, std)
    - Growth rates (1-year, 5-year differences)
    - Regional averages
    - Distance from HDI frontier
    - Composite indices (human capital, governance, digitalization)
    """)

    if not cv_df.empty:
        st.subheader("Cross-Validation Results")
        st.dataframe(cv_df.style.format("{:.6f}"), use_container_width=True)


if __name__ == "__main__":
    main()
