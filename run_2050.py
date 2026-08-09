import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

from config import config
from data.collection import load_dataset
from data.countries import COUNTRY_NAMES, classify_archetype, INCOME_GROUP_LABELS
from data.future import (
    FUTURE_VARS,
    compute_digital_infrastructure_development,
    compute_future_readiness,
    estimate_future_factor,
)
from data.reference import REFERENCE_HDI_2024
from data.undp_hdi import UNDP_HDI_COUNTRIES_193, load_undp_hdi_country_rows
from src.hdi_calculator import hdi_calc
from src.forecasting.realistic import (
    forecast_all_countries,
    compute_hdi_with_factors,
    compute_uncertainty_range,
    explain_hdi_projection,
    get_recovery_potential,
    get_resource_volatility_profile,
    compute_resource_drag,
    compute_industrialization_acceleration,
    compute_trajectory_hdi_effect,
    compute_developing_catchup_readiness,
    compute_growth_prospect_score,
    compute_developed_demographic_adaptation,
    INSTITUTIONAL_EFFICIENCY,
    TRAJECTORY_CLASSES,
    get_trajectory_class,
)
from data.stability import apply_state_capacity_adjustments, get_institutional_efficiency, get_state_capacity
from data.technology import compute_technology_factor
from data.demographics import (
    compute_demographic_factor,
    compute_demographic_profile,
    get_aging_penalty,
    migration_buffer_intensity,
    sub_replacement_depth,
)

print("Loading dataset...")
raw_df = load_dataset()

print("Building feature matrix...")
from src.feature_engineering import build_feature_matrix
df = build_feature_matrix(raw_df)

print("Forecasting to 2050...")
forecast_years = np.arange(config.HIST_END + 1, config.FORECAST_END + 1)
fc_vars = ["life_exp", "expected_school", "mean_school", "gni_ppp",
           "internet", "fertility", "urbanization", "gov_effectiveness",
           "corruption", "trade_openness", "co2_per_capita",
           "renewable_share", "eci", "physicians", "health_exp",
           "population", *FUTURE_VARS]

base_forecasts = forecast_all_countries(df, forecast_years, fc_vars)
base_forecasts = apply_state_capacity_adjustments(base_forecasts)

print("Computing HDI with new formula...")

latest_by_country = df.sort_values("year", ascending=False).groupby("country_id").first()
undp_rows = load_undp_hdi_country_rows()
if len(undp_rows) != 193:
    raise RuntimeError(f"Expected 193 UNDP HDI rows, got {len(undp_rows)}")

historical_population = (
    df[df["population"].notna()]
    .sort_values("year")
    .groupby("country_id")
    .tail(1)
    .set_index("country_id")[["year", "population"]]
)
forecast_population = (
    base_forecasts[base_forecasts["population"].notna()]
    .set_index(["country_id", "year"])["population"]
)
population_weights_path = Path("data/population_weights_2024_2050.csv")
if population_weights_path.exists():
    population_weights = pd.read_csv(population_weights_path).set_index("ISO3")
else:
    population_weights = pd.DataFrame()


def population_for(country_id: str, year: int) -> tuple[float, str]:
    if not population_weights.empty and country_id in population_weights.index:
        column = f"Population_{year}"
        if column in population_weights.columns and pd.notna(population_weights.at[country_id, column]):
            return float(population_weights.at[country_id, column]), f"un_wpp_owid_{year}"

    if year <= config.HIST_END and country_id in historical_population.index:
        row = historical_population.loc[country_id]
        return float(row["population"]), f"observed_or_latest_{int(row['year'])}"

    if (country_id, year) in forecast_population.index:
        return float(forecast_population.loc[(country_id, year)]), f"forecast_{year}"

    if (country_id, config.HIST_END + 1) in forecast_population.index:
        return float(forecast_population.loc[(country_id, config.HIST_END + 1)]), f"forecast_{config.HIST_END + 1}_proxy"

    if country_id in historical_population.index:
        row = historical_population.loc[country_id]
        return float(row["population"]), f"observed_or_latest_{int(row['year'])}_proxy"

    return np.nan, "missing"


def clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def health_index_undp(life_expectancy: float) -> float:
    return clamp01((life_expectancy - 20.0) / (88.0 - 20.0))


def education_index_undp(expected_years: float, mean_years: float) -> float:
    expected_index = clamp01(expected_years / 18.0)
    mean_index = clamp01(mean_years / 16.0)
    return clamp01(np.sqrt(expected_index * mean_index))


def income_index_undp(gni_ppp: float) -> float:
    ln_min = np.log(100.0)
    ln_max = np.log(105000.0)
    ln_value = np.log(max(float(gni_ppp), 1.0))
    return clamp01((ln_value - ln_min) / (ln_max - ln_min))


def projected_component_index(
    baseline: float,
    hdi_gain: float,
    contribution: float,
    positive_core: float,
) -> float:
    baseline = clamp01(baseline)
    contribution = float(contribution)
    if positive_core > 1e-9:
        share = max(0.0, contribution) / positive_core
    else:
        share = 1.0 / 3.0

    if hdi_gain >= 0:
        headroom = max(0.001, 1.0 - baseline)
        raw_lift = hdi_gain * share * 2.15
        saturated_lift = headroom * (1.0 - np.exp(-raw_lift / headroom))
        drag = min(0.0, contribution) * 1.15
        return clamp01(baseline + saturated_lift + drag)
    return clamp01(baseline + contribution * 1.45)


def reconcile_component_indices_to_hdi(
    health_index: float,
    education_index: float,
    income_index: float,
    target_hdi: float,
) -> tuple[float, float, float]:
    """Scale 2050 component indices so their geometric mean equals target HDI."""
    target_hdi = clamp01(target_hdi)
    values = np.array([
        max(1e-6, clamp01(health_index)),
        max(1e-6, clamp01(education_index)),
        max(1e-6, clamp01(income_index)),
    ], dtype=float)
    target_log_sum = 3.0 * np.log(max(target_hdi, 1e-6))

    low, high = -20.0, 20.0
    for _ in range(90):
        mid = (low + high) / 2.0
        adjusted = np.clip(values * np.exp(mid), 1e-6, 1.0)
        if np.log(adjusted).sum() < target_log_sum:
            low = mid
        else:
            high = mid

    adjusted = np.clip(values * np.exp((low + high) / 2.0), 1e-6, 1.0)
    return tuple(float(v) for v in adjusted)


def apply_stable_rank_clusters(results: pd.DataFrame, tolerance: float = 0.002) -> pd.DataFrame:
    """Rank by 2050 HDI while treating tiny differences as statistical ties."""
    ranked = results.sort_values(
        ["HDI_2050", "UNDP_Rank_2023"],
        ascending=[False, True],
    ).reset_index(drop=True)

    ordered_parts = []
    cluster_id = 0
    cluster_start = None
    cluster_rows = []

    for _, row in ranked.iterrows():
        hdi = float(row["HDI_2050"])
        if cluster_start is None or cluster_start - hdi > tolerance:
            if cluster_rows:
                cluster = pd.DataFrame(cluster_rows).sort_values("UNDP_Rank_2023")
                cluster["RankCluster"] = cluster_id
                cluster["RankClusterSize"] = len(cluster)
                ordered_parts.append(cluster)
            cluster_id += 1
            cluster_start = hdi
            cluster_rows = [row]
        else:
            cluster_rows.append(row)

    if cluster_rows:
        cluster = pd.DataFrame(cluster_rows).sort_values("UNDP_Rank_2023")
        cluster["RankCluster"] = cluster_id
        cluster["RankClusterSize"] = len(cluster)
        ordered_parts.append(cluster)

    stable = pd.concat(ordered_parts, ignore_index=True)
    stable.insert(0, "Rank", np.arange(1, len(stable) + 1))
    stable["RankClusterLabel"] = "Cluster " + stable["RankCluster"].astype(str)
    return stable


def add_rank_uncertainty(results: pd.DataFrame) -> pd.DataFrame:
    """Approximate rank range using each country's P10/P90 against peer medians."""
    median_values = results["HDI_P50"].to_numpy(dtype=float)
    best_ranks = []
    worst_ranks = []
    for idx, row in results.reset_index(drop=True).iterrows():
        peer_medians = np.delete(median_values, idx)
        best_ranks.append(int(1 + np.sum(peer_medians > float(row["HDI_P90"]))))
        worst_ranks.append(int(1 + np.sum(peer_medians > float(row["HDI_P10"]))))
    results = results.copy()
    results["Rank_Best_P90"] = best_ranks
    results["Rank_Worst_P10"] = worst_ranks
    return results


def add_development_momentum(results: pd.DataFrame) -> pd.DataFrame:
    """Make visible which countries are moving fastest in ranking terms."""
    results = results.copy()
    results["RankChange_2023_to_2050"] = (
        results["UNDP_Rank_2023"].astype(int) - results["Rank"].astype(int)
    )
    results["RankChangeDisplay"] = results["RankChange_2023_to_2050"].map(
        lambda value: f"+{int(value)}" if value > 0 else str(int(value))
    )

    future_cols = [col for col in FUTURE_VARS if col in results.columns]
    industrial_cols = [
        col for col in [
            "robot_density", "semiconductor_production", "green_energy_investment",
            "ev_adoption", "battery_capacity", "high_speed_rail",
            "startup_ecosystem", "venture_capital", "cloud_adoption",
            "ai_adoption_index",
        ]
        if col in results.columns
    ]
    if industrial_cols:
        results["IndustrializationSignal"] = results[industrial_cols].mean(axis=1)
    elif future_cols:
        results["IndustrializationSignal"] = results[future_cols].mean(axis=1)
    else:
        results["IndustrializationSignal"] = 0.0

    gain_norm = (
        (results["HDI_2050_Gain"] - results["HDI_2050_Gain"].min()) /
        max(results["HDI_2050_Gain"].max() - results["HDI_2050_Gain"].min(), 1e-9)
    )
    rank_up_norm = np.clip(results["RankChange_2023_to_2050"] / 45.0, 0.0, 1.0)
    future_readiness = results.get("FutureReadiness", pd.Series(0.0, index=results.index))
    digital_infra = results.get("DigitalInfraDevelopment", pd.Series(0.0, index=results.index))
    resource_drag = results.get("ResourceDrag", pd.Series(0.0, index=results.index))
    demographic_dividend = results.get("DemographicDividend", pd.Series(0.0, index=results.index))
    human_capital_absorption = results.get("HumanCapitalAbsorption", pd.Series(0.0, index=results.index))
    dependency_pressure = results.get("DependencyPressure", pd.Series(0.0, index=results.index))
    developing_stage_boost = np.where(results["HDI_Baseline"] < 0.85, 0.12, 0.0)
    results["DevelopmentMomentumScore"] = (
        100.0 * np.clip(
            0.30 * gain_norm +
            0.18 * rank_up_norm +
            0.22 * results["IndustrializationSignal"] +
            0.13 * future_readiness +
            0.11 * digital_infra +
            0.10 * demographic_dividend +
            0.08 * human_capital_absorption +
            developing_stage_boost -
            0.08 * dependency_pressure -
            0.10 * resource_drag,
            0.0,
            1.0,
        )
    )
    results["IndustrializingMover"] = (
        (
            (results["IndustrializationSignal"] >= 0.38) &
            (results["HDI_2050_Gain"] >= 0.035)
        ) |
        (
            (future_readiness >= 0.35) &
            (results["RankChange_2023_to_2050"] >= 5)
        )
    )
    results["DevelopmentMomentumTier"] = pd.cut(
        results["DevelopmentMomentumScore"],
        bins=[-0.1, 30, 42, 52, 100],
        labels=["stable_or_slow", "emerging_mover", "fast_converger", "rapid_industrializer"],
    ).astype(str)
    results.loc[
        results["IndustrializingMover"] &
        (results["DevelopmentMomentumTier"] == "emerging_mover"),
        "DevelopmentMomentumTier",
    ] = "fast_converger"

    return results

results = []
for undp_row in undp_rows:
    cid = undp_row["iso3"]
    row = latest_by_country.loc[cid] if cid in latest_by_country.index else pd.Series(dtype="object")

    # Anchor the forecast on the official UNDP HDR 2025 release.
    # The 2025 report's Table 1 reports HDI and component values for 2023.
    ref = REFERENCE_HDI_2024.get(cid, {})
    hdi_current = float(undp_row["hdi_2023"])
    hdi_legacy_2024 = float(ref.get("hdi", hdi_current))
    life_exp_current = float(undp_row["life_exp_2023"])
    expected_school_current = float(undp_row["expected_school_2023"])
    mean_school_current = float(undp_row["mean_school_2023"])
    gni_ppp_current = float(undp_row["gni_ppp_2023"])
    population_2024, population_2024_source = population_for(cid, config.HIST_END)
    population_2050, population_2050_source = population_for(cid, config.FORECAST_END)
    if row.empty:
        row = pd.Series({
            "country_id": cid,
            "year": config.HIST_END,
            "life_exp": life_exp_current,
            "expected_school": expected_school_current,
            "mean_school": mean_school_current,
            "gni_ppp": gni_ppp_current,
            "internet": 0.45,
            "urbanization": 0.55,
            "gov_effectiveness": 0.0,
            "corruption": 0.0,
            "gini": 0.38,
            "infant_mortality": 25.0,
            "climate_risk": 0.30,
            "fertility": 2.3,
            "political_stability": 0.0,
            "rule_of_law": 0.0,
            "physicians": 1.5,
            "health_exp": 6.0,
            "trade_openness": 0.55,
            "renewable_share": 0.20,
            "eci": 0.0,
            "rd_expenditure": 0.5,
        })
    for var in FUTURE_VARS:
        if var not in row.index:
            row[var] = estimate_future_factor(cid, row, config.HIST_END, var)
    future_readiness = compute_future_readiness(row)
    digital_infra_development = compute_digital_infrastructure_development(
        cid,
        row,
        config.FORECAST_END,
    )
    recovery_potential = get_recovery_potential(cid, hdi_current, config.FORECAST_END)
    hdi_baseline_recomputed = float(hdi_calc.compute_hdi(
        pd.Series([life_exp_current]),
        pd.Series([expected_school_current]),
        pd.Series([mean_school_current]),
        pd.Series([gni_ppp_current]),
    ).iloc[0])

    # Compute HDI 2050 using component-based forecasting
    # Pass 2024 values as "current" - the function forecasts forward
    hdi_2050 = compute_hdi_with_factors(
        cid, hdi_current, config.FORECAST_END,
        gni_ppp_current=gni_ppp_current,
        life_exp_current=life_exp_current,
        expected_school_current=expected_school_current,
        mean_school_current=mean_school_current,
        future_readiness_current=future_readiness,
        digital_infra_development_current=digital_infra_development,
        gini_current=float(row.get("gini", 0.38)),
        infant_mortality_current=float(row.get("infant_mortality", 25.0)),
        climate_risk_current=float(row.get("climate_risk", 0.30)),
        fertility_current=float(row.get("fertility", 2.3)),
        urbanization_current=float(row.get("urbanization", 0.55)),
        political_stability_current=float(row.get("political_stability", 0.0)),
        rule_of_law_current=float(row.get("rule_of_law", 0.0)),
        physicians_current=float(row.get("physicians", 1.5)),
        health_exp_current=float(row.get("health_exp", 6.0)),
        renewable_share_current=float(row.get("renewable_share", 0.20)),
    )

    # Get component factors for diagnostics
    inst_eff = INSTITUTIONAL_EFFICIENCY.get(cid, 0.90)
    tech_factor = compute_technology_factor(cid, config.FORECAST_END)
    demo_factor = compute_demographic_factor(cid, config.FORECAST_END)
    migration_buffer_2050 = migration_buffer_intensity(cid, config.FORECAST_END)
    sub_replacement_depth_2050 = sub_replacement_depth(cid)
    resource_profile = get_resource_volatility_profile(cid)
    state_capacity = get_state_capacity(cid)
    capacity_score = float(np.clip(np.mean([
        state_capacity.get("stability", 0.5),
        state_capacity.get("conflict", 0.65),
        state_capacity.get("corruption", 0.45),
        state_capacity.get("governance", 0.48),
        state_capacity.get("fragility", 0.5),
    ]), 0.0, 1.0))
    industrialization_accel = compute_industrialization_acceleration(
        cid,
        hdi_current,
        future_readiness,
        capacity_score,
        compute_resource_drag(cid, capacity_score),
    )
    baseline_edu_index = float(np.sqrt(
        np.clip(expected_school_current / 18.0, 0.0, 1.0) *
        np.clip(mean_school_current / 16.0, 0.0, 1.0)
    ))
    demographic_profile = compute_demographic_profile(
        cid,
        config.FORECAST_END,
        float(row.get("fertility", 2.3)),
        baseline_edu_index,
    )
    developed_demographic_adaptation = compute_developed_demographic_adaptation(
        cid,
        hdi_current,
        capacity_score,
        future_readiness,
        demographic_profile["aging_pressure"],
    )
    adapted_aging_pressure = demographic_profile["aging_pressure"] * (
        1.0 - 0.42 * developed_demographic_adaptation
    )
    adapted_aging_penalty = get_aging_penalty(cid, config.FORECAST_END) * (
        1.0 - 0.40 * developed_demographic_adaptation
    )
    hdi_gap = max(0.0, 0.985 - hdi_current)
    demographic_hdi_adjustment = (
        hdi_gap * 0.060 * demographic_profile["demographic_dividend"] * demographic_profile["human_capital_absorption"] +
        hdi_gap * 0.025 * demographic_profile["workforce_depth"] *
            demographic_profile["workforce_momentum"] *
            demographic_profile["human_capital_absorption"] -
        hdi_gap * 0.050 * demographic_profile["dependency_pressure"] * (1.0 - 0.35 * capacity_score) -
        hdi_gap * 0.040 * adapted_aging_pressure * (1.0 - 0.35 * demographic_profile["workforce_depth"]) +
        hdi_gap * 0.040 * developed_demographic_adaptation *
            np.clip((hdi_current - 0.88) / 0.10, 0.0, 1.0) *
            (0.55 + 0.45 * capacity_score)
    )
    digital_infra_hdi_adjustment = hdi_gap * 0.040 * digital_infra_development * (
        0.45 + 0.30 * demographic_profile["human_capital_absorption"] + 0.25 * capacity_score
    )
    developing_catchup_readiness = compute_developing_catchup_readiness(
        hdi_current,
        baseline_edu_index,
        float(row.get("urbanization", 0.55)),
        float(row.get("fertility", 2.3)),
        capacity_score,
        future_readiness,
        compute_resource_drag(cid, capacity_score),
        recovery_potential,
        industrialization_accel,
    )
    growth_prospect_score = compute_growth_prospect_score(
        cid,
        hdi_current,
        capacity_score,
        future_readiness,
        float(row.get("fertility", 2.3)),
        float(row.get("urbanization", 0.55)),
        compute_resource_drag(cid, capacity_score),
        recovery_potential,
        industrialization_accel,
        developing_catchup_readiness,
        adapted_aging_penalty,
    )
    trajectory_effect = compute_trajectory_hdi_effect(
        cid,
        hdi_current,
        recovery_potential,
        capacity_score,
    )

    structural_inputs = {
        "gini_current": float(row.get("gini", 0.38)),
        "infant_mortality_current": float(row.get("infant_mortality", 25.0)),
        "climate_risk_current": float(row.get("climate_risk", 0.30)),
        "fertility_current": float(row.get("fertility", 2.3)),
        "urbanization_current": float(row.get("urbanization", 0.55)),
        "political_stability_current": float(row.get("political_stability", 0.0)),
        "rule_of_law_current": float(row.get("rule_of_law", 0.0)),
        "physicians_current": float(row.get("physicians", 1.5)),
        "health_exp_current": float(row.get("health_exp", 6.0)),
        "renewable_share_current": float(row.get("renewable_share", 0.20)),
    }

    # Compute uncertainty range and approximate driver attribution.
    uncertainty = compute_uncertainty_range(
        cid,
        hdi_2050,
        hdi_current=hdi_current,
        gini_current=structural_inputs["gini_current"],
        climate_risk_current=structural_inputs["climate_risk_current"],
        political_stability_current=structural_inputs["political_stability_current"],
        fertility_current=structural_inputs["fertility_current"],
    )
    contributions = explain_hdi_projection(
        cid,
        hdi_current,
        hdi_2050,
        gni_ppp_current=gni_ppp_current,
        life_exp_current=life_exp_current,
        expected_school_current=expected_school_current,
        mean_school_current=mean_school_current,
        future_readiness_current=future_readiness,
        digital_infra_development_current=digital_infra_development,
        **structural_inputs,
        year=config.FORECAST_END,
    )

    health_index_2025 = health_index_undp(life_exp_current)
    education_index_2025 = education_index_undp(expected_school_current, mean_school_current)
    income_index_2025 = income_index_undp(gni_ppp_current)
    positive_component_contrib = (
        max(0.0, contributions.get("Contrib_Health", 0.0)) +
        max(0.0, contributions.get("Contrib_Education", 0.0)) +
        max(0.0, contributions.get("Contrib_Income", 0.0))
    )
    health_index_2050_raw = projected_component_index(
        health_index_2025,
        hdi_2050 - hdi_current,
        contributions.get("Contrib_Health", 0.0),
        positive_component_contrib,
    )
    education_index_2050_raw = projected_component_index(
        education_index_2025,
        hdi_2050 - hdi_current,
        contributions.get("Contrib_Education", 0.0),
        positive_component_contrib,
    )
    income_index_2050_raw = projected_component_index(
        income_index_2025,
        hdi_2050 - hdi_current,
        contributions.get("Contrib_Income", 0.0),
        positive_component_contrib,
    )
    health_index_2050, education_index_2050, income_index_2050 = reconcile_component_indices_to_hdi(
        health_index_2050_raw,
        education_index_2050_raw,
        income_index_2050_raw,
        hdi_2050,
    )

    results.append({
        "ProjectionType": "scenario_projection",
        "BaselineSource": "UNDP_HDR_2025_Table_1",
        "BaselineDataYear": 2023,
        "UNDP_Rank_2023": undp_row["undp_rank_2023"],
        "UNDP_Country": undp_row["undp_country"],
        "ISO3": cid,
        "Country": COUNTRY_NAMES.get(cid, undp_row["undp_country"]),
        "HDI_Baseline": hdi_current,
        "HDI_2024_Local_Estimate": hdi_legacy_2024,
        "HDI_2024_vs_UNDP_HDR2025": hdi_legacy_2024 - hdi_current,
        "HDI_2024": hdi_legacy_2024,
        "UNDP_HDI_2023": undp_row["hdi_2023"],
        "Population_2024": population_2024,
        "Population_2024_Source": population_2024_source,
        "Population_2050": population_2050,
        "Population_2050_Source": population_2050_source,
        "HDI_Baseline_Recomputed_From_Components": hdi_baseline_recomputed,
        "HDI_Baseline_Component_Mismatch": hdi_baseline_recomputed - hdi_current,
        "HDI_2024_Recomputed_From_Components": hdi_baseline_recomputed,
        "HDI_2024_Component_Mismatch": hdi_baseline_recomputed - hdi_current,
        "HDI_2050": hdi_2050,
        "HDI_2050_Gain": hdi_2050 - hdi_current,
        "HDI_P10": uncertainty["p10"],
        "HDI_P50": uncertainty["p50"],
        "HDI_P90": uncertainty["p90"],
        "HDI_Optimistic": uncertainty["optimistic"],
        "HDI_Pessimistic": uncertainty["pessimistic"],
        "Interval_Contains_Baseline_HDI": uncertainty["pessimistic"] <= hdi_current <= uncertainty["optimistic"],
        "Interval_Contains_2024_HDI": uncertainty["pessimistic"] <= hdi_legacy_2024 <= uncertainty["optimistic"],
        "LifeExp_Baseline": life_exp_current,
        "GNI_PPP_Baseline": gni_ppp_current,
        "ExpSchool_Baseline": expected_school_current,
        "MeanSchool_Baseline": mean_school_current,
        "HealthIndex_2025": health_index_2025,
        "EducationIndex_2025": education_index_2025,
        "IncomeIndex_2025": income_index_2025,
        "HealthIndex_2050": health_index_2050,
        "EducationIndex_2050": education_index_2050,
        "IncomeIndex_2050": income_index_2050,
        "HDI_2050_Recomputed_From_Indices": (health_index_2050 * education_index_2050 * income_index_2050) ** (1.0 / 3.0),
        "HDI_2050_Index_Mismatch": ((health_index_2050 * education_index_2050 * income_index_2050) ** (1.0 / 3.0)) - hdi_2050,
        "LifeExp_2024": life_exp_current,
        "GNI_PPP_2024": gni_ppp_current,
        "ExpSchool_2024": expected_school_current,
        "MeanSchool_2024": mean_school_current,
        "InstEfficiency": inst_eff,
        "TechFactor": tech_factor,
        "DemoFactor": demo_factor,
        "SubReplacementDepth": sub_replacement_depth_2050,
        "MigrationBufferIntensity": migration_buffer_2050,
        "MedianAge_2024": demographic_profile["median_age_2024"],
        "MedianAge_2050": demographic_profile["median_age_2050"],
        "MedianAgeShift_2024_to_2050": demographic_profile["median_age_shift"],
        "Youth014Pct_2024": demographic_profile["youth_0_14_pct_2024"],
        "Youth014Pct_2050": demographic_profile["youth_0_14_pct_2050"],
        "WorkingAgePct_2024": demographic_profile["working_age_pct_2024"],
        "WorkingAgePct_2050": demographic_profile["working_age_pct_2050"],
        "WorkforceChangePP_2024_to_2050": demographic_profile["workforce_change_pp"],
        "Elderly65PlusPct_2024": demographic_profile["elderly_65_plus_pct_2024"],
        "Elderly65PlusPct_2050": demographic_profile["elderly_65_plus_pct_2050"],
        "Elderly65PlusChangePP_2024_to_2050": demographic_profile["elderly_change_pp"],
        "ChildDependencyRatio": demographic_profile["child_dependency_ratio"],
        "OldAgeDependencyRatio": demographic_profile["old_age_dependency_ratio"],
        "TotalDependencyRatio": demographic_profile["total_dependency_ratio"],
        "AgePyramidScore": demographic_profile["age_pyramid_score"],
        "WorkforceDepth": demographic_profile["workforce_depth"],
        "WorkforceMomentum": demographic_profile["workforce_momentum"],
        "YouthDependencyPressure": demographic_profile["youth_dependency_pressure"],
        "AgingPressure": demographic_profile["aging_pressure"],
        "DevelopedDemographicAdaptation": developed_demographic_adaptation,
        "AdaptedAgingPressure": adapted_aging_pressure,
        "FertilityWindow": demographic_profile["fertility_window"],
        "HumanCapitalAbsorption": demographic_profile["human_capital_absorption"],
        "DemographicDividend": demographic_profile["demographic_dividend"],
        "DependencyPressure": demographic_profile["dependency_pressure"],
        "DemographicHDIMultiplier": demographic_profile["demographic_hdi_multiplier"],
        "DemographicHDIAdjustment": demographic_hdi_adjustment,
        "FutureReadiness": future_readiness,
        "DigitalInfraDevelopment": digital_infra_development,
        "DigitalInfraHDIAdjustment": digital_infra_hdi_adjustment,
        "RecoveryPotential2050": recovery_potential,
        "ResourceDependence": resource_profile["dependence"],
        "GrowthVolatility": resource_profile["volatility"],
        "EconomicDiversification": resource_profile["diversification"],
        "ResourceDrag": compute_resource_drag(cid, capacity_score),
        "IndustrializationHDIAcceleration": industrialization_accel,
        "DevelopingCatchupReadiness": developing_catchup_readiness,
        "GrowthProspectScore": growth_prospect_score,
        "LowGrowthProspectDrag": 1.0 - growth_prospect_score,
        "TrajectoryHDISpeedMultiplier": trajectory_effect["speed"],
        "TrajectoryHDICapMultiplier": trajectory_effect["cap"],
        "TrajectoryHDIShockMultiplier": trajectory_effect["shock"],
        "Gini_2024": float(row.get("gini", 0.38)),
        "InfantMortality_2024": float(row.get("infant_mortality", 25.0)),
        "ClimateRisk_2024": float(row.get("climate_risk", 0.30)),
        "Fertility_2024": float(row.get("fertility", 2.3)),
        "Urbanization_2024": float(row.get("urbanization", 0.55)),
        "PoliticalStability_2024": float(row.get("political_stability", 0.0)),
        "RuleOfLaw_2024": float(row.get("rule_of_law", 0.0)),
        "Physicians_2024": float(row.get("physicians", 1.5)),
        "HealthExp_2024": float(row.get("health_exp", 6.0)),
        "RenewableShare_2024": float(row.get("renewable_share", 0.20)),
        **contributions,
        **{var: float(row.get(var, 0.0)) for var in FUTURE_VARS},
        "Trajectory": get_trajectory_class(cid),
    })

results_df = pd.DataFrame(results)
round_cols = [
    "HDI_Baseline", "HDI_2024_Local_Estimate", "HDI_2024_vs_UNDP_HDR2025",
    "HDI_2024", "UNDP_HDI_2023",
    "Population_2024", "Population_2050",
    "HDI_Baseline_Recomputed_From_Components", "HDI_Baseline_Component_Mismatch",
    "HDI_2024_Recomputed_From_Components", "HDI_2024_Component_Mismatch",
    "HDI_2050", "HDI_2050_Gain", "HDI_P10", "HDI_P50", "HDI_P90",
    "HDI_Optimistic", "HDI_Pessimistic",
    "HDI_2050_Recomputed_From_Indices", "HDI_2050_Index_Mismatch",
    "LifeExp_Baseline", "GNI_PPP_Baseline", "ExpSchool_Baseline", "MeanSchool_Baseline",
    "HealthIndex_2025", "EducationIndex_2025", "IncomeIndex_2025",
    "HealthIndex_2050", "EducationIndex_2050", "IncomeIndex_2050",
    "LifeExp_2024", "GNI_PPP_2024", "ExpSchool_2024", "MeanSchool_2024",
    "InstEfficiency", "TechFactor", "DemoFactor",
    "SubReplacementDepth", "MigrationBufferIntensity",
    "MedianAge_2024", "MedianAge_2050", "MedianAgeShift_2024_to_2050",
    "Youth014Pct_2024", "Youth014Pct_2050",
    "WorkingAgePct_2024", "WorkingAgePct_2050", "WorkforceChangePP_2024_to_2050",
    "Elderly65PlusPct_2024", "Elderly65PlusPct_2050", "Elderly65PlusChangePP_2024_to_2050",
    "ChildDependencyRatio", "OldAgeDependencyRatio", "TotalDependencyRatio",
    "AgePyramidScore", "WorkforceDepth", "WorkforceMomentum",
    "YouthDependencyPressure", "AgingPressure", "FertilityWindow",
    "DevelopedDemographicAdaptation", "AdaptedAgingPressure",
    "HumanCapitalAbsorption", "DemographicDividend", "DependencyPressure",
    "DemographicHDIMultiplier", "DemographicHDIAdjustment",
    "FutureReadiness", "DigitalInfraDevelopment", "DigitalInfraHDIAdjustment",
    "RecoveryPotential2050",
    "ResourceDependence", "GrowthVolatility", "EconomicDiversification", "ResourceDrag",
    "IndustrializationHDIAcceleration", "DevelopingCatchupReadiness",
    "GrowthProspectScore", "LowGrowthProspectDrag",
    "TrajectoryHDISpeedMultiplier", "TrajectoryHDICapMultiplier", "TrajectoryHDIShockMultiplier",
    "Gini_2024", "InfantMortality_2024", "ClimateRisk_2024", "Fertility_2024",
    "Urbanization_2024", "PoliticalStability_2024", "RuleOfLaw_2024",
    "Physicians_2024", "HealthExp_2024", "RenewableShare_2024",
    "Contrib_Income", "Contrib_Education", "Contrib_Health",
    "Contrib_Governance", "Contrib_Demographics", "Contrib_Technology",
    "Contrib_Recovery", "Contrib_Inequality", "Contrib_Climate",
    "Contrib_ShockRisk",
    *FUTURE_VARS,
]
for col in round_cols:
    if col in results_df.columns:
        results_df[col] = results_df[col].round(4)
for col in ["Population_2024", "Population_2050"]:
    if col in results_df.columns:
        results_df[col] = results_df[col].round(0).astype("Int64")
component_index_cols = [
    "HealthIndex_2025", "EducationIndex_2025", "IncomeIndex_2025",
    "HealthIndex_2050", "EducationIndex_2050", "IncomeIndex_2050",
]
for col in component_index_cols:
    if col in results_df.columns:
        results_df[col] = results_df[col].round(6)
if {"HealthIndex_2050", "EducationIndex_2050", "IncomeIndex_2050", "HDI_2050"}.issubset(results_df.columns):
    adjusted_indices = results_df.apply(
        lambda row: reconcile_component_indices_to_hdi(
            row["HealthIndex_2050"],
            row["EducationIndex_2050"],
            row["IncomeIndex_2050"],
            row["HDI_2050"],
        ),
        axis=1,
        result_type="expand",
    )
    results_df[["HealthIndex_2050", "EducationIndex_2050", "IncomeIndex_2050"]] = adjusted_indices.round(6)
    results_df["HDI_2050_Recomputed_From_Indices"] = (
        results_df["HealthIndex_2050"] *
        results_df["EducationIndex_2050"] *
        results_df["IncomeIndex_2050"]
    ) ** (1.0 / 3.0)
    results_df["HDI_2050_Index_Mismatch"] = (
        results_df["HDI_2050_Recomputed_From_Indices"] - results_df["HDI_2050"]
    )
    results_df["HDI_2050_Recomputed_From_Indices"] = results_df["HDI_2050_Recomputed_From_Indices"].round(6)
    results_df["HDI_2050_Index_Mismatch"] = results_df["HDI_2050_Index_Mismatch"].round(8)
results_df = results_df[results_df["ISO3"].isin(UNDP_HDI_COUNTRIES_193)].copy()
missing_undp = set(UNDP_HDI_COUNTRIES_193) - set(results_df["ISO3"])
if missing_undp:
    raise RuntimeError(f"Missing UNDP HDI countries: {sorted(missing_undp)}")
results_df = add_rank_uncertainty(results_df)
results_df = apply_stable_rank_clusters(results_df)
results_df = add_development_momentum(results_df)
for col in ["IndustrializationSignal", "DevelopmentMomentumScore"]:
    results_df[col] = results_df[col].round(2)
front_cols = [
    "Rank", "UNDP_Rank_2023", "RankChange_2023_to_2050", "RankChangeDisplay",
    "DevelopmentMomentumScore", "DevelopmentMomentumTier", "IndustrializingMover",
    "IndustrializationSignal",
    "RankCluster", "RankClusterSize", "RankClusterLabel",
]
results_df = results_df[
    front_cols + [col for col in results_df.columns if col not in front_cols]
]
if len(results_df) != 193:
    raise RuntimeError(f"Expected 193 countries in output, got {len(results_df)}")

output_path = config.OUTPUT_DIR / "hdi_2050_rankings.csv"
results_df.to_csv(output_path, index=False)

print("Rankings saved to " + str(output_path))
print()
print("=== Output audit ===")
gain = results_df["HDI_2050_Gain"]
print(f"Projection type: scenario_projection (not a statistically validated prediction model)")
print("Baseline source: UNDP HDR 2025 Table 1 (latest measured HDI/component year: 2023)")
print(f"Countries improved: {(gain > 0).sum()} | flat: {(gain == 0).sum()} | declined: {(gain < 0).sum()}")
print(f"Baseline HDI below pessimistic bound: {(results_df['HDI_Baseline'] < results_df['HDI_Pessimistic']).sum()}")
print(f"Mean abs component mismatch: {results_df['HDI_Baseline_Component_Mismatch'].abs().mean():.4f}")
print(f"Max abs component mismatch: {results_df['HDI_Baseline_Component_Mismatch'].abs().max():.4f}")
print(f"Rapid industrializers: {(results_df['DevelopmentMomentumTier'] == 'rapid_industrializer').sum()}")
print()
print("Top 100:")
print("{:>4}  {:>6}  {:<3}  {:<28} {:>8} {:>8} {:>8} {:>8} {:>10} {:>12}".format(
    "Rank", "Move", "ISO", "Country", "BaseHDI", "HDI2050", "Gain", "Momentum", "Optim.", "Mismatch"
))
for rank, row in results_df.head(100).iterrows():
    cid = row["ISO3"]
    cname = row["Country"]
    hdi = row["HDI_2050"]
    hdi_base = row["HDI_Baseline"]
    gain_val = row["HDI_2050_Gain"]
    print("{:4d}  {:>6}  {:<3}  {:<28} {:8.4f} {:8.4f} {:+8.4f} {:8.2f} {:10.4f} {:+12.4f}".format(
        int(row["Rank"]), row["RankChangeDisplay"], cid, cname[:28], hdi_base, hdi, gain_val,
        row["DevelopmentMomentumScore"], row["HDI_Optimistic"], row["HDI_Baseline_Component_Mismatch"],
    ))

print()
print("=== Highest Development Momentum ===")
momentum_cols = [
    "Rank", "RankChangeDisplay", "ISO3", "Country", "HDI_Baseline", "HDI_2050",
    "HDI_2050_Gain", "DevelopmentMomentumScore", "DevelopmentMomentumTier",
]
print(results_df.sort_values("DevelopmentMomentumScore", ascending=False).head(25)[momentum_cols].to_string(index=False))

print()
print("=== Countries of Interest ===")
focus = ["ISR", "KAZ", "ARE", "SAU", "QAT", "KWT", "BHR", "OMN", "SRB", "HRV", "BIH", "MKD", "MNE", "ALB"]
for _, row in results_df[results_df["ISO3"].isin(focus)].iterrows():
    rank = int(row["Rank"])
    cid = row["ISO3"]
    cname = row["Country"]
    hdi = row["HDI_2050"]
    hdi_2024 = row["HDI_2024"]
    gain = row["HDI_2050_Gain"]
    inst = row["InstEfficiency"]
    tech = row["TechFactor"]
    demo = row["DemoFactor"]
    print("Rank {:3d}: {} {}: {:.4f} (gain: +{:.4f}) [inst={:.2f}, tech={:.2f}, demo={:.2f}]".format(
        rank, cid, cname, hdi, gain, inst, tech, demo))
