"""Ethnic composition projections to 2050 -- evidence-based AI model.

Standalone runner built on ``data/ethnicity_model.py`` (the evidence-based
demographic model: national TFR from UN WPP, per-group TFR estimation, age
momentum, corridor migration, fertility convergence, assimilation). Migration intake is
**demographic-pressure-driven**: countries with below-replacement fertility and
projected population decline (aging / shrinking workforces) need to import
labour, so their inward-migration intensity rises over the projection window,
scaled by policy openness. Reads UN population projections
(``data/population_weights_2024_2050.csv``) for the pressure term and for
absolute group sizes, and writes:

    data/output/ethnic_composition_2050_ai.csv      (long form, all groups)
    data/output/ethnic_composition_2050_ai_wide.csv (top groups per country)
    data/output/ethnic_composition_2050_ai_model.csv (model parameters per
        group: TFR 2024, TFR 2050, dev decomposition) -- full transparency.
    data/output/migration_corridors_2050.csv (directional origin-destination
        expansion signals with low/high migration sensitivity).

The old profile-based runner (``run_ethnicity_2050.py``) is kept unchanged;
this runner produces auditable evidence-based outputs in parallel.
"""

import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import config  # noqa: E402
import data.ethnicity_model as model  # noqa: E402
from data.countries import COUNTRY_NAMES  # noqa: E402
from data.undp_hdi import UNDP_HDI_COUNTRIES_193  # noqa: E402

POP_WEIGHTS = Path("data/population_weights_2024_2050.csv")
HDI_RANKINGS = Path("data/output/hdi_2050_rankings.csv")
MIGRATION_SCENARIO = "baseline"
FERTILITY_CONVERGENCE = 0.6
ASSIMILATION_DEFAULT = 0.0010


def clamp01(value: float | None) -> float:
    """Clamp a numeric score to [0, 1], treating missing values as 0."""
    if value is None or not pd.notna(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def load_population_maps() -> tuple[dict, dict]:
    """Return ({iso3: population_2024}, {iso3: population_2050})."""
    if not POP_WEIGHTS.exists():
        return {}, {}
    df = pd.read_csv(POP_WEIGHTS)
    return (dict(zip(df["ISO3"], df["Population_2024"])),
            dict(zip(df["ISO3"], df["Population_2050"])))


def load_hdi_context() -> dict:
    """Country-level HDI context used for demographic cross-validation."""
    if not HDI_RANKINGS.exists():
        return {}
    cols = [
        "ISO3", "HDI_2050_Gain", "HealthIndex_2050",
        "EducationIndex_2050", "IncomeIndex_2050", "Urbanization_2024",
        "WorkingAgePct_2050", "HumanCapitalAbsorption",
        "Youth014Pct_2024", "Youth014Pct_2050", "WorkingAgePct_2024",
        "Elderly65PlusPct_2024", "Elderly65PlusPct_2050",
        "DependencyPressure", "AgingPressure", "EducationIndex_2025",
        "DigitalInfraDevelopment", "ClimateRisk_2024", "ResourceDrag",
        "Contrib_Climate",
    ]
    df = pd.read_csv(HDI_RANKINGS, usecols=lambda c: c in cols)
    return df.set_index("ISO3").to_dict("index")


def urbanization_layer(urbanization: float | None) -> str:
    if urbanization is None or not pd.notna(urbanization):
        return "unknown"
    if urbanization >= 0.78:
        return "metropolitan-heavy"
    if urbanization >= 0.58:
        return "urbanizing-majority"
    if urbanization >= 0.38:
        return "mixed-rural-urban"
    return "rural-heavy"


def build_table(pop2024: dict, pop2050: dict, hdi_context: dict) -> pd.DataFrame:
    rows = []
    for iso3 in sorted(UNDP_HDI_COUNTRIES_193):
        entries = model._normalised_entries(iso3)
        names = [e[0] for e in entries]
        shares_2024 = [e[1] for e in entries]
        shares_frac_2024 = [s / 100.0 for s in shares_2024]
        ctx = hdi_context.get(iso3, {})
        projection_age_context = {
            "youth_share_2024": ctx.get("Youth014Pct_2024"),
            "youth_share_2050": ctx.get("Youth014Pct_2050"),
            "working_age_share_2024": ctx.get("WorkingAgePct_2024"),
            "working_age_share_2050": ctx.get("WorkingAgePct_2050"),
            "elderly_share_2024": ctx.get("Elderly65PlusPct_2024"),
            "elderly_share_2050": ctx.get("Elderly65PlusPct_2050"),
        }
        education_context = ctx.get("EducationIndex_2050")
        income_context = ctx.get("IncomeIndex_2050")

        proj = model.project_ethnic_composition(
            iso3,
            migration_scenario=MIGRATION_SCENARIO,
            fertility_convergence=FERTILITY_CONVERGENCE,
            pop_2024=pop2024.get(iso3),
            pop_2050=pop2050.get(iso3),
            education_index=education_context,
            income_index=income_context,
            **projection_age_context,
        )
        closed_migration_proj = model.project_ethnic_composition(
            iso3,
            migration_scale=0.0,
            fertility_convergence=FERTILITY_CONVERGENCE,
            pop_2024=pop2024.get(iso3),
            pop_2050=pop2050.get(iso3),
            education_index=education_context,
            income_index=income_context,
            **projection_age_context,
        )
        low_migration_proj = model.project_ethnic_composition(
            iso3,
            migration_scale=0.70,
            fertility_convergence=min(0.85, FERTILITY_CONVERGENCE + 0.05),
            pop_2024=pop2024.get(iso3),
            pop_2050=pop2050.get(iso3),
            education_index=education_context,
            income_index=income_context,
            **projection_age_context,
        )
        high_migration_proj = model.project_ethnic_composition(
            iso3,
            migration_scale=1.35,
            fertility_convergence=max(0.35, FERTILITY_CONVERGENCE - 0.05),
            pop_2024=pop2024.get(iso3),
            pop_2050=pop2050.get(iso3),
            education_index=education_context,
            income_index=income_context,
            **projection_age_context,
        )
        without_ssa_source_pool_proj = model.project_ethnic_composition(
            iso3,
            migration_scenario=MIGRATION_SCENARIO,
            fertility_convergence=FERTILITY_CONVERGENCE,
            pop_2024=pop2024.get(iso3),
            pop_2050=pop2050.get(iso3),
            ssa_source_pool_scale=0.0,
            education_index=education_context,
            income_index=income_context,
            **projection_age_context,
        )

        pressure = model.demographic_pressure(
            iso3, pop2024.get(iso3), pop2050.get(iso3))
        birth_replacement_pressure = model.europe_birth_replacement_pressure(
            iso3, pop2024.get(iso3), pop2050.get(iso3))
        birth_replacement_response = model.europe_birth_replacement_migration_boost(
            iso3, 1.0, pop2024.get(iso3), pop2050.get(iso3))
        mig_intensity_2050 = model.effective_migration_intensity(
            iso3, 1.0, pop2024.get(iso3), pop2050.get(iso3))
        skilled_source_pressure = model.skilled_migration_source_pressure(iso3)
        skilled_program_intensity = model.skilled_migration_program_intensity(iso3)
        broad_labor_source_pressure = model.broad_labor_migration_source_pressure(iso3)
        broad_labor_program_intensity = model.broad_labor_migration_program_intensity(iso3)
        ssa_late_destination_response = model.ssa_late_migration_destination_response(iso3)
        policy_openness = model.MIGRATION_POLICY_OPENNESS.get(iso3, 1.0)
        intermarriage = model.INTERMARRIAGE_INDEX.get(iso3, 0.0)
        assimilation = model.COUNTRY_ASSIMILATION.get(
            iso3, model.DEFAULT_ASSIMILATION_RATE)
        urbanization = ctx.get("Urbanization_2024")
        hdi_gain = ctx.get("HDI_2050_Gain")
        hdi_health = ctx.get("HealthIndex_2050")
        hdi_education = ctx.get("EducationIndex_2050")
        hdi_income = ctx.get("IncomeIndex_2050")
        hdi_human_capital = ctx.get("HumanCapitalAbsorption")
        hdi_dependency = ctx.get("DependencyPressure")
        hdi_aging = ctx.get("AgingPressure")
        hdi_working_age = ctx.get("WorkingAgePct_2050")
        hdi_education_2025 = ctx.get("EducationIndex_2025")
        hdi_digital = ctx.get("DigitalInfraDevelopment")
        hdi_climate_risk = ctx.get("ClimateRisk_2024")
        hdi_resource_drag = ctx.get("ResourceDrag")
        hdi_contrib_climate = ctx.get("Contrib_Climate")
        diversity_2024 = 1.0 - sum(s * s for s in shares_frac_2024)
        policy_feedback = policy_openness * (
            1.0 + pressure * 0.35 + birth_replacement_response * 0.20)
        urban_absorption_pressure = (
            pressure * (1.0 - float(urbanization))
            if urbanization is not None and pd.notna(urbanization) else None
        )
        diversity_2050_base = 1.0 - sum(float(v) * float(v) for v in proj.values())
        projected_migration_linked_share = sum(
            float(proj.get(entry[0], 0.0)) for entry in entries
            if entry[2] == "immigrant")
        migration_soft_cap = model.migration_stock_soft_cap(iso3)
        migration_saturation_factor = max(0.22, min(
            1.0,
            1.0 - 0.62 * (
                projected_migration_linked_share /
                max(migration_soft_cap, 0.05)) ** 1.6,
        ))
        polarization_rq_2050_base = sum(
            4.0 * float(v) * ((1.0 - float(v)) ** 2)
            for v in proj.values()
        )
        human_capital_value = (
            float(hdi_human_capital)
            if hdi_human_capital is not None and pd.notna(hdi_human_capital)
            else 0.0
        )
        # Identity composition is descriptive context, never a direct penalty.
        # The service-delivery gap is derived only from observable capacity and
        # access conditions that policy can change.
        structural_inequality_drag_base = clamp01(
            0.30 * (urban_absorption_pressure or 0.0) +
            0.28 * max(0.0, 1.0 - policy_openness) +
            0.22 * max(0.0, 1.0 - human_capital_value) +
            0.12 * clamp01(hdi_dependency) +
            0.08 * clamp01(hdi_climate_risk)
        )

        anchor_name = max(entries, key=lambda e: e[1])[0]
        brazil_intermarriage_multiplier = 1.0 + min(
            2.0,
            diversity_2024 * 1.15 +
            (float(urbanization) if urbanization is not None and pd.notna(urbanization) else 0.45) * 0.55 +
            intermarriage * 2.5
        )
        brazil_mixed_multiplier = 1.25 + min(2.0, diversity_2024 * 1.4 + intermarriage * 3.0)
        brazil_assimilation_rate = max(0.0, assimilation * 0.45)
        brazil_fertility_convergence = min(0.9, FERTILITY_CONVERGENCE + 0.18)
        brazil_proj = model.project_ethnic_composition(
            iso3,
            migration_scenario=MIGRATION_SCENARIO,
            fertility_convergence=brazil_fertility_convergence,
            assimilation_rate=brazil_assimilation_rate,
            intermarriage_multiplier=brazil_intermarriage_multiplier,
            mixed_identity_multiplier=brazil_mixed_multiplier,
            education_index=hdi_education,
            income_index=hdi_income,
            structural_inequality_drag=structural_inequality_drag_base,
            pop_2024=pop2024.get(iso3),
            pop_2050=pop2050.get(iso3),
            **projection_age_context,
        )

        # Parameter detail rows for the transparency file
        for name, share2024, profile, tfr2024 in entries:
            tfr2050 = model._group_tfr_2050(iso3, name, tfr2024)
            share2050 = proj[name] * 100.0
            closed_migration_share2050 = closed_migration_proj.get(name, proj[name]) * 100.0
            pop_2024 = share2024 / 100.0 * pop2024[iso3] if iso3 in pop2024 else None
            pop_2050 = share2050 / 100.0 * pop2050[iso3] if iso3 in pop2050 else None
            diversity_2050 = diversity_2050_base
            polarization_rq_2050 = polarization_rq_2050_base
            effective_groups_2050 = 1.0 / max(
                1e-9, sum(float(v) * float(v) for v in proj.values()))
            structural_inequality_drag = structural_inequality_drag_base
            brazil_share2050 = brazil_proj.get(name, proj[name]) * 100.0
            low_migration_share2050 = low_migration_proj.get(name, proj[name]) * 100.0
            high_migration_share2050 = high_migration_proj.get(name, proj[name]) * 100.0
            without_ssa_source_pool_share2050 = (
                without_ssa_source_pool_proj.get(name, proj[name]) * 100.0)
            migration_low = min(low_migration_share2050, high_migration_share2050)
            migration_high = max(low_migration_share2050, high_migration_share2050)
            corridor = model.migration_corridor_diagnostics(
                iso3,
                name,
                profile,
                share2024 / 100.0,
                1.0,
                pressure,
                birth_replacement_pressure,
            )
            migration_channel_active = bool(
                (iso3, name) in model.GROUP_MIGRATION_INTENSITY or
                (iso3, name) in model.GROUP_SKILLED_MIGRATION_SURGE or
                (iso3, name) in model.GROUP_BROAD_LABOR_MIGRATION_SURGE or
                (iso3, name) in model.GROUP_SSA_SOURCE_POOL_SURGE or
                profile == "immigrant"
            )
            brazil_diversity_2050 = 1.0 - sum(float(v) * float(v) for v in brazil_proj.values())
            education_value = clamp01(
                hdi_education
                if hdi_education is not None and pd.notna(hdi_education)
                else hdi_education_2025
            )
            digital_value = clamp01(hdi_digital)
            dependency_value = clamp01(hdi_dependency)
            urban_value = clamp01(urbanization)
            climate_risk_value = clamp01(hdi_climate_risk)
            climate_drag_value = max(
                0.0,
                clamp01(-(hdi_resource_drag or 0.0)) +
                clamp01(-(hdi_contrib_climate or 0.0)),
            ) / 2.0
            mobility_convergence = clamp01(
                0.18 + 0.36 * human_capital_value +
                0.24 * education_value + 0.12 * digital_value +
                0.10 * policy_openness - 0.14 * dependency_value -
                0.10 * structural_inequality_drag
            )
            subnational_concentration = clamp01(
                0.36 * urban_value + 0.24 * pressure +
                0.18 * mig_intensity_2050 / 3.0 +
                0.14 * (urban_absorption_pressure or 0.0) +
                0.08 * climate_risk_value
            )
            climate_migration_stress = clamp01(
                0.38 * climate_risk_value + 0.24 * climate_drag_value +
                0.18 * pressure + 0.10 * (urban_absorption_pressure or 0.0) +
                0.10 * skilled_source_pressure
            )
            language_integration = clamp01(
                0.18 + 0.26 * education_value + 0.22 * digital_value +
                0.20 * policy_openness + 0.14 * human_capital_value
            )
            language_friction = clamp01(
                0.28 * max(0.0, 1.0 - policy_openness) +
                0.24 * max(0.0, 1.0 - education_value) +
                0.20 * max(0.0, 1.0 - digital_value) +
                0.16 * subnational_concentration +
                0.12 * (urban_absorption_pressure or 0.0)
            )
            rows.append({
                "ISO3": iso3,
                "Country": COUNTRY_NAMES.get(iso3, iso3),
                "Group": name,
                "Profile": profile,
                "Baseline_Evidence_Status": "compiled_category_baseline_source_review_required",
                "Projection_Evidence_Status": "cohort_informed_scenario_not_official_forecast",
                "Uncertainty_Evidence_Status": "migration_sensitivity_range_not_calibrated_probability",
                "TFR_2024": round(tfr2024, 2),
                "TFR_2050": round(tfr2050, 2),
                "Nat_TFR_2024": round(model.NATIONAL_TFR_2024.get(iso3, 2.1), 2),
                "Nat_TFR_2050": round(model.NATIONAL_TFR_2050.get(iso3, 2.1), 2),
                "Share_2024_pct": round(share2024, 2),
                "Share_2050_pct": round(share2050, 2),
                "Change_pp": round(share2050 - share2024, 2),
                "Share_2050_WithoutMigrationResponse_pct": round(closed_migration_share2050, 2),
                "Migration_Attributed_Change_pp": round(
                    share2050 - closed_migration_share2050, 2),
                "Share_2050_WithoutSSASourcePoolTransition_pct": round(
                    without_ssa_source_pool_share2050, 2),
                "SSA_SourcePoolTransition_Attributed_Change_pp": round(
                    share2050 - without_ssa_source_pool_share2050, 2),
                "Share_2050_LowMigrationScenario_pct": round(low_migration_share2050, 2),
                "Share_2050_HighMigrationScenario_pct": round(high_migration_share2050, 2),
                "Share_2050_Migration_P10_pct": round(migration_low, 2),
                "Share_2050_Migration_P90_pct": round(migration_high, 2),
                "Migration_Uncertainty_Width_pp": round(migration_high - migration_low, 2),
                "Brazilification_Share_2050_pct": round(brazil_share2050, 2),
                "Brazilification_Change_pp": round(brazil_share2050 - share2024, 2),
                "Brazilification_Delta_vs_Baseline_pp": round(brazil_share2050 - share2050, 2),
                "Pop_2024": None if pop_2024 is None else round(pop_2024),
                "Pop_2050": None if pop_2050 is None else round(pop_2050),
                "Demographic_Pressure": round(pressure, 3),
                "Birth_Replacement_Pressure_2050": round(birth_replacement_pressure, 3),
                "Europe_Migration_Response_2050": round(birth_replacement_response, 3),
                "Migration_Intensity_2050": round(mig_intensity_2050, 3),
                "Migration_Origin_Candidates": corridor["origins"],
                "Migration_Channel_Active": migration_channel_active,
                "Migration_SourceSupply_2050": round(float(corridor["source_supply"]), 3),
                "Migration_BroadLaborMobility_2050": round(
                    float(corridor["broad_labor_mobility"]), 3),
                "Migration_SkillReadiness_2050": round(float(corridor["skill_readiness"]), 3),
                "Migration_ClimatePressure_2050": round(float(corridor["climate_pressure"]), 3),
                "Migration_DestinationPull_2050": round(float(corridor["destination_pull"]), 3),
                "Migration_DiasporaNetwork_2050": round(float(corridor["diaspora_network"]), 3),
                "Migration_CorridorAffinity_2050": round(float(corridor["corridor_affinity"]), 3),
                "Migration_SettlementRetention_2050": round(float(corridor["settlement_retention"]), 3),
                "Migration_CorridorMultiplier_2050": round(float(corridor["corridor_multiplier"]), 3),
                "Skilled_Migration_SourcePressure_2050": round(skilled_source_pressure, 3),
                "Skilled_Migration_ProgramIntensity_2050": round(skilled_program_intensity, 3),
                "BroadLabor_Migration_SourcePressure_2050": round(
                    broad_labor_source_pressure, 3),
                "BroadLabor_Migration_ProgramIntensity_2050": round(
                    broad_labor_program_intensity, 3),
                "SSA_LateMigration_DestinationResponse_2050": round(
                    ssa_late_destination_response, 3),
                "SSA_SourcePoolTransition_2050": round(
                    model.late_ssa_source_pool_transition(1.0), 3),
                "SSA_SourcePoolCapacity_2050": round(
                    model.ssa_source_pool_capacity(), 3),
                "AgeStructure_Youth014_2024": round(
                    float(ctx.get("Youth014Pct_2024", 0.0) or 0.0), 3),
                "AgeStructure_Youth014_2050": round(
                    float(ctx.get("Youth014Pct_2050", 0.0) or 0.0), 3),
                "AgeStructure_WorkingAge_2024": round(
                    float(ctx.get("WorkingAgePct_2024", 0.0) or 0.0), 3),
                "AgeStructure_WorkingAge_2050": round(
                    float(ctx.get("WorkingAgePct_2050", 0.0) or 0.0), 3),
                "AgeStructure_Elderly65Plus_2024": round(
                    float(ctx.get("Elderly65PlusPct_2024", 0.0) or 0.0), 3),
                "AgeStructure_Elderly65Plus_2050": round(
                    float(ctx.get("Elderly65PlusPct_2050", 0.0) or 0.0), 3),
                "Migration_Stock_SoftCap_2050": round(
                    migration_soft_cap, 3),
                "Migration_Linked_ResidentShare_2050": round(
                    projected_migration_linked_share, 3),
                "Migration_Saturation_Factor_2050": round(
                    migration_saturation_factor, 3),
                "Migration_Absorption_Capacity_2050": round(
                    model.migration_absorption_capacity(
                        iso3, education_context, income_context), 3),
                "Policy_Openness": round(policy_openness, 3),
                "Policy_Feedback_2050": round(policy_feedback, 3),
                "Intermarriage_Coefficient": round(intermarriage, 3),
                "Assimilation_Coefficient": round(assimilation, 4),
                "Diversity_Index_2024": round(diversity_2024, 4),
                "Diversity_Index_2050": round(diversity_2050, 4),
                "Diversity_Index_Change": round(diversity_2050 - diversity_2024, 4),
                "Ethnic_Fractionalization_2050": round(diversity_2050, 4),
                "Ethnolinguistic_Polarization_RQ_2050": round(polarization_rq_2050, 4),
                "Structural_Inequality_Drag_2050": round(structural_inequality_drag, 4),
                "Brazilification_IntermarriageMultiplier": round(brazil_intermarriage_multiplier, 3),
                "Brazilification_MixedIdentityMultiplier": round(brazil_mixed_multiplier, 3),
                "Brazilification_AssimilationRate": round(brazil_assimilation_rate, 4),
                "Brazilification_Diversity_Index_2050": round(brazil_diversity_2050, 4),
                "Inclusive_Mobility_Share_2050_pct": round(brazil_share2050, 2),
                "Inclusive_Mobility_Change_pp": round(brazil_share2050 - share2024, 2),
                "Inclusive_Mobility_Delta_vs_Baseline_pp": round(brazil_share2050 - share2050, 2),
                "Inclusive_Mobility_IdentityFormationMultiplier": round(brazil_intermarriage_multiplier, 3),
                "Inclusive_Mobility_MixedIdentityMultiplier": round(brazil_mixed_multiplier, 3),
                "Inclusive_Mobility_IdentityTransitionRate": round(brazil_assimilation_rate, 4),
                "Inclusive_Mobility_CompositionDiversity_2050": round(brazil_diversity_2050, 4),
                "Intergenerational_Mobility_Convergence_2050": round(mobility_convergence, 4),
                "Subnational_Regional_Concentration_2050": round(subnational_concentration, 4),
                "Climate_Migration_Stress_2050": round(climate_migration_stress, 4),
                "Language_Cultural_Integration_2050": round(language_integration, 4),
                "Language_Cultural_Friction_2050": round(language_friction, 4),
                "Effective_Groups_2050": round(effective_groups_2050, 2),
                "Urbanization_2024": None if urbanization is None or not pd.notna(urbanization) else round(float(urbanization), 3),
                "Urbanization_Layer": urbanization_layer(urbanization),
                "Urban_Absorption_Pressure": None if urban_absorption_pressure is None else round(urban_absorption_pressure, 3),
                "HDI_2050_Gain_Link": None if hdi_gain is None or not pd.notna(hdi_gain) else round(float(hdi_gain), 4),
                "HDI_HealthIndex_2050_Link": None if hdi_health is None or not pd.notna(hdi_health) else round(float(hdi_health), 4),
                "HDI_EducationIndex_2050_Link": None if hdi_education is None or not pd.notna(hdi_education) else round(float(hdi_education), 4),
                "HDI_IncomeIndex_2050_Link": None if hdi_income is None or not pd.notna(hdi_income) else round(float(hdi_income), 4),
                "HDI_HumanCapitalAbsorption_Link": None if hdi_human_capital is None or not pd.notna(hdi_human_capital) else round(float(hdi_human_capital), 4),
                "HDI_DependencyPressure_Link": None if hdi_dependency is None or not pd.notna(hdi_dependency) else round(float(hdi_dependency), 4),
                "HDI_AgingPressure_Link": None if hdi_aging is None or not pd.notna(hdi_aging) else round(float(hdi_aging), 4),
                "HDI_WorkingAgePct_2050_Link": None if hdi_working_age is None or not pd.notna(hdi_working_age) else round(float(hdi_working_age), 4),
                "Anchor": name == anchor_name,
            })
    return pd.DataFrame(rows)


def main():
    pop2024, pop2050 = load_population_maps()
    hdi_context = load_hdi_context()
    table = build_table(pop2024, pop2050, hdi_context)

    print("=" * 78)
    print("  POPULATION AND IDENTITY CONTEXT TO 2050 -- SCENARIO MODEL")
    print("  Engine: data/ethnicity_model.py (national TFR + per-group TFR +")
    print("          age momentum + migration + fertility convergence + identity-category transition)")
    print(f"  Migration scenario: {MIGRATION_SCENARIO} | "
          f"fertility convergence: {FERTILITY_CONVERGENCE}")
    print("=" * 78)

    # ---- validation -------------------------------------------------------
    missing = set(UNDP_HDI_COUNTRIES_193) - set(table["ISO3"])
    if missing:
        raise RuntimeError(f"Missing ethnic composition for: {sorted(missing)}")
    for col in ["Share_2024_pct", "Share_2050_pct"]:
        sums = table.groupby("ISO3")[col].sum()
        bad = sums[abs(sums - 100.0) > 0.15]
        if not bad.empty:
            raise RuntimeError(f"{col} not summing to 100 for: {bad.to_dict()}")

    out_dir = config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    long_path = out_dir / "ethnic_composition_2050_ai.csv"
    model_path = out_dir / "ethnic_composition_2050_ai_model.csv"
    public_path = out_dir / "demographic_context_2050.csv"
    table.drop(columns=["Profile", "TFR_2024", "TFR_2050",
                        "Nat_TFR_2024", "Nat_TFR_2050",
                        "Brazilification_Share_2050_pct",
                        "Brazilification_Change_pp",
                        "Brazilification_Delta_vs_Baseline_pp",
                        "Demographic_Pressure", "Migration_Intensity_2050",
                        "Skilled_Migration_SourcePressure_2050",
                        "Skilled_Migration_ProgramIntensity_2050",
                        "Policy_Openness", "Policy_Feedback_2050",
                        "Intermarriage_Coefficient", "Assimilation_Coefficient",
                        "Diversity_Index_2024", "Diversity_Index_2050",
                        "Diversity_Index_Change", "Ethnic_Fractionalization_2050",
                        "Ethnolinguistic_Polarization_RQ_2050",
                        "Structural_Inequality_Drag_2050",
                        "Brazilification_IntermarriageMultiplier",
                        "Brazilification_MixedIdentityMultiplier",
                        "Brazilification_AssimilationRate",
                        "Brazilification_Diversity_Index_2050",
                        "Intergenerational_Mobility_Convergence_2050",
                        "Subnational_Regional_Concentration_2050",
                        "Climate_Migration_Stress_2050",
                        "Language_Cultural_Integration_2050",
                        "Language_Cultural_Friction_2050",
                        "Effective_Groups_2050",
                        "Urbanization_2024", "Urbanization_Layer",
                        "Urban_Absorption_Pressure", "HDI_2050_Gain_Link",
                        "HDI_HealthIndex_2050_Link",
                        "HDI_EducationIndex_2050_Link",
                        "HDI_IncomeIndex_2050_Link",
                        "HDI_HumanCapitalAbsorption_Link",
                        "HDI_DependencyPressure_Link", "HDI_AgingPressure_Link",
                        "HDI_WorkingAgePct_2050_Link"]).to_csv(long_path, index=False)
    table.to_csv(model_path, index=False)

    # Publish an ethics-safe research table. Legacy field names remain only in
    # the internal model file for reproducibility and backward compatibility.
    public_drop = [c for c in table.columns if c.startswith("Brazilification_")]
    public_table = table.drop(columns=public_drop).rename(columns={
        "Profile": "Projection_Profile",
        "Anchor": "Baseline_Reference_Category",
        "Intermarriage_Coefficient": "Mixed_Identity_Recognition_Rate",
        "Assimilation_Coefficient": "Identity_Category_Transition_Rate",
        "Ethnic_Fractionalization_2050": "Composition_Diversity_Index_2050",
        "Ethnolinguistic_Polarization_RQ_2050": "Composition_Concentration_Index_2050",
        "Structural_Inequality_Drag_2050": "Inclusive_Service_Delivery_Gap_2050",
        "Language_Cultural_Integration_2050": "Language_Access_Capacity_2050",
        "Language_Cultural_Friction_2050": "Language_Access_Gap_2050",
    })
    public_table["Projection_Profile"] = public_table["Projection_Profile"].replace({
        "majority": "largest_baseline_category",
        "immigrant": "migration_linked",
        "assimilating": "identity_category_transition",
        "high_fertility": "higher_fertility_path",
        "low_fertility": "lower_fertility_path",
    })
    public_table.to_csv(public_path, index=False)

    # Corridor-level research output. This ranks destination-group pathways by
    # future expansion conditions; it is a directional scenario index, not a
    # forecast of exact migrant counts or a claim about individual behavior.
    corridor_table = table.loc[
        table["Migration_Origin_Candidates"].ne("unspecified") &
        table["Migration_Channel_Active"],
        [
            "ISO3", "Country", "Group", "Migration_Origin_Candidates",
            "Share_2024_pct", "Share_2050_pct", "Change_pp", "Pop_2024", "Pop_2050",
            "Share_2050_LowMigrationScenario_pct",
            "Share_2050_HighMigrationScenario_pct",
            "Share_2050_Migration_P10_pct", "Share_2050_Migration_P90_pct",
            "Migration_Uncertainty_Width_pp", "Migration_SourceSupply_2050",
            "Migration_BroadLaborMobility_2050",
            "Migration_SkillReadiness_2050", "Migration_ClimatePressure_2050",
            "Migration_DestinationPull_2050", "Migration_DiasporaNetwork_2050",
            "Migration_CorridorAffinity_2050",
            "Migration_SettlementRetention_2050",
            "Migration_CorridorMultiplier_2050", "Migration_Intensity_2050",
            "Birth_Replacement_Pressure_2050", "Europe_Migration_Response_2050",
            "BroadLabor_Migration_SourcePressure_2050",
            "BroadLabor_Migration_ProgramIntensity_2050",
            "SSA_LateMigration_DestinationResponse_2050",
            "SSA_SourcePoolTransition_2050", "SSA_SourcePoolCapacity_2050",
            "SSA_SourcePoolTransition_Attributed_Change_pp",
            "Migration_Channel_Active",
        ],
    ].copy()
    corridor_table = corridor_table.rename(columns={
        "ISO3": "Destination_ISO3",
        "Country": "Destination",
        "Group": "Destination_Group",
        "Migration_Origin_Candidates": "Origin_Candidates",
    })
    corridor_table["Corridor_Expansion_Index_2050"] = (
        corridor_table["Migration_CorridorMultiplier_2050"] *
        corridor_table["Migration_Intensity_2050"] *
        (0.55 + 0.45 * corridor_table["Migration_SkillReadiness_2050"])
    ).clip(0.0, 5.0).round(3)
    corridor_table["Projected_Group_Population_Change_pct"] = (
        100.0 * (corridor_table["Pop_2050"] / corridor_table["Pop_2024"] - 1.0)
    ).replace([float("inf"), float("-inf")], pd.NA).round(1)
    corridor_table["Expected_Presence_2050"] = corridor_table.apply(
        lambda row: (
            "strong expansion" if row["Corridor_Expansion_Index_2050"] >= 1.65
            and pd.notna(row["Projected_Group_Population_Change_pct"])
            and row["Projected_Group_Population_Change_pct"] >= 25.0
            else "moderate expansion" if row["Corridor_Expansion_Index_2050"] >= 1.15
            and pd.notna(row["Projected_Group_Population_Change_pct"])
            and row["Projected_Group_Population_Change_pct"] >= 5.0
            else "established/gradual"
        ),
        axis=1,
    )
    corridor_table["Interpretation"] = (
        "Directional corridor scenario based on source labor supply, skills, "
        "destination demand, policy, diaspora networks, climate pressure and "
        "settlement retention; not an official bilateral flow forecast."
    )
    corridor_table = corridor_table.sort_values(
        ["Corridor_Expansion_Index_2050", "Migration_Uncertainty_Width_pp"],
        ascending=False,
    )
    corridor_path = out_dir / "migration_corridors_2050.csv"
    corridor_table.to_csv(corridor_path, index=False)

    # wide summary: top 5 groups per country by 2050 share
    wide_rows = []
    for iso3, grp in table.groupby("ISO3"):
        top = grp.sort_values("Share_2050_pct", ascending=False).head(5)
        row = {"ISO3": iso3, "Country": top["Country"].iloc[0]}
        for i, (_, r) in enumerate(top.iterrows(), start=1):
            row[f"Top{i}"] = r["Group"]
            row[f"Top{i}_2050_pct"] = round(r["Share_2050_pct"], 1)
        wide_rows.append(row)
    wide = pd.DataFrame(wide_rows).sort_values("ISO3")
    wide_path = out_dir / "ethnic_composition_2050_ai_wide.csv"
    wide.to_csv(wide_path, index=False)

    print(f"\n  Long-form table : {long_path}  ({len(table)} rows)")
    print(f"  Model detail    : {model_path}")
    print(f"  Public context  : {public_path}")
    print(f"  Migration paths : {corridor_path}  ({len(corridor_table)} corridors)")
    print(f"  Wide summary    : {wide_path}  ({len(wide)} countries)")

    # ---- audit ------------------------------------------------------------
    print("\n  === AUDIT ===")
    print(f"  Countries projected        : {table['ISO3'].nunique()}")
    print(f"  Ethnic groups in table     : {len(table)}")
    print(f"  Median |change| per group  : {table['Change_pp'].abs().median():.2f} pp")
    print(f"  Mean |change| per group    : {table['Change_pp'].abs().mean():.2f} pp")

    anchor_change = table.groupby("ISO3").apply(
        lambda g: g.loc[g["Anchor"], "Change_pp"].sum(), include_groups=False
    )
    print("\n  Top 10 baseline-reference category share declines:")
    print(anchor_change.nsmallest(10).to_string())
    print("\n  Top 10 baseline-reference category share gains:")
    print(anchor_change.nlargest(10).to_string())

    print("\n  Largest category gains (pp, 2024->2050):")
    print(table.sort_values("Change_pp", ascending=False).head(12)[
        ["ISO3", "Country", "Group", "Share_2024_pct", "Share_2050_pct", "Change_pp"]
    ].to_string(index=False))

    print("\n  Largest category losses (pp, 2024->2050):")
    print(table.sort_values("Change_pp").head(12)[
        ["ISO3", "Country", "Group", "Share_2024_pct", "Share_2050_pct", "Change_pp"]
    ].to_string(index=False))

    print("\n  === DEMOGRAPHIC PRESSURE ===")
    print("  Countries projected to need the most inward migration by 2050 "
          "(high fertility shortfall + population decline):")
    press = table.groupby("ISO3", as_index=False).first()[
        ["ISO3", "Country", "Demographic_Pressure", "Migration_Intensity_2050",
         "Skilled_Migration_SourcePressure_2050",
         "Skilled_Migration_ProgramIntensity_2050",
         "Policy_Openness"]].sort_values(
        "Demographic_Pressure", ascending=False).head(15)
    print(press.to_string(index=False))
    print("\n  Largest migration-linked category gains in aging countries (pp):")
    aging = table[(table["Demographic_Pressure"] > 0.4)
                  & (table["Profile"] == "immigrant")].sort_values(
        "Change_pp", ascending=False)
    print(aging.head(10)[
        ["ISO3", "Country", "Group", "Share_2024_pct", "Share_2050_pct", "Change_pp"]
    ].to_string(index=False))

    print(f"\n  Population totals: sum(Pop_2050) = {table['Pop_2050'].sum():,.0f} "
          f"(world 2050 ~9.7bn; listed groups only)")
    print("  Done.")


if __name__ == "__main__":
    main()
