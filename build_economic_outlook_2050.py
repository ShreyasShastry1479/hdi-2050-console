"""Build the 193-country economic outlook used by the dashboard.

Observed GDP baselines come from the World Bank API when available. The output
keeps a constant-2021-dollar audit layer for HDI income-index reconciliation and
also reports inflation-adjusted 2050-dollar scenario values. These are not
official World Bank, IMF, or UN forecasts.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent
HDI_PATH = ROOT / "data" / "output" / "hdi_2050_rankings.csv"
OUTPUT_PATH = ROOT / "data" / "output" / "economic_outlook_2050.csv"
BASELINE_CACHE_PATH = ROOT / "data" / "economic_world_bank_baselines.csv"
WORLD_BANK_URL = "https://api.worldbank.org/v2/country/{countries}/indicator/{indicator}"
INDICATORS = {
    "GDP_PPP_PerCapita_Observed": "NY.GDP.PCAP.PP.CD",
    "GDP_Nominal_PerCapita_Observed": "NY.GDP.PCAP.CD",
}

# Both CPI-U observations use the same BLS 2013=100 reference. International
# dollars are benchmarked to US purchasing power, so this common US-dollar price
# factor converts the constant-2021 HDI income anchor into a 2050-price view.
CPI_U_2021 = 116.318
CPI_U_2025 = 138.289
LONG_RUN_INFLATION = 0.020
LOW_INFLATION = 0.015
HIGH_INFLATION = 0.030
INFLATION_YEARS_2025_TO_2050 = 25
OBSERVED_PRICE_FACTOR_2021_TO_2025 = CPI_U_2025 / CPI_U_2021
PRICE_FACTOR_2021_TO_2050 = OBSERVED_PRICE_FACTOR_2021_TO_2025 * (1.0 + LONG_RUN_INFLATION) ** INFLATION_YEARS_2025_TO_2050
PRICE_FACTOR_LOW_2021_TO_2050 = OBSERVED_PRICE_FACTOR_2021_TO_2025 * (1.0 + LOW_INFLATION) ** INFLATION_YEARS_2025_TO_2050
PRICE_FACTOR_HIGH_2021_TO_2050 = OBSERVED_PRICE_FACTOR_2021_TO_2025 * (1.0 + HIGH_INFLATION) ** INFLATION_YEARS_2025_TO_2050


def clamp(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))


def fetch_world_bank_baselines(iso3_codes: list[str]) -> pd.DataFrame:
    country_path = ";".join(iso3_codes)
    records: dict[str, dict[str, object]] = {iso3: {"ISO3": iso3} for iso3 in iso3_codes}
    for output_column, indicator in INDICATORS.items():
        response = requests.get(
            WORLD_BANK_URL.format(countries=country_path, indicator=indicator),
            params={"format": "json", "date": "2022:2024", "per_page": 20000},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        rows = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
        latest: dict[str, tuple[int, float]] = {}
        for row in rows:
            iso3 = str(row.get("countryiso3code") or "")
            value = row.get("value")
            if iso3 not in records or value is None:
                continue
            year = int(row["date"])
            if iso3 not in latest or year > latest[iso3][0]:
                latest[iso3] = (year, float(value))
        for iso3, (year, value) in latest.items():
            records[iso3][output_column] = value
            records[iso3][f"{output_column}_Year"] = year
    result = pd.DataFrame(records.values())
    result.to_csv(BASELINE_CACHE_PATH, index=False)
    return result


def load_world_bank_baselines(iso3_codes: list[str], offline: bool) -> pd.DataFrame:
    if not offline:
        try:
            return fetch_world_bank_baselines(iso3_codes)
        except (requests.RequestException, ValueError) as exc:
            print(f"World Bank refresh failed; using cached baselines: {exc}")
    if BASELINE_CACHE_PATH.exists():
        return pd.read_csv(BASELINE_CACHE_PATH)
    return pd.DataFrame({"ISO3": iso3_codes})


def projection_cagr(row: pd.Series, ppp_per_capita: float) -> float:
    """Estimate annual real GDP-per-capita growth from existing model drivers."""
    log_income = math.log(max(ppp_per_capita, 1_000.0))
    convergence = clamp((math.log(85_000.0) - log_income) / (math.log(85_000.0) - math.log(1_000.0)), 0.0, 1.0)
    institution = clamp((float(row.InstEfficiency) - 0.72) / 0.43, 0.0, 1.0)
    workforce_gain = clamp(float(row.WorkforceChangePP_2024_to_2050) / 12.0, -1.0, 1.0)
    aging = clamp(float(row.AdaptedAgingPressure), 0.0, 1.0)

    annual = (
        0.0030
        + 0.0120 * float(row.GrowthProspectScore)
        + 0.0200 * float(row.DevelopingCatchupReadiness)
        + 0.0140 * float(row.IndustrializationHDIAcceleration)
        + 0.0060 * float(row.FutureReadiness)
        + 0.0050 * float(row.DemographicDividend)
        + 0.0060 * convergence
        + 0.0070 * convergence * float(row.GrowthProspectScore)
        + 0.0040 * float(row.HumanCapitalAbsorption)
        + 0.0020 * float(row.DigitalInfraDevelopment)
        + 0.0040 * institution
        + 0.0020 * workforce_gain
        - 0.0120 * float(row.ResourceDrag)
        - 0.0040 * float(row.GrowthVolatility)
        - 0.0040 * float(row.DependencyPressure)
        - 0.0020 * aging
    )
    return clamp(annual, -0.008, 0.055)


def ppp_from_income_index(income_index: float) -> float:
    """Invert the income-index transform used by the national HDI model."""
    index = clamp(income_index, 0.0, 1.0)
    return float(math.exp(math.log(100.0) + index * (math.log(105_000.0) - math.log(100.0))))


def income_index_from_ppp(ppp_per_capita: float) -> float:
    return clamp(
        (math.log(max(ppp_per_capita, 1.0)) - math.log(100.0))
        / (math.log(105_000.0) - math.log(100.0)),
        0.0,
        1.0,
    )


def nearest_ratio_proxy(rows: pd.DataFrame, target: pd.Series) -> float:
    valid = rows.dropna(subset=["GDP_PPP_PerCapita_Observed", "GDP_Nominal_PerCapita_Observed"]).copy()
    if valid.empty:
        return 0.45
    target_income = math.log(max(float(target.GNI_PPP_2024), 500.0))
    valid["distance"] = (
        (np.log(valid["GDP_PPP_PerCapita_Observed"].clip(lower=500)) - target_income).abs()
        + 0.35 * (valid["ResourceDependence"] - float(target.ResourceDependence)).abs()
        + 0.25 * (valid["EconomicDiversification"] - float(target.EconomicDiversification)).abs()
    )
    peers = valid.nsmallest(12, "distance")
    ratio = (peers["GDP_Nominal_PerCapita_Observed"] / peers["GDP_PPP_PerCapita_Observed"]).median()
    return clamp(float(ratio), 0.08, 1.25)


def build_projection(offline: bool = False) -> pd.DataFrame:
    hdi = pd.read_csv(HDI_PATH)
    if len(hdi) != 193:
        raise RuntimeError(f"Expected 193 HDI rows, found {len(hdi)}")

    baselines = load_world_bank_baselines(hdi.ISO3.tolist(), offline)
    rows = hdi.merge(baselines, on="ISO3", how="left")
    results: list[dict[str, object]] = []

    for _, row in rows.iterrows():
        ppp_observed = row.get("GDP_PPP_PerCapita_Observed")
        ppp_proxy = not pd.notna(ppp_observed)
        ppp_observed = float(ppp_observed) if pd.notna(ppp_observed) else float(row.GNI_PPP_2024) * 1.02
        ppp_year = int(row.get("GDP_PPP_PerCapita_Observed_Year", 2024)) if pd.notna(row.get("GDP_PPP_PerCapita_Observed_Year")) else 2024

        nominal_observed = row.get("GDP_Nominal_PerCapita_Observed")
        nominal_proxy = not pd.notna(nominal_observed)
        if nominal_proxy:
            nominal_ratio = nearest_ratio_proxy(rows, row)
            nominal_observed = ppp_observed * nominal_ratio
            nominal_year = 2024
        else:
            nominal_observed = float(nominal_observed)
            nominal_year = int(row.get("GDP_Nominal_PerCapita_Observed_Year", 2024))

        structural_cagr = projection_cagr(row, ppp_observed)
        population_2025 = float(row.Population_2024) * (float(row.Population_2050) / max(float(row.Population_2024), 1.0)) ** (1.0 / 26.0)
        ppp_2025 = ppp_observed * (1.0 + structural_cagr) ** max(0, 2025 - ppp_year)
        nominal_2025 = nominal_observed * (1.0 + structural_cagr) ** max(0, 2025 - nominal_year)
        ppp_2025_2021_dollars = ppp_2025 / OBSERVED_PRICE_FACTOR_2021_TO_2025
        nominal_2025_2021_dollars = nominal_2025 / OBSERVED_PRICE_FACTOR_2021_TO_2025
        income_index_2050 = clamp(float(row.IncomeIndex_2050), 0.0, 1.0)
        ppp_2050_2021_dollars = round(ppp_from_income_index(income_index_2050), 2)
        ppp_2050 = ppp_2050_2021_dollars * PRICE_FACTOR_2021_TO_2050
        cagr = (ppp_2050_2021_dollars / max(ppp_2025_2021_dollars, 1.0)) ** (1.0 / 25.0) - 1.0
        recomputed_income_index_2050 = income_index_from_ppp(ppp_2050_2021_dollars)

        ratio_2025 = clamp(nominal_2025 / max(ppp_2025, 1.0), 0.08, 1.25)
        convergence = clamp((math.log(85_000.0) - math.log(max(ppp_2025, 1_000.0))) / (math.log(85_000.0) - math.log(1_000.0)), 0.0, 1.0)
        institution = clamp((float(row.InstEfficiency) - 0.72) / 0.43, 0.0, 1.0)
        price_level_gap = clamp((0.90 - ratio_2025) / 0.70, 0.0, 1.0)
        price_level_elasticity = (0.12 + 0.10 * institution + 0.06 * convergence) * price_level_gap
        ratio_2050 = ratio_2025 * (ppp_2050_2021_dollars / max(ppp_2025_2021_dollars, 1.0)) ** price_level_elasticity
        ratio_2050 *= 1.0 - 0.05 * float(row.GrowthVolatility) - 0.04 * float(row.ResourceDrag)
        ratio_2050 = clamp(ratio_2050, max(0.08, ratio_2025 * 0.78), 1.15)
        nominal_2050_2021_dollars = ppp_2050_2021_dollars * ratio_2050
        nominal_2050 = nominal_2050_2021_dollars * PRICE_FACTOR_2021_TO_2050

        uncertainty = clamp(
            0.004 + 0.005 * float(row.GrowthVolatility) + 0.004 * float(row.ResourceDrag) + 0.003 * float(row.DependencyPressure),
            0.005,
            0.015,
        )
        ppp_p10_2021_dollars = ppp_2025_2021_dollars * (1.0 + max(-0.02, cagr - uncertainty)) ** 25
        ppp_p90_2021_dollars = ppp_2025_2021_dollars * (1.0 + min(0.07, cagr + uncertainty)) ** 25
        nominal_p10_2021_dollars = ppp_p10_2021_dollars * clamp(ratio_2050 * 0.90, 0.07, 1.15)
        nominal_p90_2021_dollars = ppp_p90_2021_dollars * clamp(ratio_2050 * 1.10, 0.08, 1.20)
        ppp_p10 = ppp_p10_2021_dollars * PRICE_FACTOR_LOW_2021_TO_2050
        ppp_p90 = ppp_p90_2021_dollars * PRICE_FACTOR_HIGH_2021_TO_2050
        nominal_p10 = nominal_p10_2021_dollars * PRICE_FACTOR_LOW_2021_TO_2050
        nominal_p90 = nominal_p90_2021_dollars * PRICE_FACTOR_HIGH_2021_TO_2050

        total_ppp_2025 = ppp_2025 * population_2025
        total_nominal_2025 = nominal_2025 * population_2025
        total_ppp_2050 = ppp_2050 * float(row.Population_2050)
        total_nominal_2050 = nominal_2050 * float(row.Population_2050)
        total_ppp_2050_2021_dollars = ppp_2050_2021_dollars * float(row.Population_2050)
        total_nominal_2050_2021_dollars = nominal_2050_2021_dollars * float(row.Population_2050)
        economic_strength = clamp(
            100.0 * (
                0.68 * clamp((math.log10(max(total_nominal_2050_2021_dollars, 1e8)) - 8.0) / 7.0, 0.0, 1.0)
                + 0.12 * float(row.EconomicDiversification)
                + 0.10 * float(row.FutureReadiness)
                + 0.10 * float(row.HumanCapitalAbsorption)
            ),
            0.0,
            100.0,
        )

        results.append({
            "ISO3": row.ISO3,
            "Country": row.Country,
            "Population_2025": round(population_2025),
            "Population_2050": round(float(row.Population_2050)),
            "GDP_PPP_PerCapita_2025": round(ppp_2025, 2),
            "GDP_PPP_PerCapita_2025_2021_IntlDollar": round(ppp_2025_2021_dollars, 2),
            "GDP_PPP_PerCapita_2050": round(ppp_2050, 2),
            "GDP_PPP_PerCapita_2050_2021_IntlDollar": round(ppp_2050_2021_dollars, 2),
            "GDP_Nominal_PerCapita_2025": round(nominal_2025, 2),
            "GDP_Nominal_PerCapita_2025_2021_USD": round(nominal_2025_2021_dollars, 2),
            "GDP_Nominal_PerCapita_2050": round(nominal_2050, 2),
            "GDP_Nominal_PerCapita_2050_2021_USD": round(nominal_2050_2021_dollars, 2),
            "GDP_PPP_Total_2025": round(total_ppp_2025, 2),
            "GDP_PPP_Total_2050": round(total_ppp_2050, 2),
            "GDP_PPP_Total_2050_2021_IntlDollar": round(total_ppp_2050_2021_dollars, 2),
            "GDP_Nominal_Total_2025": round(total_nominal_2025, 2),
            "GDP_Nominal_Total_2050": round(total_nominal_2050, 2),
            "GDP_Nominal_Total_2050_2021_USD": round(total_nominal_2050_2021_dollars, 2),
            "GDP_PPP_PerCapita_P10_2050": round(ppp_p10, 2),
            "GDP_PPP_PerCapita_P90_2050": round(ppp_p90, 2),
            "GDP_Nominal_PerCapita_P10_2050": round(nominal_p10, 2),
            "GDP_Nominal_PerCapita_P90_2050": round(nominal_p90, 2),
            "Real_GDP_PerCapita_CAGR_2025_2050": round(cagr, 6),
            "Structural_GDP_PerCapita_CAGR_Signal": round(structural_cagr, 6),
            "Income_Index_2050": round(income_index_2050, 6),
            "Income_Index_2050_Recomputed_From_PPP": round(recomputed_income_index_2050, 6),
            "Income_Index_2050_PPP_Mismatch": round(recomputed_income_index_2050 - income_index_2050, 8),
            "CPI_U_2021_Annual_Average_2013_Base": CPI_U_2021,
            "CPI_U_2025_Annual_Average_2013_Base": CPI_U_2025,
            "Observed_Inflation_Factor_2021_2025": round(OBSERVED_PRICE_FACTOR_2021_TO_2025, 9),
            "Long_Run_USD_Inflation_Assumption_2026_2050": LONG_RUN_INFLATION,
            "Inflation_Factor_2021_to_2050": round(PRICE_FACTOR_2021_TO_2050, 9),
            "Inflation_Factor_Low_2021_to_2050": round(PRICE_FACTOR_LOW_2021_TO_2050, 9),
            "Inflation_Factor_High_2021_to_2050": round(PRICE_FACTOR_HIGH_2021_TO_2050, 9),
            "PPP_to_Nominal_Ratio_2025": round(ratio_2025, 5),
            "PPP_to_Nominal_Ratio_2050": round(ratio_2050, 5),
            "Economic_Strength_Score_2050": round(economic_strength, 2),
            "Growth_Prospect_Score": row.GrowthProspectScore,
            "Industrialization_Signal": row.IndustrializationSignal,
            "Future_Readiness": row.FutureReadiness,
            "Demographic_Dividend": row.DemographicDividend,
            "Resource_Drag": row.ResourceDrag,
            "Growth_Volatility": row.GrowthVolatility,
            "Trajectory": row.Trajectory,
            "Baseline_Source": "World Bank API" if not (ppp_proxy or nominal_proxy) else "World Bank API with HDI-model proxy",
            "PPP_Baseline_Year": ppp_year,
            "Nominal_Baseline_Year": nominal_year,
            "PPP_Baseline_Proxy": ppp_proxy,
            "Nominal_Baseline_Proxy": nominal_proxy,
            "Projection_Type": "Independent scenario projection; constant 2021 audit values and inflation-adjusted 2050-dollar values",
        })

    output = pd.DataFrame(results)
    rank_columns = {
        "GDP_PPP_PerCapita_2050": "Rank_GDP_PPP_PerCapita_2050",
        "GDP_Nominal_PerCapita_2050": "Rank_GDP_Nominal_PerCapita_2050",
        "GDP_PPP_Total_2050": "Rank_GDP_PPP_Total_2050",
        "GDP_Nominal_Total_2050": "Rank_GDP_Nominal_Total_2050",
        "Economic_Strength_Score_2050": "Rank_Economic_Resilience_2050",
    }
    for value_column, rank_column in rank_columns.items():
        output[rank_column] = output[value_column].rank(method="min", ascending=False).astype(int)

    output["Rank_Economic_Strength_2050"] = output["Rank_GDP_Nominal_Total_2050"]
    front = ["Rank_Economic_Strength_2050", "ISO3", "Country"]
    output = output[front + [column for column in output.columns if column not in front]]
    output = output.sort_values("Rank_Economic_Strength_2050").reset_index(drop=True)

    if output.ISO3.nunique() != 193 or len(output) != 193:
        raise RuntimeError("Economic output does not contain 193 unique countries")
    numeric = output.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise RuntimeError("Economic output contains invalid numeric values")
    strictly_positive = [
        column for column in output.columns
        if column.startswith("GDP_") and "Mismatch" not in column
    ] + ["Population_2025", "Population_2050"]
    if (output[strictly_positive] <= 0).any().any():
        raise RuntimeError("Economic output contains nonpositive GDP or population values")
    if output["Income_Index_2050_PPP_Mismatch"].abs().max() > 1e-5:
        raise RuntimeError("2050 PPP per-capita values do not reconcile with the HDI income index")
    ppp_error = (
        (output.GDP_PPP_PerCapita_2050 * output.Population_2050 - output.GDP_PPP_Total_2050).abs()
        / output.GDP_PPP_Total_2050.clip(lower=1.0)
    ).max()
    nominal_error = (
        (output.GDP_Nominal_PerCapita_2050 * output.Population_2050 - output.GDP_Nominal_Total_2050).abs()
        / output.GDP_Nominal_Total_2050.clip(lower=1.0)
    ).max()
    if ppp_error > 1e-4 or nominal_error > 1e-4:
        raise RuntimeError("GDP totals do not reconcile with population and per-capita values")
    inflation_error = (
        output.GDP_PPP_PerCapita_2050_2021_IntlDollar * PRICE_FACTOR_2021_TO_2050
        - output.GDP_PPP_PerCapita_2050
    ).abs().max()
    if inflation_error > 0.02:
        raise RuntimeError("2050-dollar PPP values do not reconcile with the inflation factor")

    output.to_csv(OUTPUT_PATH, index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="Use the checked-in World Bank baseline cache")
    args = parser.parse_args()
    output = build_projection(offline=args.offline)
    print(f"Saved {len(output)} countries to {OUTPUT_PATH}")
    print(output[["Rank_Economic_Strength_2050", "ISO3", "Country", "GDP_Nominal_Total_2050", "GDP_PPP_PerCapita_2050", "Real_GDP_PerCapita_CAGR_2025_2050"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
