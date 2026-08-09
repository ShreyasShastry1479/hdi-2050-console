"""Future-oriented development factors for 2050 HDI projections.

All variables are normalized to 0-1, where 1 represents frontier-level
capability or adoption. These are scenario/projection inputs, not observed
official HDI components.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.technology import get_ai_readiness, get_digital_gov


FUTURE_VARS = [
    "ai_adoption_index",
    "robot_density",
    "semiconductor_production",
    "green_energy_investment",
    "ev_adoption",
    "battery_capacity",
    "high_speed_rail",
    "startup_ecosystem",
    "venture_capital",
    "cloud_adoption",
]


COUNTRY_OVERRIDES = {
    "USA": {"semiconductor_production": 0.75, "venture_capital": 0.95, "cloud_adoption": 0.95, "startup_ecosystem": 0.95},
    "CHN": {"robot_density": 0.78, "semiconductor_production": 0.82, "battery_capacity": 0.95, "ev_adoption": 0.85, "high_speed_rail": 0.95},
    "KOR": {"robot_density": 0.95, "semiconductor_production": 0.88, "battery_capacity": 0.82, "cloud_adoption": 0.82},
    "JPN": {"robot_density": 0.90, "semiconductor_production": 0.70, "battery_capacity": 0.60, "high_speed_rail": 0.88},
    "TWN": {"semiconductor_production": 0.98, "robot_density": 0.76},
    "SGP": {"semiconductor_production": 0.60, "startup_ecosystem": 0.86, "venture_capital": 0.78, "cloud_adoption": 0.92},
    "DEU": {"robot_density": 0.82, "green_energy_investment": 0.78, "ev_adoption": 0.72, "high_speed_rail": 0.72},
    "NLD": {"green_energy_investment": 0.72, "startup_ecosystem": 0.74, "cloud_adoption": 0.82},
    "SWE": {"green_energy_investment": 0.86, "ev_adoption": 0.82, "startup_ecosystem": 0.76, "cloud_adoption": 0.84},
    "NOR": {"green_energy_investment": 0.88, "ev_adoption": 0.95},
    "FIN": {"green_energy_investment": 0.78, "cloud_adoption": 0.82},
    "DNK": {"green_energy_investment": 0.88, "ev_adoption": 0.80},
    "FRA": {"high_speed_rail": 0.82, "green_energy_investment": 0.70, "ev_adoption": 0.66},
    "GBR": {"startup_ecosystem": 0.82, "venture_capital": 0.82, "cloud_adoption": 0.88},
    "ISR": {"startup_ecosystem": 0.92, "venture_capital": 0.86, "ai_adoption_index": 0.82},
    "IND": {"startup_ecosystem": 0.72, "venture_capital": 0.58, "cloud_adoption": 0.62, "battery_capacity": 0.55},
    "VNM": {"robot_density": 0.42, "semiconductor_production": 0.45, "battery_capacity": 0.42},
    "MYS": {"semiconductor_production": 0.58, "robot_density": 0.42},
    "THA": {"robot_density": 0.42, "ev_adoption": 0.42},
    "IDN": {"battery_capacity": 0.48, "ev_adoption": 0.38},
    "ARE": {"ai_adoption_index": 0.78, "green_energy_investment": 0.72, "cloud_adoption": 0.78},
    "SAU": {"ai_adoption_index": 0.68, "green_energy_investment": 0.68, "cloud_adoption": 0.66},
}


def _income_score(row: pd.Series) -> float:
    gni = float(row.get("gni_ppp", 12000.0))
    return float(np.clip((np.log1p(max(gni, 300.0)) - np.log1p(1000.0)) / (np.log1p(90000.0) - np.log1p(1000.0)), 0, 1))


def _industrial_score(row: pd.Series) -> float:
    eci = float(row.get("eci", 0.0))
    trade = float(row.get("trade_openness", 0.5))
    rd = float(row.get("rd_expenditure", 0.5))
    return float(np.clip(0.45 * ((eci + 1.5) / 3.5) + 0.25 * min(trade / 1.5, 1) + 0.30 * min(rd / 5.0, 1), 0, 1))


def estimate_future_factor(country_id: str, row: pd.Series, year: int, var: str) -> float:
    """Estimate one future-oriented variable on a 0-1 scale."""
    income = _income_score(row)
    industrial = _industrial_score(row)
    internet = float(np.clip(row.get("internet", 0.4), 0, 1))
    urban = float(np.clip(row.get("urbanization", 0.5), 0, 1))
    renewable = float(np.clip(row.get("renewable_share", 0.2), 0, 1))
    rd = float(np.clip(row.get("rd_expenditure", 0.5) / 5.0, 0, 1))
    gov = float(np.clip((row.get("gov_effectiveness", 0.0) + 2.0) / 4.0, 0, 1))
    ai = float(get_ai_readiness(country_id, year))
    digital_gov = float(get_digital_gov(country_id, year))

    base = {
        "ai_adoption_index": 0.55 * ai + 0.25 * internet + 0.20 * gov,
        "robot_density": 0.55 * industrial + 0.25 * income + 0.20 * rd,
        "semiconductor_production": 0.60 * industrial + 0.25 * rd + 0.15 * gov,
        "green_energy_investment": 0.40 * income + 0.30 * gov + 0.30 * renewable,
        "ev_adoption": 0.45 * income + 0.25 * urban + 0.20 * green_energy_base(income, gov, renewable) + 0.10 * internet,
        "battery_capacity": 0.50 * industrial + 0.25 * green_energy_base(income, gov, renewable) + 0.25 * rd,
        "high_speed_rail": 0.40 * income + 0.35 * urban + 0.25 * gov,
        "startup_ecosystem": 0.35 * ai + 0.25 * rd + 0.20 * income + 0.20 * gov,
        "venture_capital": 0.35 * income + 0.30 * startup_base(ai, rd, income, gov) + 0.20 * gov + 0.15 * digital_gov,
        "cloud_adoption": 0.40 * ai + 0.30 * internet + 0.20 * income + 0.10 * digital_gov,
    }[var]

    override = COUNTRY_OVERRIDES.get(country_id, {}).get(var)
    if override is not None:
        base = 0.55 * base + 0.45 * override

    t = np.clip((year - 1990) / (2050 - 1990), 0, 1)
    adoption_curve = 1.0 / (1.0 + np.exp(-8 * (t - 0.58)))
    early_floor = 0.15 + 0.55 * t
    return float(np.clip(base * (early_floor + (1 - early_floor) * adoption_curve), 0, 1))


def green_energy_base(income: float, gov: float, renewable: float) -> float:
    return float(np.clip(0.40 * income + 0.30 * gov + 0.30 * renewable, 0, 1))


def startup_base(ai: float, rd: float, income: float, gov: float) -> float:
    return float(np.clip(0.35 * ai + 0.25 * rd + 0.20 * income + 0.20 * gov, 0, 1))


def add_future_oriented_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Add all Tier 13 future variables to a country-year dataframe."""
    result = df.copy()
    for var in FUTURE_VARS:
        if var in result.columns and result[var].notna().all():
            continue
        result[var] = [
            estimate_future_factor(row["country_id"], row, int(row["year"]), var)
            for _, row in result.iterrows()
        ]
    return result


def compute_future_readiness(row: pd.Series) -> float:
    values = [float(row.get(var, 0.0)) for var in FUTURE_VARS]
    return float(np.clip(np.mean(values), 0, 1))


DIGITAL_INFRA_ACCELERATOR_PRIOR = {
    # Large or fast-digitizing developing economies.
    "IND": 0.92, "IDN": 0.88, "PHL": 0.84, "VNM": 0.86, "BGD": 0.82,
    "PAK": 0.70, "NPL": 0.68, "LKA": 0.66, "KHM": 0.70, "LAO": 0.62,
    "MYS": 0.74, "THA": 0.70, "CHN": 0.70,
    # Africa with mobile money, digital public infrastructure, or fast catch-up potential.
    "KEN": 0.82, "RWA": 0.80, "GHA": 0.72, "NGA": 0.78, "SEN": 0.68,
    "TZA": 0.66, "UGA": 0.64, "ETH": 0.68, "CIV": 0.62, "MAR": 0.68,
    "EGY": 0.66, "TUN": 0.66, "ZAF": 0.66,
    # Gulf and small high-investment states.
    "ARE": 0.82, "SAU": 0.78, "QAT": 0.74, "BHR": 0.72, "OMN": 0.68,
    # Latin America / Central Asia digital catch-up.
    "COL": 0.66, "MEX": 0.64, "BRA": 0.62, "PER": 0.62, "CHL": 0.64,
    "KAZ": 0.68, "UZB": 0.66, "GEO": 0.66,
}


def compute_digital_infrastructure_development(country_id: str, row: pd.Series, year: int = 2050) -> float:
    """Estimate digital infrastructure development momentum on a 0-1 scale.

    This is different from frontier tech readiness. It rewards countries that
    are plausibly building broadband, cloud, digital government, AI capacity,
    and internet access quickly enough to accelerate HDI delivery by 2050.
    """
    internet = float(np.clip(row.get("internet", 0.45), 0.0, 1.0))
    broadband = float(np.clip(row.get("broadband", 10.0) / 50.0, 0.0, 1.0))
    urban = float(np.clip(row.get("urbanization", 0.55), 0.10, 0.98))
    gov = float(np.clip((row.get("gov_effectiveness", 0.0) + 2.0) / 4.0, 0.0, 1.0))
    income = _income_score(row)
    ai = float(get_ai_readiness(country_id, year))
    digital_gov = float(get_digital_gov(country_id, year))
    cloud = float(row.get("cloud_adoption", estimate_future_factor(country_id, row, year, "cloud_adoption")))
    startup = float(row.get("startup_ecosystem", estimate_future_factor(country_id, row, year, "startup_ecosystem")))
    prior = DIGITAL_INFRA_ACCELERATOR_PRIOR.get(country_id, 0.50)

    current_access = 0.62 * internet + 0.38 * broadband
    buildout_room = 1.0 - current_access
    platform_capacity = np.clip(
        0.25 * ai +
        0.22 * digital_gov +
        0.18 * cloud +
        0.14 * startup +
        0.11 * urban +
        0.10 * gov,
        0.0,
        1.0,
    )
    affordability_room = 1.0 - np.clip(income, 0.0, 1.0)
    leapfrog_potential = np.clip(0.70 * buildout_room + 0.30 * affordability_room, 0.0, 1.0)

    score = (
        0.36 * platform_capacity +
        0.26 * leapfrog_potential +
        0.22 * prior +
        0.16 * current_access
    )
    return float(np.clip(score, 0.0, 1.0))
