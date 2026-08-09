"""Realistic HDI forecasting with country-specific trajectories.

NEW FORMULA (v3):
    HDI2050 = HDI_current + Economic_Catchup + Education_Improvement + Health_Improvement + Governance_Effect + Demographic_Dividend + Tech_Adoption - Climate_Penalty

Key improvements:
- Logarithmic GDP scaling (ln(GDP) not linear)
- HDI growth decay: growth = current_growth × (1-HDI)^2
- GDP × Governance interaction (nonlinear)
- Country-specific trajectory classes
- African/Asian catch-up acceleration
- Education logistic saturation
"""

import numpy as np
import pandas as pd

from data.future import FUTURE_VARS


# ============================================================
# COUNTRY TRAJECTORY CLASSES
# ============================================================
# Each country gets a trajectory type that determines its HDI growth path

TRAJECTORY_CLASSES = {
    # Stagnant: high HDI, near biological limit
    # Expected gain: 0.003-0.010 over 26 years
    "stagnant": {
        "countries": [
            "JPN", "ITA", "GRC", "PRT", "CUB",
        ],
        "hdi_gain_range": (0.003, 0.010),
        "gni_multiplier": 0.6,
        "health_multiplier": 0.5,
        "education_multiplier": 0.5,
    },

    # Slow-growth: already high HDI, limited catch-up potential
    # Expected gain: 0.008-0.020
    "slow_growth": {
        "countries": [
            "CHE", "NOR", "ISL", "SWE", "DNK", "FIN", "IRL",
            "NLD", "BEL", "AUT", "DEU", "FRA", "GBR", "LUX",
            "CAN", "AUS", "NZL", "SGP", "USA",
        ],
        "hdi_gain_range": (0.008, 0.020),
        "gni_multiplier": 0.7,
        "health_multiplier": 0.6,
        "education_multiplier": 0.6,
    },

    # Moderate: medium HDI, steady catch-up
    # Expected gain: 0.020-0.050
    "moderate": {
        "countries": [
            "CZE", "SVN", "MLT", "EST", "LTU", "LVA", "SVK", "HUN",
            "POL", "HRV", "BGR", "ROU", "CYP", "CHL", "URY", "ARG",
            "PRT", "ESP", "ITA", "KOR", "ISR",
        ],
        "hdi_gain_range": (0.020, 0.050),
        "gni_multiplier": 0.85,
        "health_multiplier": 0.75,
        "education_multiplier": 0.75,
    },

    # Fast-converger: developing, strong institutions
    # Expected gain: 0.040-0.100
    "fast_converger": {
        "countries": [
            "CHN", "RUS", "BRA", "MEX", "TUR", "THA", "MYS",
            "COL", "DOM", "SRB", "MNE", "MKD", "ALB",
            "BIH", "GEO", "ARM", "AZE", "KAZ", "UZB", "IRN",
            "JOR", "TUN", "MAR", "EGY", "CRI", "PAN", "GUY",
        ],
        "hdi_gain_range": (0.040, 0.100),
        "gni_multiplier": 1.0,
        "health_multiplier": 1.0,
        "education_multiplier": 1.0,
    },

    # Accelerator: high growth potential, demographic dividend
    # Expected gain: 0.080-0.180
    "accelerator": {
        "countries": [
            "IND", "BGD", "IDN", "VNM", "PHL", "EGY", "MAR",
            "GHA", "KEN", "SEN", "CIV", "CMR", "AGO", "MOZ",
            "ZMB", "NAM", "BWA", "BTN", "MDV",
            "GTM", "SLV", "HND", "NIC", "LKA", "NPL",
        ],
        "hdi_gain_range": (0.080, 0.180),
        "gni_multiplier": 1.2,
        "health_multiplier": 1.3,
        "education_multiplier": 1.3,
    },

    # Frontier-jumper: Africa with massive catch-up potential
    # Expected gain: 0.100-0.250
    "frontier_jumper": {
        "countries": [
            "NGA", "ETH", "TZA", "RWA", "UGA", "MYS", "IDN",
            "PAK", "MMR", "KHM", "LAO", "MNG", "KGZ", "TJK",
            "TGO", "BEN", "BFA", "MLI", "NER", "TCD", "GIN",
            "SLE", "LBR", "GMB", "GNB", "MWI", "MDG", "SSD",
            "COD", "COG", "GAB", "BDI", "CAF", "DJI", "COM",
            "MRT", "STP", "GNQ", "CPV", "FJI", "SLB", "VUT",
            "WSM", "TON", "KIR", "MHL", "FSM", "PLW", "NRU", "TUV",
        ],
        "hdi_gain_range": (0.100, 0.250),
        "gni_multiplier": 1.4,
        "health_multiplier": 1.5,
        "education_multiplier": 1.5,
    },

    # Conflict-recovery: countries that may stabilize
    # Expected gain: 0.060-0.150
    "conflict_recovery": {
        "countries": [
            "IRQ", "SYR", "YEM", "AFG", "SOM", "SSD", "SDN",
            "LBY", "LBN", "UKR", "VEN", "HTI", "BLR", "PRK",
            "ERI", "MOZ", "MMR",
        ],
        "hdi_gain_range": (0.060, 0.150),
        "gni_multiplier": 0.9,
        "health_multiplier": 1.1,
        "education_multiplier": 1.1,
    },
}

# Reverse lookup: country -> trajectory class
COUNTRY_TRAJECTORY = {}
for traj_class, info in TRAJECTORY_CLASSES.items():
    for c in info["countries"]:
        COUNTRY_TRAJECTORY[c] = traj_class

RECOVERY_POTENTIAL_2050 = {
    # Current conflict/fragile states: do not assume 2024 conditions persist to 2050.
    "SYR": 0.70, "YEM": 0.68, "AFG": 0.62, "SOM": 0.62, "SSD": 0.65,
    "SDN": 0.62, "PSE": 0.62, "LBY": 0.65, "LBN": 0.58, "HTI": 0.58,
    "MMR": 0.58, "UKR": 0.72, "IRQ": 0.58, "VEN": 0.55, "COD": 0.55,
    "CAF": 0.52, "TCD": 0.50, "MLI": 0.52, "BFA": 0.48, "NER": 0.48,
    "ERI": 0.45, "PRK": 0.35,
    # Developing-world catch-up where stability is mixed but long-run upside exists.
    "ETH": 0.58, "NGA": 0.58, "PAK": 0.45, "MOZ": 0.45, "ZWE": 0.44,
    "GIN": 0.42, "SLE": 0.42, "LBR": 0.42, "BDI": 0.42, "CMR": 0.40,
}


def get_trajectory_class(country_id: str) -> str:
    """Get the trajectory class for a country."""
    return COUNTRY_TRAJECTORY.get(country_id, "moderate")


def get_recovery_potential(country_id: str, current_hdi: float, year: int = 2050) -> float:
    """Return a 0-1 long-run recovery/catch-up potential for developing countries."""
    horizon_progress = np.clip((year - 2024) / (2050 - 2024), 0.0, 1.0)
    base = RECOVERY_POTENTIAL_2050.get(country_id, 0.0)
    if current_hdi < 0.55:
        base = max(base, 0.38)
    elif current_hdi < 0.70:
        base = max(base, 0.28)
    elif current_hdi < 0.80:
        base = max(base, 0.16)
    return float(np.clip(base * horizon_progress, 0.0, 1.0))


def get_continuous_trajectory_modifier(country_id: str, current_hdi: float, inst_efficiency: float) -> float:
    """Get a continuous trajectory modifier (0.6-1.5) based on country characteristics.
    
    Instead of discrete classes, this function returns a continuous value that
    captures the complex interplay of:
    - Current HDI level (convergence potential)
    - Institutional efficiency (governance quality)
    - Country-specific factors
    """
    # Base modifier from HDI level (convergence theory)
    # Lower HDI = higher potential growth (catch-up effect)
    if current_hdi < 0.4:
        hdi_modifier = 1.4  # Very low HDI, high catch-up potential
    elif current_hdi < 0.55:
        hdi_modifier = 1.3
    elif current_hdi < 0.70:
        hdi_modifier = 1.2
    elif current_hdi < 0.80:
        hdi_modifier = 1.1
    elif current_hdi < 0.85:
        hdi_modifier = 1.0
    elif current_hdi < 0.90:
        hdi_modifier = 0.9
    else:
        hdi_modifier = 0.8  # Near frontier, slower growth

    # Institutional efficiency modifier
    # Better institutions = faster development
    if inst_efficiency >= 1.10:
        inst_modifier = 1.1
    elif inst_efficiency >= 1.05:
        inst_modifier = 1.05
    elif inst_efficiency >= 1.00:
        inst_modifier = 1.0
    elif inst_efficiency >= 0.95:
        inst_modifier = 0.95
    elif inst_efficiency >= 0.90:
        inst_modifier = 0.9
    else:
        inst_modifier = 0.85

    # Country-specific adjustments (conservative)
    country_adjustments = {
        # Frontier jumpers (exceptional growth)
        "CHN": 1.05, "IND": 1.05, "VNM": 1.05, "BGD": 1.02, "IDN": 1.02,
        "RWA": 1.05, "ETH": 1.06, "KEN": 1.02, "GHA": 1.02,
        
        # Stagnant/conflict
        "AFG": 0.75, "YEM": 0.75, "SOM": 0.75, "SSD": 0.80, "SYR": 0.80,
        "VEN": 0.85, "LBY": 0.85, "IRQ": 0.90,
        
        # Rich country specifics
        "JPN": 0.90,  # Demographics
        "USA": 0.95,  # Innovation but inequality
        "SGP": 1.05,  # Exceptional governance
        "ARE": 1.02,  # Diversification efforts
        
        # Eastern Europe
        "POL": 1.00, "CZE": 0.98, "ROU": 1.00, "BGR": 0.98,
        
        # Latin America
        "BRA": 0.98, "MEX": 0.98, "COL": 1.00, "PER": 1.00, "CHL": 0.98,
    }
    country_adj = country_adjustments.get(country_id, 1.0)

    # Combine modifiers (multiplicative)
    final_modifier = hdi_modifier * inst_modifier * country_adj
    
    # Clip to reasonable range
    return np.clip(final_modifier, 0.6, 1.5)


def get_frontier_saturation_factor(current_hdi: float) -> float:
    """Return a nonlinear persistence factor for countries near the HDI ceiling.

    This makes gains increasingly hard above 0.85 and especially above 0.94,
    matching the way HDI historically saturates near 1.0.
    """
    if current_hdi < 0.60:
        return 1.00
    if current_hdi < 0.75:
        return 0.96
    if current_hdi < 0.85:
        return 0.78
    if current_hdi < 0.90:
        return 0.58
    if current_hdi < 0.94:
        return 0.32
    if current_hdi < 0.965:
        return 0.13
    return 0.04


def get_stage_gain_cap(current_hdi: float) -> float:
    """Cap plausible 2023-2050 median HDI gains by development stage."""
    if current_hdi >= 0.965:
        return 0.004
    if current_hdi >= 0.940:
        return 0.009
    if current_hdi >= 0.900:
        return 0.020
    if current_hdi >= 0.850:
        return 0.040
    if current_hdi >= 0.750:
        return 0.100
    if current_hdi >= 0.600:
        return 0.145
    return 0.205


def get_catchup_horizon_multiplier(current_hdi: float) -> float:
    """Boost convergence for non-frontier countries over a full generation."""
    if current_hdi < 0.60:
        return 1.35
    if current_hdi < 0.75:
        return 1.27
    if current_hdi < 0.85:
        return 1.18
    if current_hdi < 0.90:
        return 1.08
    return 1.00


MICROSTATE_ISO3 = {
    "AND", "LIE", "LUX", "SMR", "MCO", "KNA", "ATG", "DMA", "GRD",
    "LCA", "VCT", "BRB", "PLW", "NRU", "TUV", "MHL", "FSM",
}


# ============================================================
# INSTITUTIONAL EFFICIENCY (how well GDP converts to HDI)
# ============================================================

INSTITUTIONAL_EFFICIENCY = {
    # Very high (1.10-1.15)
    "SGP": 1.15, "DNK": 1.12, "NOR": 1.12, "FIN": 1.12, "ISL": 1.12,
    "SWE": 1.12, "CHE": 1.12, "NLD": 1.11, "DEU": 1.11, "NZL": 1.11,
    "AUS": 1.10, "CAN": 1.10, "GBR": 1.10, "IRL": 1.10, "AUT": 1.10,
    "LUX": 1.10, "BEL": 1.10, "FRA": 1.10, "JPN": 1.10, "KOR": 1.10,

    # High (1.05-1.09)
    "USA": 1.08, "ESP": 1.07, "PRT": 1.07, "CZE": 1.07, "EST": 1.07,
    "LTU": 1.07, "LVA": 1.06, "SVN": 1.07, "MLT": 1.07, "CYP": 1.06,
    "ISR": 1.08, "CHL": 1.06, "URY": 1.06, "POL": 1.06, "HRV": 1.06,
    "SVK": 1.06, "HUN": 1.05, "BGR": 1.05, "ROU": 1.05, "ITA": 1.05,
    "GRC": 1.04, "QAT": 1.05, "ARE": 1.08, "SAU": 1.02, "KWT": 1.02,
    "BHR": 1.03, "OMN": 1.02,

    # Medium (0.95-1.04)
    "MYS": 1.02, "THA": 1.01, "COL": 0.98, "PER": 0.97, "BRA": 0.98,
    "MEX": 0.97, "CRI": 1.01, "PAN": 1.00, "CHN": 1.05, "RUS": 0.92,
    "TUR": 0.95, "JOR": 1.00, "MAR": 0.96, "TUN": 0.97, "EGY": 0.93,
    "KAZ": 1.00, "GEO": 0.98, "ARM": 0.97, "SRB": 0.97, "MNE": 0.96,
    "MKD": 0.96, "ALB": 0.95, "BIH": 0.93, "NAM": 1.00, "BWA": 1.01,
    "MUS": 1.02, "FJI": 0.98, "DOM": 0.96, "JAM": 0.97, "TTO": 0.97,
    "GUY": 0.97, "PRY": 0.92, "ECU": 0.91, "BLZ": 0.96, "GTM": 0.93, "SLV": 0.94,
    "HND": 0.91, "NIC": 0.92, "BOL": 0.92, "IDN": 0.98, "PHL": 0.96,
    "VNM": 0.97, "LKA": 0.93, "BGD": 0.92, "IND": 0.93, "UKR": 0.88,
    "MDA": 0.92, "KGZ": 0.93, "UZB": 0.90, "TJK": 0.88, "AZE": 0.90,
    "IRN": 0.85, "LBN": 0.85, "DZA": 0.88, "BGR": 1.05,

    # Low (0.80-0.94)
    "PAK": 0.85, "NPL": 0.88, "KEN": 0.90, "GHA": 0.92,
    "SEN": 0.91, "TZA": 0.89, "RWA": 0.95, "UGA": 0.88, "ETH": 0.87,
    "NGA": 0.84, "CMR": 0.82, "CIV": 0.83, "AGO": 0.78, "MOZ": 0.80,
    "ZMB": 0.85, "ZWE": 0.78, "MWI": 0.82, "MDG": 0.80, "BFA": 0.78,
    "MLI": 0.78, "NER": 0.75, "TCD": 0.75, "GIN": 0.80, "SLE": 0.78,
    "LBR": 0.78, "BEN": 0.85, "TGO": 0.85, "GMB": 0.83, "GNB": 0.82,
    "MRT": 0.85, "COM": 0.85, "DJI": 0.85, "SWZ": 0.88, "LSO": 0.85,
    "GAB": 0.85, "COG": 0.78, "GNQ": 0.75, "STP": 0.88, "CPV": 0.95,

    # Very low (0.70-0.79)
    "IRQ": 0.78, "SYR": 0.70, "YEM": 0.72, "AFG": 0.72, "SSD": 0.70,
    "SOM": 0.70, "LBY": 0.75, "SDN": 0.72, "ERI": 0.72, "COD": 0.72,
    "CAF": 0.72, "HTI": 0.75, "VEN": 0.72, "CUB": 0.80,
    "PRK": 0.70, "BLR": 0.82, "MMR": 0.75, "TLS": 0.85, "PSE": 0.78,
}


# ============================================================
# HDI COMPONENT FORECASTING (for component-level analysis)
# ============================================================

FRONTIER_2050 = {
    "life_exp": 87.0, "expected_school": 18.0, "mean_school": 14.0,
    "gni_ppp": 120000.0, "internet": 0.98, "fertility": 1.4,
    "urbanization": 0.90, "gov_effectiveness": 1.8, "corruption": 1.8,
    "trade_openness": 2.0, "co2_per_capita": 2.0, "renewable_share": 0.80,
    "eci": 3.0, "physicians": 6.0, "health_exp": 12.0, "population": None,
    "gini": 0.20, "infant_mortality": 2.0, "rule_of_law": 2.0,
    "political_stability": 1.0, "rd_expenditure": 5.0,
    "dependency_ratio": 0.35, "broadband": 50.0, "climate_risk": 0.05,
    "ai_adoption_index": 0.98, "robot_density": 0.95,
    "semiconductor_production": 0.95, "green_energy_investment": 0.95,
    "ev_adoption": 0.98, "battery_capacity": 0.95,
    "high_speed_rail": 0.95, "startup_ecosystem": 0.95,
    "venture_capital": 0.95, "cloud_adoption": 0.98,
}

FLOOR = {
    "life_exp": 42.0, "expected_school": 2.0, "mean_school": 0.5,
    "gni_ppp": 400.0, "internet": 0.0, "fertility": 1.0,
    "urbanization": 0.10, "gov_effectiveness": -2.0, "corruption": -2.0,
    "trade_openness": 0.15, "co2_per_capita": 0.05, "renewable_share": 0.0,
    "eci": 0.0, "physicians": 0.05, "health_exp": 1.0, "population": 10000,
    "gini": 0.15, "infant_mortality": 1.0, "rule_of_law": -2.5,
    "political_stability": -3.0, "rd_expenditure": 0.0,
    "dependency_ratio": 0.20, "broadband": 0.0, "climate_risk": 0.0,
    "ai_adoption_index": 0.0, "robot_density": 0.0,
    "semiconductor_production": 0.0, "green_energy_investment": 0.0,
    "ev_adoption": 0.0, "battery_capacity": 0.0,
    "high_speed_rail": 0.0, "startup_ecosystem": 0.0,
    "venture_capital": 0.0, "cloud_adoption": 0.0,
}


def forecast_variable_realistic(
    values: np.ndarray,
    years_observed: np.ndarray,
    forecast_years: np.ndarray,
    var_name: str,
    country_id: str = "",
) -> np.ndarray:
    """Forecast a single variable for one country."""
    valid = ~np.isnan(values)
    if valid.sum() < 3:
        last = float(values[valid][-1]) if valid.any() else FLOOR.get(var_name, 0)
        return np.full(len(forecast_years), last)

    vals = values[valid]
    yrs = years_observed[valid]
    frontier = FRONTIER_2050.get(var_name)
    floor_val = FLOOR.get(var_name, 0)
    last_val = float(vals[-1])

    recent = vals[-min(10, len(vals)):]
    recent_years = yrs[-min(10, len(yrs)):]
    if len(recent) >= 3:
        slopes = np.diff(recent) / np.maximum(np.diff(recent_years), 1)
        avg_slope = np.median(slopes[-5:])
    else:
        avg_slope = (recent[-1] - recent[0]) / max(recent_years[-1] - recent_years[0], 1)

    if frontier is None:
        trend = np.polyfit(yrs - yrs[0], vals, min(2, len(vals) - 1))
        t_fc = forecast_years - yrs[0]
        return np.polyval(trend, t_fc)

    # GNI: logarithmic growth with convergence
    if var_name == "gni_ppp":
        traj = get_trajectory_class(country_id) if country_id else "moderate"
        traj_info = TRAJECTORY_CLASSES.get(traj, TRAJECTORY_CLASSES["moderate"])
        gni_mult = traj_info["gni_multiplier"]

        # Moderate base growth: 2.5% for developing, lower for rich
        # Historical average for rich countries is ~1.5-2%
        # Developing countries can grow 3-5% but slow as they catch up
        if gni_mult >= 1.2:  # Frontier-jumper/accelerator
            base_growth = 0.040
        elif gni_mult >= 1.0:  # Fast-converger
            base_growth = 0.035
        elif gni_mult >= 0.85:  # Moderate
            base_growth = 0.028
        elif gni_mult >= 0.7:  # Slow-growth
            base_growth = 0.020
        else:  # Stagnant
            base_growth = 0.012

        preds = np.zeros(len(forecast_years))
        current = last_val
        for i, year in enumerate(forecast_years):
            gap_ratio = max(0, (frontier - current) / max(frontier - floor_val, 1))
            slowdown = 0.3 + 0.7 * gap_ratio
            target_growth = base_growth * slowdown
            target_growth = np.clip(target_growth, -0.005, 0.12)
            current = current * (1 + target_growth)
            current = np.clip(current, floor_val, frontier)
            preds[i] = current
        return preds

    # Life expectancy: country-specific differentiation
    if var_name == "life_exp":
        # Country-specific life expectancy modifiers
        # Accounts for obesity, smoking, healthcare quality, violence, etc.
        LE_MODIFIERS = {
            # High obesity/smoking penalty
            "USA": -1.5, "MEX": -1.0, "BRA": -0.5, "AUS": -0.5, "NZL": -0.5,
            "GBR": -0.5, "CAN": -0.3, "CHL": -0.3, "ARG": -0.5, "URY": -0.3,
            # Violence/accident penalty
            "BRA": -1.0, "MEX": -0.8, "COL": -0.5, "VEN": -0.5, "ZAF": -1.0,
            "RUS": -0.8, "UKR": -0.5, "BLR": -0.3,
            # Healthcare excellence bonus
            "JPN": 1.5, "SGP": 1.0, "KOR": 0.8, "CHE": 0.8, "ESP": 0.5,
            "ITA": 0.5, "FRA": 0.5, "AUS": 0.3, "SWE": 0.3, "NOR": 0.3,
            "ISR": 0.5, "CUB": 0.5, "CHL": 0.3, "CRI": 0.3,
            # Diet/lifestyle bonus
            "JPN": 1.0, "KOR": 0.5, "ESP": 0.3, "ITA": 0.3, "GRC": 0.3,
            # Conflict/war penalty
            "SYR": -3.0, "YEM": -2.5, "AFG": -2.0, "SOM": -2.0, "SSD": -2.0,
            "IRQ": -1.5, "LBY": -1.0, "UKR": -1.0,
            # HIV/AIDS penalty
            "ZAF": -2.0, "SWZ": -2.0, "LSO": -1.5, "BWA": -1.0, "MOZ": -1.0,
            "ZMB": -1.0, "ZWE": -0.8, "MWI": -0.8, "UGA": -0.8,
            # Altitude/geography bonus
            "BOL": 0.3, "ECU": 0.3, "PER": 0.3, "COL": 0.3,
            # Healthcare access penalty (rural/poor)
            "IND": -1.0, "BGD": -0.8, "PAK": -1.0, "NPL": -0.8, "MMR": -0.5,
            "IDN": -0.5, "PHL": -0.5, "VNM": -0.3, "THA": -0.3,
            # Mental health/substance abuse
            "RUS": -1.0, "UKR": -0.5, "BLR": -0.3, "LTU": -0.3, "LVA": -0.3,
            "EST": -0.3, "USA": -0.5, "GBR": -0.3,
        }
        le_modifier = LE_MODIFIERS.get(country_id, 0.0)
        base_growth = 0.12 + le_modifier * 0.01  # Modifier affects growth rate
        base_growth = np.clip(base_growth, 0.02, 0.20)
        preds = np.zeros(len(forecast_years))
        current = last_val
        for i, year in enumerate(forecast_years):
            remaining = max(frontier - current, 0)
            if remaining < 1.0:
                delta = 0.01
            elif remaining < 3.0:
                delta = base_growth * 0.3
            elif remaining < 8.0:
                delta = base_growth * 0.6
            else:
                delta = base_growth
            current = min(current + delta, frontier)
            preds[i] = current
        return preds

    # Education (expected + mean): country-specific reform paths
    if var_name in ("expected_school", "mean_school"):
        # Country-specific education reform capacity
        EDU_REFORM = {
            # Strong reformers (rapid improvement)
            "CHN": 1.3, "IND": 1.2, "VNM": 1.2, "IDN": 1.1, "THA": 1.1,
            "MYS": 1.1, "PHL": 1.0, "BGD": 1.1, "RWA": 1.2, "ETH": 1.1,
            "KEN": 1.0, "GHA": 1.0, "SEN": 1.0, "TZA": 1.0, "UGA": 1.0,
            # Education system quality leaders
            "FIN": 1.1, "SGP": 1.1, "JPN": 1.0, "KOR": 1.1, "CAN": 1.0,
            "AUS": 1.0, "NZL": 1.0, "NLD": 1.0, "BEL": 1.0, "CHE": 1.0,
            # Slow reformers (structural issues)
            "FRA": 0.8, "ITA": 0.7, "ESP": 0.8, "PRT": 0.8, "GRC": 0.7,
            "BRA": 0.8, "MEX": 0.8, "COL": 0.8, "PER": 0.8, "ARG": 0.8,
            "CHL": 0.9, "URY": 0.9, "CRI": 0.9, "PAN": 0.9,
            # Conflict/disruption
            "SYR": 0.4, "YEM": 0.4, "AFG": 0.5, "SOM": 0.4, "SSD": 0.4,
            "IRQ": 0.6, "LBY": 0.5, "UKR": 0.7, "MMR": 0.5,
            # Saturation (already high, limited room)
            "USA": 0.85, "GBR": 0.85, "DEU": 0.85, "NOR": 0.9, "SWE": 0.9,
            "DNK": 0.9, "ISL": 0.9, "IRL": 0.9, "LUX": 0.9,
            # Cultural factors
            "JPN": 1.0, "KOR": 1.1, "CHN": 1.2,  # Strong education culture
            "IND": 1.1,  # Education valued but infrastructure limited
            "SAU": 0.9, "ARE": 0.9, "QAT": 0.9, "KWT": 0.9,  # Oil wealth education
        }
        edu_mult = EDU_REFORM.get(country_id, 1.0)
        if var_name == "mean_school":
            edu_mult *= 0.9  # Mean years changes slower
        base = 0.10 * edu_mult
        preds = np.zeros(len(forecast_years))
        current = last_val
        for i, year in enumerate(forecast_years):
            gap = frontier - current
            # Logistic saturation: slows as approaches max
            if gap < 0.5:
                delta = 0.02
            elif gap < 2.0:
                delta = base * 0.6
            elif gap < 5.0:
                delta = base
            else:
                delta = base * 1.2
            current = np.clip(current + delta, floor_val, frontier)
            preds[i] = current
        return preds

    # Internet & urbanization: S-curve
    if var_name in ("internet", "urbanization"):
        preds = np.zeros(len(forecast_years))
        current = last_val
        for i, year in enumerate(forecast_years):
            gap = frontier - current
            progress = (current - floor_val) / max(frontier - floor_val, 1)
            speed = 0.08 * np.sin(np.pi * progress)
            speed = np.clip(speed, 0.01, 0.10)
            delta = gap * speed
            current = np.clip(current + delta, floor_val, frontier)
            preds[i] = current
        return preds

    # Fertility
    if var_name == "fertility":
        target = 1.8
        preds = np.zeros(len(forecast_years))
        current = last_val
        for i, year in enumerate(forecast_years):
            gap = target - current
            delta = gap * 0.03
            current = np.clip(current + delta, 1.2, 6.0)
            preds[i] = current
        return preds

    # Governance
    if var_name in ("gov_effectiveness", "corruption"):
        preds = np.zeros(len(forecast_years))
        current = last_val
        for i, year in enumerate(forecast_years):
            gap = frontier - current
            delta = gap * 0.02
            current = np.clip(current + delta, floor_val, frontier)
            preds[i] = current
        return preds

    # CO2
    if var_name == "co2_per_capita":
        direction = -1 if last_val > 5 else (1 if last_val < 1 else 0)
        preds = np.zeros(len(forecast_years))
        current = last_val
        for i, year in enumerate(forecast_years):
            delta = direction * 0.05 if direction != 0 else 0
            current = np.clip(current + delta, floor_val, frontier)
            preds[i] = current
        return preds

    # Renewables
    if var_name == "renewable_share":
        preds = np.zeros(len(forecast_years))
        current = last_val
        for i, year in enumerate(forecast_years):
            gap = frontier - current
            years_from_now = year - 2024
            acceleration = 1.0 + 0.05 * years_from_now
            delta = gap * 0.03 * acceleration
            delta = max(delta, 0.005)
            current = np.clip(current + delta, floor_val, frontier)
            preds[i] = current
        return preds

    # Gini: decline toward 0.20 (more equal)
    if var_name == "gini":
        preds = np.zeros(len(forecast_years))
        current = last_val
        for i, year in enumerate(forecast_years):
            gap = frontier - current
            delta = gap * 0.015
            current = np.clip(current + delta, floor_val, frontier)
            preds[i] = current
        return preds

    # Infant mortality: decline toward 2.0
    if var_name == "infant_mortality":
        preds = np.zeros(len(forecast_years))
        current = last_val
        for i, year in enumerate(forecast_years):
            gap = current - frontier
            if gap > 1.0:
                delta = -gap * 0.04
            else:
                delta = -0.05
            current = np.clip(current + delta, frontier, floor_val)
            preds[i] = current
        return preds

    # Rule of law: improve toward 2.0
    if var_name == "rule_of_law":
        preds = np.zeros(len(forecast_years))
        current = last_val
        for i, year in enumerate(forecast_years):
            gap = frontier - current
            delta = gap * 0.012
            current = np.clip(current + delta, floor_val, frontier)
            preds[i] = current
        return preds

    # Political stability: improve toward 1.0
    if var_name == "political_stability":
        preds = np.zeros(len(forecast_years))
        current = last_val
        for i, year in enumerate(forecast_years):
            gap = frontier - current
            delta = gap * 0.010
            current = np.clip(current + delta, floor_val, frontier)
            preds[i] = current
        return preds

    # R&D expenditure: grow toward 5.0%
    if var_name == "rd_expenditure":
        preds = np.zeros(len(forecast_years))
        current = last_val
        for i, year in enumerate(forecast_years):
            gap = frontier - current
            delta = gap * 0.025
            current = np.clip(current + delta, floor_val, frontier)
            preds[i] = current
        return preds

    # Dependency ratio: complex - some aging, some young
    if var_name == "dependency_ratio":
        preds = np.zeros(len(forecast_years))
        current = last_val
        for i, year in enumerate(forecast_years):
            gap = frontier - current
            delta = gap * 0.008
            current = np.clip(current + delta, floor_val, frontier)
            preds[i] = current
        return preds

    # Broadband: S-curve toward 50.0
    if var_name == "broadband":
        preds = np.zeros(len(forecast_years))
        current = last_val
        for i, year in enumerate(forecast_years):
            gap = frontier - current
            progress = (current - floor_val) / max(frontier - floor_val, 1)
            speed = 0.06 * np.sin(np.pi * progress)
            speed = max(speed, 0.01)
            delta = gap * speed
            current = np.clip(current + delta, floor_val, frontier)
            preds[i] = current
        return preds

    # Climate risk: slow improvement toward 0.05
    if var_name == "climate_risk":
        preds = np.zeros(len(forecast_years))
        current = last_val
        for i, year in enumerate(forecast_years):
            gap = current - frontier
            if gap > 0.01:
                delta = -gap * 0.008
            else:
                delta = -0.001
            current = np.clip(current + delta, frontier, floor_val)
            preds[i] = current
        return preds

    if var_name in FUTURE_VARS:
        speed_map = {
            "ai_adoption_index": 0.10,
            "cloud_adoption": 0.09,
            "startup_ecosystem": 0.055,
            "venture_capital": 0.05,
            "green_energy_investment": 0.055,
            "ev_adoption": 0.075,
            "robot_density": 0.055,
            "semiconductor_production": 0.04,
            "battery_capacity": 0.05,
            "high_speed_rail": 0.035,
        }
        preds = np.zeros(len(forecast_years))
        current = float(np.clip(last_val, floor_val, frontier))
        speed = speed_map[var_name]
        fragile_future_states = {"AFG", "YEM", "SOM", "SSD", "SYR", "HTI", "CAF"}
        if country_id in {"USA", "CHN", "KOR", "JPN", "DEU", "SGP", "TWN", "ISR"}:
            speed *= 1.25
        for i, year in enumerate(forecast_years):
            effective_speed = speed
            if country_id in fragile_future_states:
                recovery = get_recovery_potential(country_id, 0.50, int(year))
                effective_speed *= 0.65 + 0.45 * recovery
            progress = (current - floor_val) / max(frontier - floor_val, 1e-6)
            s_curve = max(0.01, np.sin(np.pi * np.clip(progress, 0.02, 0.98)))
            delta = (frontier - current) * effective_speed * s_curve
            current = float(np.clip(current + delta, floor_val, frontier))
            preds[i] = current
        return preds

    # Population
    if var_name == "population":
        if last_val > 0 and len(recent) >= 3:
            growth_rates = recent[1:] / np.maximum(recent[:-1], 1) - 1
            valid_growth = growth_rates[np.isfinite(growth_rates)]
            avg_growth = np.median(valid_growth[-5:]) if len(valid_growth) > 0 else 0.01
            avg_growth = np.clip(avg_growth, -0.01, 0.03)
        else:
            avg_growth = 0.01
        preds = np.zeros(len(forecast_years))
        current = last_val
        for i, year in enumerate(forecast_years):
            current = current * (1 + avg_growth)
            current = max(current, 10000)
            preds[i] = current
        return preds

    # Default
    t = np.arange(len(vals), dtype=float)
    coeffs = np.polyfit(t, vals, min(2, len(vals) - 1))
    t_fc = np.arange(len(vals), len(vals) + len(forecast_years), dtype=float)
    preds = np.polyval(coeffs, t_fc)
    if frontier is not None and floor_val is not None:
        preds = np.clip(preds, floor_val, frontier)
    return preds


def forecast_all_countries(
    df: pd.DataFrame,
    forecast_years: np.ndarray,
    fc_vars: list,
) -> pd.DataFrame:
    """Forecast all variables for all countries."""
    countries = df["country_id"].unique()
    forecast_rows = []
    total = len(countries)

    for idx, country_id in enumerate(countries):
        if (idx + 1) % 30 == 0 or idx == 0:
            print(f"  Forecasting {idx+1}/{total} countries...")
        cdf = df[df["country_id"] == country_id].sort_values("year")
        country_name = cdf["country_name"].iloc[0] if "country_name" in cdf.columns else country_id
        archetype = cdf["archetype"].iloc[0] if "archetype" in cdf.columns else "lower_middle"

        country_forecasts = {}
        for var in fc_vars:
            series = cdf[var].values
            years_obs = cdf["year"].values.astype(float)
            country_forecasts[var] = forecast_variable_realistic(
                series, years_obs, forecast_years, var, country_id
            )

        for year_idx, year in enumerate(forecast_years):
            row = {
                "country_id": country_id, "country_name": country_name,
                "archetype": archetype, "year": int(year),
            }
            for var in fc_vars:
                row[var] = float(country_forecasts[var][year_idx])
            forecast_rows.append(row)

    return pd.DataFrame(forecast_rows)


# ============================================================
# HDI FORMULA (v4): Country-Specific with Nonlinear Effects
# ============================================================

# Country-specific HDI boost multipliers for key countries
# These encode structural advantages/disadvantages beyond the formula
COUNTRY_HDI_BOOST = {
    # China: infrastructure + education + tech investment
    "CHN": 0.030,
    # India: demographic dividend + digital leapfrogging
    "IND": 0.012,
    # Bangladesh: garment industry + microfinance
    "BGD": 0.008,
    # Vietnam: manufacturing hub
    "VNM": 0.008,
    # Indonesia: growing middle class
    "IDN": 0.006,
    # Philippines: OFW remittances + BPO
    "PHL": 0.005,
    # Ethiopia: population-scale catch-up, urbanization, energy, and education runway
    "ETH": 0.014,
    # Rwanda: governance-driven development
    "RWA": 0.006,
    # Kenya: East African tech hub
    "KEN": 0.005,
    # Ghana: stable democracy
    "GHA": 0.005,
    # Senegal: stable growth
    "SEN": 0.004,
    # Tanzania: young population
    "TZA": 0.004,
    # Nigeria: 2040s demographic dividend + human-capital scale
    "NGA": 0.014,
    # Kazakhstan: Eurasian hub
    "KAZ": 0.004,
    # Turkey: customs union
    "TUR": 0.004,
    # UAE: diversification
    "ARE": 0.006,
    # Saudi Arabia: Vision 2030
    "SAU": 0.003,
    # Qatar: LNG wealth + education
    "QAT": 0.005,
    # Singapore: tech + governance
    "SGP": 0.003,
    # Israel: tech innovation
    "ISR": 0.002,
    # South Korea: tech leadership
    "KOR": 0.002,
    # Guyana: oil windfall can finance health, education, and infrastructure
    # if partially converted into public investment despite concentration risk.
    "GUY": 0.024,
    # Poland: EU convergence
    "POL": 0.003,
    # Czechia: EU convergence
    "CZE": 0.002,
    # Romania: EU convergence
    "ROU": 0.003,
    # Croatia: EU convergence
    "HRV": 0.002,
    # Montenegro: EU candidate
    "MNE": 0.002,
    # Serbia: EU candidate
    "SRB": 0.002,
    # Albania: EU candidate
    "ALB": 0.002,
    # North Macedonia: EU candidate
    "MKD": 0.002,
    # Bosnia: EU candidate
    "BIH": 0.002,
    # Georgia: EU candidate
    "GEO": 0.002,
    # Moldova: EU candidate
    "MDA": 0.002,
    # Jordan: stability
    "JOR": 0.002,
    # Morocco: reforms
    "MAR": 0.003,
    # Tunisia: stronger services platform, but constrained by macro/scale limits
    "TUN": -0.004,
    # Egypt: infrastructure
    "EGY": 0.003,
    # Uzbekistan: reforms
    "UZB": 0.003,
}

# Country-specific aging penalties (additional to demographic factor)
COUNTRY_AGING_PENALTY = {
    "JPN": -0.015,  # Severe aging, shrinking workforce
    "ITA": -0.012,  # Severe aging
    "DEU": -0.008,  # Aging
    "FIN": -0.008,  # Aging
    "GRC": -0.010,  # Aging + economic
    "PRT": -0.008,  # Aging
    "ESP": -0.008,  # Aging
    "KOR": -0.015,  # Severe aging, lowest fertility
    "CHN": -0.004,  # One-child legacy, partly offset by automation/state capacity
    "THA": -0.005,  # Starting to age
    "SGP": -0.003,  # Aging offset by immigration
    "AUT": -0.005,  # Aging
    "BEL": -0.005,  # Aging
    "NLD": -0.005,  # Aging
    "SWE": -0.005,  # Aging
    "DNK": -0.005,  # Aging
    "NOR": -0.005,  # Aging
    "CHE": -0.005,  # Aging
    "GBR": -0.005,  # Aging
    "FRA": -0.005,  # Aging
    "CAN": -0.005,  # Aging
    "AUS": -0.005,  # Aging
    "NZL": -0.005,  # Aging
    "USA": -0.005,  # Aging
    "CZE": -0.005,  # Aging
    "POL": -0.008,  # Rapid aging, emigration
    "HUN": -0.005,  # Aging
    "SVK": -0.005,  # Aging
    "HRV": -0.005,  # Aging
    "SVN": -0.005,  # Aging
    "EST": -0.005,  # Aging
    "LTU": -0.005,  # Aging
    "LVA": -0.005,  # Aging
    "BGR": -0.008,  # Rapid aging, emigration
    "ROU": -0.005,  # Aging
    "JAM": -0.003,  # Aging
    "TTO": -0.003,  # Aging
    "CUB": -0.005,  # Aging
    "RUS": -0.008,  # Demographic crisis
    "UKR": -0.008,  # War + demographic crisis
    "BLR": -0.005,  # Aging
    "SRB": -0.005,  # Aging
    "MNE": -0.003,  # Aging
    "MKD": -0.003,  # Aging
    "BIH": -0.003,  # Aging
    "ALB": -0.003,  # Aging
}


EXPECTED_REGRESSION_RISK = {
    # Central forecast drag from severe current fragility, institutional rupture,
    # sanctions/isolation, or long-running macro crisis. P90 can still recover.
    "VEN": 0.030, "HTI": 0.028, "PRK": 0.026, "LBN": 0.024,
    "SDN": 0.024, "SSD": 0.024, "YEM": 0.022, "AFG": 0.022,
    "SOM": 0.022, "SYR": 0.020, "MMR": 0.020, "ERI": 0.020,
    "CAF": 0.018, "TCD": 0.018, "MLI": 0.016, "BFA": 0.016,
    "NER": 0.016, "COD": 0.016, "ZWE": 0.014, "CUB": 0.012,
    # Demographic/frontier saturation risks.
    "JPN": 0.010, "KOR": 0.008, "ITA": 0.008, "GRC": 0.006,
    "PRT": 0.006, "ESP": 0.005, "DEU": 0.005, "TUN": 0.006,
    # South American reform/macro stagnation risk: plausible upside exists,
    # but without reform the median path should not assume smooth convergence.
    "ARG": 0.010, "BOL": 0.008, "ECU": 0.007, "PRY": 0.007,
    "PER": 0.006, "BRA": 0.005, "COL": 0.004, "SUR": 0.006,
}


RESOURCE_VOLATILITY_PROFILES = {
    # dependence: resource export/fiscal dependence, volatility: boom-bust exposure,
    # diversification: how much non-resource capacity offsets the risk.
    "SAU": {"dependence": 0.80, "volatility": 0.60, "diversification": 0.55},
    "QAT": {"dependence": 0.88, "volatility": 0.50, "diversification": 0.58},
    "KWT": {"dependence": 0.92, "volatility": 0.58, "diversification": 0.40},
    "ARE": {"dependence": 0.55, "volatility": 0.38, "diversification": 0.78},
    "BHR": {"dependence": 0.45, "volatility": 0.35, "diversification": 0.68},
    "OMN": {"dependence": 0.72, "volatility": 0.55, "diversification": 0.50},
    "IRQ": {"dependence": 0.88, "volatility": 0.78, "diversification": 0.22},
    "IRN": {"dependence": 0.68, "volatility": 0.75, "diversification": 0.40},
    "DZA": {"dependence": 0.78, "volatility": 0.68, "diversification": 0.32},
    "LBY": {"dependence": 0.90, "volatility": 0.88, "diversification": 0.20},
    "RUS": {"dependence": 0.65, "volatility": 0.72, "diversification": 0.52},
    "KAZ": {"dependence": 0.58, "volatility": 0.58, "diversification": 0.50},
    "AZE": {"dependence": 0.76, "volatility": 0.70, "diversification": 0.32},
    "TKM": {"dependence": 0.86, "volatility": 0.76, "diversification": 0.18},
    "NGA": {"dependence": 0.58, "volatility": 0.72, "diversification": 0.34},
    "AGO": {"dependence": 0.82, "volatility": 0.82, "diversification": 0.22},
    "GAB": {"dependence": 0.70, "volatility": 0.65, "diversification": 0.35},
    "GNQ": {"dependence": 0.92, "volatility": 0.90, "diversification": 0.15},
    "COG": {"dependence": 0.78, "volatility": 0.78, "diversification": 0.25},
    "TCD": {"dependence": 0.58, "volatility": 0.78, "diversification": 0.15},
    "SSD": {"dependence": 0.70, "volatility": 0.90, "diversification": 0.08},
    "VEN": {"dependence": 0.86, "volatility": 0.95, "diversification": 0.18},
    "TTO": {"dependence": 0.70, "volatility": 0.62, "diversification": 0.45},
    "GUY": {"dependence": 0.72, "volatility": 0.68, "diversification": 0.36},
    "SUR": {"dependence": 0.58, "volatility": 0.68, "diversification": 0.30},
    "BRN": {"dependence": 0.86, "volatility": 0.58, "diversification": 0.35},
    "NOR": {"dependence": 0.42, "volatility": 0.25, "diversification": 0.88},
    "CAN": {"dependence": 0.28, "volatility": 0.25, "diversification": 0.88},
    "AUS": {"dependence": 0.35, "volatility": 0.30, "diversification": 0.82},
    # Non-oil macro volatility / commodity-cycle exposure.
    "ARG": {"dependence": 0.35, "volatility": 0.78, "diversification": 0.55},
    "TUR": {"dependence": 0.20, "volatility": 0.65, "diversification": 0.62},
    "LBN": {"dependence": 0.18, "volatility": 0.90, "diversification": 0.35},
    "ZWE": {"dependence": 0.45, "volatility": 0.82, "diversification": 0.25},
    "ZMB": {"dependence": 0.55, "volatility": 0.70, "diversification": 0.28},
    "MNG": {"dependence": 0.62, "volatility": 0.72, "diversification": 0.25},
    "CHL": {"dependence": 0.45, "volatility": 0.42, "diversification": 0.68},
    "PER": {"dependence": 0.38, "volatility": 0.52, "diversification": 0.54},
    "BRA": {"dependence": 0.22, "volatility": 0.42, "diversification": 0.66},
    "COL": {"dependence": 0.28, "volatility": 0.48, "diversification": 0.58},
    "ECU": {"dependence": 0.46, "volatility": 0.58, "diversification": 0.42},
    "BOL": {"dependence": 0.42, "volatility": 0.62, "diversification": 0.38},
    "PRY": {"dependence": 0.24, "volatility": 0.50, "diversification": 0.46},
}


def get_resource_volatility_profile(country_id: str) -> dict:
    """Return resource dependence and macro volatility assumptions."""
    return RESOURCE_VOLATILITY_PROFILES.get(
        country_id,
        {"dependence": 0.12, "volatility": 0.22, "diversification": 0.60},
    )


def compute_resource_drag(country_id: str, capacity_score: float) -> float:
    """Translate resource dependence into a 0-1 drag on HDI conversion."""
    profile = get_resource_volatility_profile(country_id)
    dependence = profile["dependence"]
    volatility = profile["volatility"]
    diversification = profile["diversification"]
    concentration = dependence * volatility * (1.0 - 0.55 * diversification)
    institutional_buffer = 1.0 - 0.35 * np.clip(capacity_score, 0.0, 1.0)
    return float(np.clip(concentration * institutional_buffer, 0.0, 1.0))


INDUSTRIALIZATION_ACCELERATION_PRIOR = {
    # Large-scale manufacturing, digitalization, infrastructure, or export-led
    # convergence potential. This is an accelerator, not a direct HDI bonus.
    "CHN": 0.72, "IND": 0.78, "IDN": 0.76, "VNM": 0.74, "BGD": 0.66,
    "PHL": 0.62, "MYS": 0.58, "THA": 0.52, "TUR": 0.42,
    "RWA": 0.58, "SEN": 0.54, "TZA": 0.54, "ETH": 0.62, "KEN": 0.54,
    "GHA": 0.48, "NGA": 0.54, "UGA": 0.46, "CMR": 0.42, "CIV": 0.44, "MOZ": 0.44,
    "PAK": 0.52, "NPL": 0.44, "KHM": 0.50, "LAO": 0.46,
    "MAR": 0.46, "EGY": 0.46, "TUN": 0.44, "ROU": 0.36, "POL": 0.32,
    "MEX": 0.42, "COL": 0.32, "PER": 0.30, "DOM": 0.36,
    "GUY": 0.42, "BRA": 0.30, "ECU": 0.28, "PRY": 0.26, "BOL": 0.24,
}


REFORM_STAGNATION_DRAG = {
    # Reform-limited South American median paths. These are not pessimistic
    # collapse assumptions; they just keep catch-up from looking automatic.
    "ARG": 0.18,
    "BOL": 0.16,
    "ECU": 0.14,
    "PRY": 0.14,
    "PER": 0.12,
    "BRA": 0.11,
    "COL": 0.09,
    "SUR": 0.12,
}


OIL_WINDFALL_CONVERSION = {
    # Scale of plausible 2024-2050 public-investment conversion from resource
    # windfalls after volatility and institutional absorption are considered.
    "GUY": 0.42,
}


def compute_industrialization_acceleration(
    country_id: str,
    hdi_current: float,
    future_readiness_current: float | None,
    capacity_score: float,
    resource_drag: float,
) -> float:
    """Return a 0-1 accelerator for rapid industrializing HDI catch-up."""
    future_readiness = 0.30 if future_readiness_current is None else float(np.clip(future_readiness_current, 0.0, 1.0))
    prior = INDUSTRIALIZATION_ACCELERATION_PRIOR.get(country_id, 0.18)
    development_headroom = np.clip((0.88 - hdi_current) / 0.34, 0.0, 1.0)
    institutional_absorption = 0.55 + 0.45 * np.clip(capacity_score, 0.0, 1.0)
    concentration_penalty = 1.0 - 0.45 * np.clip(resource_drag, 0.0, 1.0)
    accelerator = (
        0.52 * prior +
        0.26 * future_readiness +
        0.22 * development_headroom
    )
    accelerator *= institutional_absorption * concentration_penalty
    if hdi_current >= 0.90:
        accelerator *= 0.25
    elif hdi_current >= 0.85:
        accelerator *= 0.55
    return float(np.clip(accelerator, 0.0, 1.0))


TRAJECTORY_HDI_EFFECTS = {
    "stagnant": {
        "speed": 0.72, "gdp": 0.78, "education": 0.82, "health": 0.82,
        "future": 0.88, "shock": 1.12, "cap": 0.70,
    },
    "slow_growth": {
        "speed": 0.82, "gdp": 0.84, "education": 0.88, "health": 0.88,
        "future": 0.92, "shock": 1.04, "cap": 0.82,
    },
    "moderate": {
        "speed": 1.00, "gdp": 1.00, "education": 1.00, "health": 1.00,
        "future": 1.00, "shock": 1.00, "cap": 1.00,
    },
    "fast_converger": {
        "speed": 1.08, "gdp": 1.06, "education": 1.06, "health": 1.03,
        "future": 1.08, "shock": 0.96, "cap": 1.08,
    },
    "accelerator": {
        "speed": 1.16, "gdp": 1.10, "education": 1.12, "health": 1.06,
        "future": 1.10, "shock": 0.94, "cap": 1.16,
    },
    "frontier_jumper": {
        "speed": 1.22, "gdp": 1.12, "education": 1.18, "health": 1.10,
        "future": 1.08, "shock": 0.96, "cap": 1.22,
    },
    "conflict_recovery": {
        "speed": 0.92, "gdp": 0.90, "education": 0.98, "health": 1.00,
        "future": 0.92, "shock": 1.10, "cap": 0.95,
    },
}


def compute_trajectory_hdi_effect(
    country_id: str,
    hdi_current: float,
    recovery_potential: float,
    capacity_score: float,
) -> dict:
    """Translate trajectory labels into smooth HDI model multipliers."""
    traj_class = get_trajectory_class(country_id)
    effect = TRAJECTORY_HDI_EFFECTS.get(traj_class, TRAJECTORY_HDI_EFFECTS["moderate"]).copy()

    if traj_class == "conflict_recovery":
        recovery = np.clip(recovery_potential, 0.0, 1.0)
        capacity = np.clip(capacity_score, 0.0, 1.0)
        recovery_conversion = recovery * (0.55 + 0.45 * capacity)
        effect["speed"] = 0.82 + 0.48 * recovery_conversion
        effect["gdp"] = 0.82 + 0.34 * recovery_conversion
        effect["education"] = 0.90 + 0.34 * recovery_conversion
        effect["health"] = 0.92 + 0.32 * recovery_conversion
        effect["future"] = 0.86 + 0.28 * recovery_conversion
        effect["shock"] = 1.18 - 0.34 * recovery_conversion
        effect["cap"] = 0.92 + 0.35 * recovery_conversion

    if country_id == "CHN":
        effect["speed"] *= 1.10
        effect["gdp"] *= 1.06
        effect["future"] *= 1.12
        effect["shock"] *= 0.92
        effect["cap"] *= 1.10
    elif country_id == "NGA":
        effect["speed"] *= 1.14
        effect["gdp"] *= 1.08
        effect["education"] *= 1.16
        effect["health"] *= 1.06
        effect["future"] *= 1.08
        effect["shock"] *= 0.96
        effect["cap"] *= 1.14
    elif country_id == "ETH":
        effect["speed"] *= 1.10
        effect["gdp"] *= 1.08
        effect["education"] *= 1.14
        effect["health"] *= 1.05
        effect["future"] *= 1.08
        effect["shock"] *= 0.98
        effect["cap"] *= 1.12
    elif country_id == "GUY":
        effect["speed"] *= 1.12
        effect["gdp"] *= 1.10
        effect["education"] *= 1.04
        effect["health"] *= 1.04
        effect["shock"] *= 0.98
        effect["cap"] *= 1.12
    elif country_id == "TUN":
        effect["speed"] *= 0.92
        effect["gdp"] *= 0.94
        effect["future"] *= 0.94
        effect["shock"] *= 1.06
        effect["cap"] *= 0.92

    reform_drag = REFORM_STAGNATION_DRAG.get(country_id, 0.0)
    if reform_drag:
        effect["speed"] *= 1.0 - 0.42 * reform_drag
        effect["gdp"] *= 1.0 - 0.35 * reform_drag
        effect["education"] *= 1.0 - 0.18 * reform_drag
        effect["future"] *= 1.0 - 0.25 * reform_drag
        effect["shock"] *= 1.0 + 0.35 * reform_drag
        effect["cap"] *= 1.0 - 0.40 * reform_drag

    if hdi_current >= 0.90:
        effect["speed"] *= 0.90
        effect["cap"] *= 0.88
    elif hdi_current < 0.60 and traj_class in ("accelerator", "frontier_jumper", "conflict_recovery"):
        effect["cap"] *= 1.08

    return {key: float(np.clip(value, 0.55, 1.35)) for key, value in effect.items()}


def compute_developing_catchup_readiness(
    hdi_current: float,
    edu_index: float,
    urbanization: float,
    fertility: float,
    capacity_score: float,
    future_readiness_current: float | None,
    resource_drag: float,
    recovery_potential: float,
    industrialization_accel: float,
) -> float:
    """Estimate whether a country can convert catch-up potential into HDI gains."""
    if hdi_current >= 0.88:
        return 0.0

    future_readiness = 0.30 if future_readiness_current is None else float(np.clip(future_readiness_current, 0.0, 1.0))
    headroom = np.clip((0.88 - hdi_current) / 0.46, 0.0, 1.0)
    education_platform = np.clip(edu_index / 0.72, 0.0, 1.0)
    urban_platform = np.clip((urbanization - 0.25) / 0.55, 0.0, 1.0)
    demographic_absorption = 1.0 - 0.35 * np.clip((fertility - 3.0) / 3.0, 0.0, 1.0)
    institutional_absorption = 0.40 + 0.60 * np.clip(capacity_score, 0.0, 1.0)
    volatility_absorption = 1.0 - 0.45 * np.clip(resource_drag, 0.0, 1.0)

    readiness = (
        0.22 * headroom +
        0.20 * education_platform +
        0.16 * urban_platform +
        0.18 * future_readiness +
        0.14 * industrialization_accel +
        0.10 * recovery_potential
    )
    readiness *= demographic_absorption * institutional_absorption * volatility_absorption
    return float(np.clip(readiness, 0.0, 1.0))


LOW_GROWTH_PROSPECT_RISK_PRIOR = {
    # Structural stagnation / aging / macro crisis / sanctions / conflict risk.
    "JPN": 0.55, "ITA": 0.52, "GRC": 0.50, "PRT": 0.38, "KOR": 0.34,
    "RUS": 0.62, "BLR": 0.56, "UKR": 0.46, "CUB": 0.58, "PRK": 0.78,
    "VEN": 0.76, "LBN": 0.72, "HTI": 0.70, "MMR": 0.64,
    "SYR": 0.64, "YEM": 0.66, "AFG": 0.66, "SOM": 0.68, "SSD": 0.70,
    "SDN": 0.62, "ERI": 0.60, "CAF": 0.56, "TCD": 0.58, "MLI": 0.54,
    "BFA": 0.52, "NER": 0.50, "NGA": 0.34, "AGO": 0.48, "GNQ": 0.58,
    "COG": 0.48, "LBY": 0.58, "IRQ": 0.48, "IRN": 0.52, "DZA": 0.42,
    "ARG": 0.46, "TUR": 0.34, "ZWE": 0.54, "PAK": 0.34,
    "BOL": 0.40, "ECU": 0.36, "PRY": 0.34, "PER": 0.32,
    "BRA": 0.30, "COL": 0.28, "SUR": 0.38, "GUY": 0.22, "ETH": 0.36,
}


DEVELOPED_DEMOGRAPHIC_ADAPTATION_PRIOR = {
    # Ability to soften aging through skilled migration, higher participation,
    # family policy, automation, and active workforce reskilling. This is a
    # marginal resilience channel for already-developed countries, not a
    # catch-up accelerator.
    "CAN": 0.86, "AUS": 0.84, "NZL": 0.78, "USA": 0.72, "GBR": 0.70,
    "CHE": 0.76, "DEU": 0.70, "NLD": 0.74, "SWE": 0.76, "NOR": 0.74,
    "DNK": 0.72, "FIN": 0.66, "AUT": 0.68, "BEL": 0.66, "FRA": 0.66,
    "IRL": 0.78, "LUX": 0.82, "SGP": 0.82, "ARE": 0.80, "QAT": 0.76,
    "KWT": 0.62, "BHR": 0.68, "OMN": 0.60, "ISR": 0.70,
    "JPN": 0.46, "KOR": 0.50, "ITA": 0.42, "ESP": 0.52, "PRT": 0.46,
    "GRC": 0.38, "CZE": 0.52, "SVN": 0.54, "EST": 0.58, "LTU": 0.48,
    "LVA": 0.46, "POL": 0.44, "HRV": 0.42, "SVK": 0.46, "HUN": 0.42,
    "MLT": 0.68, "CYP": 0.58,
}


def compute_developed_demographic_adaptation(
    country_id: str,
    hdi_current: float,
    capacity_score: float,
    future_readiness_current: float | None,
    aging_pressure: float,
) -> float:
    """Return 0-1 adaptation capacity for high-HDI demographic headwinds."""
    if hdi_current < 0.88:
        return 0.0

    prior = DEVELOPED_DEMOGRAPHIC_ADAPTATION_PRIOR.get(country_id, 0.34)
    future_readiness = 0.40 if future_readiness_current is None else float(np.clip(future_readiness_current, 0.0, 1.0))
    state_capacity = float(np.clip(capacity_score, 0.0, 1.0))
    aging_need = float(np.clip(aging_pressure, 0.0, 1.0))

    adaptation = (
        0.38 * prior +
        0.30 * state_capacity +
        0.20 * future_readiness +
        0.12 * aging_need
    )
    return float(np.clip(adaptation, 0.0, 1.0))


def compute_growth_prospect_score(
    country_id: str,
    hdi_current: float,
    capacity_score: float,
    future_readiness_current: float | None,
    fertility: float,
    urbanization: float,
    resource_drag: float,
    recovery_potential: float,
    industrialization_accel: float,
    catchup_readiness: float,
    aging_penalty_structural: float,
) -> float:
    """Return 0-1 long-run growth prospect quality for HDI conversion."""
    future_readiness = 0.30 if future_readiness_current is None else float(np.clip(future_readiness_current, 0.0, 1.0))
    traj_class = get_trajectory_class(country_id)
    risk_prior = LOW_GROWTH_PROSPECT_RISK_PRIOR.get(country_id, 0.18)
    if traj_class == "stagnant":
        risk_prior = max(risk_prior, 0.50)
    elif traj_class == "slow_growth":
        risk_prior = max(risk_prior, 0.36)
    elif traj_class == "conflict_recovery":
        risk_prior = max(risk_prior, 0.42)

    demographic_score = 1.0 - 0.40 * np.clip((fertility - 3.0) / 3.0, 0.0, 1.0)
    urban_score = np.clip((urbanization - 0.25) / 0.55, 0.0, 1.0)
    frontier_inertia = np.clip((hdi_current - 0.88) / 0.10, 0.0, 1.0)
    aging_drag = np.clip(aging_penalty_structural, 0.0, 1.0)
    demographic_adaptation = compute_developed_demographic_adaptation(
        country_id,
        hdi_current,
        capacity_score,
        future_readiness_current,
        aging_drag,
    )
    aging_drag *= 1.0 - 0.45 * demographic_adaptation

    opportunity = (
        0.24 * capacity_score +
        0.18 * future_readiness +
        0.17 * catchup_readiness +
        0.14 * industrialization_accel +
        0.12 * urban_score +
        0.08 * demographic_score +
        0.07 * recovery_potential +
        0.05 * demographic_adaptation * frontier_inertia
    )
    risk = (
        0.34 * risk_prior +
        0.24 * resource_drag +
        0.18 * (1.0 - capacity_score) +
        0.12 * aging_drag +
        0.12 * frontier_inertia
    )
    return float(np.clip(opportunity - risk + 0.36, 0.0, 1.0))


def compute_hdi_with_factors(
    country_id: str,
    hdi_current: float,
    year: int = 2050,
    gni_ppp_current: float = 30000.0,
    life_exp_current: float = 72.0,
    expected_school_current: float = 13.0,
    mean_school_current: float = 8.0,
    future_readiness_current: float | None = None,
    digital_infra_development_current: float | None = None,
    gini_current: float | None = None,
    infant_mortality_current: float | None = None,
    climate_risk_current: float | None = None,
    fertility_current: float | None = None,
    urbanization_current: float | None = None,
    political_stability_current: float | None = None,
    rule_of_law_current: float | None = None,
    physicians_current: float | None = None,
    health_exp_current: float | None = None,
    renewable_share_current: float | None = None,
) -> float:
    """Compute HDI2050 using convergence model (v6).

    HDI_2050 = HDI_current + Gap × (1 - exp(-Speed × Years))

    Where:
      Gap = 0.985 - HDI_current  (remaining headroom)
      Speed = GDP × Edu × Health × Gov × Demo  (0.0 to ~0.5)
      Years = 26 (2024 → 2050)

    This ensures:
      - Natural slowdown near frontier (Gap shrinks)
      - No triple compounding of components
      - Single convergence curve per country
    """
    from data.demographics import (
        compute_demographic_factor,
        compute_demographic_profile,
        get_aging_penalty,
        get_median_age,
        get_working_age_pct,
        get_youth_dividend,
    )
    from data.technology import compute_technology_factor
    from data.stability import get_governance_multiplier, get_state_capacity

    inst_eff = get_governance_multiplier(country_id, year)
    state = get_state_capacity(country_id)
    capacity_score = float(np.clip(np.mean([
        state.get("stability", 0.5),
        state.get("conflict", 0.65),
        state.get("corruption", 0.45),
        state.get("governance", 0.48),
        state.get("fragility", 0.5),
    ]), 0.0, 1.0))
    demo_factor = compute_demographic_factor(country_id, year)
    tech_factor = compute_technology_factor(country_id, year)
    years = year - 2024  # 26
    recovery_potential = get_recovery_potential(country_id, hdi_current, year)
    traj_class = get_trajectory_class(country_id)
    median_age_2050 = get_median_age(country_id, year)
    working_age_2050 = get_working_age_pct(country_id, year) / 100.0
    youth_dividend = get_youth_dividend(country_id, year)
    aging_penalty_structural = get_aging_penalty(country_id, year)
    saturation_factor = get_frontier_saturation_factor(hdi_current)
    resource_profile = get_resource_volatility_profile(country_id)
    resource_drag = compute_resource_drag(country_id, capacity_score)
    industrialization_accel = compute_industrialization_acceleration(
        country_id,
        hdi_current,
        future_readiness_current,
        capacity_score,
        resource_drag,
    )
    trajectory_effect = compute_trajectory_hdi_effect(
        country_id,
        hdi_current,
        recovery_potential,
        capacity_score,
    )

    gini = 0.38 if gini_current is None else float(np.clip(gini_current, 0.20, 0.65))
    infant_mortality = 25.0 if infant_mortality_current is None else float(np.clip(infant_mortality_current, 1.0, 120.0))
    climate_risk = 0.30 if climate_risk_current is None else float(np.clip(climate_risk_current, 0.0, 1.0))
    fertility = 2.3 if fertility_current is None else float(np.clip(fertility_current, 1.0, 7.0))
    urbanization = 0.56 if urbanization_current is None else float(np.clip(urbanization_current, 0.10, 0.98))
    political_stability = 0.0 if political_stability_current is None else float(np.clip(political_stability_current, -2.5, 2.5))
    rule_of_law = 0.0 if rule_of_law_current is None else float(np.clip(rule_of_law_current, -2.5, 2.5))
    physicians = 1.5 if physicians_current is None else float(np.clip(physicians_current, 0.05, 8.0))
    health_exp = 6.0 if health_exp_current is None else float(np.clip(health_exp_current, 1.0, 18.0))
    renewable_share = 0.20 if renewable_share_current is None else float(np.clip(renewable_share_current, 0.0, 1.0))
    digital_infra_development = (
        0.35 if digital_infra_development_current is None
        else float(np.clip(digital_infra_development_current, 0.0, 1.0))
    )
    exp_idx = np.clip(expected_school_current / 18.0, 0, 1)
    mean_idx = np.clip(mean_school_current / 16.0, 0, 1)
    edu_index = np.sqrt(exp_idx * mean_idx)
    demographic_profile = compute_demographic_profile(country_id, year, fertility, edu_index)
    age_pyramid_score = demographic_profile["age_pyramid_score"]
    workforce_depth = demographic_profile["workforce_depth"]
    workforce_momentum = demographic_profile["workforce_momentum"]
    human_capital_absorption = demographic_profile["human_capital_absorption"]
    demographic_dividend = demographic_profile["demographic_dividend"]
    dependency_pressure = demographic_profile["dependency_pressure"]
    youth_dependency_pressure = demographic_profile["youth_dependency_pressure"]
    aging_pressure = demographic_profile["aging_pressure"]
    demographic_hdi_multiplier = demographic_profile["demographic_hdi_multiplier"]
    developed_demographic_adaptation = compute_developed_demographic_adaptation(
        country_id,
        hdi_current,
        capacity_score,
        future_readiness_current,
        aging_pressure,
    )
    adapted_aging_pressure = aging_pressure * (1.0 - 0.42 * developed_demographic_adaptation)
    adapted_aging_penalty_structural = aging_penalty_structural * (1.0 - 0.40 * developed_demographic_adaptation)
    demographic_hdi_multiplier *= 1.0 + 0.045 * developed_demographic_adaptation * np.clip((hdi_current - 0.88) / 0.10, 0.0, 1.0)
    demographic_hdi_multiplier = float(np.clip(demographic_hdi_multiplier, 0.72, 1.20))
    developing_catchup_readiness = compute_developing_catchup_readiness(
        hdi_current,
        edu_index,
        urbanization,
        fertility,
        capacity_score,
        future_readiness_current,
        resource_drag,
        recovery_potential,
        industrialization_accel,
    )
    growth_prospect_score = compute_growth_prospect_score(
        country_id,
        hdi_current,
        capacity_score,
        future_readiness_current,
        fertility,
        urbanization,
        resource_drag,
        recovery_potential,
        industrialization_accel,
        developing_catchup_readiness,
        adapted_aging_penalty_structural,
    )
    low_growth_drag = 1.0 - growth_prospect_score

    HDI_CEILING = 0.985
    gap = max(0, HDI_CEILING - hdi_current)

    if gap <= 0.001:
        return min(hdi_current + 0.001, HDI_CEILING)

    # ================================================================
    # SPEED FACTOR 1: GDP GROWTH → economic development speed
    # Maps annual GDP growth to a speed contribution (0.02 - 0.40)
    # ================================================================
    GDP_GROWTH_RATES = {
        # Frontier jumpers
        "CHN": 0.038, "IND": 0.055, "VNM": 0.048, "BGD": 0.050, "IDN": 0.042,
        "RWA": 0.055, "ETH": 0.058, "KEN": 0.050, "GHA": 0.045, "SEN": 0.045,
        "TZA": 0.050, "UGA": 0.050, "CMR": 0.040, "CIV": 0.040, "AGO": 0.035,
        "MOZ": 0.045, "ZMB": 0.040, "NAM": 0.035, "BWA": 0.035,
        # Fast convergers
        "PHL": 0.045, "MAR": 0.035, "EGY": 0.035, "PER": 0.035, "COL": 0.035,
        "DOM": 0.035, "GTM": 0.035, "SLV": 0.035, "HND": 0.035, "NIC": 0.035,
        "PRY": 0.030, "BOL": 0.030, "ECU": 0.030, "BTN": 0.035, "MDV": 0.035,
        "LKA": 0.030, "NPL": 0.040, "KHM": 0.045, "LAO": 0.045,
        # Moderate growth
        "THA": 0.028, "MYS": 0.030, "MEX": 0.022, "BRA": 0.018, "TUR": 0.028,
        "KAZ": 0.030, "UZB": 0.035, "GEO": 0.030, "ARM": 0.030, "AZE": 0.030,
        "JOR": 0.025, "TUN": 0.022, "DZA": 0.022, "IRN": 0.020, "IRQ": 0.030,
        "CRI": 0.025, "PAN": 0.025, "CHL": 0.022, "URY": 0.018, "ARG": 0.018,
        # Slow growth (rich countries)
        "USA": 0.015, "GBR": 0.012, "DEU": 0.010, "FRA": 0.010, "ITA": 0.005,
        "JPN": 0.005, "KOR": 0.012, "CAN": 0.012, "AUS": 0.015, "NZL": 0.012,
        "CHE": 0.010, "NOR": 0.008, "SWE": 0.010, "DNK": 0.010, "NLD": 0.010,
        "BEL": 0.010, "AUT": 0.010, "FIN": 0.008, "ISR": 0.018, "SGP": 0.020,
        "ARE": 0.022, "QAT": 0.020, "SAU": 0.018, "KWT": 0.010, "BHR": 0.012,
        "OMN": 0.012, "ISL": 0.012, "IRL": 0.015, "LUX": 0.010,
        # Eastern Europe
        "POL": 0.018, "CZE": 0.012, "ROU": 0.018, "BGR": 0.012, "HRV": 0.012,
        "SVN": 0.012, "SVK": 0.012, "HUN": 0.010, "EST": 0.012, "LTU": 0.012,
        "LVA": 0.012, "MLT": 0.012, "CYP": 0.012, "PRT": 0.010, "ESP": 0.010,
        "GRC": 0.008, "SRB": 0.012, "MNE": 0.012, "MKD": 0.015, "ALB": 0.018,
        "BIH": 0.012, "MDA": 0.018, "UKR": 0.018, "BLR": 0.008,
        # Russia/Central Asia
        "RUS": 0.008, "KGZ": 0.022, "TJK": 0.025, "TKM": 0.020,
        # Conflict/stagnant
        "AFG": 0.015, "YEM": 0.010, "SOM": 0.020, "SSD": 0.015, "SYR": 0.010,
        "VEN": 0.010, "LBY": 0.015, "HTI": 0.015, "CUB": 0.008,
        "MMR": 0.015, "PRK": 0.008,
        # Fragile states
        "NGA": 0.052, "CAF": 0.015, "TCD": 0.020, "COD": 0.020, "GIN": 0.025, "SLE": 0.020,
        "LBR": 0.020, "BEN": 0.035, "TGO": 0.035, "GMB": 0.030, "GNB": 0.025,
        "MRT": 0.025, "ERI": 0.010, "DJI": 0.025, "COM": 0.025, "STP": 0.025,
        "SWZ": 0.020, "LSO": 0.020, "MDG": 0.030, "MWI": 0.030,
        "ZWE": 0.015, "COG": 0.020, "GAB": 0.020, "GNQ": 0.015,
        "CPV": 0.020, "MUS": 0.020,
        # Pacific
        "FJI": 0.020, "TON": 0.015, "SLB": 0.025, "VUT": 0.025, "PNG": 0.025,
        "WSM": 0.020, "KIR": 0.020, "MHL": 0.020, "NRU": 0.015, "PLW": 0.015,
        "FSM": 0.020, "TUV": 0.020,
        # Resource windfall
        "GUY": 0.045, "SUR": 0.030,
    }
    gdp_growth = GDP_GROWTH_RATES.get(country_id, 0.020)
    if recovery_potential > 0:
        developing_floor = 0.028 if hdi_current < 0.65 else 0.022
        gdp_growth = max(gdp_growth, developing_floor)
        gdp_growth += 0.018 * recovery_potential
    gdp_growth *= 1.0 + 0.16 * industrialization_accel
    gdp_growth *= trajectory_effect["gdp"]
    gdp_growth *= 0.78 + 0.34 * growth_prospect_score
    gdp_growth *= 0.90 + 0.20 * human_capital_absorption + 0.08 * workforce_momentum
    gdp_growth *= 0.94 + 0.16 * digital_infra_development
    gdp_growth *= 1.0 - 0.22 * resource_drag

    # Map GDP growth to speed: higher growth → faster convergence
    # 0.5% → 0.03, 1.5% → 0.08, 3% → 0.15, 5% → 0.25, 6% → 0.30
    gdp_speed = 0.02 + (gdp_growth - 0.005) * 5.0
    gdp_speed = np.clip(gdp_speed, 0.02, 0.35)
    income_index = np.clip(
        (np.log(max(gni_ppp_current, 100.0)) - np.log(100.0)) /
        (np.log(75000.0) - np.log(100.0)),
        0.0,
        1.0,
    )
    human_development_platform = np.clip(
        0.50 * hdi_current +
        0.32 * edu_index +
        0.18 * np.clip((life_exp_current - 55.0) / 32.0, 0.0, 1.0),
        0.0,
        1.0,
    )
    high_income_diminishing = 1.0 - 0.50 * np.clip((gni_ppp_current - 55000.0) / 70000.0, 0.0, 1.0)
    if gni_ppp_current > 75000.0:
        high_income_diminishing *= 0.35
    elif gni_ppp_current > 50000.0:
        high_income_diminishing *= 0.55
    if country_id in MICROSTATE_ISO3 and gni_ppp_current > 45000.0:
        high_income_diminishing *= 0.20
    institutional_conversion = 0.45 + 0.35 * capacity_score + 0.20 * np.clip((inst_eff - 0.70) / 0.45, 0.0, 1.0)
    inequality_conversion = 1.0 - 0.45 * np.clip((gini - 0.32) / 0.25, 0.0, 1.0)
    gdp_speed *= high_income_diminishing * institutional_conversion * inequality_conversion
    gdp_speed *= 0.35 + 0.65 * (1.0 - income_index)
    gdp_speed *= 1.0 + 0.12 * industrialization_accel
    gdp_speed *= trajectory_effect["gdp"]
    gdp_speed *= 0.82 + 0.26 * growth_prospect_score
    gdp_speed *= 0.88 + 0.18 * workforce_depth + 0.10 * human_capital_absorption
    gdp_speed *= 0.92 + 0.18 * digital_infra_development
    gdp_speed *= 1.0 - 0.35 * resource_drag

    # ================================================================
    # SPEED FACTOR 2: EDUCATION level → headroom for improvement
    # Countries with low education have more room to grow
    # ================================================================
    # Use current education index as basis
    exp_idx = np.clip(expected_school_current / 18.0, 0, 1)
    mean_idx = np.clip(mean_school_current / 16.0, 0, 1)
    edu_index = np.sqrt(exp_idx * mean_idx)
    developing_catchup_readiness = compute_developing_catchup_readiness(
        hdi_current,
        edu_index,
        urbanization,
        fertility,
        capacity_score,
        future_readiness_current,
        resource_drag,
        recovery_potential,
        industrialization_accel,
    )

    # Low education → high speed (room to improve), high → low (saturation)
    # edu_index 0.3 → 0.20, 0.5 → 0.15, 0.7 → 0.10, 0.9 → 0.04
    edu_speed = max(0.02, 0.25 * (1.0 - edu_index))

    # Country-specific education reform capacity
    EDU_REFORM = {
        "CHN": 1.3, "IND": 1.35, "VNM": 1.2, "IDN": 1.1, "THA": 1.1,
        "MYS": 1.0, "BGD": 1.1, "RWA": 1.15, "ETH": 1.2, "KEN": 1.0,
        "GHA": 1.0, "SEN": 1.0, "TZA": 1.0, "UGA": 1.0,
        "PHL": 1.0, "COL": 0.9, "PER": 0.9, "BRA": 0.8, "MEX": 0.8,
        "FRA": 0.7, "ITA": 0.7, "ESP": 0.8, "PRT": 0.8, "GRC": 0.7,
        "SYR": 0.3, "YEM": 0.3, "AFG": 0.3, "SOM": 0.3, "SSD": 0.3,
        "IRQ": 0.5, "LBY": 0.4, "UKR": 0.7, "MMR": 0.4,
        "USA": 0.85, "GBR": 0.85, "DEU": 0.85, "NOR": 0.9, "SWE": 0.9,
        "DNK": 0.9, "ISL": 0.9, "IRL": 0.9, "LUX": 0.9,
        "NGA": 1.12, "GUY": 1.0, "ARE": 1.1, "QAT": 1.05, "SGP": 1.0,
        "BLR": 0.8, "PRK": 0.4,
    }
    edu_reform = EDU_REFORM.get(country_id, 1.0)
    if recovery_potential > 0:
        edu_reform = max(edu_reform, 0.65 + 0.60 * recovery_potential)
    edu_speed *= edu_reform
    institution_for_education = 0.70 + 0.30 * capacity_score
    demographic_absorption = 0.76 + 0.18 * workforce_depth + 0.18 * human_capital_absorption - 0.12 * youth_dependency_pressure
    urban_absorption = 0.85 + 0.15 * np.clip((urbanization - 0.35) / 0.40, 0.0, 1.0)
    edu_speed *= institution_for_education * demographic_absorption * urban_absorption
    edu_speed *= 1.0 + 0.14 * industrialization_accel
    edu_speed *= 0.90 + 0.22 * digital_infra_development
    edu_speed *= trajectory_effect["education"]

    # ================================================================
    # SPEED FACTOR 3: HEALTH / life expectancy → headroom
    # Low LE → fast improvement, high LE → near biological limit
    # ================================================================
    # LE 50 → 0.20, LE 65 → 0.15, LE 75 → 0.08, LE 82 → 0.03
    le_headroom = max(0, (87.0 - life_exp_current) / 45.0)
    health_speed = le_headroom * 0.22

    # Conflict is a near-term drag, not a permanent 2050 assumption.
    if traj_class == "conflict_recovery":
        health_speed *= 0.65 + 0.45 * recovery_potential
    elif traj_class == "stagnant":
        health_speed *= 0.6
    healthcare_quality = np.clip(
        0.45 * (1.0 - infant_mortality / 80.0) +
        0.30 * (physicians / 4.0) +
        0.25 * (health_exp / 12.0),
        0.25,
        1.20,
    )
    health_speed *= healthcare_quality * (1.0 - 0.18 * climate_risk)
    health_speed *= 0.94 + 0.14 * digital_infra_development
    health_speed *= trajectory_effect["health"]

    # ================================================================
    # SPEED FACTOR 4: GOVERNANCE → institutional capacity
    # Strong institutions → faster development; weak → bottleneck
    # ================================================================
    # inst_eff 1.12 → 0.12, 1.00 → 0.08, 0.90 → 0.05, 0.75 → 0.02
    gov_speed = 0.03 + (inst_eff - 0.70) * 0.35
    gov_speed = np.clip(gov_speed, 0.02, 0.15)
    gov_speed += 0.035 * recovery_potential
    gov_speed = np.clip(gov_speed, 0.02, 0.18)
    rule_stability = np.clip(((rule_of_law + political_stability) / 2.0 + 2.5) / 5.0, 0.0, 1.0)
    gov_speed *= 0.65 + 0.35 * rule_stability

    # ================================================================
    # SPEED FACTOR 5: DEMOGRAPHICS → young = dividend, old = drag
    # ================================================================
    # demo_factor 1.2 → 0.08, 1.0 → 0.05, 0.8 → 0.02
    demo_speed = (
        0.012 +
        0.050 * demographic_dividend +
        0.028 * human_capital_absorption +
        0.018 * workforce_momentum
    )
    fertility_pressure = np.clip((fertility - 2.1) / 3.0, 0.0, 1.0)
    frontier_age_pressure = np.clip((median_age_2050 - 42.0) / 14.0, 0.0, 1.0)
    adapted_frontier_age_pressure = frontier_age_pressure * (1.0 - 0.42 * developed_demographic_adaptation)
    demo_speed += 0.025 * youth_dividend + 0.012 * age_pyramid_score
    demo_speed *= 1.0 - 0.34 * dependency_pressure - 0.16 * adapted_frontier_age_pressure
    demo_speed *= 1.0 + 0.12 * developed_demographic_adaptation
    if country_id == "NGA":
        demo_speed *= 1.28
    demo_speed = np.clip(demo_speed, 0.004, 0.13)

    # ================================================================
    # SPEED FACTOR 6: FUTURE READINESS -> 2050 productive capacity
    # ================================================================
    # Captures AI, automation, chips, batteries, green investment, EVs,
    # rail, startups, venture capital, and cloud adoption. This is more
    # important for 2050 than for explaining current HDI.
    if future_readiness_current is None:
        future_speed = 0.05
    else:
        future_readiness_s = np.clip(future_readiness_current, 0, 1)
        future_speed = 0.02 + (1.0 / (1.0 + np.exp(-8.0 * (future_readiness_s - 0.45)))) * 0.11
    future_speed = np.clip(future_speed, 0.02, 0.18)
    infrastructure_quality = np.clip(0.45 * urbanization + 0.35 * renewable_share + 0.20 * tech_factor, 0.20, 1.10)
    future_speed *= 0.75 + 0.35 * infrastructure_quality
    future_speed *= 1.0 + 0.35 * industrialization_accel
    future_speed *= 0.88 + 0.32 * digital_infra_development
    future_speed *= trajectory_effect["future"]

    # ================================================================
    # COMBINE SPEED FACTORS (weighted sum, not product)
    # Each factor contributes independently to convergence speed
    # ================================================================
    total_speed = (
        gdp_speed * 0.11 +      # GDP growth, with strong diminishing returns
        edu_speed * 0.29 +      # Education headroom and absorption
        health_speed * 0.15 +   # Health headroom and healthcare quality
        gov_speed * 0.18 +      # Governance/state capacity
        demo_speed * 0.16 +     # Age pyramid, workforce, and human capital
        future_speed * 0.09 +   # Future-oriented productive capacity
        capacity_score * 0.02   # Conflict/corruption/stability backbone
    )

    # Scale: total_speed ranges ~0.01 to ~0.12
    total_speed = total_speed * 0.16
    total_speed *= get_catchup_horizon_multiplier(hdi_current)
    total_speed *= 1.0 + 0.20 * industrialization_accel
    total_speed *= 0.92 + 0.34 * developing_catchup_readiness
    total_speed *= 0.88 + 0.24 * human_development_platform
    total_speed *= trajectory_effect["speed"]
    total_speed *= 0.84 + 0.28 * growth_prospect_score
    total_speed *= demographic_hdi_multiplier
    total_speed *= 0.94 + 0.16 * digital_infra_development
    total_speed *= saturation_factor
    total_speed = np.clip(total_speed, 0.001, 0.055)

    # Gap-dependent dampening: countries far from frontier converge slower
    # This reflects structural barriers (institutions, infrastructure, human capital)
    # that GDP growth alone doesn't overcome.
    # KEY: governance quality modulates dampening — strong institutions
    # reduce barriers, fragile states face full penalty.
    # inst_mod: 0.70 inst → 0.0, 0.95 → 0.556, 1.15 → 1.0
    inst_mod = np.clip((inst_eff - 0.70) / (1.15 - 0.70), 0.0, 1.0)
    inst_mod = max(float(inst_mod), 0.25 + 0.55 * recovery_potential)

    if gap > 0.35:
        base_damp = 0.60
    elif gap > 0.25:
        base_damp = 0.75
    elif gap > 0.15:
        base_damp = 0.85
    elif gap > 0.05:
        base_damp = 0.95
    else:
        base_damp = 1.0

    # Interpolate between base_damp and 1.0 based on institutional quality,
    # allowing long-run recovery to reduce today's fragility penalty.
    recovery_floor = min(0.92, base_damp + 0.22 * recovery_potential)
    total_speed *= max(recovery_floor, base_damp + (1.0 - base_damp) * inst_mod)
    shock_risk = (
        0.30 * (1.0 - state.get("conflict", 0.65)) +
        0.20 * (1.0 - state.get("inflation_stability", 0.65)) +
        0.20 * climate_risk +
        0.15 * np.clip((gini - 0.40) / 0.20, 0.0, 1.0) +
        0.15 * (1.0 - capacity_score)
    )
    shock_risk += 0.18 * resource_drag
    shock_risk += 0.05 * resource_profile["dependence"] * (1.0 - renewable_share)
    shock_risk += 0.16 * low_growth_drag
    shock_risk += 0.10 * dependency_pressure + 0.06 * adapted_aging_pressure
    shock_risk -= 0.06 * digital_infra_development * capacity_score
    shock_risk *= trajectory_effect["shock"]
    total_speed *= 1.0 - 0.28 * np.clip(shock_risk, 0.0, 1.0)

    # ================================================================
    # APPLY CONVERGENCE FORMULA
    # HDI_2050 = HDI_current + Gap × (1 - exp(-Speed × Years))
    # ================================================================
    hdi_gain = gap * (1.0 - np.exp(-total_speed * years))

    # Country-specific adjustments (additive to gain, small)
    country_adjust = COUNTRY_HDI_BOOST.get(country_id, 0.0)
    aging_adjust = COUNTRY_AGING_PENALTY.get(country_id, 0.0) * (1.0 - 0.55 * developed_demographic_adaptation)

    recovery_bonus = gap * 0.05 * recovery_potential
    climate_penalty = gap * 0.05 * climate_risk
    inequality_penalty = gap * 0.06 * np.clip((gini - 0.38) / 0.22, 0.0, 1.0)
    aging_penalty = gap * 0.04 * adapted_aging_penalty_structural
    social_mobility_bonus = gap * 0.025 * np.clip((0.40 - gini) / 0.15, 0.0, 1.0) * capacity_score
    oil_windfall_conversion_bonus = (
        gap *
        0.075 *
        OIL_WINDFALL_CONVERSION.get(country_id, 0.0) *
        np.clip(capacity_score, 0.0, 1.0) *
        (1.0 - 0.45 * resource_drag)
    )
    high_fertility_penalty = hdi_current * 0.010 * fertility_pressure * (1.0 - capacity_score)
    dependency_penalty = gap * 0.050 * dependency_pressure * (1.0 - 0.35 * capacity_score)
    demographic_dividend_bonus = gap * 0.060 * demographic_dividend * human_capital_absorption
    workforce_bonus = gap * 0.025 * workforce_depth * workforce_momentum * human_capital_absorption
    human_capital_scale_bonus = (
        gap *
        0.055 *
        (1.0 if country_id == "NGA" else 0.0) *
        human_capital_absorption *
        (0.55 + 0.45 * demographic_dividend) *
        (0.70 + 0.30 * digital_infra_development)
    )
    developed_demographic_fix_bonus = (
        gap *
        0.040 *
        developed_demographic_adaptation *
        np.clip((hdi_current - 0.88) / 0.10, 0.0, 1.0) *
        (0.55 + 0.45 * capacity_score)
    )
    frontier_demographic_retention_bonus = (
        hdi_current *
        0.006 *
        developed_demographic_adaptation *
        np.clip((hdi_current - 0.90) / 0.07, 0.0, 1.0)
    )
    aging_workforce_penalty = gap * 0.040 * adapted_aging_pressure * (1.0 - 0.35 * workforce_depth)
    digital_infra_bonus = gap * 0.040 * digital_infra_development * (
        0.45 + 0.30 * human_capital_absorption + 0.25 * capacity_score
    )
    platform_persistence_bonus = gap * 0.050 * np.clip((hdi_current - 0.72) / 0.18, 0.0, 1.0) * human_development_platform
    expected_shock_loss = hdi_current * 0.020 * np.clip(shock_risk, 0.0, 1.0)
    expected_resource_loss = hdi_current * 0.010 * resource_drag
    expected_regression_loss = hdi_current * EXPECTED_REGRESSION_RISK.get(country_id, 0.0)
    frontier_aging_loss = hdi_current * 0.008 * adapted_aging_pressure * np.clip((hdi_current - 0.86) / 0.12, 0.0, 1.0)
    recovery_bonus *= max(0.40, saturation_factor)
    social_mobility_bonus *= max(0.40, saturation_factor)
    oil_windfall_conversion_bonus *= max(0.40, saturation_factor)
    country_adjust *= 0.50 * max(0.15, saturation_factor)
    hdi_gain += (
        country_adjust + aging_adjust + recovery_bonus + social_mobility_bonus + oil_windfall_conversion_bonus +
        developed_demographic_fix_bonus + frontier_demographic_retention_bonus -
        climate_penalty - inequality_penalty - aging_penalty -
        high_fertility_penalty - dependency_penalty + demographic_dividend_bonus + workforce_bonus + human_capital_scale_bonus -
        aging_workforce_penalty + digital_infra_bonus + platform_persistence_bonus - expected_shock_loss -
        expected_resource_loss - expected_regression_loss - frontier_aging_loss
    )

    # Allow central declines when structural risks outweigh catch-up.
    # The floor keeps median deterioration plausible, not catastrophic.
    max_decline = hdi_current * (0.015 + 0.040 * np.clip(shock_risk, 0.0, 1.0))
    if hdi_gain > 0:
        catchup_headroom = np.clip((0.88 - hdi_current) / 0.46, 0.0, 1.0)
        low_base_catchup_constraint = 0.86 + 0.18 * human_development_platform + 0.08 * capacity_score
        mid_hdi_persistence = 0.92 + 0.12 * human_development_platform if hdi_current >= 0.74 else 1.0
        cap_multiplier = (
            0.92 +
            0.28 * industrialization_accel * catchup_headroom +
            0.30 * developing_catchup_readiness * catchup_headroom
        )
        cap_multiplier *= 0.82 + 0.32 * growth_prospect_score
        cap_multiplier *= 0.88 + 0.24 * demographic_dividend + 0.16 * human_capital_absorption - 0.16 * dependency_pressure
        cap_multiplier *= 0.92 + 0.18 * digital_infra_development
        cap_multiplier *= low_base_catchup_constraint
        cap_multiplier *= mid_hdi_persistence
        hdi_gain = min(hdi_gain, get_stage_gain_cap(hdi_current) * cap_multiplier * trajectory_effect["cap"])
    hdi_gain = np.clip(hdi_gain, -max_decline, gap)

    hdi_2050 = hdi_current + hdi_gain

    # Hard bounds
    hdi_2050 = float(np.clip(hdi_2050, 0.250, HDI_CEILING))

    return hdi_2050


def compute_uncertainty_range(
    country_id: str,
    hdi_baseline: float,
    hdi_current: float | None = None,
    gini_current: float | None = None,
    climate_risk_current: float | None = None,
    political_stability_current: float | None = None,
    fertility_current: float | None = None,
) -> dict:
    """Compute optimistic and pessimistic HDI estimates.

    Optimistic = baseline + upside potential (things go well)
    Pessimistic = baseline - downside risk (things go badly)
    """
    if hdi_baseline >= 0.97:
        p10 = hdi_baseline - 0.004
        p90 = min(hdi_baseline + 0.002, 0.985)
        return {
            "optimistic": p90,
            "baseline": hdi_baseline,
            "pessimistic": p10,
            "p10": p10,
            "p50": hdi_baseline,
            "p90": p90,
        }

    traj_class = get_trajectory_class(country_id)
    inst_eff = INSTITUTIONAL_EFFICIENCY.get(country_id, 0.90)
    from data.stability import get_state_capacity
    state = get_state_capacity(country_id)
    capacity_score = float(np.clip(np.mean([
        state.get("stability", 0.5),
        state.get("conflict", 0.65),
        state.get("corruption", 0.45),
        state.get("governance", 0.48),
        state.get("fragility", 0.5),
    ]), 0.0, 1.0))
    gini = 0.38 if gini_current is None else float(np.clip(gini_current, 0.20, 0.65))
    climate_risk = 0.30 if climate_risk_current is None else float(np.clip(climate_risk_current, 0.0, 1.0))
    political_stability = 0.0 if political_stability_current is None else float(np.clip(political_stability_current, -2.5, 2.5))
    fertility = 2.3 if fertility_current is None else float(np.clip(fertility_current, 1.0, 7.0))

    # Upside and downside as fraction of remaining gap to 1.0
    # Higher HDI = less room on both sides
    if hdi_baseline < 0.50:
        upside_mult = 0.35
        downside_mult = 0.10
    elif hdi_baseline < 0.65:
        upside_mult = 0.30
        downside_mult = 0.08
    elif hdi_baseline < 0.80:
        upside_mult = 0.25
        downside_mult = 0.06
    elif hdi_baseline < 0.90:
        upside_mult = 0.15
        downside_mult = 0.04
    else:
        upside_mult = 0.08
        downside_mult = 0.02

    # Weak governance = more downside risk
    if inst_eff < 0.85:
        upside_mult *= 1.2
        downside_mult *= 1.5

    # Conflict/recovery = wider range
    if traj_class in ("frontier_jumper", "conflict_recovery"):
        upside_mult *= 1.3
        downside_mult *= 1.5
    elif traj_class == "accelerator":
        upside_mult *= 1.15
        downside_mult *= 1.2

    structural_risk = (
        0.25 * (1.0 - capacity_score) +
        0.20 * climate_risk +
        0.20 * np.clip((gini - 0.38) / 0.22, 0.0, 1.0) +
        0.20 * np.clip((-political_stability) / 2.5, 0.0, 1.0) +
        0.15 * np.clip((fertility - 2.5) / 3.0, 0.0, 1.0)
    )
    upside_mult *= 1.0 + 0.18 * structural_risk
    downside_mult *= 1.0 + 0.75 * structural_risk

    gap = 1.0 - hdi_baseline
    optimistic = hdi_baseline + gap * upside_mult
    pessimistic = hdi_baseline - hdi_baseline * downside_mult
    if hdi_current is not None:
        shock_floor = hdi_current - (0.015 + 0.08 * structural_risk)
        pessimistic = min(pessimistic, shock_floor)

    p90 = min(optimistic, 0.98)
    p10 = max(pessimistic, 0.250)
    return {
        "optimistic": p90,
        "baseline": hdi_baseline,
        "pessimistic": p10,
        "p10": p10,
        "p50": hdi_baseline,
        "p90": p90,
    }


def explain_hdi_projection(
    country_id: str,
    hdi_current: float,
    hdi_2050: float,
    *,
    gni_ppp_current: float = 30000.0,
    life_exp_current: float = 72.0,
    expected_school_current: float = 13.0,
    mean_school_current: float = 8.0,
    future_readiness_current: float | None = None,
    digital_infra_development_current: float | None = None,
    gini_current: float | None = None,
    infant_mortality_current: float | None = None,
    climate_risk_current: float | None = None,
    fertility_current: float | None = None,
    urbanization_current: float | None = None,
    political_stability_current: float | None = None,
    rule_of_law_current: float | None = None,
    physicians_current: float | None = None,
    health_exp_current: float | None = None,
    renewable_share_current: float | None = None,
    year: int = 2050,
) -> dict:
    """Approximate additive driver contributions for audit/explainability.

    These are not SHAP values. They allocate the projected HDI gain across the
    same structural channels used by the convergence model, then normalize to
    the final gain so the columns are easy to audit country by country.
    """
    from data.demographics import compute_demographic_profile, get_aging_penalty, get_youth_dividend
    from data.stability import get_governance_multiplier, get_state_capacity
    from data.technology import compute_technology_factor

    gain = float(hdi_2050 - hdi_current)
    if abs(gain) < 1e-9:
        return {
            "Contrib_Income": 0.0,
            "Contrib_Education": 0.0,
            "Contrib_Health": 0.0,
            "Contrib_Governance": 0.0,
            "Contrib_Demographics": 0.0,
            "Contrib_Technology": 0.0,
            "Contrib_Recovery": 0.0,
            "Contrib_Inequality": 0.0,
            "Contrib_Climate": 0.0,
            "Contrib_ShockRisk": 0.0,
        }

    inst_eff = get_governance_multiplier(country_id, year)
    state = get_state_capacity(country_id)
    capacity_score = float(np.clip(np.mean([
        state.get("stability", 0.5),
        state.get("conflict", 0.65),
        state.get("corruption", 0.45),
        state.get("governance", 0.48),
        state.get("fragility", 0.5),
    ]), 0.0, 1.0))
    gini = 0.38 if gini_current is None else float(np.clip(gini_current, 0.20, 0.65))
    climate_risk = 0.30 if climate_risk_current is None else float(np.clip(climate_risk_current, 0.0, 1.0))
    infant_mortality = 25.0 if infant_mortality_current is None else float(np.clip(infant_mortality_current, 1.0, 120.0))
    fertility = 2.3 if fertility_current is None else float(np.clip(fertility_current, 1.0, 7.0))
    political_stability = 0.0 if political_stability_current is None else float(np.clip(political_stability_current, -2.5, 2.5))
    rule_of_law = 0.0 if rule_of_law_current is None else float(np.clip(rule_of_law_current, -2.5, 2.5))
    physicians = 1.5 if physicians_current is None else float(np.clip(physicians_current, 0.05, 8.0))
    health_exp = 6.0 if health_exp_current is None else float(np.clip(health_exp_current, 1.0, 18.0))
    urbanization = 0.56 if urbanization_current is None else float(np.clip(urbanization_current, 0.10, 0.98))
    renewable_share = 0.20 if renewable_share_current is None else float(np.clip(renewable_share_current, 0.0, 1.0))
    future_readiness = 0.30 if future_readiness_current is None else float(np.clip(future_readiness_current, 0.0, 1.0))
    digital_infra_development = (
        0.35 if digital_infra_development_current is None
        else float(np.clip(digital_infra_development_current, 0.0, 1.0))
    )

    income_score = 1.0 - np.clip(
        (np.log(max(gni_ppp_current, 100.0)) - np.log(100.0)) /
        (np.log(75000.0) - np.log(100.0)),
        0.0,
        1.0,
    )
    edu_index = np.sqrt(np.clip(expected_school_current / 18.0, 0, 1) * np.clip(mean_school_current / 16.0, 0, 1))
    demographic_profile = compute_demographic_profile(country_id, year, fertility, edu_index)
    health_score = np.clip((87.0 - life_exp_current) / 45.0, 0.0, 1.0)
    healthcare_score = np.clip(
        0.45 * (1.0 - infant_mortality / 80.0) +
        0.30 * (physicians / 4.0) +
        0.25 * (health_exp / 12.0),
        0.0,
        1.2,
    )
    gov_score = np.clip((inst_eff - 0.70) / 0.45, 0.0, 1.0) * 0.6 + capacity_score * 0.4
    rule_stability = np.clip(((rule_of_law + political_stability) / 2.0 + 2.5) / 5.0, 0.0, 1.0)
    tech_score = np.clip(
        0.48 * future_readiness +
        0.27 * digital_infra_development +
        0.17 * compute_technology_factor(country_id, year) +
        0.08 * renewable_share,
        0.0,
        1.2,
    )
    demo_score = np.clip(
        0.58 * demographic_profile["demographic_dividend"] +
        0.27 * demographic_profile["human_capital_absorption"] -
        0.30 * demographic_profile["dependency_pressure"] -
        0.18 * demographic_profile["aging_pressure"] +
        0.15 * get_youth_dividend(country_id, year),
        -1.0,
        1.0,
    )
    recovery_score = get_recovery_potential(country_id, hdi_current, year)
    resource_profile = get_resource_volatility_profile(country_id)
    resource_drag = compute_resource_drag(country_id, capacity_score)
    shock_risk = (
        0.35 * (1.0 - state.get("conflict", 0.65)) +
        0.25 * climate_risk +
        0.20 * np.clip((gini - 0.40) / 0.20, 0.0, 1.0) +
        0.20 * (1.0 - capacity_score)
    )
    shock_risk += 0.18 * resource_drag
    shock_risk += 0.05 * resource_profile["dependence"] * (1.0 - renewable_share)

    raw = {
        "Contrib_Income": 0.18 * income_score * (0.55 + 0.45 * gov_score),
        "Contrib_Education": 0.22 * (1.0 - edu_index) * (0.65 + 0.35 * rule_stability),
        "Contrib_Health": 0.16 * health_score * healthcare_score,
        "Contrib_Governance": 0.15 * gov_score,
        "Contrib_Demographics": 0.22 * demo_score,
        "Contrib_Technology": 0.13 * tech_score * (0.60 + 0.40 * urbanization),
        "Contrib_Recovery": 0.06 * recovery_score,
        "Contrib_Inequality": -0.08 * np.clip((gini - 0.36) / 0.24, 0.0, 1.0),
        "Contrib_Climate": -0.06 * climate_risk,
        "Contrib_ShockRisk": -0.08 * shock_risk - 0.03 * get_aging_penalty(country_id, year),
    }
    if gain < 0:
        # Preserve intuitive signs for decline cases: positive drivers can still
        # exist, but negative risk/aging/climate channels dominate the outcome.
        gross_total = sum(abs(v) for v in raw.values())
        scale = abs(gain) / gross_total if gross_total > 1e-9 else 0.0
    else:
        positive_total = sum(v for v in raw.values() if v > 0)
        negative_total = sum(v for v in raw.values() if v < 0)
        net_total = positive_total + negative_total
        scale = gain / net_total if abs(net_total) > 1e-9 else 0.0
    return {k: float(v * scale) for k, v in raw.items()}
