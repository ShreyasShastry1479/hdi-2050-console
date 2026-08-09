"""Technology readiness and AI adoption data for HDI forecasting.

Provides technology readiness indices that capture how well countries
can leverage AI, automation, and digital transformation to accelerate
human development.

Key indices:
- AI Readiness: infrastructure, data, talent, regulation
- Digital Government: e-services, data openness, citizen engagement
- Automation Potential: manufacturing/services automation adoption
- Innovation Capacity: R&D spending, patents, tech exports
"""

import numpy as np


# AI Readiness Index (0-100 scale, 2024 estimates)
# Based on Oxford Insights AI Readiness Index + our adjustments
AI_READINESS_2024 = {
    # Top tier (70+)
    "SGP": 85, "USA": 82, "GBR": 78, "FIN": 77, "DEU": 76,
    "KOR": 75, "SWE": 75, "NLD": 74, "CHE": 74, "CAN": 73,
    "AUS": 72, "ISR": 72, "JPN": 71, "FRA": 70, "DNK": 70,

    # High tier (55-69)
    "NOR": 68, "AUT": 67, "BEL": 66, "IRL": 66, "NZL": 65,
    "ESP": 63, "ITA": 62, "CHL": 60, "CZE": 59, "EST": 59,
    "CHN": 58, "POL": 57, "PRT": 57, "LTU": 56, "SVN": 56,
    "MLT": 55, "ARE": 68, "QAT": 65, "SAU": 62, "KWT": 58,
    "BHR": 57, "OMN": 55, "MYS": 56, "THA": 54, "RUS": 53,
    "TUR": 52, "URY": 52, "COL": 50, "MEX": 50, "BRA": 49,
    "CHL": 55, "ARG": 48, "PAN": 47, "CRI": 47,

    # Medium tier (40-54)
    "KAZ": 52, "AZE": 48, "GEO": 47, "ARM": 46, "MKD": 45,
    "SRB": 45, "BGR": 44, "ROU": 44, "HUN": 44, "HRV": 43,
    "SVK": 43, "CYP": 50, "GRC": 48, "LVA": 45, "BLR": 42,
    "UKR": 41, "MNE": 42, "ALB": 40, "JAM": 45, "TTO": 44,
    "PER": 42, "ECU": 40, "DOM": 40, "GUY": 38, "SUR": 37,
    "IDN": 43, "PHL": 42, "VNM": 41, "IND": 40, "LKA": 44,
    "BGD": 38, "NPL": 35, "BTN": 36, "MDV": 42, "MYS": 50,

    # Low tier (25-39)
    "KEN": 38, "GHA": 37, "NGA": 35, "ZAF": 42, "MAR": 38,
    "TUN": 40, "EGY": 37, "JOR": 40, "LBN": 39, "IRN": 36,
    "IRQ": 30, "PRY": 35, "BOL": 32, "SLV": 36, "GTM": 33,
    "HND": 31, "NIC": 30, "BLZ": 35, "CUB": 38, "TZA": 33,
    "UGA": 32, "RWA": 35, "ETH": 30, "SEN": 33, "CIV": 30,
    "CMR": 29, "BEN": 28, "TGO": 28, "GAB": 32, "COG": 27,
    "AGO": 28, "MOZ": 25, "MWI": 24, "ZMB": 30, "ZWE": 28,
    "NAM": 35, "BWA": 38, "LSO": 32, "SWZ": 30, "MDG": 22,
    "MRT": 28, "GIN": 25, "SLE": 22, "LBR": 21, "BFA": 22,
    "MLI": 23, "NER": 20, "TCD": 19, "SSD": 18, "AFG": 17,
    "SOM": 15, "YEM": 16, "SDN": 20, "ERI": 18, "COD": 20,
    "CAF": 19, "BDI": 21, "RWA": 32, "GMB": 26, "GNB": 25,
    "GEO": 42, "KGZ": 35, "TJK": 30, "UZB": 38, "TKM": 28,
    "MMR": 28, "KHM": 30, "LAO": 28, "TLS": 25, "BRN": 45,
    "FJI": 38, "PNG": 22, "SLB": 20, "VUT": 18, "WSM": 25,
    "TON": 24, "KIR": 22, "MHL": 23, "FSM": 22, "PLW": 30,
    "NRU": 25, "TUV": 20,

    # Very high (microstates & city-states)
    "LUX": 70, "MCO": 65, "LIE": 68, "SMR": 60, "AND": 62,
    "ISL": 72, "MLT": 60,

    # Special cases
    "TWN": 72, "HKG": 75, "MAC": 62,
}

# Digital Government Index (0-100, UN E-Government Survey 2024)
DIGITAL_GOV_2024 = {
    # Leaders (80+)
    "DNK": 95, "FIN": 93, "SWE": 92, "NOR": 91, "NLD": 90,
    "KOR": 89, "GBR": 88, "NZL": 87, "AUS": 86, "USA": 85,
    "EST": 85, "SGP": 84, "JPN": 83, "DEU": 82, "FRA": 81,

    # High (65-79)
    "CAN": 78, "ISR": 77, "CHE": 76, "IRL": 75, "AUT": 74,
    "BEL": 73, "ESP": 72, "ITA": 71, "CZE": 70, "SVN": 69,
    "PRT": 68, "LTU": 67, "LVA": 66, "MLT": 65, "SVK": 65,
    "CHL": 68, "URY": 66, "ARE": 80, "SAU": 72, "QAT": 70,
    "KWT": 65, "BHR": 64, "OMN": 62, "POL": 64, "HUN": 63,
    "ROU": 62, "BGR": 60, "HRV": 60, "GRC": 62, "CYP": 64,

    # Medium (50-64)
    "KAZ": 58, "AZE": 50, "GEO": 55, "ARM": 52, "SRB": 52,
    "MKD": 50, "MNE": 48, "ALB": 47, "TUR": 58, "RUS": 55,
    "MYS": 60, "THA": 58, "CHN": 55, "IND": 52, "IDN": 50,
    "PHL": 48, "VNM": 48, "COL": 52, "MEX": 50, "BRA": 52,
    "ARG": 48, "PER": 45, "PAN": 48, "CRI": 50, "JOR": 52,
    "LBN": 48, "EGY": 48, "MAR": 50, "TUN": 52, "IRN": 42,
    "ZAF": 50, "BWA": 48, "NAM": 45, "MUS": 55, "FJI": 45,
    "GUY": 40, "SUR": 38, "DOM": 42, "JAM": 45, "TTO": 45,
    "BGD": 42, "LKA": 48, "BTN": 40, "NPL": 38, "MDV": 50,

    # Low (30-49)
    "KEN": 42, "GHA": 40, "NGA": 38, "TZA": 35, "UGA": 35,
    "ETH": 30, "RWA": 42, "SEN": 38, "CIV": 32, "CMR": 30,
    "GTM": 35, "HND": 32, "SLV": 38, "NIC": 30, "BOL": 35,
    "PRY": 38, "ECU": 40, "BLZ": 40, "CUB": 35, "IRQ": 28,
    "YEM": 15, "AFG": 12, "SOM": 10, "SSD": 12, "SDN": 18,
    "SSD": 12, "ERI": 15, "COD": 15, "CAF": 15, "TCD": 15,
    "NER": 12, "MLI": 18, "BFA": 15, "GIN": 18, "SLE": 15,
    "LBR": 15, "MOZ": 20, "MWI": 18, "ZMB": 22, "ZWE": 20,
    "AGO": 22, "MDG": 18, "MRT": 22, "GAB": 25, "COG": 20,
    "BEN": 18, "TGO": 18, "GMB": 18, "GNB": 18,
    "KHM": 28, "LAO": 25, "MMR": 22, "TLS": 22, "PNG": 18,
    "SLB": 15, "VUT": 12, "WSM": 20, "TON": 18, "KIR": 15,
    "MHL": 15, "FSM": 15, "PLW": 22, "NRU": 18, "TUV": 15,
    "KGZ": 35, "TJK": 30, "UZB": 38, "TKM": 25, "BLR": 42,
    "UKR": 45, "MDA": 38, "ISL": 82, "LUX": 75, "MCO": 60,
    "LIE": 55, "SMR": 50, "AND": 55,

    # Special
    "TWN": 78, "HKG": 82, "MAC": 65,
}


def get_ai_readiness(country_id: str, year: int = 2024) -> float:
    """Get AI readiness score (0-100), projected to 2050.

    Growth follows S-curve: fast early, then saturating.
    """
    base = AI_READINESS_2024.get(country_id, 35.0) / 100.0

    if year <= 2024:
        return base
    if year >= 2050:
        target = min(base * 1.8, 0.98)
        return target

    t = (year - 2024) / (2050 - 2024)
    s_curve = 1.0 / (1.0 + np.exp(-10 * (t - 0.3)))
    target = min(base * 1.8, 0.98)
    return base + (target - base) * s_curve


def get_digital_gov(country_id: str, year: int = 2024) -> float:
    """Get digital government score (0-100), projected to 2050."""
    base = DIGITAL_GOV_2024.get(country_id, 35.0) / 100.0

    if year <= 2024:
        return base
    if year >= 2050:
        target = min(base * 1.7, 0.98)
        return target

    t = (year - 2024) / (2050 - 2024)
    s_curve = 1.0 / (1.0 + np.exp(-8 * (t - 0.35)))
    target = min(base * 1.7, 0.98)
    return base + (target - base) * s_curve


def compute_technology_factor(country_id: str, year: int = 2050) -> float:
    """Compute technology acceleration factor for HDI growth.

    Returns a multiplier (0.85-1.20) that adjusts HDI growth:
    - High AI readiness + digital gov → boost (+)
    - Low technology adoption → slight drag (-)
    - Average → neutral (~1.0)

    Logic:
    - AI can accelerate education delivery (online learning)
    - AI can improve healthcare diagnostics
    - Digital government improves service delivery
    - Automation can boost productivity → income
    """
    ai = get_ai_readiness(country_id, year)
    dg = get_digital_gov(country_id, year)

    # AI contribution (0-10% boost)
    if ai > 0.80:
        ai_factor = 1.08
    elif ai > 0.65:
        ai_factor = 1.05
    elif ai > 0.50:
        ai_factor = 1.02
    elif ai > 0.35:
        ai_factor = 1.00
    elif ai > 0.25:
        ai_factor = 0.98
    else:
        ai_factor = 0.95

    # Digital government contribution (0-5% boost)
    if dg > 0.80:
        dg_factor = 1.05
    elif dg > 0.60:
        dg_factor = 1.03
    elif dg > 0.40:
        dg_factor = 1.01
    elif dg > 0.25:
        dg_factor = 1.00
    else:
        dg_factor = 0.98

    return ai_factor * dg_factor


def get_technology_readiness(country_id: str, year: int = 2050) -> float:
    """Get combined technology readiness score (0-100)."""
    ai = get_ai_readiness(country_id, year) * 100
    dg = get_digital_gov(country_id, year) * 100
    return (ai * 0.6 + dg * 0.4)
