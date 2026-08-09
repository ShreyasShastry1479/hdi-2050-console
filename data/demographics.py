"""Demographic data and factors for HDI forecasting.

Provides age structure, dependency ratios, and demographic dividend
calculations for 193 UN member states. Used to adjust HDI growth
projections based on population dynamics.

Key concepts:
- Youth bulge: high % of population 15-24 → potential dividend if educated
- Aging penalty: median age >40 → slower growth due to dependency burden
- Working-age advantage: high % 15-64 → demographic dividend window
"""

import numpy as np

from data.ethnicity_model import MIGRATION_POLICY_OPENNESS, national_tfr


# Median age in 2024 (approximate, from UN World Population Prospects)
MEDIAN_AGE_2024 = {
    # Africa - young populations
    "UGA": 15.7, "MDG": 19.8, "MOZ": 17.2, "MAL": 16.5, "BFA": 17.3,
    "RWA": 19.7, "BDI": 17.0, "NER": 14.8, "MLI": 16.0,
    "TCD": 15.8, "SOM": 16.7, "SSD": 17.3, "AFG": 17.6, "ETH": 17.8,
    "GIN": 18.0, "SEN": 18.3, "CIV": 18.7, "CMR": 18.7, "TZA": 17.2,
    "KEN": 20.0, "GHA": 20.9, "NGA": 18.1, "AGO": 16.7, "COD": 16.8,
    "ZMB": 17.6, "ZWE": 18.7, "MWI": 17.2, "NAM": 21.8, "BWA": 24.5,
    "LSO": 22.6, "SWZ": 20.3, "ERI": 19.7, "DJI": 24.9, "LBY": 29.0,
    "TUN": 33.0, "DZA": 29.2, "MAR": 30.2, "EGY": 24.1, "SDN": 19.7,
    "SSD": 17.3, "LBR": 18.3, "SLE": 19.1, "GNB": 18.7, "GNQ": 22.3,
    "GAB": 22.5, "COG": 19.0, "CAF": 17.6, "BEN": 18.4, "TGO": 19.4,
    "GMB": 17.8, "MRT": 20.1, "CPV": 26.5, "SYC": 38.4, "MUS": 35.5,
    "COM": 20.4, "STP": 19.0,

    # Asia - mixed
    "IND": 28.2, "PAK": 20.4, "BGD": 27.6, "LKA": 34.0, "NPL": 24.6,
    "BTN": 28.1, "MDV": 30.3, "AFG": 17.6, "MMR": 29.0, "THA": 40.1,
    "VNM": 31.9, "KHM": 26.4, "LAO": 24.4, "MYS": 30.3, "SGP": 42.2,
    "IDN": 29.7, "PHL": 25.7, "BRN": 31.2, "TLS": 20.8,

    # East Asia
    "CHN": 39.0, "JPN": 48.6, "KOR": 43.7, "TWN": 42.5, "MNG": 28.3,
    "PRK": 35.1, "HKG": 45.6, "MACO": 39.2,

    # Central Asia
    "KAZ": 30.8, "UZB": 27.8, "TKM": 27.5, "KGZ": 27.1, "TJK": 22.4,
    "AZE": 30.3, "GEO": 38.7, "ARM": 33.5,

    # Middle East
    "IRQ": 20.0, "IRN": 32.0, "TUR": 32.2, "SYR": 23.0, "JOR": 24.5,
    "LBN": 30.5, "ISR": 30.4, "PSE": 18.5, "YEM": 17.8, "SAU": 31.8,
    "ARE": 33.5, "QAT": 33.7, "KWT": 36.8, "BHR": 32.1, "OMN": 30.6,
    "KSA": 31.8,

    # Europe
    "ISL": 37.0, "NOR": 39.8, "SWE": 39.6, "FIN": 43.1, "DNK": 42.0,
    "IRL": 38.2, "GBR": 40.5, "NLD": 42.8, "BEL": 41.6, "LUX": 39.5,
    "FRA": 41.7, "DEU": 45.7, "AUT": 43.2, "CHE": 42.7, "ITA": 48.0,
    "ESP": 45.5, "PRT": 46.2, "GRC": 45.6, "CYP": 38.2, "MLT": 41.5,
    "SVN": 44.9, "HRV": 44.3, "SRB": 42.5, "BIH": 42.3, "MNE": 39.5,
    "MKD": 39.0, "ALB": 38.0, "XKX": 29.5, "BGR": 45.4, "ROU": 43.2,
    "HUN": 43.6, "SVK": 41.2, "CZE": 43.3, "POL": 42.0, "LTU": 44.6,
    "LVA": 44.4, "EST": 42.4, "BLR": 40.3, "UKR": 41.2, "MDA": 38.0,
    "RUS": 39.6, "AUT": 43.2,

    # North America
    "USA": 38.5, "CAN": 41.8, "MEX": 29.3,

    # Central America & Caribbean
    "CRI": 34.0, "PAN": 30.2, "GTM": 22.9, "HND": 24.5, "SLV": 28.1,
    "NIC": 26.5, "BLZ": 25.5, "CUB": 42.2, "JAM": 31.5, "HTI": 24.1,
    "DOM": 27.6, "TTO": 37.8, "BHS": 32.5, "BRB": 39.5, "ATG": 35.0,
    "DMA": 34.5, "GRD": 33.0, "KNA": 35.5, "LCA": 34.0, "VCT": 33.5,

    # South America
    "BRA": 34.3, "ARG": 31.7, "CHL": 35.5, "URY": 35.8, "PRY": 26.5,
    "BOL": 25.2, "PER": 31.0, "ECU": 28.8, "COL": 31.5, "VEN": 28.5,
    "GUY": 27.5, "SUR": 29.5,

    # Oceania
    "AUS": 37.9, "NZL": 37.7, "FJI": 28.5, "PNG": 22.4, "SLB": 20.5,
    "VUT": 21.5, "WSM": 22.5, "TON": 22.0, "KIR": 23.5, "MHL": 22.5,
    "FSM": 22.0, "PLW": 30.5, "NRU": 26.5, "TUV": 24.0, "ASM": 25.5,
    "GUM": 30.5, "MNP": 32.5, "NCL": 33.5, "PYF": 34.5,

    # Post-Soviet
    "UKR": 41.2, "BLR": 40.3, "MDA": 38.0, "RUS": 39.6,
}

# Working-age population percentage (15-64) in 2024
WORKING_AGE_PCT_2024 = {
    # Africa - low working age share (high youth dependency)
    "UGA": 54.0, "MDG": 57.0, "MOZ": 55.0, "MAL": 53.0, "BFA": 54.5,
    "RWA": 57.0, "BDI": 54.0, "NER": 51.0, "MLI": 53.5,
    "TCD": 52.0, "SOM": 53.5, "SSD": 54.0, "AFG": 55.5, "ETH": 56.5,
    "GIN": 56.0, "SEN": 57.5, "CIV": 57.0, "CMR": 57.5, "TZA": 56.0,
    "KEN": 60.0, "GHA": 61.5, "NGA": 54.5, "AGO": 55.0, "COD": 54.0,
    "ZMB": 55.5, "ZWE": 58.0, "MWI": 54.5, "NAM": 62.0, "BWA": 65.5,
    "LSO": 63.0, "SWZ": 61.0, "ERI": 56.5, "DJI": 63.5, "LBY": 66.0,
    "TUN": 67.0, "DZA": 64.5, "MAR": 64.0, "EGY": 62.0, "SDN": 55.5,
    "LBR": 56.0, "SLE": 56.5, "GNB": 56.0, "GNQ": 60.0,
    "GAB": 61.0, "COG": 57.0, "CAF": 54.0, "BEN": 56.5, "TGO": 57.5,
    "GMB": 55.5, "MRT": 58.0, "CPV": 66.5, "SYC": 68.0, "MUS": 67.5,
    "COM": 58.5, "STP": 57.0,

    # Asia
    "IND": 67.5, "PAK": 57.5, "BGD": 67.0, "LKA": 65.5, "NPL": 65.0,
    "BTN": 68.0, "MDV": 66.5, "MMR": 67.0, "THA": 64.5, "VNM": 69.5,
    "KHM": 66.0, "LAO": 65.5, "MYS": 69.5, "SGP": 73.0, "IDN": 68.5,
    "PHL": 65.0, "BRN": 70.5, "TLS": 62.5,

    # East Asia
    "CHN": 69.0, "JPN": 59.0, "KOR": 69.5, "TWN": 70.0, "MNG": 67.5,
    "PRK": 66.0,

    # Central Asia
    "KAZ": 67.5, "UZB": 65.5, "TKM": 65.0, "KGZ": 64.5, "TJK": 62.0,
    "AZE": 66.5, "GEO": 63.5, "ARM": 65.5,

    # Middle East
    "IRQ": 57.5, "IRN": 65.0, "TUR": 67.5, "SYR": 55.5, "JOR": 62.5,
    "LBN": 63.5, "ISR": 66.5, "PSE": 56.0, "YEM": 55.0, "SAU": 70.5,
    "ARE": 83.0, "QAT": 85.0, "KWT": 74.0, "BHR": 73.5, "OMN": 72.0,

    # Europe
    "ISL": 67.0, "NOR": 66.5, "SWE": 65.0, "FIN": 63.5, "DNK": 65.0,
    "IRL": 66.5, "GBR": 64.0, "NLD": 65.5, "BEL": 64.5, "LUX": 67.0,
    "FRA": 63.0, "DEU": 64.5, "AUT": 65.5, "CHE": 66.0, "ITA": 60.0,
    "ESP": 62.5, "PRT": 62.0, "GRC": 61.5, "CYP": 67.5, "MLT": 66.5,
    "SVN": 64.5, "HRV": 63.5, "SRB": 62.5, "BIH": 62.0, "MNE": 63.5,
    "MKD": 63.0, "ALB": 63.5, "BGR": 61.5, "ROU": 63.0, "HUN": 63.5,
    "SVK": 64.0, "CZE": 64.5, "POL": 64.0, "LTU": 63.0, "LVA": 63.5,
    "EST": 63.5, "BLR": 63.5, "UKR": 62.5, "MDA": 63.0, "RUS": 65.0,

    # Americas
    "USA": 65.0, "CAN": 65.5, "MEX": 67.0, "CRI": 68.5, "PAN": 68.0,
    "GTM": 64.0, "HND": 63.5, "SLV": 65.5, "NIC": 64.5, "BLZ": 65.0,
    "CUB": 62.0, "JAM": 67.0, "HTI": 63.5, "DOM": 67.5, "TTO": 67.0,
    "BRA": 69.5, "ARG": 65.5, "CHL": 69.0, "URY": 66.5, "PRY": 65.5,
    "BOL": 64.0, "PER": 67.5, "ECU": 66.5, "COL": 68.0, "VEN": 64.0,
    "GUY": 66.0, "SUR": 66.5,

    # Oceania
    "AUS": 65.5, "NZL": 65.0, "FJI": 66.5, "PNG": 62.5, "SLB": 61.0,
    "VUT": 61.5, "WSM": 63.5, "TON": 63.0, "KIR": 63.5, "MHL": 63.0,
    "FSM": 62.0, "PLW": 68.0, "NRU": 65.0, "TUV": 64.0,
}

WORKING_AGE_PCT_2050_OVERRIDES = {
    # Nigeria's 2040s/2050 transition should begin converting today's youth
    # bulge into a larger labor-force cohort. The generic aging formula treats
    # all median-age increases as reducing workers, which is wrong for very
    # young countries moving into their demographic-dividend window.
    "NGA": 57.5,
}

# Projected median age in 2050 (UN WPP medium variant projections)
MEDIAN_AGE_2050 = {
    # Africa - still young but aging
    "UGA": 19.5, "MDG": 24.0, "MOZ": 21.5, "MAL": 20.5, "BFA": 21.5,
    "RWA": 24.5, "BDI": 21.0, "NER": 18.5, "MLI": 20.0,
    "TCD": 19.5, "SOM": 20.5, "SSD": 21.0, "AFG": 22.0, "ETH": 22.5,
    "GIN": 22.5, "SEN": 23.0, "CIV": 23.5, "CMR": 23.5, "TZA": 22.0,
    "KEN": 25.5, "GHA": 26.5, "NGA": 22.5, "AGO": 21.5, "COD": 21.0,
    "ZMM": 22.5, "ZWE": 24.0, "MWI": 21.5, "NAM": 28.0, "BWA": 30.5,
    "LSO": 28.0, "SWZ": 26.0, "ERI": 24.0, "DJI": 30.0, "LBY": 34.5,
    "TUN": 39.0, "DZA": 35.0, "MAR": 36.0, "EGY": 29.5, "SDN": 24.0,
    "LBR": 23.0, "SLE": 23.5, "GNB": 23.0, "GNQ": 28.0,
    "GAB": 28.0, "COG": 24.0, "CAF": 21.5, "BEN": 23.0, "TGO": 24.0,
    "GMB": 22.0, "MRT": 25.5, "CPV": 34.0, "SYC": 43.0, "MUS": 41.5,
    "COM": 25.5, "STP": 24.0,

    # Asia - aging rapidly
    "IND": 35.0, "PAK": 27.0, "BGD": 36.5, "LKA": 40.5, "NPL": 32.0,
    "BTN": 35.5, "MDV": 38.0, "MMR": 35.5, "THA": 47.0, "VNM": 40.0,
    "KHM": 33.0, "LAO": 31.5, "MYS": 38.0, "SGP": 50.0, "IDN": 37.0,
    "PHL": 32.5, "BRN": 39.0, "TLS": 28.0,

    # East Asia - very old
    "CHN": 50.0, "JPN": 54.5, "KOR": 53.0, "TWN": 52.0, "MNG": 36.0,
    "PRK": 42.0,

    # Central Asia
    "KAZ": 37.5, "UZB": 34.0, "TKM": 34.0, "KGZ": 33.5, "TJK": 29.0,
    "AZE": 37.0, "GEO": 45.0, "ARM": 41.0,

    # Middle East
    "IRQ": 26.5, "IRN": 40.0, "TUR": 40.5, "SYR": 30.0, "JOR": 32.0,
    "LBN": 38.0, "ISR": 35.5, "PSE": 24.0, "YEM": 23.0, "SAU": 38.5,
    "ARE": 40.0, "QAT": 40.5, "KWT": 42.0, "BHR": 38.5, "OMN": 37.0,

    # Europe - very old
    "ISL": 42.5, "NOR": 45.0, "SWE": 44.5, "FIN": 48.0, "DNK": 46.5,
    "IRL": 43.5, "GBR": 45.5, "NLD": 47.5, "BEL": 46.0, "LUX": 44.0,
    "FRA": 46.0, "DEU": 51.0, "AUT": 48.5, "CHE": 47.5, "ITA": 53.5,
    "ESP": 51.0, "PRT": 51.5, "GRC": 51.0, "CYP": 44.0, "MLT": 47.0,
    "SVN": 50.0, "HRV": 49.5, "SRB": 48.5, "BIH": 48.0, "MNE": 45.5,
    "MKD": 45.0, "ALB": 44.5, "BGR": 51.0, "ROU": 49.0, "HUN": 49.0,
    "SVK": 47.0, "CZE": 48.5, "POL": 48.0, "LTU": 50.0, "LVA": 50.0,
    "EST": 48.5, "BLR": 46.5, "UKR": 47.0, "MDA": 44.5, "RUS": 46.0,

    # Americas
    "USA": 42.0, "CAN": 46.5, "MEX": 37.5, "CRI": 42.0, "PAN": 38.5,
    "GTM": 30.5, "HND": 31.0, "SLV": 35.0, "NIC": 33.5, "BLZ": 33.0,
    "CUB": 50.0, "JAM": 39.5, "HTI": 31.5, "DOM": 36.0, "TTO": 44.5,
    "BRA": 43.5, "ARG": 40.0, "CHL": 44.0, "URY": 44.5, "PRY": 34.5,
    "BOL": 32.5, "PER": 39.5, "ECU": 37.0, "COL": 40.0, "VEN": 36.5,
    "GUY": 35.5, "SUR": 37.5,

    # Oceania
    "AUS": 42.5, "NZL": 42.0, "FJI": 35.5, "PNG": 28.5, "SLB": 27.0,
    "VUT": 27.5, "WSM": 30.0, "TON": 29.5, "KIR": 30.0, "MHL": 29.5,
    "FSM": 28.5, "PLW": 37.0, "NRU": 33.5, "TUV": 31.0,
}


# ---------------------------------------------------------------------------
# Sub-replacement migration buffer
# ---------------------------------------------------------------------------
# Countries whose projected national fertility dips below replacement during
# the 2025-2050 window run a labour shortfall and are modelled as importing
# migrant workers to buffer their working-age population. The buffer follows
# the same pressure/ramp/openness pattern as the ethnicity engine's
# demographic-driven migration boost (see ``data/ethnicity_model.py``).
# ---------------------------------------------------------------------------

# Sub-replacement TFR threshold. A country whose projected national TFR dips
# below this level at any point in 2024-2050 is treated as running a labour
# shortfall and receives a migration-inflow buffer.
SUBR_REPLACEMENT_TFR_THRESHOLD = 1.8

# Floor of the fertility clamp (mirrors the forecast engine's 1.2 floor).
_FERTILITY_FLOOR = 1.2

# Maximum working-age share boost (as a fraction, e.g. 0.06 = 6pp) that a
# fully-active migration buffer can add by 2050. Migrant inflows are
# concentrated in the 15-64 bracket, so the buffer primarily lifts the
# working-age share (which mechanically lowers both youth and elderly
# dependency pressure).
MAX_MIGRATION_BUFFER_WORKING_AGE_BOOST = 0.06

# Realistic ceiling for the resulting working-age share (Gulf expatriate
# economies already sit near 0.83-0.85; the buffer should not push past this).
_MAX_WORKING_AGE_SHARE = 0.80


def sub_replacement_depth(country_id: str) -> float:
    """How far the projected national TFR falls below the threshold in 2024-2050.

    Uses the UN-WPP-anchored national TFR path from ``data.ethnicity_model``
    (``NATIONAL_TFR_2024`` → ``NATIONAL_TFR_2050`` interpolated per year),
    which covers all 193 UNDP countries and is far more reliable than the
    dataset's stale/missing fertility column.

    Returns a depth in [0, 1], where 1.0 means the projected TFR bottoms out
    at or below the 1.2 floor and 0.0 means it never dips below the
    sub-replacement threshold (no migration buffer).
    """
    min_tfr = min(
        float(np.clip(national_tfr(country_id, year), _FERTILITY_FLOOR, 6.0))
        for year in range(2024, 2051)
    )
    depth = (SUBR_REPLACEMENT_TFR_THRESHOLD - min_tfr) / (
        SUBR_REPLACEMENT_TFR_THRESHOLD - _FERTILITY_FLOOR
    )
    return float(np.clip(depth, 0.0, 1.0))


def migration_buffer_intensity(country_id: str, year: int = 2050) -> float:
    """Strength of the migration-inflow buffer in a given projection year.

    Active only for countries whose projected fertility falls below the
    sub-replacement threshold at any point in 2024-2050. The inflow ramps
    from 0 (2024) to full strength in 2050 as the labour shortage deepens,
    and is scaled by the country's migration-policy openness (reuses
    ``data.ethnicity_model.MIGRATION_POLICY_OPENNESS``).

    Returns
    -------
    float
        Buffer intensity in [0, 1].
    """
    depth = sub_replacement_depth(country_id)
    if depth <= 0.0:
        return 0.0
    progress = float(np.clip((year - 2024) / (2050 - 2024), 0.0, 1.0))
    openness = MIGRATION_POLICY_OPENNESS.get(country_id, 1.0)
    return float(np.clip(depth * openness * progress, 0.0, 1.0))


def get_median_age(country_id: str, year: int = 2024) -> float:
    """Get median age, interpolated between 2024 and 2050."""
    age_2024 = MEDIAN_AGE_2024.get(country_id, 28.0)
    age_2050 = MEDIAN_AGE_2050.get(country_id, 35.0)

    if year <= 2024:
        return age_2024
    if year >= 2050:
        return age_2050

    t = (year - 2024) / (2050 - 2024)
    return age_2024 + t * (age_2050 - age_2024)


def get_working_age_pct(country_id: str, year: int = 2024) -> float:
    """Get working-age population percentage, projected to 2050."""
    wa_2024 = WORKING_AGE_PCT_2024.get(country_id, 65.0)
    override_2050 = WORKING_AGE_PCT_2050_OVERRIDES.get(country_id)

    age_2024 = MEDIAN_AGE_2024.get(country_id, 28.0)
    age_2050 = MEDIAN_AGE_2050.get(country_id, 35.0)
    aging_rate = (age_2050 - age_2024) / 26.0

    if year <= 2024:
        return wa_2024
    if override_2050 is not None:
        t = min((year - 2024) / (2050 - 2024), 1.0)
        return wa_2024 + t * (override_2050 - wa_2024)
    if year >= 2050:
        aging_delta = aging_rate * 26.0
        wa_2050 = wa_2024 - aging_delta * 0.8
        return max(wa_2050, 50.0)

    t = (year - 2024) / (2050 - 2024)
    aging_delta = aging_rate * (year - 2024)
    wa = wa_2024 - aging_delta * 0.8
    return max(wa, 50.0)


def estimate_age_group_shares(country_id: str, year: int = 2050,
                              migration_buffer: float = 0.0) -> dict:
    """Estimate broad age-group shares from working-age share and median age.

    The project does not ship full age-pyramid data, so this derives a smooth
    approximation:
    - 15-64 comes from the projected working-age series.
    - 65+ rises continuously with median age.
    - 0-14 receives the remaining non-working-age share.

    ``migration_buffer`` (0-1) optionally adds a working-age lift from
    sub-replacement migration inflows, mechanically lowering the youth and
    elderly shares as well.
    """
    median_age = get_median_age(country_id, year)
    working_age_share = get_working_age_pct(country_id, year) / 100.0
    if migration_buffer > 0.0:
        working_age_share = min(
            working_age_share + MAX_MIGRATION_BUFFER_WORKING_AGE_BOOST * migration_buffer,
            _MAX_WORKING_AGE_SHARE,
        )
    non_working_share = np.clip(1.0 - working_age_share, 0.0, 0.55)

    elderly_tilt = 1.0 / (1.0 + np.exp(-(median_age - 37.0) / 6.5))
    elderly_share = np.clip(non_working_share * elderly_tilt, 0.02, non_working_share * 0.92)
    youth_share = np.clip(non_working_share - elderly_share, 0.02, 0.48)

    total = youth_share + working_age_share + elderly_share
    return {
        "youth_0_14_share": float(youth_share / total),
        "working_15_64_share": float(working_age_share / total),
        "elderly_65_plus_share": float(elderly_share / total),
    }


def compute_demographic_factor(country_id: str, year: int = 2050) -> float:
    """Compute demographic factor for HDI growth.

    Returns a multiplier (0.7 - 1.2) that adjusts HDI growth:
    - Youth bulge + improving education → dividend (+)
    - Rapid aging → penalty (-)
    - Stable demographics → neutral (~1.0)
    - Sub-replacement fertility → migration buffer lifts working-age share (+)

    Logic:
    - Countries with median age 20-30 in 2050 get dividend bonus
    - Countries with median age >45 get aging penalty
    - Countries with median age <18 may have too much youth dependency
    """
    migration_buffer = migration_buffer_intensity(country_id, year)
    shares = estimate_age_group_shares(country_id, year, migration_buffer)
    youth_share = shares["youth_0_14_share"]
    working_share = shares["working_15_64_share"]
    elderly_share = shares["elderly_65_plus_share"]

    working_bonus = 1.0 + 0.65 * (working_share - 0.62)
    youth_drag = 1.0 - 0.45 * max(youth_share - 0.24, 0.0)
    elderly_drag = 1.0 - 0.70 * max(elderly_share - 0.14, 0.0)
    return float(np.clip(working_bonus * youth_drag * elderly_drag, 0.72, 1.18))


def get_age_group_shares(country_id: str, year: int = 2050) -> dict:
    """Public wrapper for broad age-group shares."""
    return estimate_age_group_shares(country_id, year)


def get_dependency_ratio_from_age_shares(country_id: str, year: int = 2050) -> float:
    """Return dependents per working-age person using estimated age shares."""
    shares = estimate_age_group_shares(country_id, year)
    dependents = shares["youth_0_14_share"] + shares["elderly_65_plus_share"]
    workers = max(shares["working_15_64_share"], 1e-6)
    return float(dependents / workers)


def get_aging_penalty(country_id: str, year: int = 2050) -> float:
    """Return a penalty factor (0.0-0.3) for how much aging reduces growth.

    Used specifically for GNI and health growth adjustments.
    Japan/Italy/Korea get higher penalties.
    """
    median_age = get_median_age(country_id, year)

    if median_age < 35:
        return 0.0
    elif median_age < 40:
        return 0.05
    elif median_age < 45:
        return 0.10
    elif median_age < 50:
        return 0.15
    else:
        return 0.25


def get_youth_dividend(country_id: str, year: int = 2050) -> float:
    """Return a dividend factor (0.0-0.3) for youth-driven growth potential.

    Countries with young, educated populations get higher dividends.
    """
    median_age = get_median_age(country_id, year)
    wa_pct = get_working_age_pct(country_id, year)

    if median_age < 20:
        return 0.05  # Too young
    elif median_age < 25:
        dividend = 0.15
    elif median_age < 30:
        dividend = 0.20
    elif median_age < 35:
        dividend = 0.10
    else:
        dividend = 0.0

    if wa_pct > 65:
        dividend *= 1.2
    elif wa_pct < 55:
        dividend *= 0.7

    return dividend


def compute_demographic_profile(
    country_id: str,
    year: int = 2050,
    fertility: float = 2.3,
    education_index: float = 0.65,
    migration_buffer: float | None = None,
) -> dict:
    """Return auditable demographic channels for long-run HDI conversion.

    The age pyramid is not treated as automatically good or bad. A youthful
    population becomes a dividend when paired with education and a large
    working-age share; very high fertility or severe aging become drags.
    Sub-replacement countries receive a working-age migration buffer that
    ramps up to 2050 (see ``migration_buffer_intensity``).
    """
    if migration_buffer is None:
        migration_buffer = migration_buffer_intensity(country_id, year)
    median_age_2024 = get_median_age(country_id, 2024)
    median_age = get_median_age(country_id, year)
    age_shares_2024 = estimate_age_group_shares(country_id, 2024, 0.0)
    age_shares = estimate_age_group_shares(country_id, year, migration_buffer)
    youth_share_2024 = age_shares_2024["youth_0_14_share"]
    working_age_2024 = age_shares_2024["working_15_64_share"]
    elderly_share_2024 = age_shares_2024["elderly_65_plus_share"]
    youth_share = age_shares["youth_0_14_share"]
    working_age = age_shares["working_15_64_share"]
    elderly_share = age_shares["elderly_65_plus_share"]
    fertility = float(np.clip(fertility, 1.0, 7.0))
    education_index = float(np.clip(education_index, 0.0, 1.0))

    child_dependency_ratio = youth_share / max(working_age, 1e-6)
    old_age_dependency_ratio = elderly_share / max(working_age, 1e-6)
    total_dependency_ratio = (youth_share + elderly_share) / max(working_age, 1e-6)
    age_pyramid_score = np.clip(
        1.0 -
        1.35 * abs(working_age - 0.66) -
        0.90 * max(youth_share - 0.30, 0.0) -
        1.15 * max(elderly_share - 0.18, 0.0),
        0.0,
        1.0,
    )
    workforce_depth = np.clip((working_age - 0.50) / 0.22, 0.0, 1.0)
    workforce_change = working_age - working_age_2024
    workforce_momentum = np.clip((workforce_change + 0.08) / 0.16, 0.0, 1.0)
    aging_pressure = np.clip((elderly_share - 0.12) / 0.22, 0.0, 1.0)
    youth_dependency = np.clip((youth_share - 0.22) / 0.22, 0.0, 1.0)
    youth_dependency *= 0.70 + 0.30 * np.clip((fertility - 1.8) / 2.4, 0.0, 1.0)
    fertility_window = 1.0 - np.clip(abs(fertility - 2.1) / 2.4, 0.0, 1.0)

    human_capital_absorption = np.clip(
        0.48 * education_index +
        0.24 * workforce_depth +
        0.18 * fertility_window +
        0.10 * age_pyramid_score -
        0.20 * youth_dependency -
        0.14 * old_age_dependency_ratio,
        0.0,
        1.0,
    )
    demographic_dividend = np.clip(
        0.22 * age_pyramid_score +
        0.38 * workforce_depth +
        0.18 * workforce_momentum +
        0.20 * human_capital_absorption -
        0.26 * youth_dependency -
        0.34 * aging_pressure,
        0.0,
        1.0,
    )
    dependency_pressure = np.clip(
        0.42 * youth_dependency +
        0.36 * aging_pressure +
        0.30 * np.clip((total_dependency_ratio - 0.48) / 0.42, 0.0, 1.0),
        0.0,
        1.0,
    )
    demographic_hdi_multiplier = np.clip(
        0.82 +
        0.24 * demographic_dividend +
        0.16 * human_capital_absorption -
        0.18 * dependency_pressure,
        0.72,
        1.18,
    )

    return {
        "median_age_2024": float(median_age_2024),
        "median_age_2050": float(median_age),
        "median_age_shift": float(median_age - median_age_2024),
        "youth_0_14_pct_2024": float(youth_share_2024 * 100.0),
        "youth_0_14_pct_2050": float(youth_share * 100.0),
        "working_age_pct_2024": float(working_age_2024 * 100.0),
        "working_age_pct_2050": float(working_age * 100.0),
        "workforce_change_pp": float(workforce_change * 100.0),
        "elderly_65_plus_pct_2024": float(elderly_share_2024 * 100.0),
        "elderly_65_plus_pct_2050": float(elderly_share * 100.0),
        "elderly_change_pp": float((elderly_share - elderly_share_2024) * 100.0),
        "child_dependency_ratio": float(child_dependency_ratio),
        "old_age_dependency_ratio": float(old_age_dependency_ratio),
        "total_dependency_ratio": float(total_dependency_ratio),
        "age_pyramid_score": float(age_pyramid_score),
        "workforce_depth": float(workforce_depth),
        "workforce_momentum": float(workforce_momentum),
        "youth_dependency_pressure": float(youth_dependency),
        "aging_pressure": float(aging_pressure),
        "fertility_window": float(fertility_window),
        "human_capital_absorption": float(human_capital_absorption),
        "demographic_dividend": float(demographic_dividend),
        "dependency_pressure": float(dependency_pressure),
        "demographic_hdi_multiplier": float(demographic_hdi_multiplier),
        "sub_replacement_depth": float(sub_replacement_depth(country_id)),
        "migration_buffer": float(migration_buffer),
    }
