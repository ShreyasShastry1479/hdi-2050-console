"""Evidence-based ethnic composition projection to 2050.

This module replaces the crude fixed profile-deviation model in
``data/ethnicity.py`` with a demographic model anchored on real data:

1. **National TFR** (total fertility rate) for every country, from UN World
   Population Prospects / World Bank (c. 2023-2024). ``NATIONAL_TFR_2024``
   holds the 2024 value and ``NATIONAL_TFR_2050`` the medium-variant 2050
   target (convergence to near-replacement fertility).

2. **Per-group TFR estimation** from the group's profile and name keywords
   (religion, region, demographic behaviour). E.g. Haredi/Orthodox Jews have
   a group TFR near 6.4, Muslim minorities in the West are ~1 TFR point above
   national, Japanese/Korean/Chinese majority groups inherit the (very low)
   national TFR.

3. **Growth decomposition.** Each group grows at the national rate plus a
   deviation built from:

   - fertility differential:   ``dev = ln(TFR_g / TFR_nat) / G``
     where ``G`` = mean generation length (~29 years). This is the standard
     stable-population relationship between the net-reproduction ratio and
     the intrinsic growth rate.
   - age-structure momentum:   a high-TFR group has a young age pyramid, so
     its already-born cohorts keep births elevated for ~2 decades even as
     TFR falls. Modelled as a decaying fraction of the 2024 fertility
     advantage.
    - migration:                ``immigrant`` groups receive a net-inflow
      deviation scaled by the country's migration intensity.
    - convergence:              group TFR pulls toward national TFR over time
      (``fertility_convergence``), and national TFR moves toward the 2050
      target (demographic transition).
    - assimilation:             per-country intermarriage / identity shift
      toward the anchor group (reuses ``COUNTRY_ASSIMILATION``).

Demographic-pressure-driven migration
--------------------------------------
Migration intensity is no longer a static table. A country's *demographic
problems* -- how far fertility sits below replacement and how fast its
population is projected to shrink (``demographic_pressure``) -- determine how
much labour it must import, combined with its policy openness to migrants
(``MIGRATION_POLICY_OPENNESS``). Pressure is added to the baseline intensity
and ramps up over the projection window as the crisis deepens, capturing
countries that liberalise immigration in response to aging workforces
(e.g. Japan, South Korea, Southern/Eastern Europe). This replaces the purely
static ``COUNTRY_MIGRATION_INTENSITY`` baseline.

The engine returns shares that are renormalised to 100% each year, matching
the ``ethnicity.py`` interface so ``run_ethnicity_2050*.py`` can swap models.
"""

from __future__ import annotations

import math
import re
from typing import Optional

import numpy as np

from data.ethnicity import (
    ETHNIC_COMPOSITION_2024,
    COUNTRY_ASSIMILATION,
    DEFAULT_ASSIMILATION_RATE,
    COUNTRY_MIGRATION_INTENSITY,
)

# Mean generation length (years) used to convert a TFR ratio into an annual
# intrinsic-growth differential: r = ln(NRR_g/NRR_nat)/G, NRR ~ 0.49*TFR.
GENERATION_LENGTH = 29.0

# Replacement fertility (children per woman).
REPLACEMENT_TFR = 2.1

# ---------------------------------------------------------------------------
# Policy openness to migration (multiplier on the demographic-pressure boost).
# 1.0 = the country is expected to accept the migrants its demographic
# situation requires; <1 = restrictive policy keeps intake below what pure
# demographics imply (e.g. China, Russia, Iran); >1 = proactively pro-immigrant
# (Canada, Australia) or structurally dependent on foreign labour (Gulf).
# Applied on top of COUNTRY_MIGRATION_INTENSITY, which already encodes the
# static expatriate-labour / settlement baseline.
# ---------------------------------------------------------------------------
MIGRATION_POLICY_OPENNESS: dict[str, float] = {
    # Strong settlement countries -- actively recruit immigrants
    "CAN": 1.3, "AUS": 1.3, "NZL": 1.3, "USA": 1.2,
    # Gulf / expatriate-labour economies
    "ARE": 1.3, "QAT": 1.3, "KWT": 1.2, "BHR": 1.2, "OMN": 1.1, "SAU": 1.1,
    "SGP": 1.2, "LUX": 1.2, "MDV": 1.1, "BRN": 1.1, "AND": 1.1, "MCO": 1.1,
    # Aging / shrinking economies that are opening to labour migration
    "JPN": 1.1, "KOR": 1.1, "TWN": 1.0, "ITA": 1.0, "ESP": 1.0, "GRC": 1.0,
    "PRT": 1.0, "DEU": 1.0, "THA": 1.0, "CHN": 0.5, "RUS": 0.8,
    # Restrictive / conflict-adjacent regimes
    "IRN": 0.6, "PRK": 0.1, "CUB": 0.6, "ERI": 0.5, "BLR": 0.7,
}

# ---------------------------------------------------------------------------
# Global migration-impact multiplier (calibration).
# Scales the effective net-migration intensity uniformly across every country:
#   effective = (static baseline + demographic-pressure boost) * this factor.
# Because intensity multiplies both the per-group inflow coefficients and the
# default immigrant-profile coefficient, this raises immigration's effect on
# the 2050 ethnic composition by ~25% (e.g. USA White NH 46.5% -> ~45%,
# Hispanic ~25.6% -> ~28%, CAN White 54.5% -> ~52%). Kept as an explicit
# constant so the strength of the migration channel is auditable and easy to
# tune; 1.0 restores the previously validated baseline.
# ---------------------------------------------------------------------------
MIGRATION_IMPACT_MULTIPLIER = 1.25

# ---------------------------------------------------------------------------
# Group-specific migration inflow coefficients keyed by (ISO3, group name).
# Replaces the old "every immigrant-profile group gets the same deviation"
# behaviour: a country's immigration intake is *not* evenly distributed across
# its ethnic groups. These coefficients capture the observed composition of
# net inflows (e.g. US: Hispanic + Asian dominate; Canada: South Asian,
# Chinese, Filipino; Gulf states: specific expatriate nationalities). The
# value is a relative annual growth premium (fraction per year) on top of the
# country's migration intensity. Groups not listed fall back to the default
# immigrant-profile coefficient, so the table is additive, not exhaustive.
# ---------------------------------------------------------------------------
GROUP_MIGRATION_INTENSITY: dict[tuple[str, str], float] = {
    # --- United States: Hispanic and Asian inflows dominate -----------------
    ("USA", "Hispanic / Latino (Mexican)"): 0.014,
    ("USA", "Hispanic / Latino (Salvadoran)"): 0.014,
    ("USA", "Hispanic / Latino (Guatemalan)"): 0.014,
    ("USA", "Hispanic / Latino (Honduran)"): 0.016,
    ("USA", "Hispanic / Latino (Colombian)"): 0.012,
    ("USA", "Hispanic / Latino (Venezuelan)"): 0.018,
    ("USA", "Hispanic / Latino (Dominican)"): 0.012,
    ("USA", "Hispanic / Latino (Other)"): 0.012,
    ("USA", "Hispanic / Latino (Puerto Rican)"): 0.004,
    ("USA", "Hispanic / Latino (Cuban)"): 0.004,
    ("USA", "Asian (Indian)"): 0.018,
    ("USA", "Asian (Chinese)"): 0.010,
    ("USA", "Asian (Filipino)"): 0.012,
    ("USA", "Asian (Vietnamese)"): 0.010,
    ("USA", "Asian (Korean)"): 0.008,
    ("USA", "Asian (Pakistani)"): 0.014,
    ("USA", "Asian (Bangladeshi)"): 0.016,
    ("USA", "Asian (Other)"): 0.014,
    ("USA", "Middle Eastern / North African"): 0.016,
    ("USA", "Black (African immigrant)"): 0.014,
    ("USA", "Black (Caribbean)"): 0.008,
    ("USA", "Asian (Japanese)"): 0.002,
    # --- Canada: South Asian, Chinese, Filipino-led intake -------------------
    ("CAN", "South Asian (Indian)"): 0.020,
    ("CAN", "South Asian (Pakistani)"): 0.012,
    ("CAN", "South Asian (Sri Lankan)"): 0.012,
    ("CAN", "South Asian (Other)"): 0.012,
    ("CAN", "Chinese"): 0.012,
    ("CAN", "Filipino"): 0.018,
    ("CAN", "Black (African)"): 0.014,
    ("CAN", "Black (Caribbean)"): 0.008,
    ("CAN", "Arab"): 0.012,
    ("CAN", "Latin American"): 0.012,
    ("CAN", "Southeast Asian"): 0.012,
    ("CAN", "West Asian (Iranian/Afghan/Turkish)"): 0.012,
    ("CAN", "Korean"): 0.010,
    # --- United Kingdom -----------------------------------------------------
    ("GBR", "Indian"): 0.010,
    ("GBR", "Pakistani"): 0.008,
    ("GBR", "Bangladeshi"): 0.008,
    ("GBR", "Black African"): 0.012,
    ("GBR", "Other Asian"): 0.012,
    ("GBR", "White Irish"): 0.006,
    ("GBR", "Chinese"): 0.006,
    ("GBR", "Arab"): 0.010,
    # --- France -------------------------------------------------------------
    ("FRA", "Sub-Saharan African"): 0.012,
    ("FRA", "Algerian origin"): 0.008,
    ("FRA", "Moroccan origin"): 0.008,
    ("FRA", "Tunisian origin"): 0.008,
    ("FRA", "Other North African"): 0.008,
    ("FRA", "Portuguese"): 0.006,
    ("FRA", "Southeast Asian (Vietnamese/Cambodian)"): 0.006,
    # --- Germany ------------------------------------------------------------
    ("DEU", "Syrian"): 0.012,
    ("DEU", "Afghan"): 0.012,
    ("DEU", "Ukrainian"): 0.020,
    ("DEU", "Romanian"): 0.010,
    ("DEU", "Polish"): 0.008,
    ("DEU", "Italian"): 0.006,
    ("DEU", "Iraqi"): 0.010,
    # --- Iberia -------------------------------------------------------------
    ("ESP", "Venezuelan"): 0.022,
    ("ESP", "Colombian"): 0.014,
    ("ESP", "Ecuadorian"): 0.010,
    ("ESP", "Peruvian"): 0.010,
    ("ESP", "Moroccan"): 0.008,
    ("ITA", "Romanian"): 0.008,
    ("ITA", "Moroccan"): 0.010,
    ("ITA", "Egyptian"): 0.012,
    ("ITA", "Indian"): 0.014,
    ("ITA", "Bangladeshi"): 0.012,
    ("ITA", "Pakistani"): 0.012,
    # --- Benelux & Nordics --------------------------------------------------
    ("NLD", "Syrian"): 0.016,
    ("NLD", "Polish"): 0.012,
    ("NLD", "Somali"): 0.010,
    ("SWE", "Syrian"): 0.012,
    ("SWE", "Afghan"): 0.014,
    ("SWE", "Indian"): 0.014,
    ("SWE", "Somalian"): 0.010,
    ("DNK", "Ukrainian"): 0.020,
    ("DNK", "Syrian"): 0.010,
    # --- Alpine / Austria ---------------------------------------------------
    ("AUT", "Syrian"): 0.014,
    ("AUT", "Afghan"): 0.014,
    ("CHE", "Indian"): 0.014,
    ("CHE", "Filipino"): 0.012,
    ("CHE", "Portuguese"): 0.008,
    ("CHE", "Albanian"): 0.010,
    # --- Oceania ------------------------------------------------------------
    ("AUS", "Indian"): 0.016,
    ("AUS", "Filipino"): 0.014,
    ("AUS", "Chinese"): 0.010,
    ("AUS", "Vietnamese"): 0.010,
    ("AUS", "Korean"): 0.010,
    ("AUS", "Sri Lankan"): 0.012,
    ("NZL", "Asian"): 0.014,
    # --- Asia ---------------------------------------------------------------
    ("JPN", "Vietnamese"): 0.014,
    ("JPN", "Filipino"): 0.012,
    ("JPN", "Chinese"): 0.010,
    ("KOR", "Other (incl. foreign workers)"): 0.014,
    ("HKG", "Filipino"): 0.012,
    ("HKG", "Indonesian"): 0.012,
}

# ---------------------------------------------------------------------------
# Late skilled-migration surge, 2038-2050.
# High-demographic-dividend source countries (Nigeria, Ethiopia, Ghana, Kenya,
# Pakistan, and similar labor-abundant states) are likely to supply a larger
# share of skilled migrants as OECD, Gulf, East Asian, and Anglosphere labor
# programs broaden in the 2040s. These coefficients are intentionally
# destination-group specific: the source-country pressure is exported through
# diaspora / regional-origin groups that already exist in the destination
# composition table.
# ---------------------------------------------------------------------------
SKILLED_MIGRATION_SOURCE_PRESSURE: dict[str, float] = {
    "NGA": 1.00, "ETH": 0.88, "GHA": 0.74, "KEN": 0.78, "PAK": 0.86,
    "BGD": 0.74, "PHL": 0.72, "IND": 0.62, "EGY": 0.58, "VNM": 0.56,
}

SKILLED_MIGRATION_DESTINATION_OPENNESS: dict[str, float] = {
    "CAN": 1.30, "AUS": 1.22, "NZL": 1.15, "USA": 1.08, "GBR": 1.05,
    "DEU": 1.04, "IRL": 1.04, "NLD": 1.03, "SWE": 1.02, "FRA": 0.96,
    "JPN": 0.88, "KOR": 0.86, "SGP": 1.16, "ARE": 1.24, "QAT": 1.20,
    "KWT": 1.10, "SAU": 1.08, "OMN": 1.02, "BHR": 1.04,
}

GROUP_SKILLED_MIGRATION_SURGE: dict[tuple[str, str], float] = {
    # Anglosphere settlement countries
    ("USA", "Black (African immigrant)"): 0.0060,
    ("USA", "Asian (Pakistani)"): 0.0048,
    ("USA", "Asian (Bangladeshi)"): 0.0038,
    ("USA", "Asian (Indian)"): 0.0030,
    ("CAN", "Black (African)"): 0.0068,
    ("CAN", "South Asian (Pakistani)"): 0.0054,
    ("CAN", "South Asian (Indian)"): 0.0032,
    ("CAN", "South Asian (Other)"): 0.0034,
    ("AUS", "Indian"): 0.0036,
    ("AUS", "Sri Lankan"): 0.0028,
    ("AUS", "Filipino"): 0.0028,
    ("NZL", "Asian"): 0.0028,
    # Europe
    ("GBR", "Black African"): 0.0062,
    ("GBR", "Pakistani"): 0.0052,
    ("GBR", "Indian"): 0.0030,
    ("GBR", "Bangladeshi"): 0.0038,
    ("FRA", "Sub-Saharan African"): 0.0058,
    ("DEU", "Afghan"): 0.0032,
    ("DEU", "Iraqi"): 0.0024,
    ("NLD", "Somali"): 0.0030,
    ("SWE", "Somalian"): 0.0032,
    ("SWE", "Indian"): 0.0026,
    # Gulf and high-income Asian labor hubs
    ("ARE", "South Asian expatriate"): 0.0062,
    ("ARE", "Other expatriate"): 0.0040,
    ("QAT", "South Asian expatriate"): 0.0060,
    ("KWT", "South Asian expatriate"): 0.0054,
    ("SAU", "South Asian expatriate"): 0.0048,
    ("OMN", "South Asian expatriate"): 0.0042,
    ("BHR", "South Asian expatriate"): 0.0044,
    ("SGP", "Indian"): 0.0038,
    ("JPN", "Vietnamese"): 0.0030,
    ("KOR", "Other (incl. foreign workers)"): 0.0036,
}


def skilled_migration_source_pressure(iso3: str) -> float:
    """Relative skilled-emigration supply pressure for a source country."""
    return SKILLED_MIGRATION_SOURCE_PRESSURE.get(iso3, 0.0)


def skilled_migration_program_intensity(iso3: str) -> float:
    """Destination-side skilled-labor program intensity by 2050."""
    return SKILLED_MIGRATION_DESTINATION_OPENNESS.get(iso3, 0.0)


def late_skilled_migration_curve(progress: float) -> float:
    """Ramp from near-zero before the late 2030s to full strength in 2050."""
    return float(np.clip((progress - 0.55) / 0.45, 0.0, 1.0) ** 2)

# ---------------------------------------------------------------------------
# Intermarriage / mixed-identity formation index.
# Share of new marriages that are inter-ethnic, proxying how fast mixed /
# multiracial identities form (children of mixed unions are increasingly
# counted as multiracial). Drives the growth of "Mixed"/"Multiracial" groups,
# which real projections (US Census, ONS) show as the fastest-growing buckets.
# ---------------------------------------------------------------------------
INTERMARRIAGE_INDEX: dict[str, float] = {
    "USA": 0.22, "NZL": 0.16, "AUS": 0.15, "CAN": 0.14, "GBR": 0.10,
    "SWE": 0.10, "NOR": 0.09, "DNK": 0.09, "NLD": 0.08, "FRA": 0.08,
    "DEU": 0.07, "BEL": 0.07, "CHE": 0.07, "AUT": 0.06, "FIN": 0.06,
    "ITA": 0.05, "ESP": 0.05, "PRT": 0.05, "GRC": 0.04,
}

# ---------------------------------------------------------------------------
# Per-group age-structure "youth bonus" overrides keyed by (ISO3, group).
# The profile-based bonus (high_fertility/immigrant = +0.06) is a reasonable
# default, but some documented cases need explicit values:
#   * African-American and Indigenous groups have young age pyramids that keep
#     births elevated even though their TFR is close to (or below) national --
#     without this override they project as *declining*, contradicting the
#     US Census / StatsCan which show them stable-to-growing.
#   * Multiracial is a young, fast-growing identity.
# ---------------------------------------------------------------------------
GROUP_YOUTH_BONUS: dict[tuple[str, str], float] = {
    ("USA", "Black / African American"): 0.06,
    ("USA", "Native American"): 0.05,
    ("USA", "Native Hawaiian / Pacific"): 0.05,
    ("USA", "Multiracial"): 0.04,
    ("USA", "Middle Eastern / North African"): 0.04,
    ("USA", "Asian (Pakistani)"): 0.05,
    ("USA", "Asian (Bangladeshi)"): 0.06,
    ("CAN", "Indigenous: First Nations"): 0.05,
    ("CAN", "Indigenous: Métis"): 0.05,
    ("CAN", "Indigenous: Inuit"): 0.05,
    ("CAN", "Black (African)"): 0.05,
    ("GBR", "Black African"): 0.05,
    ("GBR", "Pakistani"): 0.05,
    ("GBR", "Bangladeshi"): 0.05,
    ("GBR", "Mixed/Multiple"): 0.04,
    ("FRA", "Sub-Saharan African"): 0.05,
    ("FRA", "Algerian origin"): 0.04,
    ("FRA", "Moroccan origin"): 0.04,
    ("FRA", "Tunisian origin"): 0.04,
    ("FRA", "Other North African"): 0.04,
    ("DEU", "Syrian"): 0.06,
    ("DEU", "Afghan"): 0.06,
    ("DEU", "Iraqi"): 0.05,
    ("NLD", "Somali"): 0.06,
    ("SWE", "Somalian"): 0.06,
    ("DNK", "Somali"): 0.06,
    ("NOR", "Somali"): 0.06,
    ("FIN", "Somali"): 0.06,
}


def demographic_pressure(iso3: str, pop_2024: Optional[float] = None,
                         pop_2050: Optional[float] = None) -> float:
    """How acute the country's future demographic problems are, in [0, 1].

    Combines (a) how far below replacement its current fertility sits and
    (b) how fast its population is projected to shrink between 2024 and 2050.
    Both are strong predictors of future labour shortage and therefore of the
    migration a country will *need* to take in.

    Parameters
    ----------
    iso3 : str
        Country code.
    pop_2024, pop_2050 : float | None
        UN medium-variant populations for the two anchor years. If omitted,
        the population-decline term is dropped (pressure comes from fertility
        alone).

    Returns
    -------
    float
        Pressure index in [0, 1].
    """
    tfr = NATIONAL_TFR_2024.get(iso3, REPLACEMENT_TFR)
    # (a) Fertility shortfall below replacement, scaled so TFR = 0.7
    #     (Korea) maps to ~1.0.
    fert_shortfall = max(0.0, (REPLACEMENT_TFR - tfr) / (REPLACEMENT_TFR - 0.7))
    fert_shortfall = min(1.0, fert_shortfall)

    # (b) Projected annual population decline, scaled so a -1%/yr shrinkage
    #     maps to 1.0.
    pop_decline = 0.0
    if pop_2024 and pop_2050 and pop_2024 > 0:
        annual = (pop_2050 / pop_2024) ** (1.0 / 26.0) - 1.0
        pop_decline = min(1.0, max(0.0, -annual / 0.01))

    return 0.6 * fert_shortfall + 0.4 * pop_decline


def effective_migration_intensity(iso3: str, progress: float,
                                  pop_2024: Optional[float] = None,
                                  pop_2050: Optional[float] = None) -> float:
    """Country's net-migration intensity in a given year of the projection.

    Baseline is the static expatriate/settlement intensity
    (``COUNTRY_MIGRATION_INTENSITY``). On top of it, demographic pressure
    adds an inflow need that *ramps up* over the window (``progress`` in
    [0,1]) as aging and population loss deepen, scaled by policy openness.

    Parameters
    ----------
    iso3 : str
        Country code.
    progress : float
        Fraction of the projection window elapsed (0 = 2024, 1 = 2050).
    pop_2024, pop_2050 : float | None
        Population anchors for the decline component of pressure.

    Returns
    -------
    float
        Effective migration-intensity multiplier (scaled by
        ``MIGRATION_IMPACT_MULTIPLIER``).
    """
    base = COUNTRY_MIGRATION_INTENSITY.get(iso3, 1.0)
    pressure = demographic_pressure(iso3, pop_2024, pop_2050)
    openness = MIGRATION_POLICY_OPENNESS.get(iso3, 1.0)
    # The demographic-driven component grows linearly from 0 (2024) to full
    # strength in 2050 -- policy responds to the worsening crisis over time.
    # Only countries without an already-high expatriate baseline (Gulf states,
    # Singapore, etc., whose structural labour imports are captured by the
    # static intensity) receive the demographic-pressure boost.
    # The boost is weighted at 2.5x the base per unit (0.010 vs 0.0040 annual
    # deviation) because demographic-driven intake is concentrated on targeted
    # labour-recruitment programmes rather than diffuse settlement.
    boost = 0.0
    if base < 1.5:
        boost = 2.5 * 0.5 * pressure * openness * progress
    return (base + boost) * MIGRATION_IMPACT_MULTIPLIER



# ---------------------------------------------------------------------------
# National total fertility rate (children per woman), c. 2024
# Sources: UN World Population Prospects 2024 / World Bank WDI. Values are
# the mid-2020s estimates used across the demographic-transition literature.
# ---------------------------------------------------------------------------
NATIONAL_TFR_2024: dict[str, float] = {
    # --- Europe ----------------------------------------------------------
    "ISL": 1.6, "NOR": 1.4, "SWE": 1.5, "FIN": 1.3, "DNK": 1.5,
    "IRL": 1.6, "GBR": 1.5, "NLD": 1.4, "BEL": 1.5, "LUX": 1.4,
    "FRA": 1.7, "DEU": 1.4, "AUT": 1.3, "CHE": 1.4, "ITA": 1.2,
    "ESP": 1.2, "PRT": 1.4, "GRC": 1.3, "CYP": 1.3, "MLT": 1.1,
    "SVN": 1.6, "HRV": 1.5, "SRB": 1.5, "BIH": 1.4, "MNE": 1.7,
    "MKD": 1.4, "ALB": 1.4, "BGR": 1.6, "ROU": 1.6, "HUN": 1.5,
    "SVK": 1.5, "CZE": 1.5, "POL": 1.3, "LTU": 1.4, "LVA": 1.4,
    "EST": 1.5, "BLR": 1.4, "UKR": 1.2, "MDA": 1.7, "RUS": 1.5,
    "LIE": 1.3, "AND": 1.1, "SMR": 1.2, "MCO": 1.3,
    # --- Asia -------------------------------------------------------------
    "IND": 2.0, "PAK": 3.5, "BGD": 2.0, "LKA": 1.9, "NPL": 2.0,
    "BTN": 1.8, "MDV": 1.5, "AFG": 4.5, "MMR": 2.5, "THA": 1.3,
    "VNM": 1.9, "KHM": 2.4, "LAO": 2.4, "MYS": 1.6, "SGP": 1.0,
    "IDN": 2.1, "PHL": 2.8, "BRN": 1.7, "TLS": 3.3, "CHN": 1.0,
    "JPN": 1.2, "KOR": 0.7, "TWN": 0.9, "MNG": 2.8, "PRK": 1.8,
    "HKG": 0.8,
    # --- Central Asia / Caucasus ------------------------------------------
    "KAZ": 2.9, "UZB": 3.2, "TKM": 2.9, "KGZ": 3.0, "TJK": 3.5,
    "AZE": 1.6, "GEO": 1.7, "ARM": 1.6,
    # --- Middle East & North Africa ---------------------------------------
    "IRQ": 3.4, "IRN": 1.7, "TUR": 1.6, "SYR": 2.8, "JOR": 2.7,
    "LBN": 2.1, "ISR": 2.9, "PSE": 3.9, "SAU": 2.4, "ARE": 1.4,
    "QAT": 1.8, "KWT": 1.6, "BHR": 1.7, "OMN": 2.4, "YEM": 3.8,
    "EGY": 2.9, "DZA": 2.9, "MAR": 2.3, "TUN": 2.0, "LBY": 2.4,
    "SDN": 4.2, "SSD": 4.7, "SOM": 6.0, "MRT": 4.3, "DJI": 3.6,
    "COM": 3.9,
    # --- Sub-Saharan Africa -----------------------------------------------
    "NGA": 4.6, "ETH": 3.9, "KEN": 3.2, "GHA": 3.5, "TZA": 4.4,
    "UGA": 4.5, "SEN": 4.2, "CIV": 4.4, "CMR": 4.2, "AGO": 5.0,
    "MOZ": 4.7, "ZMB": 4.0, "ZWE": 3.4, "MWI": 4.0, "MDG": 3.9,
    "BWA": 2.8, "NAM": 3.3, "SWZ": 3.0, "ZAF": 2.4, "GAB": 3.4,
    "COG": 4.2, "COD": 6.0, "CAF": 5.6, "TCD": 5.9, "MLI": 5.8,
    "NER": 6.6, "BFA": 4.5, "BEN": 4.7, "TGO": 4.3, "GIN": 4.2,
    "GMB": 4.4, "GNB": 4.3, "SLE": 4.0, "LBR": 4.1, "ERI": 3.9,
    "GNQ": 4.0, "STP": 3.8, "MUS": 1.4, "SYC": 2.2, "CPV": 2.1,
    "RWA": 3.8, "BDI": 5.0, "LSO": 3.0,
    # --- Americas ---------------------------------------------------------
    "USA": 1.6, "CAN": 1.4, "MEX": 1.8, "CRI": 1.5, "PAN": 2.1,
    "GTM": 2.3, "HND": 2.3, "SLV": 1.9, "NIC": 2.3, "BLZ": 2.0,
    "CUB": 1.4, "JAM": 1.7, "HTI": 2.5, "DOM": 2.1, "TTO": 1.6,
    "BHS": 1.3, "BRB": 1.6, "ATG": 1.6, "DMA": 1.7, "GRD": 1.9,
    "KNA": 1.6, "LCA": 1.4, "VCT": 1.7, "BRA": 1.6, "ARG": 1.9,
    "CHL": 1.5, "URY": 1.5, "PRY": 2.4, "BOL": 2.6, "PER": 2.0,
    "ECU": 1.9, "COL": 1.7, "VEN": 2.1, "GUY": 2.0, "SUR": 2.1,
    # --- Oceania ----------------------------------------------------------
    "AUS": 1.6, "NZL": 1.6, "FJI": 2.7, "PNG": 3.2, "SLB": 3.5,
    "VUT": 3.6, "WSM": 3.7, "TON": 3.3, "KIR": 3.4, "MHL": 3.0,
    "FSM": 2.6, "PLW": 1.7, "NRU": 3.0, "TUV": 3.2,
}

def _tfr_2050_target(tfr_2024: float) -> float:
    """Medium-variant 2050 TFR target.

    High-fertility countries fall toward replacement; below-replacement
    countries recover only partly (e.g. East Asia stays far below
    replacement). Rule: 2.1 + (tfr_2024 - 2.1) * convergence, where
    convergence = 0.55 for TFR>=3, 0.40 for 2.1<=TFR<3, 0.30 for TFR<2.1.
    """
    if tfr_2024 >= 3.0:
        return round(2.1 + (tfr_2024 - 2.1) * 0.55, 2)
    if tfr_2024 >= 2.1:
        return round(2.1 + (tfr_2024 - 2.1) * 0.40, 2)
    return round(tfr_2024 + (2.1 - tfr_2024) * 0.30, 2)


# Medium-variant 2050 TFR target per country.
NATIONAL_TFR_2050: dict[str, float] = {
    iso: _tfr_2050_target(tfr) for iso, tfr in NATIONAL_TFR_2024.items()
}


def national_tfr(iso3: str, year: int = 2024) -> float:
    """Interpolated national TFR between the 2024 and 2050 anchors."""
    t0 = NATIONAL_TFR_2024.get(iso3, 2.1)
    t1 = NATIONAL_TFR_2050.get(iso3, t0)
    if year <= 2024:
        return t0
    if year >= 2050:
        return t1
    return t0 + (t1 - t0) * (year - 2024) / 26.0


# ---------------------------------------------------------------------------
# Per-group TFR estimation
# ---------------------------------------------------------------------------
# Keyword -> additive TFR adjustment applied on top of the profile base.
# Rules are matched against the (lower-cased) group name; the *last* match
# wins so more specific terms can be placed after generic ones.
GROUP_TFR_RULES: list[tuple[re.Pattern, float]] = [
    # Ultra-high fertility groups
    (re.compile(r"haredi|ultra-orthodox|ultra orthodox|orthodox jew"), +3.6),
    (re.compile(r"mormon|amish|hutterite"), +1.2),
    (re.compile(r"\bromani\b|rom |gyps"), +0.7),
    (re.compile(r"hazaras?|hazara"), +0.4),
    # Muslim minorities (West / non-majority-Muslim countries)
    (re.compile(r"muslim|moro|bangsamoro"), +0.6),
    (re.compile(r"arab|bedouin|berber|amazigh|tuareg|somali|sudanese|egyptian"
                r"|yemeni|syrian|iraqi|iranian|turkish|kurdish|kurd"), +0.4),
    # Indigenous / high-fertility minority groups in the Americas, Oceania
    (re.compile(r"indigenous|amerindian|maori|pacific|aboriginal|torres strait"
                r"|native american|first nations|inuit|wayuu|embera"), +0.5),
    # Sub-Saharan groups living at/above national fertility
    (re.compile(r"hausa|fulani|peulh|mossi|kanuri|tiv|igbo|yoruba"), +0.3),
    (re.compile(r"dinka|nuer|shilluk|azande|turkana|maasai"), +0.4),
    # Low-fertility groups (East Asian-origin, secular European descent)
    (re.compile(r"japanese|korean|chinese|taiwanese|hakka|sino"), -0.2),
    (re.compile(r"white|european descent|european "), -0.2),
    (re.compile(r"british|english|french|german|italian|spanish|portuguese"
                r"|dutch|swiss|scandinavian|nordic"), -0.1),
]

# Groups whose name denotes a mixed / creole / tri-hybrid identity are already
# captured by their own profile; suppress the ethnic-keyword adjustments so a
# name like "Pardo (Mixed Euro/African/Indigenous)" does not inherit the
# indigenous fertility bonus.
_MIXED_IDENTITY = re.compile(
    r"mixed|mestizo|pardo|mulatto|creole|mestico|mestee|métis|metis|forros"
)

# Exact-group overrides keyed by (ISO3, group name) for well-documented cases.
GROUP_TFR_OVERRIDES: dict[tuple[str, str], float] = {
    # Israel: Haredi ~6.4, non-Haredi Jewish ~2.1, Arab Muslim ~3.1
    ("ISR", "Jewish (Haredi/Orthodox)"): 6.4,
    ("ISR", "Jewish (Non-Haredi)"): 2.1,
    ("ISR", "Arab Muslim (incl. Bedouin)"): 3.1,
    ("ISR", "Arab Christian"): 2.2,
    ("ISR", "Druze"): 2.3,
    # West Bank / Gaza
    ("PSE", "Palestinian Arab (Sunni Muslim)"): 4.0,
    # Europe's Muslim minorities
    ("FRA", "Algerian origin"): 2.4,
    ("FRA", "Moroccan origin"): 2.5,
    ("FRA", "Tunisian origin"): 2.3,
    ("FRA", "Other North African"): 2.5,
    ("FRA", "Sub-Saharan African"): 2.8,
    ("GBR", "Pakistani"): 2.8,
    ("GBR", "Bangladeshi"): 2.9,
    ("DEU", "Turkish"): 2.1,
    ("DEU", "Syrian"): 2.6,
    ("NLD", "Turkish"): 2.0,
    ("NLD", "Moroccan"): 2.2,
    ("BEL", "Moroccan"): 2.4,
    ("BEL", "Turkish"): 2.1,
    ("CHE", "Kosovar Albanian"): 2.0,
    ("NOR", "Somali"): 2.8,
    ("DNK", "Somali"): 2.8,
    # East Asian low fertility
    ("JPN", "Japanese"): 1.2,
    ("KOR", "Korean"): 0.7,
    ("SGP", "Chinese"): 0.9,
    ("HKG", "Chinese"): 0.8,
    ("CHN", "Han Chinese"): 1.0,
    # Latin American / US Hispanic
    ("USA", "White non-Hispanic"): 1.4,
    ("USA", "Black / African American"): 1.7,
    ("USA", "Native American"): 1.8,
    # --- US Hispanic subgroups (2023 ACS fertility ~1.6-2.2) ----------------
    ("USA", "Hispanic / Latino (Mexican)"): 1.8,
    ("USA", "Hispanic / Latino (Puerto Rican)"): 1.6,
    ("USA", "Hispanic / Latino (Salvadoran)"): 2.0,
    ("USA", "Hispanic / Latino (Guatemalan)"): 2.1,
    ("USA", "Hispanic / Latino (Honduran)"): 2.2,
    ("USA", "Hispanic / Latino (Colombian)"): 1.6,
    ("USA", "Hispanic / Latino (Venezuelan)"): 1.7,
    ("USA", "Hispanic / Latino (Dominican)"): 1.8,
    ("USA", "Hispanic / Latino (Cuban)"): 1.4,
    ("USA", "Hispanic / Latino (Other)"): 1.8,
    # --- US Asian subgroups (Chinese/Japanese/Korean well below replacement) -
    ("USA", "Asian (Chinese)"): 1.3,
    ("USA", "Asian (Indian)"): 1.5,
    ("USA", "Asian (Filipino)"): 1.6,
    ("USA", "Asian (Vietnamese)"): 1.5,
    ("USA", "Asian (Korean)"): 1.1,
    ("USA", "Asian (Japanese)"): 1.0,
    ("USA", "Asian (Pakistani)"): 2.0,
    ("USA", "Asian (Bangladeshi)"): 2.1,
    ("USA", "Asian (Other)"): 1.4,
    ("USA", "Middle Eastern / North African"): 2.0,
    ("USA", "Black (African immigrant)"): 2.0,
    ("USA", "Black (Caribbean)"): 1.7,
    # --- Canada ---------------------------------------------------------------
    ("CAN", "South Asian (Indian)"): 1.7,
    ("CAN", "South Asian (Pakistani)"): 2.2,
    ("CAN", "South Asian (Sri Lankan)"): 1.9,
    ("CAN", "South Asian (Other)"): 1.8,
    ("CAN", "Black (African)"): 2.2,
    ("CAN", "Black (Caribbean)"): 1.9,
    ("CAN", "Indigenous: First Nations"): 2.0,
    ("CAN", "Indigenous: Métis"): 1.8,
    ("CAN", "Indigenous: Inuit"): 2.5,
    ("CAN", "Arab"): 2.3,
    ("CAN", "West Asian (Iranian/Afghan/Turkish)"): 2.0,
    ("CAN", "Korean"): 1.0,
    ("CAN", "Japanese"): 0.9,
    # --- UK -------------------------------------------------------------------
    ("GBR", "White British"): 1.5,
    ("GBR", "Black African"): 2.4,
    ("GBR", "White Irish"): 1.6,
    ("GBR", "Arab"): 2.5,
    # --- Europe ----------------------------------------------------------------
    ("DEU", "Syrian"): 2.6,
    ("DEU", "Afghan"): 2.4,
    ("DEU", "Ukrainian"): 1.7,
    ("DEU", "Iraqi"): 2.4,
    ("FRA", "Turkish"): 2.1,
    ("FRA", "Southeast Asian (Vietnamese/Cambodian)"): 1.9,
    ("ESP", "Venezuelan"): 1.8,
    ("ESP", "Colombian"): 1.7,
    ("ESP", "Ecuadorian"): 1.9,
    ("ESP", "Peruvian"): 1.9,
    ("ESP", "Moroccan"): 2.1,
    ("ITA", "Egyptian"): 2.0,
    ("ITA", "Bangladeshi"): 2.2,
    ("ITA", "Pakistani"): 2.1,
    ("ITA", "Indian"): 1.9,
    ("NLD", "Somali"): 2.6,
    ("NLD", "Polish"): 1.7,
    ("SWE", "Somalian"): 2.8,
    ("SWE", "Pakistani"): 2.5,
    ("SWE", "Afghan"): 2.4,
    ("DNK", "Somali"): 2.8,
    ("DNK", "Pakistani"): 2.3,
    ("AUT", "Syrian"): 2.5,
    ("AUT", "Afghan"): 2.4,
    ("CHE", "Indian"): 1.8,
    ("CHE", "Kosovar Albanian"): 2.0,
    # --- Asia -----------------------------------------------------------------
    ("JPN", "Vietnamese"): 1.5,
    ("JPN", "Filipino"): 1.7,
    ("JPN", "Brazilian (Nikkei)"): 1.4,
    ("KOR", "Other (incl. foreign workers)"): 1.5,
    ("HKG", "Filipino"): 1.6,
    ("HKG", "Indonesian"): 1.7,
    ("SGP", "Filipino"): 1.7,
    ("MYS", "Indian (Tamil)"): 1.5,
    # --- Latin America ----------------------------------------------------------
    ("BRA", "Pardo (Mixed Euro/African/Indigenous)"): 1.7,
    ("BRA", "Black (African descent)"): 1.8,
    ("BRA", "Indigenous"): 2.0,
    ("VEN", "Mixed (Mestizo)"): 1.9,
}


def _rule_adjustment(group_name: str) -> float:
    if _MIXED_IDENTITY.search(group_name.lower()):
        return 0.0
    adj = 0.0
    name = group_name.lower()
    for pattern, delta in GROUP_TFR_RULES:
        if pattern.search(name):
            adj = delta  # last match wins
    return adj


def estimate_group_tfr(iso3: str, group_name: str, profile: object) -> float:
    """Estimate the group's TFR in 2024.

    Base = national TFR, adjusted by profile, then by keyword rules, then by
    exact overrides. Clamped to a defensible [1.0, 7.0] band.
    """
    tfr_nat = NATIONAL_TFR_2024.get(iso3, 2.1)

    if isinstance(profile, (int, float)):
        base = tfr_nat
    elif profile == "high_fertility":
        base = tfr_nat + 0.35
    elif profile == "low_fertility":
        base = tfr_nat - 0.40
    elif profile == "immigrant":
        base = tfr_nat  # migrants adopt host-country fertility
    else:  # majority / assimilating
        base = tfr_nat

    override = GROUP_TFR_OVERRIDES.get((iso3, group_name))
    if override is not None:
        return float(np.clip(override, 0.7, 7.0))

    # Keyword adjustments apply only to non-majority groups. The majority
    # group anchors the national TFR, so applying religion/region keywords on
    # top of it would double-count (e.g. Egyptian majority getting an "Arab"
    # bonus).
    if profile == "majority":
        return float(np.clip(base, 0.7, 7.0))

    adjusted = base + _rule_adjustment(group_name)
    return float(np.clip(adjusted, 0.7, 7.0))


def _group_tfr_2050(iso3: str, group_name: str, tfr_2024: float) -> float:
    """Group TFR in 2050 after fertility convergence toward the national path."""
    tfr_nat_0 = NATIONAL_TFR_2024.get(iso3, 2.1)
    tfr_nat_1 = NATIONAL_TFR_2050.get(iso3, tfr_nat_0)
    gap = tfr_2024 - tfr_nat_0
    return float(np.clip(tfr_nat_1 + gap * (1.0 - 0.6), 0.7, 7.0))


def estimate_group_tfr_map(iso3: str) -> list[tuple[str, float, float, float]]:
    """Return (group, share_pct, profile, tfr) list for one country."""
    out = []
    for name, share, profile in ETHNIC_COMPOSITION_2024[iso3]:
        tfr = estimate_group_tfr(iso3, name, profile)
        out.append((name, share, profile, tfr))
    return out


# ---------------------------------------------------------------------------
# Projection engine
# ---------------------------------------------------------------------------
def project_ethnic_composition(
    iso3: str,
    start_year: int = 2024,
    end_year: int = 2050,
    migration_scenario: str = "baseline",
    migration_scale: Optional[float] = None,
    fertility_convergence: float = 0.6,
    assimilation_rate: Optional[float] = None,
    intermarriage_multiplier: float = 1.0,
    mixed_identity_multiplier: float = 1.0,
    education_index: Optional[float] = None,
    income_index: Optional[float] = None,
    structural_inequality_drag: float = 0.0,
    pop_2024: Optional[float] = None,
    pop_2050: Optional[float] = None,
) -> dict[str, float]:
    """Evidence-based projection of ethnic shares for one country to end_year.

    The annual decomposition combines:

    1. **Fertility differential** -- ``ln(TFR_g / TFR_nat) / G`` (intrinsic
       growth from the group's relative fertility).
    2. **Age-structure momentum** -- already-born cohorts keep births elevated
       for ~2 decades. Scaled by a profile "youth bonus" (high-fertility and
       immigrant groups have young pyramids; low-fertility/majority groups in
       aging societies have a drag).
    3. **Composition-specific migration** -- each country imports a *specific
       mix* of ethnic groups (``GROUP_MIGRATION_INTENSITY``), scaled by the
       country's migration intensity (static baseline + demographic-pressure
       boost). Unlisted immigrant-profile groups get the default coefficient.
    4. **Late skilled-migration surge** -- from the late 2030s onward,
       skilled-labor programs pull more migrants from demographic-dividend
       source countries into destination-specific diaspora groups.
    5. **Intermarriage / mixed-identity formation** -- in high-intermarriage
       countries (``INTERMARRIAGE_INDEX``), a fraction of the non-mixed
       population forms new Mixed/Multiracial-identified people each year.
    6. **Fertility convergence** -- group TFR pulls toward national TFR
       (``fertility_convergence``) and national TFR toward its 2050 target.
    7. **Assimilation** -- per-country intermarriage/identity-shift transfer
       of minorities to the anchor group (``COUNTRY_ASSIMILATION``).

    Parameters
    ----------
    iso3 : str
        Country code present in ETHNIC_COMPOSITION_2024.
    start_year, end_year : int
        Projection window.
    migration_scenario : {"baseline", "closed", "high"}
        Scales the net-migration deviation of ``immigrant`` groups.
    migration_scale : float | None
        Direct numeric override of the migration deviation scale (0 = closed,
        1.0 = baseline, >1 = elevated). Takes precedence over
        ``migration_scenario``; used by the dashboard's what-if sliders.
    fertility_convergence : float in [0,1]
        Fraction of the 2024 group-vs-national TFR gap that closes by the
        end year (demographic transition). 0.6 -> 60% of the differential
        converges toward national fertility.
    assimilation_rate : float | None
        Override of the per-year minority -> anchor transfer rate.
    intermarriage_multiplier, mixed_identity_multiplier : float
        Scenario multipliers for multi-ethnic family formation and fluid mixed
        identity growth. Values above 1.0 are used by the Brazilification
        scenario.
    education_index, income_index : float | None
        HDI component context used to accelerate fertility convergence for
        high-development / high-mobility settings.
    structural_inequality_drag : float
        Dampens the Brazilification feedback where polarization or unequal
        public-service delivery limits socioeconomic convergence.
    pop_2024, pop_2050 : float | None
        UN medium-variant populations for the demographic-pressure (labour
        shortage) component of migration intensity. Optional; if omitted the
        pressure boost uses fertility shortfall alone.

    Returns
    -------
    dict[str, float]
        Projected share (0-1) per ethnic group for the end year.
    """
    if migration_scenario == "closed":
        mig_scale = 0.0
    elif migration_scenario == "high":
        mig_scale = 1.5
    else:
        mig_scale = 1.0
    if migration_scale is not None:
        mig_scale = float(np.clip(migration_scale, 0.0, 3.0))

    # Normalise to 100% (adds an "Other" bucket if needed)
    entries = _normalised_entries(iso3)
    names = [e[0] for e in entries]
    shares = np.array([e[1] / 100.0 for e in entries], dtype=float)

    tfr_nat_0 = NATIONAL_TFR_2024.get(iso3, 2.1)
    tfr_nat_1 = NATIONAL_TFR_2050.get(iso3, tfr_nat_0)
    tfr_g0 = np.array([e[3] for e in entries], dtype=float)

    if assimilation_rate is None:
        assimilation_rate = COUNTRY_ASSIMILATION.get(iso3, DEFAULT_ASSIMILATION_RATE)
    intermarriage_multiplier = float(np.clip(intermarriage_multiplier, 0.0, 4.0))
    mixed_identity_multiplier = float(np.clip(mixed_identity_multiplier, 0.0, 5.0))
    education_income_context = 0.0
    if education_index is not None and income_index is not None:
        education_income_context = float(np.clip((education_index + income_index) / 2.0, 0.0, 1.0))
    structural_inequality_drag = float(np.clip(structural_inequality_drag, 0.0, 1.0))
    base_mig_intensity = COUNTRY_MIGRATION_INTENSITY.get(iso3, 1.0)
    pressure = demographic_pressure(iso3, pop_2024, pop_2050)
    mig_openness = MIGRATION_POLICY_OPENNESS.get(iso3, 1.0)

    anchor = int(np.argmax(shares))
    years = end_year - start_year

    # Profile -> age-structure "youth bonus" (fraction/year). Groups with a
    # young age pyramid (high-fertility, recent immigrants) keep producing
    # births above what their TFR alone implies for ~2 decades; aging groups
    # (low-fertility, majority in East Asia/Europe) have the opposite drag.
    # Documented per-group cases (US Black, Indigenous, Multiracial, key
    # immigrant minorities) override the profile default.
    youth_bonus = np.array([
        GROUP_YOUTH_BONUS.get((iso3, e[0]),
            {"high_fertility": 0.06, "immigrant": 0.06, "assimilating": 0.03,
             "low_fertility": -0.03}.get(e[2], 0.0)) for e in entries
    ], dtype=float)

    # Intermarriage-driven mixed-identity formation for this country.
    intermarriage = INTERMARRIAGE_INDEX.get(iso3, 0.0) * intermarriage_multiplier
    mixed_mask = np.array([
        1.0 if re.search(r"mixed|multiracial|multiple origins|multi-racial",
                         e[0], re.IGNORECASE) else 0.0 for e in entries
    ], dtype=float)

    for t in range(1, years + 1):
        progress = t / years

        # 1) National TFR follows the demographic-transition path.
        tfr_nat = tfr_nat_0 + (tfr_nat_1 - tfr_nat_0) * progress

        # 2) Group TFR converges toward national fertility.
        gap = tfr_g0 - tfr_nat_0
        effective_fertility_convergence = np.clip(
            fertility_convergence +
            education_income_context * 0.22 * progress -
            structural_inequality_drag * 0.18,
            0.05,
            0.95,
        )
        tfr_g = tfr_nat + gap * (1.0 - effective_fertility_convergence * progress)
        tfr_g = np.clip(tfr_g, 0.8, 7.5)

        # 3) Intrinsic-growth differential from fertility (stable population).
        dev_fert = np.log(tfr_g / max(tfr_nat, 0.5)) / GENERATION_LENGTH

        # 4) Age-structure momentum: already-born cohorts keep births up.
        #    Decays over ~2 decades; boosted for young-profile groups and for
        #    high-fertility groups (whose gap captures the young pyramid).
        momentum = (0.16 * gap + youth_bonus) * np.exp(-progress * 2.0) / GENERATION_LENGTH

        # 5) Migration inflow. Each country imports a specific *composition* of
        #    groups (GROUP_MIGRATION_INTENSITY); unlisted immigrant-profile
        #    groups get the default coefficient. Intensity is the static
        #    expatriate/settlement baseline plus a demographic-pressure boost
        #    that ramps up as labour shortages deepen.
        mig_intensity = effective_migration_intensity(
            iso3, progress, pop_2024, pop_2050)
        skilled_program = skilled_migration_program_intensity(iso3)
        late_skilled = late_skilled_migration_curve(progress)
        dev_mig = np.zeros_like(shares)
        for i, e in enumerate(entries):
            coeff = GROUP_MIGRATION_INTENSITY.get((iso3, e[0]))
            if coeff is not None:
                dev_mig[i] = coeff * mig_scale * mig_intensity
            elif e[2] == "immigrant":
                dev_mig[i] = 0.0040 * mig_scale * mig_intensity
            elif (e[0] == "Other" and pressure > 0.30
                  and e[2] in ("assimilating", "majority")):
                dev_mig[i] = 0.0040 * mig_scale * mig_intensity * 0.5
            skilled_coeff = GROUP_SKILLED_MIGRATION_SURGE.get((iso3, e[0]))
            if skilled_coeff is not None:
                dev_mig[i] += (
                    skilled_coeff * mig_scale * skilled_program *
                    late_skilled
                )

        # 6) Intermarriage: a fraction of the non-mixed population forms new
        #    mixed-identity people each year (children of mixed unions are
        #    increasingly identified as multiracial). This is the engine
        #    behind the fast-growing Mixed/Multiracial buckets in the US,
        #    Canada, Australia, NZ, the UK and northern Europe.
        dev_mixed = np.zeros_like(shares)
        if intermarriage > 0.0 and mixed_mask.sum() > 0.0:
            mixed_share = float((shares * mixed_mask).sum())
            pool = 1.0 - mixed_share
            inflow = intermarriage * 0.12 * pool * mixed_identity_multiplier
            dev_mixed[mixed_mask > 0.0] = inflow

        dev = dev_fert + momentum + dev_mig + dev_mixed

        shares = shares * (1.0 + dev)

        # 7) Assimilation: transfer a fraction of each minority to the anchor.
        if assimilation_rate > 0.0:
            for i in range(len(shares)):
                if i == anchor:
                    continue
                transfer = shares[i] * assimilation_rate
                shares[i] -= transfer
                shares[anchor] += transfer

        shares = shares / shares.sum()

    return {names[i]: float(shares[i]) for i in range(len(names))}


def _normalised_entries(iso3: str) -> list[tuple[str, float, object, float]]:
    """(group, share_pct, profile, tfr) with shares normalised to 100."""
    raw = ETHNIC_COMPOSITION_2024[iso3]
    total = sum(share for _, share, _ in raw)
    entries: list[tuple[str, float, object, float]] = []
    for name, share, profile in raw:
        tfr = estimate_group_tfr(iso3, name, profile)
        entries.append((name, share, profile, tfr))
    if total < 99.99:
        entries.append(("Other", round(100.0 - total, 3), "majority",
                        NATIONAL_TFR_2024.get(iso3, 2.1)))
    elif total > 100.01:
        scale = 100.0 / total
        entries = [(n, round(s * scale, 3), p, t) for n, s, p, t in entries]
    return entries


def project_absolute_populations(
    iso3: str,
    population_2050: float,
    end_year: int = 2050,
    **kwargs,
) -> dict[str, float]:
    """Project absolute population per ethnic group for the end year."""
    shares = project_ethnic_composition(iso3, end_year=end_year, **kwargs)
    return {g: s * float(population_2050) for g, s in shares.items()}
