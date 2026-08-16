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
    - migration:                migration-linked groups receive a net-inflow
      deviation from an explicit origin-destination corridor model combining
      source workforce supply, portable-skill readiness, climate pressure,
      destination labor demand, policy openness, diaspora networks, bilateral
      affinity, and long-run settlement retention.
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

For Europe, a separate birth-cohort replacement term captures the fact that
births can continue falling even when TFR partly recovers, because fewer women
enter child-bearing ages. The resulting migration response is delayed and
constrained by country-specific recruitment and integration capacity.

Migration-corridor scenarios
----------------------------
The 2050 projection does not treat migration as a generic uplift shared by all
groups. Group labels are mapped to plausible source countries, and only groups
with an explicit migration channel receive corridor growth. Low and high
migration runs expose sensitivity to policy and shock uncertainty. The output
is directional scenario analysis rather than an official bilateral flow
forecast; native-majority groups are excluded from the corridor table.

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

# European capacity to translate worsening birth-cohort replacement into
# additional labor migration. This is intentionally separate from demographic
# need: labor shortages do not guarantee migration when politics, housing,
# integration systems, or administrative capacity constrain recruitment.
# Values are scenario multipliers, not observed migration rates.
EUROPE_BIRTH_REPLACEMENT_RESPONSE: dict[str, float] = {
    "AUT": 0.95, "BEL": 1.00, "BGR": 0.58, "HRV": 0.72,
    "CYP": 0.82, "CZE": 0.82, "DNK": 0.92, "EST": 0.78,
    "FIN": 0.92, "FRA": 1.00, "DEU": 1.12, "GRC": 0.72,
    "HUN": 0.48, "IRL": 1.12, "ITA": 0.82, "LVA": 0.62,
    "LTU": 0.66, "LUX": 1.10, "MLT": 0.92, "NLD": 1.08,
    "POL": 0.58, "PRT": 0.88, "ROU": 0.58, "SVK": 0.58,
    "SVN": 0.78, "ESP": 0.92, "SWE": 1.00,
    # Other European labor markets included in the same regional mechanism.
    "GBR": 1.06, "NOR": 0.96, "CHE": 1.05, "ISL": 0.88,
    "ALB": 0.46, "BIH": 0.42, "MKD": 0.46, "MNE": 0.52,
    "SRB": 0.48,
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
    "UGA": 0.70, "TZA": 0.68, "RWA": 0.66, "CIV": 0.62,
    "CMR": 0.64, "SEN": 0.60, "ZMB": 0.58, "ZWE": 0.62,
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


def broad_labor_migration_source_pressure(iso3: str) -> float:
    """Potential late-period workforce mobility from source demographics."""
    youth, skill, climate = SOURCE_MIGRATION_CAPACITY.get(
        iso3, (0.0, 0.0, 0.0))
    return float(np.clip(
        0.58 * youth + 0.18 * skill + 0.24 * climate,
        0.0, 1.0,
    ))


def broad_labor_migration_program_intensity(iso3: str) -> float:
    """Destination demand translated into policy-accessible labor pathways."""
    demand = DESTINATION_LABOR_DEMAND.get(iso3, 0.0)
    policy = BROAD_LABOR_DESTINATION_OPENNESS.get(iso3, 0.0)
    openness = MIGRATION_POLICY_OPENNESS.get(iso3, 1.0)
    retention = DESTINATION_SETTLEMENT_RETENTION.get(iso3, 0.78)
    return float(np.clip(
        demand * policy * min(1.0, openness / 1.15) *
        (0.55 + 0.45 * retention),
        0.0, 1.25,
    ))


def ssa_late_migration_destination_response(iso3: str) -> float:
    """Destination exposure to expanded Sub-Saharan African corridors."""
    return float(np.clip(
        broad_labor_migration_program_intensity(iso3) *
        SSA_LATE_MIGRATION_DESTINATION_EXPOSURE.get(iso3, 0.0),
        0.0, 1.0,
    ))


def late_skilled_migration_curve(progress: float) -> float:
    """Ramp from near-zero before the late 2030s to full strength in 2050."""
    return float(np.clip((progress - 0.55) / 0.45, 0.0, 1.0) ** 2)


def late_broad_labor_migration_curve(progress: float) -> float:
    """Broader labor mobility ramp from the mid-2030s through the 2040s."""
    return float(np.clip((progress - 0.46) / 0.54, 0.0, 1.0) ** 1.65)


def late_ssa_source_pool_transition(progress: float) -> float:
    """Late-horizon shift toward SSA in the global mobile-labor source pool.

    The transition starts after 2037 and accelerates through the 2040s as
    working-age growth slows across South and Southeast Asia. It is a
    conditional scenario factor, not an assumed migration rate.
    """
    x = float(np.clip((progress - 0.50) / 0.50, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


# Source-side migration capacity. The three components distinguish a large
# youth cohort from the education/credential base needed for skilled mobility
# and from displacement pressure. They are scenario inputs, not ethnic traits.
SOURCE_MIGRATION_CAPACITY: dict[str, tuple[float, float, float]] = {
    # ISO3: (working-age/youth supply, portable-skill readiness, climate stress)
    "NGA": (0.96, 0.68, 0.58), "ETH": (0.92, 0.52, 0.76),
    "GHA": (0.78, 0.75, 0.46), "KEN": (0.84, 0.78, 0.58),
    "PAK": (0.88, 0.70, 0.74), "IND": (0.66, 0.90, 0.56),
    "BGD": (0.80, 0.68, 0.84), "PHL": (0.70, 0.88, 0.64),
    "VNM": (0.62, 0.84, 0.68), "IDN": (0.72, 0.78, 0.76),
    "NPL": (0.74, 0.66, 0.74), "LKA": (0.54, 0.82, 0.54),
    "EGY": (0.72, 0.72, 0.66), "MAR": (0.60, 0.70, 0.58),
    "DZA": (0.62, 0.68, 0.60), "TUN": (0.48, 0.74, 0.52),
    "AFG": (0.90, 0.42, 0.86), "IRQ": (0.74, 0.58, 0.70),
    "SYR": (0.76, 0.60, 0.74), "UKR": (0.46, 0.82, 0.52),
    "MEX": (0.48, 0.74, 0.56), "GTM": (0.68, 0.56, 0.70),
    "HND": (0.66, 0.56, 0.72), "SLV": (0.56, 0.62, 0.66),
    "VEN": (0.52, 0.78, 0.64), "COL": (0.52, 0.78, 0.60),
    "BRA": (0.46, 0.82, 0.62), "HTI": (0.82, 0.48, 0.86),
    # Additional Sub-Saharan African sources. A large youth cohort does not
    # imply automatic emigration: skill portability, financing, policy access,
    # diaspora links and destination demand still constrain each corridor.
    "UGA": (0.92, 0.60, 0.64), "TZA": (0.90, 0.62, 0.66),
    "RWA": (0.78, 0.69, 0.58), "COD": (0.98, 0.42, 0.82),
    "CIV": (0.86, 0.64, 0.68), "CMR": (0.86, 0.66, 0.70),
    "SEN": (0.80, 0.67, 0.64), "ZMB": (0.86, 0.63, 0.66),
    "ZWE": (0.70, 0.76, 0.62), "MOZ": (0.92, 0.50, 0.80),
    "MWI": (0.94, 0.48, 0.76), "AGO": (0.88, 0.56, 0.72),
    "SDN": (0.92, 0.46, 0.88), "SOM": (0.92, 0.40, 0.90),
    "ERI": (0.82, 0.55, 0.80), "BFA": (0.94, 0.48, 0.80),
    "NER": (1.00, 0.40, 0.86), "MLI": (0.96, 0.46, 0.82),
    "GIN": (0.90, 0.52, 0.74), "SLE": (0.86, 0.54, 0.76),
    "LBR": (0.84, 0.58, 0.74), "BEN": (0.86, 0.58, 0.70),
    "TGO": (0.84, 0.58, 0.70), "GMB": (0.84, 0.55, 0.72),
}

SSA_SOURCE_POOL_COUNTRIES: tuple[str, ...] = (
    "NGA", "ETH", "GHA", "KEN", "UGA", "TZA", "RWA", "COD",
    "CIV", "CMR", "SEN", "ZMB", "ZWE", "MOZ", "MWI", "AGO",
    "SDN", "SOM", "ERI", "BFA", "NER", "MLI", "GIN", "SLE",
    "LBR", "BEN", "TGO", "GMB",
)


def ssa_source_pool_capacity() -> float:
    """Composite supply/skills/stress capacity for the SSA source pool."""
    values = [SOURCE_MIGRATION_CAPACITY[iso3] for iso3 in SSA_SOURCE_POOL_COUNTRIES]
    youth = float(np.mean([value[0] for value in values]))
    skill = float(np.mean([value[1] for value in values]))
    climate = float(np.mean([value[2] for value in values]))
    return float(np.clip(0.54 * youth + 0.30 * skill + 0.16 * climate, 0.0, 1.0))

# Late-period broad workforce pathways. This channel covers care, logistics,
# construction, hospitality, agriculture, manufacturing and skilled trades as
# well as formal professional recruitment. Values are annual share-growth
# premiums on destination categories and remain subject to destination policy,
# corridor affinity and settlement retention.
GROUP_BROAD_LABOR_MIGRATION_SURGE: dict[tuple[str, str], float] = {
    ("USA", "Black (African immigrant)"): 0.0080,
    ("CAN", "Black (African)"): 0.0100,
    ("GBR", "Black African"): 0.0120,
    ("FRA", "Sub-Saharan African"): 0.0140,
    ("DEU", "Other"): 0.0040,
    ("ITA", "Nigerian"): 0.0160, ("ITA", "Other"): 0.0030,
    ("ESP", "Other"): 0.0030,
    ("NLD", "Somali"): 0.0100, ("NLD", "Other"): 0.0030,
    ("BEL", "Sub-Saharan African"): 0.0140,
    ("SWE", "Somalian"): 0.0100, ("SWE", "Eritrean"): 0.0100,
    ("DNK", "Somali"): 0.0100,
    ("DNK", "Other Middle Eastern/African/Asian"): 0.0040,
    ("NOR", "Somali"): 0.0100, ("NOR", "Eritrean"): 0.0100,
    ("FIN", "Somali"): 0.0090,
    ("AUT", "Other"): 0.0030,
    ("CHE", "Eritrean"): 0.0100, ("CHE", "Other"): 0.0030,
    ("IRL", "Nigerian"): 0.0160, ("IRL", "African (other)"): 0.0140,
    ("PRT", "African (PALOP - Angolan/Cape Verdean)"): 0.0130,
    ("GRC", "Other"): 0.0025,
    ("AUS", "Other"): 0.0035,
    ("NZL", "MELAA (ME/LatAm/African)"): 0.0050,
    ("JPN", "Other"): 0.0040,
    ("KOR", "Other (incl. foreign workers)"): 0.0060,
    ("SGP", "Other"): 0.0040,
    ("ARE", "Other foreign"): 0.0090,
    ("QAT", "Other foreign"): 0.0090,
    ("SAU", "Sudanese"): 0.0100, ("SAU", "Other foreign"): 0.0060,
}

# Incremental late-2040s rebalancing toward the world's last large growing
# working-age pool. These coefficients act only on destination categories
# that can plausibly record new SSA-origin residents. They are policy-,
# demand-, skills- and retention-gated in the projection loop below.
GROUP_SSA_SOURCE_POOL_SURGE: dict[tuple[str, str], float] = {
    ("USA", "Black (African immigrant)"): 0.0025,
    ("CAN", "Black (African)"): 0.0040,
    ("GBR", "Black African"): 0.0025,
    ("FRA", "Sub-Saharan African"): 0.0020,
    ("DEU", "Other"): 0.0012,
    ("ITA", "Nigerian"): 0.0018, ("ITA", "Other"): 0.0006,
    ("ESP", "Other"): 0.0007,
    ("NLD", "Somali"): 0.0015, ("NLD", "Other"): 0.0006,
    ("BEL", "Sub-Saharan African"): 0.0018,
    ("SWE", "Somalian"): 0.0014, ("SWE", "Eritrean"): 0.0012,
    ("DNK", "Somali"): 0.0012, ("NOR", "Somali"): 0.0012,
    ("FIN", "Somali"): 0.0010, ("CHE", "Eritrean"): 0.0010,
    ("IRL", "Nigerian"): 0.0018, ("IRL", "African (other)"): 0.0016,
    ("PRT", "African (PALOP - Angolan/Cape Verdean)"): 0.0014,
    ("AUS", "Other"): 0.0015,
    ("NZL", "MELAA (ME/LatAm/African)"): 0.0013,
    ("JPN", "Other"): 0.0025,
    ("KOR", "Other (incl. foreign workers)"): 0.0030,
    ("HKG", "Other"): 0.0024,
    ("SGP", "Other"): 0.0022,
    ("MYS", "Other"): 0.0016,
    ("BRN", "Other foreign workers"): 0.0016,
    ("THA", "Other"): 0.0005,
    ("MDV", "Other"): 0.0008,
    ("ARE", "Other foreign"): 0.0015,
    ("QAT", "Other foreign"): 0.0014,
    ("SAU", "Sudanese"): 0.0015, ("SAU", "Other foreign"): 0.0010,
}

# Destination demand is kept separate from policy openness. A country may
# need workers but remain politically restrictive, or recruit temporary labor
# without offering durable settlement.
DESTINATION_LABOR_DEMAND: dict[str, float] = {
    "CAN": 0.96, "AUS": 0.92, "NZL": 0.82, "USA": 0.86,
    "GBR": 0.88, "DEU": 0.94, "FRA": 0.80, "NLD": 0.88,
    "IRL": 0.90, "BEL": 0.84, "AUT": 0.84, "CHE": 0.86,
    "FIN": 0.82, "SWE": 0.78, "NOR": 0.80, "DNK": 0.78,
    "ITA": 0.84, "ESP": 0.82, "PRT": 0.74, "GRC": 0.72,
    "JPN": 0.94, "KOR": 0.96, "SGP": 0.92, "TWN": 0.90,
    "HKG": 0.90, "THA": 0.82, "MYS": 0.78, "BRN": 0.80,
    "MDV": 0.76,
    "ARE": 0.96, "QAT": 0.94, "KWT": 0.88, "SAU": 0.90,
    "OMN": 0.82, "BHR": 0.84,
}

# Policy willingness to use broader labor pathways, distinct from high-skill
# points systems. Restrictive systems can have severe labor shortages while
# admitting relatively few permanent workers; temporary-labor systems receive
# lower settlement retention below.
BROAD_LABOR_DESTINATION_OPENNESS: dict[str, float] = {
    "CAN": 1.14, "AUS": 1.06, "NZL": 1.02, "USA": 0.90,
    "GBR": 1.00, "DEU": 1.04, "FRA": 0.94, "NLD": 0.98,
    "BEL": 0.96, "IRL": 1.04, "SWE": 0.94, "NOR": 0.92,
    "DNK": 0.86, "FIN": 0.88, "ITA": 1.00, "ESP": 0.98,
    "PRT": 0.92, "GRC": 0.82, "AUT": 0.88, "CHE": 0.92,
    "JPN": 0.72, "KOR": 0.78, "SGP": 0.94, "HKG": 0.86,
    "THA": 0.70, "MYS": 0.82, "BRN": 0.88, "MDV": 0.84,
    "ARE": 1.12, "QAT": 1.08, "KWT": 0.98, "SAU": 1.00,
    "OMN": 0.94, "BHR": 0.96,
}

# Share of the destination's late broad-labor response plausibly connected to
# Sub-Saharan African source corridors. This is highest where language,
# diaspora, recruitment or historical ties already lower movement costs.
SSA_LATE_MIGRATION_DESTINATION_EXPOSURE: dict[str, float] = {
    "CAN": 0.82, "USA": 0.68, "GBR": 0.96, "FRA": 1.00,
    "DEU": 0.48, "ITA": 0.78, "ESP": 0.38, "NLD": 0.76,
    "BEL": 0.94, "IRL": 0.92, "SWE": 0.84, "NOR": 0.82,
    "DNK": 0.76, "FIN": 0.68, "AUT": 0.46, "CHE": 0.64,
    "PRT": 0.86, "GRC": 0.34, "AUS": 0.42, "NZL": 0.38,
    "JPN": 0.24, "KOR": 0.28, "SGP": 0.36, "HKG": 0.30,
    "THA": 0.16, "MYS": 0.24, "BRN": 0.22, "MDV": 0.18,
    "ARE": 0.54, "QAT": 0.50, "KWT": 0.44, "SAU": 0.56,
    "OMN": 0.42, "BHR": 0.42,
}

# Fraction of a corridor's gross inflow expected to remain in the destination
# long enough to affect the 2050 resident composition. This prevents temporary
# Gulf and circular-labor systems from being treated like settlement migration.
DESTINATION_SETTLEMENT_RETENTION: dict[str, float] = {
    "ARE": 0.46, "QAT": 0.42, "KWT": 0.48, "SAU": 0.44,
    "OMN": 0.48, "BHR": 0.50, "SGP": 0.58, "HKG": 0.58,
    "THA": 0.52, "MYS": 0.56, "BRN": 0.46, "MDV": 0.44,
    "JPN": 0.68, "KOR": 0.66,
    "CAN": 0.92, "AUS": 0.90, "NZL": 0.88, "USA": 0.88,
    "GBR": 0.84, "DEU": 0.80, "FRA": 0.80,
}

# Existing networks, language, recruitment systems and travel costs. Missing
# pairs remain possible at 1.0; values above one describe established or
# institutionally favored corridors rather than deterministic flows.
CORRIDOR_AFFINITY: dict[tuple[str, str], float] = {
    ("CAN", "IND"): 1.35, ("CAN", "PAK"): 1.28, ("CAN", "PHL"): 1.28,
    ("CAN", "NGA"): 1.22, ("CAN", "GHA"): 1.16, ("CAN", "CHN"): 1.14,
    ("GBR", "IND"): 1.30, ("GBR", "PAK"): 1.32, ("GBR", "NGA"): 1.25,
    ("GBR", "GHA"): 1.18, ("GBR", "BGD"): 1.22, ("GBR", "KEN"): 1.15,
    ("USA", "MEX"): 1.36, ("USA", "GTM"): 1.24, ("USA", "HND"): 1.22,
    ("USA", "IND"): 1.20, ("USA", "PHL"): 1.18, ("USA", "NGA"): 1.12,
    ("AUS", "IND"): 1.28, ("AUS", "PHL"): 1.24, ("AUS", "VNM"): 1.18,
    ("DEU", "UKR"): 1.24, ("DEU", "SYR"): 1.18, ("DEU", "TUR"): 1.18,
    ("FRA", "DZA"): 1.30, ("FRA", "MAR"): 1.28, ("FRA", "TUN"): 1.22,
    ("ESP", "MAR"): 1.24, ("ESP", "VEN"): 1.28, ("ESP", "COL"): 1.24,
    ("ITA", "EGY"): 1.18, ("ITA", "BGD"): 1.16, ("ITA", "PAK"): 1.15,
    ("JPN", "VNM"): 1.28, ("JPN", "PHL"): 1.20, ("JPN", "IDN"): 1.16,
    ("KOR", "VNM"): 1.26, ("KOR", "PHL"): 1.18, ("KOR", "IDN"): 1.14,
    ("ARE", "IND"): 1.34, ("ARE", "PAK"): 1.30, ("ARE", "BGD"): 1.26,
    ("QAT", "IND"): 1.28, ("QAT", "PAK"): 1.26, ("SAU", "EGY"): 1.22,
    ("FRA", "SEN"): 1.25, ("FRA", "CIV"): 1.22,
    ("FRA", "CMR"): 1.20, ("FRA", "COD"): 1.16,
    ("BEL", "COD"): 1.25, ("BEL", "CMR"): 1.15,
    ("ITA", "NGA"): 1.18, ("PRT", "AGO"): 1.24,
    ("IRL", "NGA"): 1.22, ("GBR", "UGA"): 1.12,
    ("GBR", "ZWE"): 1.14, ("CAN", "KEN"): 1.15,
    ("CAN", "ETH"): 1.12, ("USA", "GHA"): 1.10,
    ("USA", "KEN"): 1.08, ("SWE", "SOM"): 1.20,
    ("SWE", "ERI"): 1.18, ("NOR", "SOM"): 1.18,
    ("DNK", "SOM"): 1.15, ("FIN", "SOM"): 1.14,
    ("ARE", "ETH"): 1.12, ("ARE", "KEN"): 1.12,
    ("QAT", "KEN"): 1.10, ("SAU", "SDN"): 1.18,
    ("JPN", "KEN"): 1.04, ("KOR", "NGA"): 1.04,
    ("HKG", "NGA"): 1.02, ("SGP", "KEN"): 1.05,
    ("MYS", "NGA"): 1.02, ("BRN", "GHA"): 1.02,
}

GROUP_MIGRATION_ORIGIN_OVERRIDES: dict[tuple[str, str], tuple[str, ...]] = {
    key: SSA_SOURCE_POOL_COUNTRIES for key in GROUP_SSA_SOURCE_POOL_SURGE
}

MIGRATION_ORIGIN_RULES: list[tuple[re.Pattern, tuple[str, ...]]] = [
    (re.compile(r"nigerian", re.I), ("NGA",)),
    (re.compile(r"west african", re.I), ("NGA", "GHA", "SEN", "CIV", "CMR")),
    (re.compile(r"sub-saharan african|black african|black \(african\)|african immigrant|african \(other\)", re.I),
     ("NGA", "GHA", "KEN", "UGA", "TZA", "ETH", "SEN", "CIV", "CMR", "COD")),
    (re.compile(r"palop|angolan|cape verdean", re.I), ("AGO", "CPV", "MOZ")),
    (re.compile(r"ethiopian|eritrean|horn of africa", re.I), ("ETH",)),
    (re.compile(r"somali", re.I), ("SOM", "ETH", "KEN")),
    (re.compile(r"pakistani", re.I), ("PAK",)),
    (re.compile(r"bangladeshi", re.I), ("BGD",)),
    (re.compile(r"indian|south asian", re.I), ("IND", "PAK", "BGD", "LKA", "NPL")),
    (re.compile(r"filipino", re.I), ("PHL",)),
    (re.compile(r"vietnamese", re.I), ("VNM",)),
    (re.compile(r"indonesian", re.I), ("IDN",)),
    (re.compile(r"egyptian", re.I), ("EGY",)),
    (re.compile(r"moroccan", re.I), ("MAR",)),
    (re.compile(r"algerian", re.I), ("DZA",)),
    (re.compile(r"tunisian", re.I), ("TUN",)),
    (re.compile(r"syrian", re.I), ("SYR",)),
    (re.compile(r"afghan", re.I), ("AFG",)),
    (re.compile(r"iraqi", re.I), ("IRQ",)),
    (re.compile(r"ukrainian", re.I), ("UKR",)),
    (re.compile(r"mexican", re.I), ("MEX",)),
    (re.compile(r"guatemalan", re.I), ("GTM",)),
    (re.compile(r"honduran", re.I), ("HND",)),
    (re.compile(r"salvadoran", re.I), ("SLV",)),
    (re.compile(r"venezuelan", re.I), ("VEN",)),
    (re.compile(r"colombian", re.I), ("COL",)),
    (re.compile(r"haitian", re.I), ("HTI",)),
    (re.compile(r"chinese", re.I), ("CHN",)),
]


def infer_migration_origins(
    group_name: str, destination_iso3: str | None = None
) -> tuple[str, ...]:
    """Infer plausible origin countries from a destination-group label."""
    if destination_iso3 is not None:
        override = GROUP_MIGRATION_ORIGIN_OVERRIDES.get(
            (destination_iso3, group_name))
        if override:
            return override
    for pattern, origins in MIGRATION_ORIGIN_RULES:
        if pattern.search(group_name):
            return origins
    return ()


def migration_corridor_diagnostics(
    destination_iso3: str,
    group_name: str,
    profile: object,
    baseline_share: float,
    progress: float,
    demographic_need: float = 0.0,
    birth_replacement_need: float = 0.0,
) -> dict[str, float | str]:
    """Return an auditable multiplier for one origin-destination corridor.

    The result combines source demographic supply and skill readiness with
    destination labor demand, policy openness, diaspora depth, bilateral
    affinity and long-run settlement retention. It intentionally avoids using
    ethnicity itself as a causal variable.
    """
    origins = infer_migration_origins(group_name, destination_iso3)
    late = late_skilled_migration_curve(progress)
    if origins:
        capacities = [SOURCE_MIGRATION_CAPACITY.get(origin, (0.50, 0.58, 0.45)) for origin in origins]
        youth = float(np.mean([item[0] for item in capacities]))
        skill = float(np.mean([item[1] for item in capacities]))
        climate = float(np.mean([item[2] for item in capacities]))
        affinity = float(np.mean([CORRIDOR_AFFINITY.get((destination_iso3, origin), 1.0) for origin in origins]))
    else:
        youth, skill, climate, affinity = (0.48, 0.56, 0.42, 1.0)

    # Skill-selective movement becomes more important in the 2040s; climate
    # stress raises regional/forced movement but is discounted for long-run
    # settlement because much displacement remains internal or temporary.
    source_supply = float(np.clip(
        0.44 * youth + (0.30 + 0.14 * late) * skill +
        (0.16 - 0.06 * late) * climate,
        0.0, 1.0,
    ))
    broad_labor_mobility = float(np.clip(
        0.58 * youth + 0.18 * skill + 0.24 * climate,
        0.0, 1.0,
    ))
    openness = MIGRATION_POLICY_OPENNESS.get(destination_iso3, 1.0)
    labor_demand = DESTINATION_LABOR_DEMAND.get(destination_iso3, 0.58)
    combined_demographic_need = float(np.clip(
        demographic_need + 0.45 * birth_replacement_need, 0.0, 1.0))
    destination_pull = float(np.clip(
        0.38 * labor_demand + 0.34 * combined_demographic_need +
        0.28 * min(1.0, openness / 1.2),
        0.0, 1.0,
    ))
    # Established diasporas reduce information, financing and credential-risk
    # costs. Log scaling avoids making already-large groups self-perpetuating.
    diaspora_network = float(np.clip(
        0.82 + 0.12 * math.log1p(max(0.0, baseline_share) * 100.0),
        0.82, 1.30,
    ))
    retention = DESTINATION_SETTLEMENT_RETENTION.get(destination_iso3, 0.78)
    corridor_multiplier = float(np.clip(
        0.55 + source_supply * destination_pull * affinity *
        diaspora_network * (0.55 + 0.55 * retention),
        0.55, 2.25,
    ))
    return {
        "origins": ";".join(origins) if origins else "unspecified",
        "source_supply": source_supply,
        "broad_labor_mobility": broad_labor_mobility,
        "skill_readiness": skill,
        "climate_pressure": climate,
        "destination_pull": destination_pull,
        "diaspora_network": diaspora_network,
        "corridor_affinity": affinity,
        "settlement_retention": retention,
        "corridor_multiplier": corridor_multiplier,
    }

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

# Age-structure deviations from each country's national pyramid. Values are
# applied directly to the 0-14, 15-64 and 65+ shares, then renormalized. They
# replace the older profile-growth shortcut with explicit proportional age
# exposure. Group-specific youth overrides above refine the 0-14 component.
PROFILE_AGE_STRUCTURE_DEVIATION: dict[object, tuple[float, float, float]] = {
    "majority": (0.00, 0.00, 0.00),
    "high_fertility": (0.085, -0.045, -0.040),
    "low_fertility": (-0.045, -0.005, 0.050),
    "immigrant": (0.015, 0.070, -0.085),
    "assimilating": (0.025, 0.010, -0.035),
}

# Soft ceilings for the combined migration-linked resident stock. The model
# does not impose a hard demographic cap: it progressively dampens additional
# inflow as housing, infrastructure and integration systems become binding.
MIGRATION_STOCK_SOFT_CAP: dict[str, float] = {
    "CAN": 0.42, "AUS": 0.44, "NZL": 0.42, "USA": 0.36,
    "GBR": 0.34, "DEU": 0.30, "FRA": 0.30, "NLD": 0.34,
    "BEL": 0.32, "IRL": 0.38, "SWE": 0.34, "NOR": 0.32,
    "DNK": 0.30, "FIN": 0.26, "ITA": 0.28, "ESP": 0.30,
    "PRT": 0.28, "AUT": 0.29, "CHE": 0.36,
    "JPN": 0.16, "KOR": 0.18, "HKG": 0.34, "SGP": 0.52,
    "MYS": 0.28, "THA": 0.20, "BRN": 0.58, "MDV": 0.48,
    "ARE": 0.86, "QAT": 0.88, "KWT": 0.82, "SAU": 0.58,
    "OMN": 0.56, "BHR": 0.64,
}


def migration_stock_soft_cap(iso3: str) -> float:
    """Return an auditable soft ceiling for migration-linked resident stock."""
    if iso3 in MIGRATION_STOCK_SOFT_CAP:
        return MIGRATION_STOCK_SOFT_CAP[iso3]
    openness = MIGRATION_POLICY_OPENNESS.get(iso3, 1.0)
    baseline = COUNTRY_MIGRATION_INTENSITY.get(iso3, 1.0)
    return float(np.clip(
        0.12 + 0.10 * min(1.0, openness / 1.2) +
        0.12 * min(1.0, baseline / 2.5),
        0.14, 0.38,
    ))


def migration_absorption_capacity(
    iso3: str,
    education_index: Optional[float] = None,
    income_index: Optional[float] = None,
) -> float:
    """Capacity to convert recruitment into durable, integrated settlement."""
    development = 0.58
    if education_index is not None and income_index is not None:
        development = float(np.clip(
            (education_index + income_index) / 2.0, 0.0, 1.0))
    openness = min(1.0, MIGRATION_POLICY_OPENNESS.get(iso3, 1.0) / 1.2)
    retention = DESTINATION_SETTLEMENT_RETENTION.get(iso3, 0.78)
    return float(np.clip(
        0.30 + 0.32 * development + 0.18 * openness + 0.20 * retention,
        0.35, 1.0,
    ))


def _share_fraction(value: Optional[float], fallback: float) -> float:
    if value is None or not np.isfinite(value):
        return fallback
    numeric = float(value)
    return float(np.clip(numeric / 100.0 if numeric > 1.0 else numeric, 0.0, 1.0))


def _normalise_age_structure(
    youth: float, working: float, elderly: float
) -> tuple[float, float, float]:
    values = np.maximum(np.array([youth, working, elderly], dtype=float), 0.005)
    values /= values.sum()
    return tuple(float(value) for value in values)


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


def europe_birth_replacement_pressure(
    iso3: str,
    pop_2024: Optional[float] = None,
    pop_2050: Optional[float] = None,
) -> float:
    """European labor-replacement pressure from smaller future birth cohorts.

    A modest TFR recovery does not immediately restore births because the
    number of women entering child-bearing ages can keep shrinking. The proxy
    therefore combines the 2050 fertility gap with projected population
    contraction. It is only active for countries listed in
    ``EUROPE_BIRTH_REPLACEMENT_RESPONSE`` and remains a scenario mechanism,
    not a forecast of exact births or migrant counts.
    """
    if iso3 not in EUROPE_BIRTH_REPLACEMENT_RESPONSE:
        return 0.0
    future_tfr = NATIONAL_TFR_2050.get(
        iso3, NATIONAL_TFR_2024.get(iso3, REPLACEMENT_TFR))
    future_gap = float(np.clip(
        (REPLACEMENT_TFR - future_tfr) / (REPLACEMENT_TFR - 0.7),
        0.0, 1.0,
    ))
    cohort_contraction = 0.0
    if pop_2024 and pop_2050 and pop_2024 > 0:
        total_change = max(0.0, 1.0 - pop_2050 / pop_2024)
        cohort_contraction = float(np.clip(total_change / 0.22, 0.0, 1.0))
    return float(np.clip(
        0.68 * future_gap + 0.32 * cohort_contraction,
        0.0, 1.0,
    ))


def europe_birth_replacement_migration_boost(
    iso3: str,
    progress: float,
    pop_2024: Optional[float] = None,
    pop_2050: Optional[float] = None,
) -> float:
    """Additional migration intensity as European birth cohorts thin.

    The response is delayed and nonlinear because labor shortages, electoral
    policy, recruitment systems, housing, and integration capacity adjust with
    a lag. Country response multipliers prevent demographic need from being
    treated as automatic migration.
    """
    response = EUROPE_BIRTH_REPLACEMENT_RESPONSE.get(iso3, 0.0)
    if response <= 0.0:
        return 0.0
    pressure = europe_birth_replacement_pressure(iso3, pop_2024, pop_2050)
    late_response = float(np.clip(progress, 0.0, 1.0) ** 1.45)
    return 0.62 * pressure * response * late_response


def effective_migration_intensity(iso3: str, progress: float,
                                  pop_2024: Optional[float] = None,
                                  pop_2050: Optional[float] = None) -> float:
    """Country's net-migration intensity in a given year of the projection.

    Baseline is the static expatriate/settlement intensity
    (``COUNTRY_MIGRATION_INTENSITY``). On top of it, demographic pressure
    adds an inflow need that *ramps up* over the window (``progress`` in
    [0,1]) as aging and population loss deepen, scaled by policy openness.
    European countries also receive a delayed birth-cohort replacement term.

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
    birth_replacement_boost = europe_birth_replacement_migration_boost(
        iso3, progress, pop_2024, pop_2050)
    return (
        base + boost + birth_replacement_boost
    ) * MIGRATION_IMPACT_MULTIPLIER



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
    ssa_source_pool_scale: float = 1.0,
    youth_share_2024: Optional[float] = None,
    youth_share_2050: Optional[float] = None,
    working_age_share_2024: Optional[float] = None,
    working_age_share_2050: Optional[float] = None,
    elderly_share_2024: Optional[float] = None,
    elderly_share_2050: Optional[float] = None,
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
    4. **Late migration surge** -- from the late 2030s onward, distinct
       skilled and broader labor-shortage programs connect young source
       regions to destination-specific diaspora or foreign-worker groups.
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
    ssa_source_pool_scale : float
        Multiplier for the late-2030s/2040s shift toward Sub-Saharan Africa
        in the global mobile-labor source pool. Zero provides an auditable
        counterfactual while leaving all other migration channels active.
    youth_share_*, working_age_share_*, elderly_share_* : float | None
        Country age-pyramid proportions for 2024 and 2050. Inputs may be
        fractions or percentages. They directly scale group cohort momentum.

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
    ssa_source_pool_scale = float(np.clip(ssa_source_pool_scale, 0.0, 2.0))

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
    birth_replacement_pressure = europe_birth_replacement_pressure(
        iso3, pop_2024, pop_2050)
    mig_openness = MIGRATION_POLICY_OPENNESS.get(iso3, 1.0)
    absorption_capacity = migration_absorption_capacity(
        iso3, education_index, income_index)
    migration_soft_cap = migration_stock_soft_cap(iso3)

    anchor = int(np.argmax(shares))
    years = end_year - start_year

    fallback_youth_0 = float(np.clip(0.18 + 0.075 * (tfr_nat_0 - 1.6), 0.08, 0.46))
    fallback_old_0 = float(np.clip(0.18 - 0.10 * (tfr_nat_0 - 1.6), 0.02, 0.34))
    fallback_work_0 = 1.0 - fallback_youth_0 - fallback_old_0
    fallback_youth_1 = float(np.clip(0.17 + 0.070 * (tfr_nat_1 - 1.6), 0.07, 0.42))
    fallback_old_1 = float(np.clip(0.21 - 0.09 * (tfr_nat_1 - 1.6), 0.03, 0.38))
    fallback_work_1 = 1.0 - fallback_youth_1 - fallback_old_1
    national_age_0 = _normalise_age_structure(
        _share_fraction(youth_share_2024, fallback_youth_0),
        _share_fraction(working_age_share_2024, fallback_work_0),
        _share_fraction(elderly_share_2024, fallback_old_0),
    )
    national_age_1 = _normalise_age_structure(
        _share_fraction(youth_share_2050, fallback_youth_1),
        _share_fraction(working_age_share_2050, fallback_work_1),
        _share_fraction(elderly_share_2050, fallback_old_1),
    )
    group_age_0 = []
    group_age_1 = []
    for entry in entries:
        dy, dw, do = PROFILE_AGE_STRUCTURE_DEVIATION.get(
            entry[2], (0.0, 0.0, 0.0))
        dy += GROUP_YOUTH_BONUS.get((iso3, entry[0]), 0.0) * 0.55
        group_age_0.append(_normalise_age_structure(
            national_age_0[0] + dy,
            national_age_0[1] + dw,
            national_age_0[2] + do,
        ))
        # Most age-structure gaps narrow as migrant-origin and minority
        # populations age and fertility converges, but do not disappear.
        group_age_1.append(_normalise_age_structure(
            national_age_1[0] + 0.38 * dy,
            national_age_1[1] + 0.38 * dw,
            national_age_1[2] + 0.38 * do,
        ))
    group_age_0 = np.asarray(group_age_0, dtype=float)
    group_age_1 = np.asarray(group_age_1, dtype=float)

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

        # 4) Explicit proportional age structure. Cohort momentum derives from
        #    the group's 0-14, working-age and 65+ shares relative to the
        #    national pyramid, rather than a categorical growth label.
        national_age = np.asarray(national_age_0) + (
            np.asarray(national_age_1) - np.asarray(national_age_0)) * progress
        group_age = group_age_0 + (group_age_1 - group_age_0) * progress
        age_exposure = (
            0.78 * (group_age[:, 0] - national_age[0]) +
            0.18 * (group_age[:, 1] - national_age[1]) -
            0.22 * (group_age[:, 2] - national_age[2])
        )
        momentum = age_exposure * np.exp(-progress * 1.35) / GENERATION_LENGTH

        # 5) Migration inflow. Each country imports a specific *composition* of
        #    groups (GROUP_MIGRATION_INTENSITY); unlisted immigrant-profile
        #    groups get the default coefficient. Intensity is the static
        #    expatriate/settlement baseline plus a demographic-pressure boost
        #    that ramps up as labour shortages deepen.
        mig_intensity = effective_migration_intensity(
            iso3, progress, pop_2024, pop_2050)
        skilled_program = skilled_migration_program_intensity(iso3)
        broad_labor_program = broad_labor_migration_program_intensity(iso3)
        late_skilled = late_skilled_migration_curve(progress)
        late_broad_labor = late_broad_labor_migration_curve(progress)
        late_ssa_pool = late_ssa_source_pool_transition(progress)
        dev_mig = np.zeros_like(shares)
        direct_ssa_inflow = np.zeros_like(shares)
        migration_linked_share = float(sum(
            shares[i] for i, entry in enumerate(entries)
            if entry[2] == "immigrant"))
        saturation_ratio = migration_linked_share / max(migration_soft_cap, 0.05)
        migration_saturation = float(np.clip(
            1.0 - 0.62 * saturation_ratio ** 1.6, 0.22, 1.0))
        for i, e in enumerate(entries):
            coeff = GROUP_MIGRATION_INTENSITY.get((iso3, e[0]))
            corridor = migration_corridor_diagnostics(
                iso3, e[0], e[2], e[1] / 100.0, progress, pressure,
                birth_replacement_pressure)
            corridor_multiplier = float(corridor["corridor_multiplier"])
            if coeff is not None:
                dev_mig[i] = (
                    coeff * mig_scale * mig_intensity * corridor_multiplier
                )
            elif e[2] == "immigrant":
                dev_mig[i] = (
                    0.0040 * mig_scale * mig_intensity *
                    corridor_multiplier
                )
            elif (e[0] == "Other" and pressure > 0.30
                  and e[2] in ("assimilating", "majority")):
                dev_mig[i] = 0.0040 * mig_scale * mig_intensity * 0.5
            skilled_coeff = GROUP_SKILLED_MIGRATION_SURGE.get((iso3, e[0]))
            if skilled_coeff is not None:
                dev_mig[i] += (
                    skilled_coeff * mig_scale * skilled_program *
                    late_skilled *
                    (0.65 + 0.55 * float(corridor["skill_readiness"])) *
                    float(corridor["settlement_retention"])
                )
            broad_labor_coeff = GROUP_BROAD_LABOR_MIGRATION_SURGE.get(
                (iso3, e[0]))
            if broad_labor_coeff is not None:
                dev_mig[i] += (
                    broad_labor_coeff * mig_scale * broad_labor_program *
                    late_broad_labor *
                    (0.68 + 0.52 * float(corridor["broad_labor_mobility"])) *
                    float(corridor["settlement_retention"])
                )
            ssa_pool_coeff = GROUP_SSA_SOURCE_POOL_SURGE.get((iso3, e[0]))
            if ssa_pool_coeff is not None:
                # This is a resident-stock inflow, not a relative growth rate
                # on the pre-existing diaspora. That distinction lets a new
                # corridor emerge from a small baseline without exploding it.
                direct_ssa_inflow[i] = (
                    ssa_pool_coeff * ssa_source_pool_scale * mig_scale *
                    ssa_late_migration_destination_response(iso3) *
                    late_ssa_pool *
                    (0.62 + 0.48 * ssa_source_pool_capacity()) *
                    float(corridor["settlement_retention"]) * 0.35
                )
            # Guard against a small baseline diaspora acquiring an implausible
            # annual growth rate from several overlapping migration channels.
            dev_mig[i] = float(np.clip(
                dev_mig[i] * migration_saturation * absorption_capacity,
                0.0, 0.036))
            direct_ssa_inflow[i] *= migration_saturation * absorption_capacity

        # 6) Intermarriage: a fraction of the non-mixed population forms new
        #    mixed-identity people each year (children of mixed unions are
        #    increasingly identified as multiracial). This is the engine
        #    behind the fast-growing Mixed/Multiracial buckets in the US,
        #    Canada, Australia, NZ, the UK and northern Europe.
        direct_mixed_inflow = np.zeros_like(shares)
        if intermarriage > 0.0 and mixed_mask.sum() > 0.0:
            mixed_share = float((shares * mixed_mask).sum())
            pool = 1.0 - mixed_share
            inflow = min(
                pool * 0.008,
                intermarriage * 0.006 * pool * mixed_identity_multiplier,
            )
            direct_mixed_inflow[mixed_mask > 0.0] = (
                inflow / max(1.0, mixed_mask.sum()))

        dev = dev_fert + momentum + dev_mig

        shares = shares * (1.0 + dev) + direct_ssa_inflow
        if direct_mixed_inflow.sum() > 0.0:
            transfer = float(direct_mixed_inflow.sum())
            nonmixed = mixed_mask == 0.0
            nonmixed_total = float(shares[nonmixed].sum())
            if nonmixed_total > transfer:
                shares[nonmixed] *= (nonmixed_total - transfer) / nonmixed_total
                shares += direct_mixed_inflow

        # 7) Identity transition applies only to migration-linked or explicitly
        #    transitional statistical categories. Long-established minorities
        #    are not mechanically transferred toward the largest category.
        if assimilation_rate > 0.0:
            for i in range(len(shares)):
                if i == anchor or entries[i][2] not in ("immigrant", "assimilating"):
                    continue
                transfer = shares[i] * assimilation_rate
                shares[i] -= transfer
                if mixed_mask.sum() > 0.0 and entries[i][2] == "immigrant":
                    mixed_transfer = transfer * min(0.65, 0.25 + intermarriage)
                    shares[mixed_mask > 0.0] += mixed_transfer / mixed_mask.sum()
                    shares[anchor] += transfer - mixed_transfer
                else:
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
