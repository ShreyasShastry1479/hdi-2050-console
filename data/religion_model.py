"""Scenario-style religious composition projection to 2050.

The model is intentionally lightweight and transparent. It provides a
country-level religion layer for the 2050 Mosaic rather than an official
demographic forecast. Baselines use broad regional defaults plus country
overrides for the largest or structurally distinctive cases, then apply
simple 2024 -> 2050 adjustments for fertility, secularization, migration, and
HDI-linked education/income convergence.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Mapping

import pandas as pd

from data.countries import COUNTRY_NAMES
from data.undp_hdi import UNDP_HDI_COUNTRIES_193

MUSLIM_2025_REFERENCE_PATH = Path(__file__).resolve().parent / "muslim_population_2025_reference.csv"
HINDU_2025_REFERENCE_PATH = Path(__file__).resolve().parent / "hindu_population_2025_reference.csv"

RELIGIONS = [
    "Catholic Christianity",
    "Protestant Christianity",
    "Orthodox Christianity",
    "Other Christianity",
    "African Independent / Syncretic Christianity",
    "Sunni Islam",
    "Shia Islam",
    "Ibadi / Other Islam",
    "Hinduism",
    "Theravada Buddhism",
    "Mahayana Buddhism",
    "Vajrayana / Other Buddhism",
    "Sikhism",
    "Bahai",
    "Zoroastrianism",
    "Folk / Traditional",
    "Indigenous / Syncretic traditions",
    "Judaism",
    "Unaffiliated",
    "Other religions",
]

MUSLIM_MAJORITY = {
    "AFG", "ALB", "DZA", "AZE", "BHR", "BGD", "BIH", "BRN", "BFA",
    "TCD", "COM", "DJI", "EGY", "GMB", "GIN", "GNB", "IDN", "IRN",
    "IRQ", "JOR", "KAZ", "KWT", "KGZ", "LBN", "LBY", "MYS", "MDV",
    "MLI", "MRT", "MAR", "NER", "OMN", "PAK", "PSE", "QAT", "SAU",
    "SEN", "SOM", "SDN", "SYR", "TJK", "TUN", "TUR", "TKM", "ARE",
    "UZB", "YEM",
}

SOUTH_ASIA = {"AFG", "BGD", "BTN", "IND", "LKA", "MDV", "NPL", "PAK"}
EAST_ASIA = {
    "CHN", "HKG", "JPN", "KOR", "MNG", "PRK", "TWN", "VNM", "KHM",
    "LAO", "MMR", "THA", "SGP",
}
LATIN_AMERICA = {
    "ARG", "ATG", "BHS", "BLZ", "BOL", "BRA", "BRB", "CHL", "COL",
    "CRI", "CUB", "DMA", "DOM", "ECU", "GRD", "GTM", "GUY", "HND",
    "HTI", "JAM", "KNA", "LCA", "MEX", "NIC", "PAN", "PER", "PRY",
    "SLV", "SUR", "TTO", "URY", "VCT", "VEN",
}
EUROPE = {
    "AND", "AUT", "BEL", "BGR", "BLR", "CHE", "CYP", "CZE", "DEU",
    "DNK", "ESP", "EST", "FIN", "FRA", "GBR", "GRC", "HRV", "HUN",
    "IRL", "ISL", "ITA", "LIE", "LTU", "LUX", "LVA", "MLT", "MCO",
    "MDA", "MKD", "MNE", "NLD", "NOR", "POL", "PRT", "ROU", "RUS",
    "SMR", "SRB", "SVK", "SVN", "SWE", "UKR",
}
SUB_SAHARAN = {
    "AGO", "BDI", "BEN", "BWA", "CAF", "CIV", "CMR", "COD", "COG",
    "CPV", "ERI", "ETH", "GAB", "GHA", "GNQ", "KEN", "LBR", "LSO",
    "MDG", "MUS", "MWI", "MOZ", "NAM", "NGA", "RWA", "SLE", "SSD",
    "STP", "SWZ", "SYC", "TGO", "TZA", "UGA", "ZAF", "ZMB", "ZWE",
}
PACIFIC = {
    "AUS", "FJI", "FSM", "KIR", "MHL", "NRU", "NZL", "PLW", "PNG",
    "SLB", "TON", "TUV", "VUT", "WSM",
}
MENA = {
    "DZA", "EGY", "ISR", "JOR", "LBN", "LBY", "MAR", "TUN", "YEM",
    "PSE", "SAU", "ARE", "QAT", "KWT", "BHR", "OMN", "IRN", "IRQ", "SYR",
}
WESTERN_EUROPE = {
    "AUT", "BEL", "CHE", "DEU", "DNK", "ESP", "FIN", "FRA", "GBR",
    "IRL", "ISL", "ITA", "LUX", "NLD", "NOR", "PRT", "SWE",
}

AFRICA_MUSLIM_BELT = {
    "BFA", "TCD", "COM", "DJI", "ERI", "GMB", "GIN", "GNB", "MLI",
    "MRT", "NER", "SDN", "SEN", "SOM",
}
AFRICA_MIXED_WEST = {"BEN", "CIV", "CMR", "GHA", "NGA", "SLE", "TGO"}
AFRICA_CHRISTIAN_CORE = {
    "AGO", "BDI", "CAF", "COD", "COG", "ETH", "GAB", "GNQ", "KEN",
    "LBR", "MDG", "MOZ", "MWI", "RWA", "SSD", "STP", "TZA", "UGA",
    "ZMB", "ZWE",
}
AFRICA_SOUTHERN = {"BWA", "LSO", "NAM", "SWZ", "ZAF"}

COUNTRY_TARGETS_2050: dict[str, dict[str, float]] = {
    # Pew-style scale correction: India remains strongly Hindu but becomes
    # the world's largest Muslim-population country by sheer population size.
    "IND": {"Hinduism": 77.0, "Islam": 17.0, "Christianity": 2.4, "Sikhism": 1.6, "Buddhism": 0.7, "Zoroastrianism": 0.05, "Other religions": 0.55, "Unaffiliated": 0.7},
    "IDN": {"Islam": 86.2, "Christianity": 10.4, "Hinduism": 1.7, "Buddhism": 0.7, "Folk / Traditional": 0.4, "Other religions": 0.6},
    # North America: reduced but still large Christian majority, larger nones,
    # and Islam overtaking Judaism in the United States.
    "USA": {"Christianity": 66.0, "Unaffiliated": 23.0, "Islam": 2.2, "Judaism": 1.4, "Hinduism": 1.5, "Buddhism": 1.1, "Other religions": 4.8},
    "CAN": {"Christianity": 60.0, "Unaffiliated": 27.0, "Islam": 7.0, "Hinduism": 3.0, "Sikhism": 2.0, "Buddhism": 1.0, "Judaism": 0.6, "Other religions": 0.4},
    # Western Europe medium-migration calibration. These country-level targets
    # mirror the 2050 medium migration path in the user's reference table; the
    # broad Islam share is split into Sunni/Shia/Ibadi downstream.
    "SWE": {"Christianity": 43.0, "Unaffiliated": 32.0, "Islam": 20.0, "Buddhism": 1.0, "Judaism": 0.3, "Hinduism": 0.4, "Other religions": 3.3},
    "FRA": {"Christianity": 50.0, "Unaffiliated": 28.0, "Islam": 17.0, "Buddhism": 1.0, "Judaism": 0.6, "Hinduism": 0.4, "Other religions": 3.0},
    "DEU": {"Christianity": 52.0, "Unaffiliated": 32.0, "Islam": 10.0, "Buddhism": 1.0, "Judaism": 0.3, "Hinduism": 0.4, "Other religions": 4.3},
    "GBR": {"Christianity": 48.0, "Unaffiliated": 28.0, "Islam": 16.0, "Hinduism": 2.2, "Sikhism": 1.2, "Buddhism": 1.0, "Judaism": 0.4, "Other religions": 3.2},
    "BEL": {"Christianity": 48.0, "Unaffiliated": 31.0, "Islam": 15.0, "Buddhism": 0.8, "Judaism": 0.4, "Hinduism": 0.5, "Other religions": 4.3},
    "NLD": {"Christianity": 46.0, "Unaffiliated": 36.0, "Islam": 12.0, "Buddhism": 0.9, "Judaism": 0.3, "Hinduism": 0.5, "Other religions": 4.3},
    "AUT": {"Christianity": 58.0, "Unaffiliated": 24.0, "Islam": 13.0, "Buddhism": 0.7, "Judaism": 0.2, "Hinduism": 0.3, "Other religions": 3.8},
    "ITA": {"Christianity": 68.0, "Unaffiliated": 18.0, "Islam": 9.0, "Buddhism": 0.8, "Judaism": 0.2, "Hinduism": 0.3, "Other religions": 3.7},
    "DNK": {"Christianity": 56.0, "Unaffiliated": 29.0, "Islam": 10.0, "Buddhism": 0.8, "Judaism": 0.2, "Hinduism": 0.3, "Other religions": 3.7},
    # Asia-Pacific Buddhist-ageing cases: Buddhist share is held flat to lower
    # because major Buddhist populations are ageing or below replacement.
    "CHN": {"Unaffiliated": 50.0, "Folk / Traditional": 24.0, "Buddhism": 16.5, "Christianity": 7.0, "Islam": 2.0, "Other religions": 0.5},
    "JPN": {"Unaffiliated": 60.0, "Buddhism": 25.0, "Folk / Traditional": 9.0, "Christianity": 1.5, "Other religions": 4.5},
    "THA": {"Buddhism": 89.0, "Islam": 6.0, "Christianity": 1.5, "Folk / Traditional": 1.2, "Other religions": 0.7, "Unaffiliated": 1.6},
    # MENA remains overwhelmingly Muslim, with smaller minority shares under
    # migration/conflict displacement and urbanization pressure.
    "EGY": {"Islam": 91.5, "Christianity": 8.0, "Other religions": 0.3, "Unaffiliated": 0.2},
    "LBN": {"Islam": 68.0, "Christianity": 27.0, "Unaffiliated": 2.0, "Other religions": 3.0},
    "ISR": {"Judaism": 72.0, "Islam": 20.0, "Christianity": 1.8, "Bahai": 0.1, "Unaffiliated": 3.2, "Other religions": 2.9},
    "IRN": {"Islam": 98.0, "Christianity": 0.25, "Bahai": 0.35, "Zoroastrianism": 0.12, "Judaism": 0.05, "Other religions": 0.33, "Unaffiliated": 0.9},
    # Sub-Saharan anchor cases: Christianity and Islam both grow rapidly, with
    # folk/traditional shares shrinking as identities consolidate.
    "NGA": {"Islam": 52.5, "Christianity": 46.5, "Folk / Traditional": 0.7, "Other religions": 0.2, "Unaffiliated": 0.1},
    "ETH": {"Christianity": 62.5, "Islam": 35.0, "Folk / Traditional": 1.8, "Other religions": 0.4, "Unaffiliated": 0.3},
    "GHA": {"Christianity": 72.0, "Islam": 22.0, "Indigenous / Syncretic traditions": 2.5, "Folk / Traditional": 1.5, "Other religions": 1.0, "Unaffiliated": 1.0},
    "KEN": {"Christianity": 83.0, "Islam": 13.0, "Indigenous / Syncretic traditions": 1.5, "Folk / Traditional": 0.7, "Other religions": 1.0, "Unaffiliated": 0.8},
    "TZA": {"Christianity": 64.0, "Islam": 34.5, "Indigenous / Syncretic traditions": 0.6, "Folk / Traditional": 0.3, "Other religions": 0.4, "Unaffiliated": 0.2},
    "UGA": {"Christianity": 84.0, "Islam": 14.0, "Indigenous / Syncretic traditions": 0.8, "Folk / Traditional": 0.4, "Other religions": 0.5, "Unaffiliated": 0.3},
    "ZAF": {"Christianity": 76.0, "Islam": 3.0, "Hinduism": 1.0, "Indigenous / Syncretic traditions": 8.0, "Folk / Traditional": 4.0, "Unaffiliated": 6.5, "Other religions": 1.5},
    "COD": {"Christianity": 92.0, "Islam": 2.0, "Indigenous / Syncretic traditions": 3.0, "Folk / Traditional": 1.8, "Other religions": 0.7, "Unaffiliated": 0.5},
    "SEN": {"Islam": 95.0, "Christianity": 4.0, "Indigenous / Syncretic traditions": 0.5, "Folk / Traditional": 0.2, "Other religions": 0.2, "Unaffiliated": 0.1},
    "SDN": {"Islam": 91.0, "Christianity": 5.0, "Indigenous / Syncretic traditions": 2.0, "Folk / Traditional": 1.0, "Other religions": 0.6, "Unaffiliated": 0.4},
    "SOM": {"Islam": 99.0, "Christianity": 0.2, "Indigenous / Syncretic traditions": 0.2, "Other religions": 0.3, "Unaffiliated": 0.3},
    "CIV": {"Islam": 44.0, "Christianity": 38.0, "Indigenous / Syncretic traditions": 10.0, "Folk / Traditional": 5.0, "Other religions": 2.0, "Unaffiliated": 1.0},
    "CMR": {"Christianity": 65.0, "Islam": 25.0, "Indigenous / Syncretic traditions": 5.0, "Folk / Traditional": 3.0, "Other religions": 1.2, "Unaffiliated": 0.8},
}

COUNTRY_BASELINES: dict[str, dict[str, float]] = {
    "IND": {"Hinduism": 79.0, "Islam": 14.5, "Christianity": 2.4, "Sikhism": 1.7, "Buddhism": 0.7, "Jainism": 0.35, "Zoroastrianism": 0.05, "Other religions": 0.55, "Unaffiliated": 0.7},
    "CHN": {"Unaffiliated": 52.0, "Folk / Traditional": 22.0, "Buddhism": 18.0, "Christianity": 5.0, "Islam": 2.0, "Other religions": 1.0},
    "USA": {"Christianity": 62.0, "Unaffiliated": 29.0, "Judaism": 2.0, "Islam": 1.4, "Hinduism": 1.1, "Buddhism": 1.0, "Other religions": 3.5},
    "GBR": {"Christianity": 46.0, "Unaffiliated": 37.0, "Islam": 6.5, "Hinduism": 1.65, "Sikhism": 0.9, "Buddhism": 0.5, "Judaism": 0.5, "Other religions": 6.95},
    "BRA": {"Christianity": 82.0, "Unaffiliated": 9.0, "Folk / Traditional": 4.0, "Other religions": 3.0, "Islam": 0.2, "Judaism": 0.1, "Buddhism": 0.2, "Hinduism": 0.1},
    "NGA": {"Islam": 52.0, "Christianity": 46.0, "Folk / Traditional": 1.5, "Other religions": 0.3, "Unaffiliated": 0.2},
    "GHA": {"Christianity": 71.0, "Islam": 20.0, "Indigenous / Syncretic traditions": 4.0, "Folk / Traditional": 3.0, "Other religions": 1.0, "Unaffiliated": 1.0},
    "KEN": {"Christianity": 84.0, "Islam": 11.0, "Indigenous / Syncretic traditions": 2.0, "Folk / Traditional": 1.0, "Other religions": 1.0, "Unaffiliated": 1.0},
    "TZA": {"Christianity": 63.0, "Islam": 35.0, "Indigenous / Syncretic traditions": 0.8, "Folk / Traditional": 0.7, "Other religions": 0.3, "Unaffiliated": 0.2},
    "UGA": {"Christianity": 84.0, "Islam": 14.0, "Indigenous / Syncretic traditions": 0.8, "Folk / Traditional": 0.5, "Other religions": 0.4, "Unaffiliated": 0.3},
    "ZAF": {"Christianity": 78.0, "Islam": 2.0, "Hinduism": 1.0, "Indigenous / Syncretic traditions": 7.0, "Folk / Traditional": 4.0, "Unaffiliated": 7.0, "Other religions": 1.0},
    "COD": {"Christianity": 92.0, "Islam": 1.8, "Indigenous / Syncretic traditions": 3.2, "Folk / Traditional": 2.0, "Other religions": 0.6, "Unaffiliated": 0.4},
    "SEN": {"Islam": 95.0, "Christianity": 4.2, "Indigenous / Syncretic traditions": 0.4, "Folk / Traditional": 0.2, "Other religions": 0.1, "Unaffiliated": 0.1},
    "SDN": {"Islam": 91.0, "Christianity": 5.0, "Indigenous / Syncretic traditions": 2.0, "Folk / Traditional": 1.0, "Other religions": 0.6, "Unaffiliated": 0.4},
    "SOM": {"Islam": 99.0, "Christianity": 0.2, "Indigenous / Syncretic traditions": 0.2, "Other religions": 0.3, "Unaffiliated": 0.3},
    "CIV": {"Islam": 44.0, "Christianity": 38.0, "Indigenous / Syncretic traditions": 11.0, "Folk / Traditional": 5.0, "Other religions": 1.2, "Unaffiliated": 0.8},
    "CMR": {"Christianity": 64.0, "Islam": 25.0, "Indigenous / Syncretic traditions": 6.0, "Folk / Traditional": 3.5, "Other religions": 1.0, "Unaffiliated": 0.5},
    "IDN": {"Islam": 87.0, "Christianity": 10.0, "Hinduism": 1.7, "Buddhism": 0.7, "Folk / Traditional": 0.3, "Other religions": 0.3},
    "PAK": {"Islam": 96.5, "Hinduism": 1.9, "Christianity": 1.3, "Other religions": 0.2, "Unaffiliated": 0.1},
    "BGD": {"Islam": 91.0, "Hinduism": 8.0, "Buddhism": 0.6, "Christianity": 0.3, "Other religions": 0.1},
    "IRN": {"Islam": 98.4, "Christianity": 0.3, "Bahai": 0.3, "Zoroastrianism": 0.1, "Judaism": 0.05, "Other religions": 0.35, "Unaffiliated": 0.5},
    "JPN": {"Unaffiliated": 57.0, "Buddhism": 28.0, "Folk / Traditional": 10.0, "Christianity": 1.5, "Other religions": 3.5},
    "RUS": {"Christianity": 71.0, "Islam": 10.0, "Unaffiliated": 15.0, "Buddhism": 1.0, "Folk / Traditional": 1.0, "Other religions": 2.0},
    "MEX": {"Christianity": 86.0, "Unaffiliated": 8.0, "Folk / Traditional": 3.0, "Other religions": 2.5, "Judaism": 0.1, "Islam": 0.1, "Buddhism": 0.2, "Hinduism": 0.1},
    "ETH": {"Christianity": 63.0, "Islam": 34.0, "Folk / Traditional": 2.5, "Other religions": 0.3, "Unaffiliated": 0.2},
    "EGY": {"Islam": 90.0, "Christianity": 9.5, "Other religions": 0.3, "Unaffiliated": 0.2},
    "PHL": {"Christianity": 86.0, "Islam": 6.0, "Folk / Traditional": 2.0, "Unaffiliated": 4.0, "Other religions": 2.0},
    "VNM": {"Folk / Traditional": 45.0, "Unaffiliated": 30.0, "Buddhism": 16.0, "Christianity": 8.0, "Other religions": 1.0},
    "THA": {"Buddhism": 92.0, "Islam": 5.0, "Christianity": 1.2, "Folk / Traditional": 1.0, "Other religions": 0.5, "Unaffiliated": 0.3},
    "ISR": {"Judaism": 73.0, "Islam": 18.0, "Christianity": 2.0, "Other religions": 4.0, "Unaffiliated": 3.0},
    "TUR": {"Islam": 88.0, "Unaffiliated": 9.0, "Christianity": 0.3, "Other religions": 2.7},
    "CAN": {"Christianity": 53.0, "Unaffiliated": 34.0, "Islam": 5.0, "Hinduism": 2.5, "Sikhism": 2.0, "Buddhism": 1.5, "Judaism": 1.0, "Other religions": 1.0},
    "AUS": {"Christianity": 44.0, "Unaffiliated": 39.0, "Islam": 3.5, "Hinduism": 3.0, "Buddhism": 2.5, "Judaism": 0.5, "Other religions": 7.5},
}

ORTHODOX_COUNTRIES = {
    "ARM", "BLR", "BGR", "CYP", "GEO", "GRC", "MDA", "MNE", "MKD",
    "ROU", "RUS", "SRB", "UKR",
}
PROTESTANT_COUNTRIES = {
    "AUS", "CAN", "DEU", "DNK", "EST", "FIN", "GBR", "ISL", "NLD",
    "NOR", "NZL", "SWE", "USA", "ZAF",
}
CATHOLIC_COUNTRIES = LATIN_AMERICA | {
    "AUT", "BEL", "CHE", "ESP", "FRA", "HRV", "HUN", "IRL", "ITA",
    "LTU", "LUX", "MLT", "POL", "PRT", "SVK", "SVN",
}
SHIA_COUNTRIES = {"IRN", "IRQ", "AZE", "BHR"}
IBADI_COUNTRIES = {"OMN"}
THERAVADA_COUNTRIES = {"KHM", "LAO", "LKA", "MMR", "THA"}
MAHAYANA_COUNTRIES = {"CHN", "HKG", "JPN", "KOR", "PRK", "SGP", "TWN", "VNM"}
VAJRAYANA_COUNTRIES = {"BTN", "MNG", "NPL"}

ALIASES = {"Jainism": "Other religions"}

# Incremental religious composition of future migrant inflows into European
# destinations. These are scenario mixes, not characteristics assigned to
# individuals. They approximate the combined origin pattern visible in the
# ethnic-corridor model: intra-European mobility, North/West/East African
# links, South Asian skilled recruitment, and humanitarian channels.
EUROPE_MIGRANT_RELIGION_MIX: dict[str, dict[str, float]] = {
    "GBR": {"Christianity": 45, "Islam": 28, "Hinduism": 12, "Sikhism": 5, "Unaffiliated": 8, "Other religions": 2},
    "FRA": {"Christianity": 43, "Islam": 39, "Unaffiliated": 11, "Hinduism": 2, "Buddhism": 2, "Other religions": 3},
    "DEU": {"Christianity": 51, "Islam": 29, "Unaffiliated": 14, "Hinduism": 2, "Buddhism": 1, "Other religions": 3},
    "ESP": {"Christianity": 68, "Islam": 18, "Unaffiliated": 10, "Hinduism": 1, "Buddhism": 1, "Other religions": 2},
    "ITA": {"Christianity": 52, "Islam": 28, "Hinduism": 7, "Sikhism": 2, "Unaffiliated": 8, "Other religions": 3},
    "NLD": {"Christianity": 45, "Islam": 29, "Unaffiliated": 18, "Hinduism": 3, "Buddhism": 2, "Other religions": 3},
    "BEL": {"Christianity": 45, "Islam": 34, "Unaffiliated": 14, "Hinduism": 2, "Buddhism": 2, "Other religions": 3},
    "SWE": {"Christianity": 42, "Islam": 35, "Unaffiliated": 16, "Hinduism": 2, "Buddhism": 2, "Other religions": 3},
    "DNK": {"Christianity": 46, "Islam": 31, "Unaffiliated": 16, "Hinduism": 2, "Buddhism": 2, "Other religions": 3},
    "AUT": {"Christianity": 48, "Islam": 34, "Unaffiliated": 12, "Hinduism": 2, "Buddhism": 1, "Other religions": 3},
    "CHE": {"Christianity": 52, "Islam": 22, "Unaffiliated": 17, "Hinduism": 3, "Buddhism": 2, "Other religions": 4},
    "IRL": {"Christianity": 57, "Islam": 17, "Hinduism": 7, "Unaffiliated": 14, "Other religions": 5},
    "PRT": {"Christianity": 66, "Islam": 13, "Unaffiliated": 12, "Hinduism": 3, "Buddhism": 2, "Other religions": 4},
    "GRC": {"Christianity": 57, "Islam": 27, "Unaffiliated": 10, "Hinduism": 1, "Other religions": 5},
}

DEFAULT_EUROPE_MIGRANT_RELIGION_MIX = {
    "Christianity": 52.0,
    "Islam": 25.0,
    "Unaffiliated": 15.0,
    "Hinduism": 2.5,
    "Sikhism": 0.8,
    "Buddhism": 1.7,
    "Other religions": 3.0,
}

# Approximate aggregate religious mix of the additional Sub-Saharan African
# late-period corridor. It is expressed directly in model denominations to
# avoid assigning destination-country Christian sect splits to origin groups.
# The mix is a regional scenario average; country corridors remain uncertain.
SSA_LATE_MIGRANT_RELIGION_MIX = {
    "Catholic Christianity": 19.0,
    "Protestant Christianity": 27.0,
    "Orthodox Christianity": 5.0,
    "African Independent / Syncretic Christianity": 7.0,
    "Other Christianity": 4.0,
    "Sunni Islam": 30.0,
    "Shia Islam": 0.5,
    "Indigenous / Syncretic traditions": 3.0,
    "Unaffiliated": 2.5,
    "Other religions": 2.0,
}


@lru_cache(maxsize=1)
def _muslim_reference_2025() -> dict[str, float]:
    """Country-level 2025 Muslim share reference supplied for projection baselines."""
    if not MUSLIM_2025_REFERENCE_PATH.exists():
        return {}
    df = pd.read_csv(MUSLIM_2025_REFERENCE_PATH)
    out = {}
    for _, row in df.iterrows():
        iso3 = str(row.get("ISO3", "")).strip()
        try:
            pct = float(row.get("Muslim_Pct_2025"))
        except (TypeError, ValueError):
            continue
        if iso3 and 0.0 <= pct <= 100.0:
            out[iso3] = pct
    return out


@lru_cache(maxsize=1)
def _hindu_reference_2025() -> dict[str, float]:
    """Country-level 2025 Hindu share reference supplied for projection baselines."""
    if not HINDU_2025_REFERENCE_PATH.exists():
        return {}
    df = pd.read_csv(HINDU_2025_REFERENCE_PATH)
    out = {}
    for _, row in df.iterrows():
        iso3 = str(row.get("ISO3", "")).strip()
        try:
            pct = float(row.get("Hindu_Pct_2025"))
        except (TypeError, ValueError):
            continue
        if iso3 and 0.0 <= pct <= 100.0:
            out[iso3] = pct
    return out


def _apply_muslim_reference_2025(shares: Mapping[str, float], iso3: str) -> dict[str, float]:
    """Apply supplied 2025 Muslim and Hindu shares to the raw baseline.

    Other religious categories are scaled proportionally so the country baseline
    still sums to 100 before downstream Sunni/Shia/Ibadi/Buddhist splitting.
    """
    muslim_ref = _muslim_reference_2025().get(iso3)
    hindu_ref = _hindu_reference_2025().get(iso3)
    if muslim_ref is None and hindu_ref is None:
        return dict(shares)

    current = {k: float(v) for k, v in shares.items()}
    islam_keys = [k for k in current if k == "Islam" or "Islam" in k]
    current_islam = sum(current[k] for k in islam_keys)
    fixed_keys = set(islam_keys)
    if "Hinduism" in current:
        fixed_keys.add("Hinduism")
    other_keys = [k for k in current if k not in fixed_keys]
    other_total = sum(current[k] for k in other_keys)

    adjusted: dict[str, float] = {}
    fixed_total = 0.0
    if muslim_ref is not None:
        if islam_keys and current_islam > 0:
            for key in islam_keys:
                adjusted[key] = muslim_ref * current[key] / current_islam
        else:
            adjusted["Islam"] = muslim_ref
        fixed_total += muslim_ref
    else:
        for key in islam_keys:
            adjusted[key] = current[key]
        fixed_total += current_islam

    if hindu_ref is not None:
        adjusted["Hinduism"] = hindu_ref
        fixed_total += hindu_ref
    elif "Hinduism" in current:
        adjusted["Hinduism"] = current["Hinduism"]
        fixed_total += current["Hinduism"]

    remaining = max(0.0, 100.0 - fixed_total)
    if other_keys and other_total > 0:
        for key in other_keys:
            adjusted[key] = current[key] * remaining / other_total
    elif remaining > 0:
        adjusted["Other religions"] = adjusted.get("Other religions", 0.0) + remaining
    return adjusted


def _christian_split(iso3: str) -> dict[str, float]:
    if iso3 in ORTHODOX_COUNTRIES:
        return {
            "Orthodox Christianity": 0.78,
            "Catholic Christianity": 0.09,
            "Protestant Christianity": 0.06,
            "Other Christianity": 0.07,
        }
    if iso3 in PROTESTANT_COUNTRIES:
        return {
            "Protestant Christianity": 0.55,
            "Catholic Christianity": 0.26,
            "Orthodox Christianity": 0.04,
            "Other Christianity": 0.15,
        }
    if iso3 in CATHOLIC_COUNTRIES:
        return {
            "Catholic Christianity": 0.72,
            "Protestant Christianity": 0.18,
            "Orthodox Christianity": 0.03,
            "Other Christianity": 0.07,
        }
    if iso3 in SUB_SAHARAN:
        return {
            "Protestant Christianity": 0.40,
            "Catholic Christianity": 0.31,
            "Orthodox Christianity": 0.06,
            "African Independent / Syncretic Christianity": 0.18,
            "Other Christianity": 0.05,
        }
    if iso3 in MENA:
        return {
            "Orthodox Christianity": 0.46,
            "Catholic Christianity": 0.24,
            "Protestant Christianity": 0.11,
            "Other Christianity": 0.19,
        }
    return {
        "Catholic Christianity": 0.42,
        "Protestant Christianity": 0.34,
        "Orthodox Christianity": 0.08,
        "Other Christianity": 0.16,
    }


def _islam_split(iso3: str) -> dict[str, float]:
    if iso3 == "IRN":
        return {"Shia Islam": 0.90, "Sunni Islam": 0.08, "Ibadi / Other Islam": 0.02}
    if iso3 == "IRQ":
        return {"Shia Islam": 0.62, "Sunni Islam": 0.36, "Ibadi / Other Islam": 0.02}
    if iso3 == "AZE":
        return {"Shia Islam": 0.65, "Sunni Islam": 0.33, "Ibadi / Other Islam": 0.02}
    if iso3 == "BHR":
        return {"Shia Islam": 0.58, "Sunni Islam": 0.40, "Ibadi / Other Islam": 0.02}
    if iso3 == "LBN":
        return {"Shia Islam": 0.38, "Sunni Islam": 0.58, "Ibadi / Other Islam": 0.04}
    if iso3 in IBADI_COUNTRIES:
        return {"Ibadi / Other Islam": 0.48, "Sunni Islam": 0.47, "Shia Islam": 0.05}
    return {"Sunni Islam": 0.88, "Shia Islam": 0.09, "Ibadi / Other Islam": 0.03}


def _buddhist_split(iso3: str) -> dict[str, float]:
    if iso3 in THERAVADA_COUNTRIES:
        return {
            "Theravada Buddhism": 0.88,
            "Mahayana Buddhism": 0.07,
            "Vajrayana / Other Buddhism": 0.05,
        }
    if iso3 in MAHAYANA_COUNTRIES:
        return {
            "Mahayana Buddhism": 0.84,
            "Theravada Buddhism": 0.06,
            "Vajrayana / Other Buddhism": 0.10,
        }
    if iso3 in VAJRAYANA_COUNTRIES:
        return {
            "Vajrayana / Other Buddhism": 0.72,
            "Mahayana Buddhism": 0.18,
            "Theravada Buddhism": 0.10,
        }
    return {
        "Mahayana Buddhism": 0.45,
        "Theravada Buddhism": 0.35,
        "Vajrayana / Other Buddhism": 0.20,
    }


def _merge_aliases(shares: Mapping[str, float], iso3: str | None = None) -> dict[str, float]:
    out = {r: 0.0 for r in RELIGIONS}
    for religion, share in shares.items():
        religion = ALIASES.get(religion, religion)
        if religion == "Christianity":
            for sect, pct in _christian_split(iso3 or "").items():
                out[sect] += float(share) * pct
        elif religion == "Islam":
            for sect, pct in _islam_split(iso3 or "").items():
                out[sect] += float(share) * pct
        elif religion == "Buddhism":
            for sect, pct in _buddhist_split(iso3 or "").items():
                out[sect] += float(share) * pct
        elif religion == "Folk / Traditional" and (iso3 or "") in SUB_SAHARAN:
            out["Folk / Traditional"] += float(share) * 0.62
            out["Indigenous / Syncretic traditions"] += float(share) * 0.38
        else:
            out[religion] = out.get(religion, 0.0) + float(share)
    total = sum(out.values()) or 1.0
    return {k: v / total * 100.0 for k, v in out.items() if v > 0.01}


def baseline_for_country(iso3: str) -> dict[str, float]:
    if iso3 in COUNTRY_BASELINES:
        return _merge_aliases(_apply_muslim_reference_2025(COUNTRY_BASELINES[iso3], iso3), iso3)
    if iso3 in SUB_SAHARAN:
        if iso3 in AFRICA_MUSLIM_BELT:
            return _merge_aliases(_apply_muslim_reference_2025({"Islam": 84, "Christianity": 8, "Indigenous / Syncretic traditions": 4, "Folk / Traditional": 3, "Other religions": 0.5, "Unaffiliated": 0.5}, iso3), iso3)
        if iso3 in AFRICA_MIXED_WEST:
            return _merge_aliases(_apply_muslim_reference_2025({"Christianity": 48, "Islam": 42, "Indigenous / Syncretic traditions": 5, "Folk / Traditional": 3, "Other religions": 1, "Unaffiliated": 1}, iso3), iso3)
        if iso3 in AFRICA_CHRISTIAN_CORE:
            return _merge_aliases(_apply_muslim_reference_2025({"Christianity": 79, "Islam": 9, "Indigenous / Syncretic traditions": 6, "Folk / Traditional": 4, "Other religions": 1, "Unaffiliated": 1}, iso3), iso3)
        if iso3 in AFRICA_SOUTHERN:
            return _merge_aliases(_apply_muslim_reference_2025({"Christianity": 76, "Islam": 2, "Indigenous / Syncretic traditions": 9, "Folk / Traditional": 6, "Unaffiliated": 5, "Other religions": 2}, iso3), iso3)
        return _merge_aliases(_apply_muslim_reference_2025({"Christianity": 62, "Islam": 28, "Indigenous / Syncretic traditions": 5, "Folk / Traditional": 3, "Other religions": 1.2, "Unaffiliated": 0.8}, iso3), iso3)
    if iso3 in MUSLIM_MAJORITY:
        return _merge_aliases(_apply_muslim_reference_2025({"Islam": 88, "Christianity": 5, "Unaffiliated": 3, "Other religions": 2, "Folk / Traditional": 2}, iso3), iso3)
    if iso3 in SOUTH_ASIA:
        return _merge_aliases(_apply_muslim_reference_2025({"Hinduism": 68, "Islam": 22, "Buddhism": 4, "Christianity": 3, "Other religions": 2, "Unaffiliated": 1}, iso3), iso3)
    if iso3 in EAST_ASIA:
        return _merge_aliases(_apply_muslim_reference_2025({"Unaffiliated": 39, "Buddhism": 30, "Folk / Traditional": 18, "Christianity": 8, "Islam": 2, "Other religions": 3}, iso3), iso3)
    if iso3 in EUROPE:
        return _merge_aliases(_apply_muslim_reference_2025({"Christianity": 66, "Unaffiliated": 24, "Islam": 6, "Buddhism": 1, "Judaism": 0.5, "Other religions": 2.5}, iso3), iso3)
    if iso3 in LATIN_AMERICA:
        return _merge_aliases(_apply_muslim_reference_2025({"Christianity": 84, "Unaffiliated": 9, "Folk / Traditional": 3, "Other religions": 3, "Islam": 0.3, "Judaism": 0.2, "Buddhism": 0.3, "Hinduism": 0.2}, iso3), iso3)
    if iso3 in PACIFIC:
        return _merge_aliases(_apply_muslim_reference_2025({"Christianity": 72, "Unaffiliated": 16, "Folk / Traditional": 6, "Hinduism": 2, "Buddhism": 1.5, "Islam": 1, "Other religions": 1.5}, iso3), iso3)
    return _merge_aliases(_apply_muslim_reference_2025({"Christianity": 55, "Islam": 18, "Unaffiliated": 15, "Buddhism": 4, "Hinduism": 3, "Folk / Traditional": 3, "Other religions": 2}, iso3), iso3)


def target_for_country_2050(iso3: str) -> tuple[dict[str, float] | None, float]:
    """Return a Pew-style 2050 calibration target plus blend weight."""
    if iso3 in COUNTRY_TARGETS_2050:
        return _merge_aliases(COUNTRY_TARGETS_2050[iso3], iso3), 1.0
    if iso3 in WESTERN_EUROPE:
        return _merge_aliases({
            "Christianity": 52.0, "Unaffiliated": 32.0, "Islam": 11.0,
            "Buddhism": 1.2, "Judaism": 0.5, "Hinduism": 0.6,
            "Other religions": 2.7,
        }, iso3), 0.78
    if iso3 in EUROPE:
        return _merge_aliases({
            "Christianity": 60.0, "Unaffiliated": 27.0, "Islam": 9.0,
            "Buddhism": 0.8, "Judaism": 0.4, "Other religions": 2.8,
        }, iso3), 0.72
    if iso3 in MENA and iso3 not in {"ISR", "LBN"}:
        return _merge_aliases({
            "Islam": 92.5, "Christianity": 4.0, "Unaffiliated": 1.2,
            "Folk / Traditional": 0.8, "Other religions": 1.5,
        }, iso3), 0.82
    if iso3 in SUB_SAHARAN:
        if iso3 in AFRICA_MUSLIM_BELT:
            return _merge_aliases({
                "Islam": 87.0, "Christianity": 7.0,
                "Indigenous / Syncretic traditions": 2.5, "Folk / Traditional": 1.5,
                "Other religions": 1.0, "Unaffiliated": 1.0,
            }, iso3), 0.64
        if iso3 in AFRICA_MIXED_WEST:
            return _merge_aliases({
                "Christianity": 50.0, "Islam": 45.0,
                "Indigenous / Syncretic traditions": 2.3, "Folk / Traditional": 1.1,
                "Other religions": 0.9, "Unaffiliated": 0.7,
            }, iso3), 0.64
        if iso3 in AFRICA_CHRISTIAN_CORE:
            return _merge_aliases({
                "Christianity": 80.0, "Islam": 11.0,
                "Indigenous / Syncretic traditions": 4.0, "Folk / Traditional": 2.5,
                "Other religions": 1.3, "Unaffiliated": 1.2,
            }, iso3), 0.64
        if iso3 in AFRICA_SOUTHERN:
            return _merge_aliases({
                "Christianity": 74.0, "Islam": 3.0, "Hinduism": 1.0,
                "Indigenous / Syncretic traditions": 8.0, "Folk / Traditional": 4.0,
                "Unaffiliated": 8.0, "Other religions": 2.0,
            }, iso3), 0.64
        return _merge_aliases({
            "Christianity": 63.0, "Islam": 30.0,
            "Indigenous / Syncretic traditions": 3.2, "Folk / Traditional": 1.7,
            "Other religions": 1.2, "Unaffiliated": 0.9,
        }, iso3), 0.60
    if iso3 in EAST_ASIA:
        return _merge_aliases({
            "Unaffiliated": 42.0, "Buddhism": 27.0, "Folk / Traditional": 18.0,
            "Christianity": 8.5, "Islam": 2.0, "Other religions": 2.5,
        }, iso3), 0.55
    return None, 0.0


def _float_ctx(ctx: Mapping[str, float], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        value = ctx.get(key)
        try:
            if value is not None and pd.notna(value):
                return float(value)
        except (TypeError, ValueError):
            continue
    return default


def _development_irreligion_pressure(iso3: str, ctx: Mapping[str, float]) -> float:
    """HDI-linked pressure toward unaffiliated identity by 2050.

    The effect is intentionally nonlinear and modest. It is strongest where
    education/income/HDI rise from developing-world baselines, weaker in places
    that are already highly secular, and damped where youth-heavy religious
    retention remains structurally strong.
    """
    hdi_now = _float_ctx(ctx, "HDI_Baseline", "HDI_2024", default=0.62)
    hdi_2050 = _float_ctx(ctx, "HDI_2050", default=hdi_now)
    edu_now = _float_ctx(ctx, "EducationIndex_2025", default=0.60)
    edu_2050 = _float_ctx(ctx, "EducationIndex_2050", default=edu_now)
    income_now = _float_ctx(ctx, "IncomeIndex_2025", default=0.60)
    income_2050 = _float_ctx(ctx, "IncomeIndex_2050", default=income_now)
    urban = _float_ctx(ctx, "Urbanization_2024", default=0.55)
    pressure = _float_ctx(ctx, "DependencyPressure", default=0.35)

    hdi_gain = max(0.0, hdi_2050 - hdi_now)
    education_gain = max(0.0, edu_2050 - edu_now)
    income_gain = max(0.0, income_2050 - income_now)
    development_2050 = (hdi_2050 + edu_2050 + income_2050 + urban) / 4.0

    catchup = max(0.0, hdi_gain - 0.035)
    high_development = max(0.0, development_2050 - 0.63)
    education_channel = max(0.0, edu_2050 - 0.62) + 0.65 * education_gain
    income_channel = 0.35 * income_gain
    urban_channel = 0.18 * max(0.0, urban - 0.50)

    raw = (
        0.55 * catchup +
        0.24 * high_development +
        0.17 * education_channel +
        income_channel +
        urban_channel
    )

    # The developing-world catch-up effect should exist, but not imply that
    # secularization mechanically follows every income gain at European speed.
    if iso3 in MUSLIM_MAJORITY:
        raw *= 0.55
    elif iso3 in SUB_SAHARAN:
        raw *= 0.72
    elif iso3 in SOUTH_ASIA:
        raw *= 0.72
    elif iso3 in LATIN_AMERICA:
        raw *= 0.90
    elif iso3 in EAST_ASIA:
        raw *= 1.05
    elif iso3 in EUROPE:
        raw *= 0.45

    youth_retention = max(0.0, pressure - 0.34)
    raw *= max(0.60, 1.0 - 0.75 * youth_retention)
    return min(0.30, max(0.0, raw))


def _apply_development_irreligion_shift(shares: Mapping[str, float], iso3: str, ctx: Mapping[str, float]) -> dict[str, float]:
    pressure = _development_irreligion_pressure(iso3, ctx)
    if pressure <= 0:
        return dict(shares)

    out = {k: float(v) for k, v in shares.items()}
    current_unaff = out.get("Unaffiliated", 0.0)
    if current_unaff >= 45.0:
        return out

    max_shift = 9.0 if iso3 not in EUROPE else 4.0
    shift = min(max_shift, max(0.0, 18.0 * pressure), max(0.0, 45.0 - current_unaff))
    if shift <= 0:
        return out

    donor_keys = [
        key for key in out
        if key != "Unaffiliated" and (
            "Christianity" in key or "Islam" in key or key in {
                "Hinduism", "Theravada Buddhism", "Mahayana Buddhism",
                "Vajrayana / Other Buddhism", "Folk / Traditional",
                "Indigenous / Syncretic traditions", "Sikhism",
                "Judaism", "Bahai", "Zoroastrianism",
            }
        )
    ]
    donor_total = sum(out[key] for key in donor_keys)
    if donor_total <= 0:
        return out

    for key in donor_keys:
        out[key] = max(0.0, out[key] - shift * out[key] / donor_total)
    out["Unaffiliated"] = current_unaff + shift
    total = sum(out.values()) or 1.0
    return {key: value / total * 100.0 for key, value in out.items() if value / total * 100.0 > 0.01}


def _europe_migration_composition_weight(iso3: str, ctx: Mapping[str, float]) -> float:
    """Incremental 2050 resident-stock weight from labor-replacement migration.

    The response is deliberately capped. A shrinking birth cohort creates
    demand for workers, but housing, policy, recruitment, return migration and
    integration capacity determine how much of that demand becomes durable
    settlement. The value is a scenario weight, not a forecast net-migration
    rate.
    """
    if iso3 not in EUROPE:
        return 0.0
    pressure = _float_ctx(ctx, "Birth_Replacement_Pressure_2050", default=0.0)
    response = _float_ctx(ctx, "Europe_Migration_Response_2050", default=0.0)
    openness = _float_ctx(ctx, "Policy_Openness", default=0.75)
    intensity = _float_ctx(ctx, "Migration_Intensity_2050", default=1.0)
    if pressure <= 0.0 or response <= 0.0:
        return 0.0
    weight = (
        0.035 * pressure +
        0.070 * response +
        0.008 * max(0.0, intensity - 1.0)
    ) * (0.65 + 0.35 * max(0.0, min(1.25, openness)))
    return min(0.065, max(0.0, weight))


def _apply_europe_migration_composition(
    shares: Mapping[str, float], iso3: str, ctx: Mapping[str, float]
) -> tuple[dict[str, float], float]:
    """Blend the central religion projection with destination inflow mix."""
    weight = _europe_migration_composition_weight(iso3, ctx)
    if weight <= 0.0:
        return dict(shares), 0.0

    broad_mix = EUROPE_MIGRANT_RELIGION_MIX.get(
        iso3, DEFAULT_EUROPE_MIGRANT_RELIGION_MIX)
    inflow_mix = _merge_aliases(broad_mix, iso3)
    religions = set(shares) | set(inflow_mix)
    blended = {
        religion: (1.0 - weight) * float(shares.get(religion, 0.0)) +
        weight * float(inflow_mix.get(religion, 0.0))
        for religion in religions
    }
    total = sum(blended.values()) or 1.0
    return ({
        religion: value / total * 100.0
        for religion, value in blended.items()
        if value / total * 100.0 > 0.01
    }, weight)


def _apply_late_ssa_migration_composition(
    shares: Mapping[str, float], iso3: str, ctx: Mapping[str, float]
) -> tuple[dict[str, float], float]:
    """Add a capped late-period SSA corridor effect to destination shares."""
    response = _float_ctx(
        ctx, "SSA_LateMigration_DestinationResponse_2050", default=0.0)
    if response <= 0.0:
        return dict(shares), 0.0
    # At full response, the incremental channel represents at most 2.2% of
    # the destination's 2050 resident stock. This is deliberately conservative
    # because much labor mobility is temporary, circular or return migration.
    weight = min(0.022, max(0.0, 0.022 * response))
    religions = set(shares) | set(SSA_LATE_MIGRANT_RELIGION_MIX)
    blended = {
        religion: (1.0 - weight) * float(shares.get(religion, 0.0)) +
        weight * float(SSA_LATE_MIGRANT_RELIGION_MIX.get(religion, 0.0))
        for religion in religions
    }
    total = sum(blended.values()) or 1.0
    return ({
        religion: value / total * 100.0
        for religion, value in blended.items()
        if value / total * 100.0 > 0.01
    }, weight)


def _religion_growth_modifier(religion: str, iso3: str, ctx: Mapping[str, float]) -> float:
    education = float(ctx.get("EducationIndex_2050", ctx.get("EducationIndex_2025", 0.65)) or 0.65)
    income = float(ctx.get("IncomeIndex_2050", 0.65) or 0.65)
    urban = float(ctx.get("Urbanization_2024", 0.55) or 0.55)
    migration = float(ctx.get("Migration_Intensity_2050", 1.0) or 1.0)
    pressure = float(ctx.get("DependencyPressure", 0.35) or 0.35)
    development = (education + income + urban) / 3.0
    secularization = max(0.0, development - 0.58)
    youth_religiosity = max(0.0, pressure - 0.32)
    migration_boost = max(0.0, migration - 1.0)

    is_christian = "Christianity" in religion
    is_muslim = "Islam" in religion
    is_buddhist = "Buddhism" in religion

    if religion == "Unaffiliated":
        development_gain = _development_irreligion_pressure(iso3, ctx)
        return 1.0 + 0.42 * secularization + 0.75 * development_gain + 0.04 * migration_boost - 0.14 * youth_religiosity
    if is_muslim:
        sect_adjustment = 0.015 if religion == "Sunni Islam" else 0.0
        return 1.0 + 0.18 * youth_religiosity + 0.05 * migration_boost - 0.07 * secularization + sect_adjustment
    if is_christian:
        sect_adjustment = 0.015 if religion == "Protestant Christianity" and iso3 in SUB_SAHARAN else 0.0
        return 1.0 - 0.11 * secularization + 0.03 * youth_religiosity + (0.02 if iso3 in SUB_SAHARAN else -0.01) + sect_adjustment
    if religion == "Hinduism" or is_buddhist:
        buddhist_ageing_drag = 0.035 if is_buddhist and iso3 in (EAST_ASIA | THERAVADA_COUNTRIES) else 0.0
        vajrayana_buffer = 0.015 if religion == "Vajrayana / Other Buddhism" and iso3 in VAJRAYANA_COUNTRIES else 0.0
        return 1.0 - 0.05 * secularization + 0.04 * youth_religiosity - buddhist_ageing_drag + vajrayana_buffer
    if religion in {"Sikhism", "Bahai", "Zoroastrianism"}:
        return 1.0 + 0.03 * migration_boost - 0.02 * secularization
    if religion in {"Folk / Traditional", "Indigenous / Syncretic traditions"}:
        return 1.0 - 0.16 * development + 0.05 * youth_religiosity
    if religion == "Judaism":
        return 1.0 + 0.03 * migration_boost
    return 1.0 + 0.02 * migration_boost - 0.02 * secularization


def project_religion_shares(iso3: str, hdi_context: Mapping[str, Mapping[str, float]] | None = None) -> dict[str, float]:
    ctx = (hdi_context or {}).get(iso3, {})
    base = baseline_for_country(iso3)
    projected = {
        religion: max(0.0, share * _religion_growth_modifier(religion, iso3, ctx))
        for religion, share in base.items()
    }
    total = sum(projected.values()) or 1.0
    normalized = {
        religion: value / total * 100.0
        for religion, value in projected.items()
        if value / total * 100.0 > 0.01
    }
    target, weight = target_for_country_2050(iso3)
    if target and weight > 0:
        religions = set(normalized) | set(target)
        blended = {
            religion: normalized.get(religion, 0.0) * (1.0 - weight) +
            target.get(religion, 0.0) * weight
            for religion in religions
        }
        blended_total = sum(blended.values()) or 1.0
        central = _apply_development_irreligion_shift(blended, iso3, ctx)
    else:
        central = _apply_development_irreligion_shift(normalized, iso3, ctx)
    european_adjusted = _apply_europe_migration_composition(
        central, iso3, ctx)[0]
    return _apply_late_ssa_migration_composition(
        european_adjusted, iso3, ctx)[0]


def build_religion_table(pop2024: Mapping[str, float], pop2050: Mapping[str, float], hdi_context: Mapping[str, Mapping[str, float]]) -> pd.DataFrame:
    rows = []
    for iso3 in sorted(UNDP_HDI_COUNTRIES_193):
        base = baseline_for_country(iso3)
        projected = project_religion_shares(iso3, hdi_context)
        ctx = hdi_context.get(iso3, {})
        migration_weight = _europe_migration_composition_weight(iso3, ctx)
        ssa_migration_weight = min(0.022, max(
            0.0, 0.022 * _float_ctx(
                ctx, "SSA_LateMigration_DestinationResponse_2050")))
        without_response_ctx = dict(ctx)
        without_response_ctx["Europe_Migration_Response_2050"] = 0.0
        projected_without_response = project_religion_shares(
            iso3, {iso3: without_response_ctx})
        without_ssa_ctx = dict(ctx)
        without_ssa_ctx["SSA_LateMigration_DestinationResponse_2050"] = 0.0
        projected_without_ssa = project_religion_shares(
            iso3, {iso3: without_ssa_ctx})
        country = COUNTRY_NAMES.get(iso3, iso3)
        for religion in sorted(set(base) | set(projected)):
            share_2024 = base.get(religion, 0.0)
            share_2050 = projected.get(religion, 0.0)
            share_without_response = projected_without_response.get(religion, 0.0)
            share_without_ssa = projected_without_ssa.get(religion, 0.0)
            if share_2024 < 0.01 and share_2050 < 0.01:
                continue
            rows.append({
                "ISO3": iso3,
                "Country": country,
                "Group": religion,
                "Profile": "religion",
                "Share_2024_pct": round(share_2024, 2),
                "Share_2050_pct": round(share_2050, 2),
                "Change_pp": round(share_2050 - share_2024, 2),
                "Share_2050_WithoutBirthReplacementMigration_pct": round(
                    share_without_response, 2),
                "BirthReplacementMigration_Attributed_Change_pp": round(
                    share_2050 - share_without_response, 2),
                "Birth_Replacement_Pressure_2050": round(
                    _float_ctx(ctx, "Birth_Replacement_Pressure_2050"), 3),
                "Europe_Migration_Response_2050": round(
                    _float_ctx(ctx, "Europe_Migration_Response_2050"), 3),
                "Migration_Composition_Weight_2050": round(migration_weight, 4),
                "SSA_LateMigration_CompositionWeight_2050": round(
                    ssa_migration_weight, 4),
                "Share_2050_WithoutSSALateMigration_pct": round(
                    share_without_ssa, 2),
                "SSA_LateMigration_Attributed_Change_pp": round(
                    share_2050 - share_without_ssa, 2),
                "Pop_2024": round(share_2024 / 100.0 * pop2024[iso3]) if iso3 in pop2024 else None,
                "Pop_2050": round(share_2050 / 100.0 * pop2050[iso3]) if iso3 in pop2050 else None,
            })
    return pd.DataFrame(rows)
