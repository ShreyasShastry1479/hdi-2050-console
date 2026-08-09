"""Build 2025 and 2050 subdivision HDI projections.

The source workbook supplies subdivision HDI values but not subdivision
population weights or subnational map geometries. This script therefore builds
modeled subdivision population weights calibrated to each country's national
2024 and 2050 population totals, reconciles 2025 subdivision HDI to the
national 2025 HDI baseline, projects 2050 values from the national HDI table,
and re-centers each country so its population-weighted subdivision mean equals
the national target.
"""

from __future__ import annotations

import csv
import json
import math
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import pycountry

from data.countries import COUNTRY_NAMES

SOURCE_XLSX = Path(r"C:\Users\raghu\Downloads\HDI 2025 By Subdivision.xlsx")
HDI_CSV = Path("data/output/hdi_2050_rankings.csv")
OUT_CSV = Path("data/output/subdivision_hdi_2025_2050.csv")
OUT_HTML = Path("web/subdivision-hdi-2025-2050.html")
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
OFFICIAL_WORLD_HDI_2025 = 0.756
OFFICIAL_WORLD_HDI_2050 = 0.794

NAME_OVERRIDES = {
    "Bahamas": "BHS",
    "Bolivia": "BOL",
    "Bosnia and Herzegovina": "BIH",
    "Brunei": "BRN",
    "Cape Verde": "CPV",
    "Congo": "COG",
    "Congo (Kinshasa)": "COD",
    "DRC": "COD",
    "Cote d'Ivoire": "CIV",
    "Côte d'Ivoire": "CIV",
    "Czech Republic": "CZE",
    "DR Congo": "COD",
    "Democratic Republic of the Congo": "COD",
    "Eswatini": "SWZ",
    "Hong Kong": "HKG",
    "Iran": "IRN",
    "Ivory Coast": "CIV",
    "Laos": "LAO",
    "Micronesia": "FSM",
    "Moldova": "MDA",
    "North Macedonia": "MKD",
    "Palestine": "PSE",
    "Russia": "RUS",
    "Sao Tome and Principe": "STP",
    "São Tomé and Príncipe": "STP",
    "South Korea": "KOR",
    "Syria": "SYR",
    "Tanzania": "TZA",
    "Turkey": "TUR",
    "UAE": "ARE",
    "United Kingdom": "GBR",
    "United States": "USA",
    "Venezuela": "VEN",
    "Vietnam": "VNM",
}
for iso3, name in COUNTRY_NAMES.items():
    NAME_OVERRIDES.setdefault(name, iso3)


SUBDIVISION_POPULATION_PRIORS = {
    "CAN": {
        "Ontario": 0.389,
        "Quebec": 0.220,
        "British Columbia": 0.138,
        "Alberta": 0.119,
        "Manitoba": 0.036,
        "Saskatchewan": 0.030,
        "Nova Scotia": 0.026,
        "New Brunswick": 0.020,
        "Newfoundland and Labrador": 0.013,
        "Prince Edward Island, Yukon Territory, Northwest Territories, Nunavut": 0.009,
    },
    "CHN": {
        "Guangdong": 0.090,
        "Shandong": 0.071,
        "Henan": 0.070,
        "Sichuan": 0.059,
        "Jiangsu": 0.060,
        "Hebei": 0.052,
        "Hunan": 0.046,
        "Zhejiang": 0.046,
        "Anhui": 0.043,
        "Hubei": 0.041,
        "Guangxi": 0.036,
        "Yunnan": 0.033,
        "Jiangxi": 0.032,
        "Liaoning": 0.030,
        "Fujian": 0.029,
        "Shaanxi": 0.028,
        "Guizhou": 0.027,
        "Shanxi": 0.025,
        "Chongqing": 0.023,
        "Heilongjiang": 0.022,
        "Xinjiang": 0.018,
        "Shanghai": 0.017,
        "Jilin": 0.017,
        "Inner Mongolia": 0.017,
        "Gansu": 0.018,
        "Beijing": 0.015,
        "Tianjin": 0.010,
        "Hainan": 0.007,
        "Ningxia": 0.005,
        "Qinghai": 0.004,
        "Tibet": 0.003,
    },
    "IND": {
        "Uttar Pradesh": 0.170,
        "Bihar": 0.100,
        "Maharashtra": 0.090,
        "West Bengal": 0.070,
        "Madhya Pradesh": 0.060,
        "Rajasthan": 0.056,
        "Tamil Nadu": 0.052,
        "Gujarat": 0.050,
        "Karnataka": 0.046,
        "Andhra Pradesh": 0.036,
        "Odisha": 0.032,
        "Orissa": 0.032,
        "Telangana": 0.027,
        "Jharkhand": 0.026,
        "Assam": 0.025,
        "Chhattisgarh": 0.022,
        "Punjab": 0.021,
        "Haryana": 0.021,
        "Kerala": 0.025,
        "New Delhi": 0.015,
        "Jammu and Kashmir": 0.009,
        "Uttaranchal": 0.008,
        "Himachal Pradesh": 0.005,
        "Tripura": 0.003,
        "Meghalaya": 0.0025,
        "Manipur": 0.002,
        "Nagaland": 0.0016,
        "Goa": 0.0011,
        "Arunachal Pradesh": 0.0011,
        "Puducherry": 0.0011,
        "Mizoram": 0.0009,
        "Chandigarth": 0.0008,
        "Sikkim": 0.0005,
        "Andaman and Nicobar Islands": 0.0003,
        "Dadra and Nagar Haveli": 0.00025,
        "Daman and Diu": 0.00020,
        "Lakshadweep": 0.00005,
    },
    "USA": {
        "California": 0.117,
        "Texas": 0.087,
        "Florida": 0.066,
        "New York": 0.058,
        "Pennsylvania": 0.038,
        "Illinois": 0.037,
        "Ohio": 0.035,
        "Georgia": 0.032,
        "North Carolina": 0.032,
        "Michigan": 0.030,
        "New Jersey": 0.028,
        "Virginia": 0.026,
        "Washington": 0.023,
        "Arizona": 0.022,
        "Tennessee": 0.021,
        "Massachusetts": 0.020,
        "Indiana": 0.020,
        "Missouri": 0.018,
        "Maryland": 0.018,
        "Wisconsin": 0.017,
        "Colorado": 0.017,
        "Minnesota": 0.017,
        "South Carolina": 0.016,
        "Alabama": 0.015,
        "Louisiana": 0.013,
        "Kentucky": 0.013,
        "Oregon": 0.013,
        "Oklahoma": 0.012,
        "Connecticut": 0.011,
        "Utah": 0.010,
        "Iowa": 0.0095,
        "Nevada": 0.0095,
        "Arkansas": 0.0090,
        "Mississippi": 0.0085,
        "Kansas": 0.0085,
        "New Mexico": 0.0063,
        "Nebraska": 0.0058,
        "Idaho": 0.0056,
        "West Virginia": 0.0052,
        "Hawaii": 0.0043,
        "New Hampshire": 0.0041,
        "Maine": 0.0041,
        "Rhode Island": 0.0033,
        "Montana": 0.0033,
        "Delaware": 0.0030,
        "South Dakota": 0.0027,
        "North Dakota": 0.0023,
        "Alaska": 0.0022,
        "District of Columbia": 0.0020,
        "Vermont": 0.0019,
        "Wyoming": 0.0017,
    },
    "BRA": {
        "Sao Paulo": 0.214,
        "Minas Gerais": 0.100,
        "Rio de Janeiro": 0.079,
        "Bahia": 0.069,
        "Parana": 0.054,
        "Rio Grande do Sul": 0.052,
        "Pernambuco": 0.044,
        "Ceara": 0.042,
        "Para": 0.040,
        "Santa Catarina": 0.036,
        "Goias": 0.034,
        "Maranhao": 0.033,
        "Amazonas": 0.020,
        "Espirito Santo": 0.019,
        "Paraiba": 0.019,
        "Mato Grosso": 0.018,
        "Rio Grande do Norte": 0.017,
        "Alagoas": 0.015,
        "Piaui": 0.015,
        "Distrito Federal": 0.014,
        "Mato Grosso do Sul": 0.013,
        "Sergipe": 0.011,
        "Rondonia": 0.008,
        "Tocantins": 0.007,
        "Acre": 0.004,
        "Amapa": 0.004,
        "Roraima": 0.003,
    },
    "NGA": {
        "Kano": 0.065,
        "Lagos": 0.060,
        "Kaduna": 0.040,
        "Katsina": 0.035,
        "Oyo": 0.034,
        "Rivers": 0.032,
        "Bauchi": 0.031,
        "Jigawa": 0.030,
        "Benue": 0.029,
        "Anambra": 0.027,
        "Borno": 0.027,
        "Delta": 0.027,
        "Imo": 0.026,
        "Niger": 0.026,
        "Akwa Ibom": 0.025,
        "Ogun": 0.025,
        "Sokoto": 0.024,
        "Ondo": 0.022,
        "Osun": 0.021,
        "Kogi": 0.020,
        "Zamfora": 0.020,
        "Enugu": 0.019,
        "Kebbi": 0.019,
        "Kwara": 0.018,
        "Adamawa": 0.018,
        "Plateau": 0.018,
        "Edo": 0.017,
        "Cross River": 0.017,
        "Abia": 0.016,
        "Yobe": 0.016,
        "Gombe": 0.015,
        "Taraba": 0.014,
        "Ebonyi": 0.013,
        "Bayelsa": 0.010,
        "Ekiti": 0.010,
        "Abuja FCT": 0.010,
        "Nassarawa": 0.010,
    },
}


def iso_for(name: str) -> str:
    name = str(name).strip()
    if name in NAME_OVERRIDES:
        return NAME_OVERRIDES[name]
    try:
        return pycountry.countries.lookup(name).alpha_3
    except LookupError:
        return ""


def col_idx(ref: str) -> int:
    letters = "".join(ch for ch in ref if ch.isalpha())
    n = 0
    for ch in letters:
        n = n * 26 + ord(ch) - 64
    return n - 1


def read_xlsx_rows(path: Path) -> list[list[str]]:
    with zipfile.ZipFile(path) as zf:
        shared = []
        root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
        for item in root.findall(NS + "si"):
            shared.append("".join(t.text or "" for t in item.iter(NS + "t")))

        sheet = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        rows = []
        for row in sheet.findall(".//" + NS + "row"):
            values = {}
            max_col = -1
            for cell in row.findall(NS + "c"):
                idx = col_idx(cell.attrib.get("r", ""))
                max_col = max(max_col, idx)
                node = cell.find(NS + "v")
                value = ""
                if node is not None:
                    value = node.text or ""
                    if cell.attrib.get("t") == "s":
                        value = shared[int(value)]
                values[idx] = value
            if values:
                rows.append([values.get(i, "") for i in range(max_col + 1)])
    return rows


def fnum(value) -> float:
    try:
        if value == "" or pd.isna(value):
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        match = re.search(r"-?\d+(?:\.\d+)?", str(value))
        return float(match.group(0)) if match else math.nan


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def finite_or(value, fallback: float) -> float:
    parsed = fnum(value)
    return parsed if math.isfinite(parsed) else fallback


def recenter(values: list[float], weights: list[float], target: float, low: float, high: float) -> list[float]:
    adjusted = [clamp(value, low, high) for value in values]
    for _ in range(12):
        current = sum(value * weight for value, weight in zip(adjusted, weights))
        residual = target - current
        if abs(residual) <= 1e-10:
            break
        movable = [
            i for i, value in enumerate(adjusted)
            if (residual > 0 and value < high - 1e-10) or (residual < 0 and value > low + 1e-10)
        ]
        if not movable:
            break
        movable_weight = sum(weights[i] for i in movable)
        if movable_weight <= 0:
            break
        shift = residual / movable_weight
        for i in movable:
            adjusted[i] = clamp(adjusted[i] + shift, low, high)
    return adjusted


def normalize_weights(weights: list[float]) -> list[float]:
    total = sum(max(0.0, weight) for weight in weights)
    if total <= 0:
        return [1.0 / len(weights)] * len(weights)
    return [max(0.0, weight) / total for weight in weights]


def calibrate_weights_to_hdi(priors: list[float], hdi_values: list[float], target: float) -> list[float]:
    """Entropy-tilt weights so the raw subdivision HDI mean matches country HDI."""
    low = min(hdi_values)
    high = max(hdi_values)
    if target <= low or target >= high or abs(high - low) < 1e-9:
        return normalize_weights(priors)

    def tilted(lam: float) -> list[float]:
        raw = [prior * math.exp(lam * value) for prior, value in zip(priors, hdi_values)]
        return normalize_weights(raw)

    lo, hi = -900.0, 900.0
    for _ in range(90):
        mid = (lo + hi) / 2
        weights = tilted(mid)
        mean = sum(weight * value for weight, value in zip(weights, hdi_values))
        if mean < target:
            lo = mid
        else:
            hi = mid
    return tilted((lo + hi) / 2)


def seeded_population_prior(rows: list[dict], iso3: str, hdi_values: list[float]) -> tuple[list[float], bool]:
    seeded = SUBDIVISION_POPULATION_PRIORS.get(iso3, {})
    if seeded:
        weights = [seeded.get(row["Subdivision"], math.nan) for row in rows]
        if all(math.isfinite(weight) and weight > 0 for weight in weights):
            return normalize_weights(weights), True

    n = len(rows)
    mean = sum(hdi_values) / n
    variance = sum((value - mean) ** 2 for value in hdi_values) / n
    stdev = math.sqrt(variance) or 1.0
    z_scores = [clamp((value - mean) / stdev, -2.25, 2.25) for value in hdi_values]
    raw = [math.exp(0.22 * z) for z in z_scores]
    floor = 0.28 / n
    return normalize_weights([max(floor, weight) for weight in normalize_weights(raw)]), False


def evolve_2050_weights(rows: list[dict], weights_2024: list[float], national: dict, hdi_values: list[float]) -> list[float]:
    pop_2024 = finite_or(national.get("Population_2024"), math.nan)
    pop_2050 = finite_or(national.get("Population_2050"), pop_2024)
    pop_growth = 0.0 if not pop_2024 or not math.isfinite(pop_2024) else clamp((pop_2050 - pop_2024) / pop_2024, -0.35, 1.75)
    urbanization = clamp(finite_or(national.get("Urbanization_2024"), 55.0) / 100.0, 0.18, 0.98)
    growth_score = clamp(finite_or(national.get("GrowthProspectScore"), 0.50), 0.0, 1.0)
    national_hdi = clamp(finite_or(national.get("HDI_Baseline"), sum(hdi_values) / len(hdi_values)), 0.30, 0.98)

    mean = sum(hdi_values) / len(hdi_values)
    variance = sum((value - mean) ** 2 for value in hdi_values) / len(hdi_values)
    stdev = math.sqrt(variance) or 1.0
    z_scores = [clamp((value - mean) / stdev, -2.2, 2.2) for value in hdi_values]

    developing_pressure = clamp((0.82 - national_hdi) / 0.42, 0.0, 1.0)
    urban_pull = 0.06 + 0.12 * urbanization + 0.07 * growth_score
    fertility_dispersion = 0.16 * max(pop_growth, 0.0) * developing_pressure
    decline_pull = 0.12 * abs(min(pop_growth, 0.0))

    raw = []
    for weight, z in zip(weights_2024, z_scores):
        low_hdi_growth = -z * fertility_dispersion
        high_hdi_migration = z * urban_pull
        aging_buffer = z * decline_pull
        raw.append(weight * math.exp(low_hdi_growth + high_hdi_migration + aging_buffer))

    return normalize_weights(raw)


def modeled_population_weights(rows: list[dict], national: dict) -> tuple[list[float], list[float], list[float], list[float], str]:
    """Approximate subdivision population shares where no source weights exist.

    The proxy uses the subdivision HDI distribution as a loose stand-in for
    urban/economic concentration, then scales concentration by national
    urbanization, population growth, and development momentum. It is normalized
    back to official national population totals.
    """
    n = len(rows)
    if n == 1:
        pop_2024 = finite_or(national.get("Population_2024"), math.nan)
        pop_2050 = finite_or(national.get("Population_2050"), pop_2024)
        return [1.0], [1.0], [pop_2024], [pop_2050], "single_subdivision_national_population_total"

    pop_2024 = finite_or(national.get("Population_2024"), math.nan)
    pop_2050 = finite_or(national.get("Population_2050"), pop_2024)
    if not math.isfinite(pop_2024) or pop_2024 <= 0:
        pop_2024 = float(n)
    if not math.isfinite(pop_2050) or pop_2050 <= 0:
        pop_2050 = pop_2024

    values = [row["Subdivision_HDI_2025_Source"] for row in rows]
    priors, seeded = seeded_population_prior(rows, rows[0]["ISO3"], values)
    weights_2024 = priors
    weights_2050 = evolve_2050_weights(rows, weights_2024, national, values)
    subdivision_pop_2024 = [pop_2024 * weight for weight in weights_2024]
    subdivision_pop_2050 = [pop_2050 * weight for weight in weights_2050]
    method = (
        ("seeded_population_prior" if seeded else "modeled_population_proxy")
        + "_population_first_evolved_to_2050_then_hdi_reconciled"
    )
    return weights_2024, weights_2050, subdivision_pop_2024, subdivision_pop_2050, method


def build_rows() -> tuple[list[dict], list[dict]]:
    raw = read_xlsx_rows(SOURCE_XLSX)
    source_rows = []
    for row in raw[1:]:
        if len(row) < 7:
            continue
        rank, subdivision, country, continent, national_hdi, national_rank, regional_hdi = (row + [None] * 7)[:7]
        if not str(rank).strip() or not re.search(r"\d", str(rank)):
            continue
        iso3 = iso_for(country)
        subdivision_hdi = fnum(regional_hdi)
        if not iso3 or not math.isfinite(subdivision_hdi):
            continue
        source_rows.append({
            "Rank_Source": int(round(fnum(rank))),
            "Subdivision": str(subdivision).strip(),
            "Country_Source": str(country).strip(),
            "ISO3": iso3,
            "Country": COUNTRY_NAMES.get(iso3, str(country).strip()),
            "Continent_Source": str(continent).strip(),
            "National_HDI_Source": fnum(national_hdi),
            "National_Rank_Source": fnum(national_rank),
            "Subdivision_HDI_2025_Source": subdivision_hdi,
        })

    hdi = pd.read_csv(HDI_CSV)
    cols = [
        "ISO3", "Country", "HDI_Baseline", "HDI_2050", "Population_2024",
        "Population_2050", "DevelopmentMomentumTier", "Trajectory",
        "Urbanization_2024", "GrowthProspectScore",
    ]
    hdi = hdi[[c for c in cols if c in hdi.columns]]
    national_by_iso = hdi.set_index("ISO3").to_dict("index")
    source_rows = [row for row in source_rows if row["ISO3"] in national_by_iso]

    by_iso: dict[str, list[dict]] = {}
    for row in source_rows:
        by_iso.setdefault(row["ISO3"], []).append(row)

    projected = []
    for iso3, rows in by_iso.items():
        national = national_by_iso[iso3]
        target_2025 = float(national.get("HDI_Baseline", rows[0]["National_HDI_Source"]))
        target_2050 = float(national.get("HDI_2050", target_2025))
        weights_2024, weights_2050, pop_2024, pop_2050, weight_method = modeled_population_weights(rows, national)
        source_mean = sum(row["Subdivision_HDI_2025_Source"] * weight for row, weight in zip(rows, weights_2024))
        shrink = max(0.42, min(0.86, 0.86 - 0.38 * max(0.0, target_2050 - target_2025)))

        preliminary = []
        for row in rows:
            deviation = row["Subdivision_HDI_2025_Source"] - source_mean
            hdi_2025 = max(0.250, min(0.995, row["Subdivision_HDI_2025_Source"] + (target_2025 - source_mean)))
            hdi_2050 = max(0.300, min(0.997, target_2050 + deviation * shrink))
            preliminary.append((row, hdi_2025, hdi_2050))

        hdi_2025_values = recenter([item[1] for item in preliminary], weights_2024, target_2025, 0.250, 0.995)
        hdi_2050_values = recenter([item[2] for item in preliminary], weights_2050, target_2050, 0.300, 0.997)

        for idx, (row, _, _) in enumerate(preliminary):
            hdi_2025 = hdi_2025_values[idx]
            hdi_2050 = hdi_2050_values[idx]
            projected.append({
                **row,
                "Subdivision_Weight_2024": weights_2024[idx],
                "Subdivision_Weight_2050": weights_2050[idx],
                "Subdivision_Weight": weights_2050[idx],
                "Weight_Method": weight_method,
                "National_HDI_2025_Target": target_2025,
                "National_HDI_2050_Target": target_2050,
                "Subdivision_HDI_2025_Reconciled": round(hdi_2025, 6),
                "Subdivision_HDI_2050_Projected": round(hdi_2050, 6),
                "Subdivision_HDI_Change_2025_to_2050": round(hdi_2050 - hdi_2025, 6),
                "Country_Subdivision_Count": len(rows),
                "Country_Source_Mean_HDI_2025": round(source_mean, 6),
                "Country_Internal_Gap_Shrink_Factor": round(shrink, 6),
                "Population_2024": national.get("Population_2024"),
                "Population_2050": national.get("Population_2050"),
                "Subdivision_Population_2024_Est": pop_2024[idx],
                "Subdivision_Population_2050_Est": pop_2050[idx],
                "Subdivision_Population_Growth_2024_to_2050": round((pop_2050[idx] / pop_2024[idx]) - 1.0, 6) if pop_2024[idx] else 0,
                "Population_Weight_Confidence": "modeled_proxy_not_official_subdivision_projection",
                "DevelopmentMomentumTier": national.get("DevelopmentMomentumTier", ""),
                "Trajectory": national.get("Trajectory", ""),
            })

    country_rows = []
    df = pd.DataFrame(projected)
    for iso3, group in df.groupby("ISO3"):
        weights_2024 = group["Subdivision_Weight_2024"].astype(float)
        weights_2050 = group["Subdivision_Weight_2050"].astype(float)
        country_rows.append({
            "ISO3": iso3,
            "Country": group["Country"].iloc[0],
            "HDI_2025_Weighted_From_Subdivisions": float((group["Subdivision_HDI_2025_Reconciled"] * weights_2024).sum()),
            "HDI_2050_Weighted_From_Subdivisions": float((group["Subdivision_HDI_2050_Projected"] * weights_2050).sum()),
            "National_HDI_2025_Target": float(group["National_HDI_2025_Target"].iloc[0]),
            "National_HDI_2050_Target": float(group["National_HDI_2050_Target"].iloc[0]),
            "Subdivision_Count": int(len(group)),
            "Subdivision_Population_2024_Est_Total": float(group["Subdivision_Population_2024_Est"].sum()),
            "Subdivision_Population_2050_Est_Total": float(group["Subdivision_Population_2050_Est"].sum()),
            "Min_2025": float(group["Subdivision_HDI_2025_Reconciled"].min()),
            "Max_2025": float(group["Subdivision_HDI_2025_Reconciled"].max()),
            "Min_2050": float(group["Subdivision_HDI_2050_Projected"].min()),
            "Max_2050": float(group["Subdivision_HDI_2050_Projected"].max()),
            "Top_2050": group.sort_values("Subdivision_HDI_2050_Projected", ascending=False).iloc[0]["Subdivision"],
            "Bottom_2050": group.sort_values("Subdivision_HDI_2050_Projected").iloc[0]["Subdivision"],
        })
    return projected, sorted(country_rows, key=lambda row: row["Country"])


def write_csv(rows: list[dict]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def global_hdi_summary() -> dict:
    hdi = pd.read_csv(HDI_CSV)
    pop_2050_mean = (hdi["HDI_2050"] * hdi["Population_2050"]).sum() / hdi["Population_2050"].sum()
    return {
        "Official_HDI_2025": OFFICIAL_WORLD_HDI_2025,
        "Official_HDI_2050": OFFICIAL_WORLD_HDI_2050,
        "Raw_Model_PopWeighted_HDI_2050": float(pop_2050_mean),
        "Country_Count": int(len(hdi)),
    }


def iso_numeric_map(countries: list[dict]) -> dict:
    mapping = {}
    for row in countries:
        country = pycountry.countries.get(alpha_3=row["ISO3"])
        if country and getattr(country, "numeric", None):
            mapping[row["ISO3"]] = str(int(country.numeric))
    return mapping


def write_html(rows: list[dict], countries: list[dict]) -> None:
    global_summary = global_hdi_summary()
    numeric_map = iso_numeric_map(countries)
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Subdivision HDI 2025-2050 Projection</title>
<style>
:root {{ color-scheme: dark; --bg:#08111f; --panel:#101c2e; --line:rgba(255,255,255,.12); --text:#eaf2ff; --muted:#9fb1c9; --cyan:#58e6ff; }}
* {{ box-sizing:border-box }} body {{ margin:0; font-family:Inter,Segoe UI,Arial,sans-serif; background:radial-gradient(circle at top left,#12345c 0,#08111f 38%,#050912 100%); color:var(--text); }}
.shell {{ width:min(1540px,94vw); margin:0 auto; padding:34px 0 48px; }}
.hero {{ display:grid; grid-template-columns:1.3fr .7fr; gap:18px; align-items:stretch; margin-bottom:18px; }}
.card,.panel {{ background:linear-gradient(180deg,rgba(255,255,255,.075),rgba(255,255,255,.035)); border:1px solid var(--line); border-radius:16px; box-shadow:0 20px 60px rgba(0,0,0,.28); backdrop-filter:blur(10px); }}
.card {{ padding:24px; }} h1 {{ margin:0 0 10px; font-size:clamp(30px,4vw,58px); letter-spacing:0; }} p {{ color:var(--muted); line-height:1.55; }}
.kicker {{ color:var(--cyan); font-size:12px; text-transform:uppercase; letter-spacing:.14em; font-weight:800; }}
.stats {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:18px; }}
.stat {{ padding:16px; border-radius:14px; background:rgba(255,255,255,.055); border:1px solid var(--line); }} .stat .v {{ font-size:24px; font-weight:800; }} .stat .k {{ color:var(--muted); font-size:12px; text-transform:uppercase; }}
.maps {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }} .panel {{ padding:14px; min-height:590px; }} .plot {{ width:100%; height:540px; display:block; background:#eef1f4; border-radius:12px; }}
.map-base {{ opacity:.23; }} .map-tint {{ fill:rgba(3,8,18,.08); }} .subpoint {{ stroke:rgba(255,255,255,.92); stroke-width:5; cursor:pointer; transition:stroke-width .15s, opacity .15s; filter:drop-shadow(0 10px 18px rgba(0,0,0,.38)); }} .subpoint:hover {{ stroke:#fff; stroke-width:10; opacity:1; }}
.legend {{ display:flex; align-items:center; gap:10px; color:var(--muted); font-size:12px; margin-top:8px; }} .legend-bar {{ height:10px; width:180px; border-radius:999px; background:linear-gradient(90deg,#4b000b,#d64d4d,#f3d17a,#f7f7f7,#66d9ff,#001052); border:1px solid var(--line); }}
.tooltip {{ position:fixed; pointer-events:none; opacity:0; z-index:20; max-width:310px; padding:10px 12px; border-radius:12px; background:rgba(5,10,20,.94); border:1px solid rgba(255,255,255,.18); box-shadow:0 16px 42px rgba(0,0,0,.35); color:var(--text); font-size:12px; line-height:1.45; }}
.toolbar {{ display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin:18px 0; }} input,select,button {{ background:#0d192a; color:var(--text); border:1px solid var(--line); border-radius:10px; padding:10px 12px; }} button {{ cursor:pointer; background:#123252; }}
.table-wrap {{ max-height:560px; overflow:auto; border:1px solid var(--line); border-radius:14px; }} table {{ width:100%; border-collapse:collapse; font-size:13px; }} th,td {{ padding:9px 10px; border-bottom:1px solid rgba(255,255,255,.08); white-space:nowrap; }} th {{ position:sticky; top:0; background:#111d30; text-align:left; z-index:1; }}
.note {{ border-left:3px solid var(--cyan); padding:12px 14px; background:rgba(88,230,255,.08); color:#cfe9ff; border-radius:10px; }}
@media(max-width:900px) {{ .hero,.maps,.stats {{ grid-template-columns:1fr; }} }}
</style></head><body><div class="shell">
<section class="hero"><div class="card"><div class="kicker">Subnational extension</div><h1>Subdivision HDI 2025 -> 2050</h1><p>Projection layer derived from the supplied subdivision HDI workbook and reconciled to the national HDI 2050 scenario table. Each country's subdivision HDI values are weighted by population-first subdivision shares and re-centered to the national HDI target.</p><div class="note">The source workbook does not include official subdivision populations or subnational map geometries. This version uses seeded population priors for major subdivision countries, a transparent modeled fallback elsewhere, and 2050 evolution based on national population growth, urbanization, and development momentum. Treat these as modeled weights, not official provincial forecasts.</div></div><div class="card"><div class="stats" style="grid-template-columns:1fr 1fr"><div class="stat"><div class="k">Subdivisions</div><div class="v">{len(rows):,}</div></div><div class="stat"><div class="k">Countries</div><div class="v">{len(countries):,}</div></div><div class="stat"><div class="k">Source</div><div class="v">2025</div></div><div class="stat"><div class="k">Projection</div><div class="v">2050</div></div></div></div></section>
<div class="stats"><div class="stat"><div class="k">Global HDI 2025 ref.</div><div class="v" id="mean2025">--</div></div><div class="stat"><div class="k">Global HDI 2050 ref.</div><div class="v" id="mean2050">--</div></div><div class="stat"><div class="k">Largest 2050 internal range</div><div class="v" id="range2050">--</div></div><div class="stat"><div class="k">Selected</div><div class="v" id="selected">World</div></div></div>
<section class="maps"><div class="panel"><h2>Subdivision HDI, 2025</h2><svg id="map2025" class="plot" viewBox="0 0 7200 3318" role="img" aria-label="2025 subdivision HDI symbol map"></svg><div class="legend"><span>0.4</span><div class="legend-bar"></div><span>1.0</span></div></div><div class="panel"><h2>Projected subdivision HDI, 2050</h2><svg id="map2050" class="plot" viewBox="0 0 7200 3318" role="img" aria-label="2050 subdivision HDI symbol map"></svg><div class="legend"><span>0.4</span><div class="legend-bar"></div><span>1.0</span></div></div></section>
<div class="toolbar"><input id="search" type="search" placeholder="Search country or subdivision"><select id="country"><option value="">All countries</option></select><button id="reset">Reset</button><a href="../data/output/subdivision_hdi_2025_2050.csv" style="color:#58e6ff">Open CSV</a></div>
<section class="panel" style="padding:16px"><h2>Subdivision Data</h2><div class="table-wrap"><table><thead><tr><th>Country</th><th>Subdivision</th><th>2050 pop est.</th><th>2050 weight</th><th>Source 2025</th><th>Reconciled 2025</th><th>Projected 2050</th><th>Change</th></tr></thead><tbody id="tbody"></tbody></table></div></section>
</div><div id="tooltip" class="tooltip"></div><script>
const countryData={json.dumps(countries)};
const rows={json.dumps(rows)};
const globalSummary={json.dumps(global_summary)};
const isoNumericByIso3={json.dumps(numeric_map)};
const fmt=v=>Number.isFinite(v)?v.toFixed(3):'--';
const fmtPop=v=>Number.isFinite(v)?Math.round(v).toLocaleString():'--';
const tooltip=document.getElementById('tooltip');
const grouped=new Map();
for (const row of rows) {{
  if (!grouped.has(row.ISO3)) grouped.set(row.ISO3, []);
  grouped.get(row.ISO3).push(row);
}}
const golden=Math.PI*(3-Math.sqrt(5));
const regionBoxes={{
  "America": {{x:390,y:390,w:2300,h:2520,label:"Americas"}},
  "Europe": {{x:2940,y:500,w:1280,h:920,label:"Europe"}},
  "Africa": {{x:3050,y:1320,w:1420,h:1420,label:"Africa"}},
  "Asia/Pacific": {{x:4200,y:320,w:2680,h:2580,label:"Asia / Pacific"}}
}};
function color(value) {{
  const stops=[
    [0.40,[75,0,11]],[0.62,[214,77,77]],[0.74,[243,209,122]],
    [0.82,[247,247,247]],[0.90,[102,217,255]],[1.00,[0,16,82]]
  ];
  const v=Math.max(0.4,Math.min(1,Number(value)));
  for (let i=1;i<stops.length;i++) {{
    if (v<=stops[i][0]) {{
      const v0=stops[i-1][0], c0=stops[i-1][1], v1=stops[i][0], c1=stops[i][1];
      const t=(v-v0)/(v1-v0 || 1);
      const rgb=c0.map((c,j)=>Math.round(c+(c1[j]-c)*t));
      return `rgb(${{rgb[0]}},${{rgb[1]}},${{rgb[2]}})`;
    }}
  }}
  return 'rgb(0,16,82)';
}}
function countryPositions() {{
  const byRegion=new Map();
  for (const country of countryData) {{
    const sample=(grouped.get(country.ISO3)||[])[0];
    const region=sample?.Continent_Source || 'Asia/Pacific';
    if (!byRegion.has(region)) byRegion.set(region, []);
    byRegion.get(region).push(country);
  }}
  const positions=new Map();
  for (const [region, countries] of byRegion) {{
    const box=regionBoxes[region] || regionBoxes["Asia/Pacific"];
    countries.sort((a,b)=>a.Country.localeCompare(b.Country));
    const cols=Math.ceil(Math.sqrt(countries.length*box.w/Math.max(1,box.h)));
    const rowsCount=Math.ceil(countries.length/cols);
    countries.forEach((country,i)=>{{
      const col=i%cols, row=Math.floor(i/cols);
      const x=box.x+120+(col+.5)*(box.w-240)/Math.max(1,cols);
      const y=box.y+120+(row+.5)*(box.h-240)/Math.max(1,rowsCount);
      positions.set(country.ISO3, {{x,y,region}});
    }});
  }}
  return positions;
}}
const positions=countryPositions();
function drawSubdivisionMap(selector, valueKey, title) {{
  const svg=document.querySelector(selector);
  svg.innerHTML='';
  const ns='http://www.w3.org/2000/svg';
  const make=(name, attrs={{}})=>{{ const el=document.createElementNS(ns,name); for (const [k,v] of Object.entries(attrs)) el.setAttribute(k,v); return el; }};
  svg.appendChild(make('image', {{class:'map-base', href:'assets/subdivision-reference-map.png', x:0, y:0, width:7200, height:3318, preserveAspectRatio:'xMidYMid meet'}}));
  svg.appendChild(make('rect', {{class:'map-tint', x:0, y:0, width:7200, height:3318}}));
  const positioned=[];
  for (const [iso, group] of grouped) {{
    const centroid=positions.get(iso);
    if (!centroid) continue;
    const count=group.length;
    const base=Math.max(42, Math.min(145, Math.sqrt(count)*18));
    group.forEach((row, i) => {{
      const ring=count===1 ? 0 : Math.sqrt((i+0.35)/count)*base;
      const angle=i*golden;
      positioned.push({{...row, x:centroid.x+Math.cos(angle)*ring, y:centroid.y+Math.sin(angle)*ring, value:Number(row[valueKey])}});
    }});
  }}
  for (const d of positioned) {{
    const node=make('circle', {{
      class:'subpoint', cx:d.x, cy:d.y,
      r:Math.max(18, Math.min(46, Math.sqrt(Number(d.Subdivision_Weight_2050 || d.Subdivision_Weight_2024 || 0.01))*150)),
      fill:color(d.value), opacity:0.88
    }});
    node.addEventListener('mouseenter', event=>showTip(event,d,title,valueKey));
    node.addEventListener('mousemove', moveTip);
    node.addEventListener('mouseleave', hideTip);
    svg.appendChild(node);
  }}
}}
function showTip(event,d,title,valueKey) {{
  tooltip.innerHTML=`<strong>${{d.Subdivision}}</strong><br>${{d.Country}} (${{d.ISO3}})<br>${{title}}: ${{fmt(Number(d[valueKey]))}}<br>2025 source: ${{fmt(Number(d.Subdivision_HDI_2025_Source))}}<br>2025 reconciled: ${{fmt(Number(d.Subdivision_HDI_2025_Reconciled))}}<br>2050 projected: ${{fmt(Number(d.Subdivision_HDI_2050_Projected))}}<br>2050 pop est.: ${{fmtPop(Number(d.Subdivision_Population_2050_Est))}}<br>2050 weight: ${{(Number(d.Subdivision_Weight_2050)*100).toFixed(2)}}%`;
  moveTip(event);
  tooltip.style.opacity='1';
}}
function moveTip(event) {{ tooltip.style.left=`${{Math.min(window.innerWidth-330,event.clientX+14)}}px`; tooltip.style.top=`${{event.clientY+14}}px`; }}
function hideTip() {{ tooltip.style.opacity='0'; }}
function fillCountries(){{ const sel=document.getElementById('country'); countryData.forEach(d=>{{ const o=document.createElement('option'); o.value=d.ISO3; o.textContent=d.Country; sel.appendChild(o); }}); }}
function renderTable(){{ const q=document.getElementById('search').value.toLowerCase().trim(); const iso=document.getElementById('country').value; const filtered=rows.filter(r=>(!iso||r.ISO3===iso)&&(!q||r.Country.toLowerCase().includes(q)||r.Subdivision.toLowerCase().includes(q)||r.ISO3.toLowerCase().includes(q))).slice(0,900); document.getElementById('tbody').innerHTML=filtered.map(r=>`<tr><td>${{r.Country}}</td><td>${{r.Subdivision}}</td><td>${{fmtPop(r.Subdivision_Population_2050_Est)}}</td><td>${{(r.Subdivision_Weight_2050*100).toFixed(2)}}%</td><td>${{fmt(r.Subdivision_HDI_2025_Source)}}</td><td>${{fmt(r.Subdivision_HDI_2025_Reconciled)}}</td><td>${{fmt(r.Subdivision_HDI_2050_Projected)}}</td><td>${{r.Subdivision_HDI_Change_2025_to_2050>=0?'+':''}}${{fmt(r.Subdivision_HDI_Change_2025_to_2050)}}</td></tr>`).join(''); document.getElementById('selected').textContent=iso?(countryData.find(d=>d.ISO3===iso)||{{Country:'--'}}).Country:'World'; }}
function stats(){{ document.getElementById('mean2025').textContent=fmt(globalSummary.Official_HDI_2025); document.getElementById('mean2050').textContent=fmt(globalSummary.Official_HDI_2050); const r=countryData.map(d=>({{c:d.Country,v:d.Max_2050-d.Min_2050}})).sort((a,b)=>b.v-a.v)[0]; document.getElementById('range2050').textContent=r.c+' '+fmt(r.v); }}
drawSubdivisionMap('#map2025','Subdivision_HDI_2025_Source','Subdivision HDI 2025');
drawSubdivisionMap('#map2050','Subdivision_HDI_2050_Projected','Projected subdivision HDI 2050');
fillCountries(); stats(); renderTable();
document.getElementById('search').addEventListener('input',renderTable); document.getElementById('country').addEventListener('change',renderTable); document.getElementById('reset').addEventListener('click',()=>{{document.getElementById('search').value='';document.getElementById('country').value='';renderTable();}});
</script></body></html>"""
    OUT_HTML.write_text(html, encoding="utf-8")


def validate(rows: list[dict]) -> list[tuple]:
    df = pd.DataFrame(rows)
    errors = []
    for iso3, group in df.groupby("ISO3"):
        weighted_2025 = (group["Subdivision_HDI_2025_Reconciled"] * group["Subdivision_Weight_2024"]).sum()
        weighted_2050 = (group["Subdivision_HDI_2050_Projected"] * group["Subdivision_Weight_2050"]).sum()
        target_2025 = group["National_HDI_2025_Target"].iloc[0]
        target_2050 = group["National_HDI_2050_Target"].iloc[0]
        weight_sum_2024 = group["Subdivision_Weight_2024"].sum()
        weight_sum_2050 = group["Subdivision_Weight_2050"].sum()
        pop_sum_2024 = group["Subdivision_Population_2024_Est"].sum()
        pop_sum_2050 = group["Subdivision_Population_2050_Est"].sum()
        national_pop_2024 = finite_or(group["Population_2024"].iloc[0], pop_sum_2024)
        national_pop_2050 = finite_or(group["Population_2050"].iloc[0], pop_sum_2050)
        if (
            abs(weighted_2025 - target_2025) > 2e-5
            or abs(weighted_2050 - target_2050) > 2e-5
            or abs(weight_sum_2024 - 1.0) > 1e-9
            or abs(weight_sum_2050 - 1.0) > 1e-9
            or abs(pop_sum_2024 - national_pop_2024) > max(2.0, national_pop_2024 * 1e-9)
            or abs(pop_sum_2050 - national_pop_2050) > max(2.0, national_pop_2050 * 1e-9)
        ):
            errors.append((iso3, weighted_2025, target_2025, weighted_2050, target_2050, pop_sum_2024, national_pop_2024, pop_sum_2050, national_pop_2050))
    return errors


def main() -> None:
    rows, countries = build_rows()
    if not rows:
        raise RuntimeError("No subdivision rows were generated.")
    errors = validate(rows)
    if errors:
        raise RuntimeError(f"Subdivision weighted means do not reconcile: {errors[:5]}")
    write_csv(rows)
    write_html(rows, countries)
    print(f"Subdivision rows: {len(rows):,}")
    print(f"Countries covered: {len(countries):,}")
    print(f"Reconciliation errors: {len(errors)}")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_HTML}")


if __name__ == "__main__":
    main()
