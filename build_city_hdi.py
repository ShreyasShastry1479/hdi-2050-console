"""Build modeled 2025-2050 HDI estimates for national capitals and major cities."""

from __future__ import annotations

import csv
import math
import os
import re
import unicodedata
import zipfile
from pathlib import Path

import pycountry


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data" / "output" / "city_hdi_2025_2050.csv"
HDI_PATH = ROOT / "data" / "output" / "hdi_2050_rankings.csv"
SUBDIVISION_PATH = ROOT / "data" / "output" / "subdivision_hdi_2025_2050.csv"
GEONAMES_ZIP = Path(os.environ.get("TEMP", ".")) / "cities15000.zip"
ADMIN1_PATH = Path(os.environ.get("TEMP", ".")) / "admin1CodesASCII.txt"

GEONAMES_COLUMNS = [
    "geoname_id", "name", "ascii_name", "alternate_names", "latitude", "longitude",
    "feature_class", "feature_code", "country_alpha2", "alternate_country_codes",
    "admin1_code", "admin2_code", "admin3_code", "admin4_code", "population",
    "elevation", "dem", "timezone", "modification_date",
]

COUNTRY_ALPHA2_OVERRIDES = {
    "COD": "CD", "COG": "CG", "KOS": "XK", "PSE": "PS",
}

SUBDIVISION_ALIASES = {
    "IND": {"delhi": "new delhi", "odisha": "orissa", "uttarakhand": "uttaranchal"},
    "NGA": {"fct": "abuja fct", "federal capital territory": "abuja fct", "abuja federal capital territory": "abuja fct", "nasarawa": "nassarawa"},
    "FRA": {"ile de france": "ile de france", "provence alpes cote d azur": "provence alpes cote dazur"},
    "BRA": {"federal district": "distrito federal"},
    "AUS": {"act": "australian capital territory"},
    "PAK": {"islamabad": "islamabad ict"},
    "ETH": {"addis ababa": "addis"},
    "USA": {"district of columbia": "district of columbia", "dc": "district of columbia"},
    "GBR": {"england": ""},
}

COORDINATE_SUBDIVISION_OVERRIDES = {
    ("USA", "Washington"): "District of Columbia",
    ("GBR", "Birmingham"): "West Midlands",
    ("GBR", "Manchester"): "North West",
    ("GBR", "Sheffield"): "Yorkshire and The Humber",
    ("GBR", "Glasgow"): "Scotland",
    ("GBR", "London"): "London",
}


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value or "").encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def alpha2_for_iso3(iso3: str) -> str | None:
    if iso3 in COUNTRY_ALPHA2_OVERRIDES:
        return COUNTRY_ALPHA2_OVERRIDES[iso3]
    country = pycountry.countries.get(alpha_3=iso3)
    return country.alpha_2 if country else None


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_admin_names() -> dict[tuple[str, str], str]:
    names: dict[tuple[str, str], str] = {}
    with ADMIN1_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            code, name, ascii_name, *_ = line.rstrip("\n").split("\t")
            country, admin1 = code.split(".", 1)
            names[(country, admin1)] = ascii_name or name
    return names


def load_geonames() -> list[dict[str, str]]:
    with zipfile.ZipFile(GEONAMES_ZIP) as archive:
        file_name = next(name for name in archive.namelist() if name.endswith(".txt"))
        with archive.open(file_name) as raw:
            rows = []
            for line in raw:
                values = line.decode("utf-8").rstrip("\n").split("\t")
                if len(values) == len(GEONAMES_COLUMNS):
                    rows.append(dict(zip(GEONAMES_COLUMNS, values)))
            return rows


def subdivision_match(city: dict[str, str], iso3: str, admin_names: dict[tuple[str, str], str], subdivisions: list[dict[str, str]]) -> tuple[dict[str, str] | None, str]:
    alpha2 = city["country_alpha2"]
    admin_name = admin_names.get((alpha2, city["admin1_code"]), "")
    candidates = {normalize(admin_name)}
    aliases = SUBDIVISION_ALIASES.get(iso3, {})
    candidates |= {aliases.get(candidate, candidate) for candidate in list(candidates)}
    city_name = normalize(city["ascii_name"] or city["name"])
    override = COORDINATE_SUBDIVISION_OVERRIDES.get((iso3, city["ascii_name"] or city["name"]))
    if override:
        exact = next((row for row in subdivisions if normalize(row["Subdivision"]) == normalize(override)), None)
        if exact:
            return exact, "matched_subdivision"
    candidates.add(aliases.get(city_name, city_name))

    best: tuple[float, dict[str, str]] | None = None
    for row in subdivisions:
        subdivision_name = normalize(row["Subdivision"])
        subdivision_alias = aliases.get(subdivision_name, subdivision_name)
        score = 0.0
        for candidate in candidates:
            if not candidate:
                continue
            if candidate == subdivision_alias:
                score = max(score, 1.0)
            elif candidate in subdivision_alias or subdivision_alias in candidate:
                score = max(score, 0.82)
            else:
                candidate_tokens = set(candidate.split())
                subdivision_tokens = set(subdivision_alias.split())
                overlap = len(candidate_tokens & subdivision_tokens) / max(1, len(candidate_tokens | subdivision_tokens))
                score = max(score, overlap)
        if score >= 0.50 and (best is None or score > best[0]):
            best = (score, row)
    if best:
        return best[1], "matched_subdivision"
    return None, "national_fallback"


def city_adjustment(city: dict[str, str], selected_rank: int, is_capital: bool, national_hdi: float) -> float:
    population = max(0, int(city["population"] or 0))
    size_signal = max(0.0, min(1.0, math.log10(max(50_000, population) / 50_000) / math.log10(200)))
    role_signal = 1.0 if is_capital else max(0.15, 0.75 - 0.12 * selected_rank)
    development_room = max(0.20, min(1.0, (0.93 - national_hdi) / 0.43))
    return (0.004 + 0.011 * size_signal + 0.006 * role_signal) * development_room


def main() -> None:
    if not GEONAMES_ZIP.exists() or not ADMIN1_PATH.exists():
        raise FileNotFoundError("Download cities15000.zip and admin1CodesASCII.txt from GeoNames before running this builder.")

    national_rows = load_csv(HDI_PATH)
    subdivision_rows = load_csv(SUBDIVISION_PATH)
    admin_names = load_admin_names()
    geonames_rows = load_geonames()
    geonames_by_country: dict[str, list[dict[str, str]]] = {}
    for row in geonames_rows:
        geonames_by_country.setdefault(row["country_alpha2"], []).append(row)

    subdivisions_by_iso: dict[str, list[dict[str, str]]] = {}
    for row in subdivision_rows:
        subdivisions_by_iso.setdefault(row["ISO3"], []).append(row)

    output_rows: list[dict[str, object]] = []
    for national in national_rows:
        iso3 = national["ISO3"]
        alpha2 = alpha2_for_iso3(iso3)
        if not alpha2:
            continue
        country_cities = geonames_by_country.get(alpha2, [])
        if not country_cities:
            continue

        capitals = [city for city in country_cities if city["feature_code"] == "PPLC"]
        if iso3 in {"ISR", "PSE"}:
            preferred_names = {"ISR": {"jerusalem"}, "PSE": {"east jerusalem"}}[iso3]
            capitals = [city for city in country_cities if normalize(city["ascii_name"] or city["name"]) in preferred_names] or capitals
        ranked = sorted(country_cities, key=lambda city: int(city["population"] or 0), reverse=True)
        selected: list[dict[str, str]] = []
        for city in capitals + ranked[:5]:
            if city["geoname_id"] not in {item["geoname_id"] for item in selected}:
                selected.append(city)
        selected = selected[:6]

        national_2025 = float(national["HDI_Baseline"])
        national_2050 = float(national["HDI_2050"])
        country_subdivisions = subdivisions_by_iso.get(iso3, [])
        for city_rank, city in enumerate(selected, start=1):
            is_capital = city in capitals
            subdivision, source_method = subdivision_match(city, iso3, admin_names, country_subdivisions)
            if subdivision:
                base_2025 = float(subdivision["Subdivision_HDI_2025_Reconciled"])
                base_2050 = float(subdivision["Subdivision_HDI_2050_Projected"])
                subdivision_name = subdivision["Subdivision"]
            else:
                base_2025 = national_2025
                base_2050 = national_2050
                subdivision_name = admin_names.get((alpha2, city["admin1_code"]), "")

            adjustment_2025 = city_adjustment(city, city_rank, is_capital, national_2025)
            adjustment_2050 = adjustment_2025 * (0.64 if national_2050 >= national_2025 else 0.82)
            city_2025 = min(0.990, max(0.350, base_2025 + adjustment_2025))
            city_2050 = min(0.990, max(0.350, base_2050 + adjustment_2050))
            confidence = "higher" if subdivision else "modeled"
            if not int(city["population"] or 0):
                confidence = "lower"

            output_rows.append({
                "City": city["ascii_name"] or city["name"],
                "Country": national["Country"],
                "ISO3": iso3,
                "Country_Alpha2": alpha2,
                "Continent": next((row["Continent_Source"] for row in country_subdivisions if row["Continent_Source"]), ""),
                "Latitude": round(float(city["latitude"]), 5),
                "Longitude": round(float(city["longitude"]), 5),
                "GeoNames_Population": int(city["population"] or 0),
                "Is_Capital": is_capital,
                "City_Rank_In_Selected_Country": city_rank,
                "Matched_Subdivision": subdivision_name,
                "City_HDI_2025": round(city_2025, 6),
                "City_HDI_2050": round(city_2050, 6),
                "City_HDI_Change": round(city_2050 - city_2025, 6),
                "National_HDI_2025": round(national_2025, 6),
                "National_HDI_2050": round(national_2050, 6),
                "Urban_Premium_2025": round(city_2025 - base_2025, 6),
                "Urban_Premium_2050": round(city_2050 - base_2050, 6),
                "Projection_Method": source_method + "_plus_bounded_urban_premium",
                "Confidence": confidence,
                "Data_Status": "modeled_city_estimate_not_official_undp_city_hdi",
                "GeoNames_ID": city["geoname_id"],
            })

    output_rows.sort(key=lambda row: (-float(row["City_HDI_2050"]), str(row["Country"]), str(row["City"])))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} city estimates across {len({row['ISO3'] for row in output_rows})} countries to {OUTPUT}")


if __name__ == "__main__":
    main()
