"""Build modeled 2025-2050 HDI estimates for national capitals and major cities."""

from __future__ import annotations

import csv
import json
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
ADM1_GEOJSON_PATH = ROOT / "web" / "assets" / "geo" / "adm1" / "world_subdivisions.geojson"
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
    "FRA": {"ile de france": "ile de france", "provence alpes cote d azur": "provence alpes cote dazur", "auvergne rhone alpes": "rhone alpes"},
    "BRA": {"federal district": "distrito federal"},
    "AUS": {"act": "australian capital territory"},
    "PAK": {"islamabad": "islamabad ict"},
    "ETH": {"addis ababa": "addis"},
    "USA": {"district of columbia": "district of columbia", "dc": "district of columbia"},
    "GBR": {"england": ""},
    "DEU": {"bavaria": "bayern", "north rhine westphalia": "nordrhein westfalen", "lower saxony": "niedersachsen"},
    "MEX": {"mexico city": "distrito federal", "ciudad de mexico": "distrito federal"},
    "IDN": {"jakarta": "dki jakarta", "yogyakarta": "di yogyakarta"},
    "ESP": {"madrid": "comunidad de madrid", "valencia": "comunidad valenciana"},
}

COORDINATE_SUBDIVISION_OVERRIDES = {
    ("USA", "Washington"): "District of Columbia",
    ("GBR", "Birmingham"): "West Midlands",
    ("GBR", "Manchester"): "North West",
    ("GBR", "Sheffield"): "Yorkshire and The Humber",
    ("GBR", "Glasgow"): "Scotland",
    ("GBR", "London"): "London",
    ("FRA", "Toulouse"): "Midi-Pyrenees",
}

REGIONAL_SUBDIVISION_ALIASES = {
    "KOR": {
        "seoul": "Capital Region", "incheon": "Capital Region", "gyeonggi": "Capital Region",
        "busan": "Gyeongnam Region", "ulsan": "Gyeongnam Region", "south gyeongsang": "Gyeongnam Region",
        "daejeon": "Chungcheong Region", "sejong": "Chungcheong Region", "north chungcheong": "Chungcheong Region", "south chungcheong": "Chungcheong Region",
        "gwangju": "Jeolla Region", "north jeolla": "Jeolla Region", "south jeolla": "Jeolla Region",
        "gangwon": "Gangwon Region", "jeju": "Jeju",
        "daegu": "Gyeongbuk Region", "north gyeongsang": "Gyeongbuk Region",
    },
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


def load_adm1_features() -> dict[str, list[dict[str, object]]]:
    with ADM1_GEOJSON_PATH.open(encoding="utf-8") as handle:
        features = json.load(handle)["features"]
    by_iso: dict[str, list[dict[str, object]]] = {}
    for feature in features:
        iso3 = feature.get("properties", {}).get("adm0_a3")
        if iso3:
            by_iso.setdefault(str(iso3), []).append(feature)
    return by_iso


def point_in_ring(longitude: float, latitude: float, ring: list[list[float]]) -> bool:
    if len(ring) < 3:
        return False
    longitudes = [point[0] for point in ring]
    crosses_dateline = max(longitudes) - min(longitudes) > 180
    x = longitude + 360 if crosses_dateline and longitude < 0 else longitude
    inside = False
    previous = ring[-1]
    previous_x = previous[0] + 360 if crosses_dateline and previous[0] < 0 else previous[0]
    previous_y = previous[1]
    for current in ring:
        current_x = current[0] + 360 if crosses_dateline and current[0] < 0 else current[0]
        current_y = current[1]
        if (current_y > latitude) != (previous_y > latitude):
            crossing_x = (previous_x - current_x) * (latitude - current_y) / (previous_y - current_y) + current_x
            if x < crossing_x:
                inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside


def point_in_polygon(longitude: float, latitude: float, polygon: list[list[list[float]]]) -> bool:
    return bool(polygon and point_in_ring(longitude, latitude, polygon[0]) and not any(
        point_in_ring(longitude, latitude, hole) for hole in polygon[1:]
    ))


def geometry_contains(geometry: dict[str, object], longitude: float, latitude: float) -> bool:
    coordinates = geometry.get("coordinates", [])
    if geometry.get("type") == "Polygon":
        return point_in_polygon(longitude, latitude, coordinates)
    if geometry.get("type") == "MultiPolygon":
        return any(point_in_polygon(longitude, latitude, polygon) for polygon in coordinates)
    return False


def locate_adm1(city: dict[str, str], iso3: str, adm1_features: dict[str, list[dict[str, object]]]) -> dict[str, object] | None:
    longitude = float(city["longitude"])
    latitude = float(city["latitude"])
    return next((
        feature for feature in adm1_features.get(iso3, [])
        if geometry_contains(feature.get("geometry", {}), longitude, latitude)
    ), None)


def feature_names(feature: dict[str, object] | None) -> list[str]:
    if not feature:
        return []
    properties = feature.get("properties", {})
    values = [properties.get("name"), properties.get("name_en")]
    values.extend(str(properties.get("name_alt") or "").split("|"))
    return [str(value) for value in values if value]


def subdivision_match(
    city: dict[str, str],
    iso3: str,
    admin_names: dict[tuple[str, str], str],
    subdivisions: list[dict[str, str]],
    adm1_features: dict[str, list[dict[str, object]]],
) -> tuple[dict[str, str] | None, str, str]:
    alpha2 = city["country_alpha2"]
    admin_name = admin_names.get((alpha2, city["admin1_code"]), "")
    located_feature = locate_adm1(city, iso3, adm1_features)
    located_names = feature_names(located_feature)
    geospatial_name = located_names[0] if located_names else admin_name
    candidates = {normalize(admin_name), *(normalize(name) for name in located_names)}
    aliases = SUBDIVISION_ALIASES.get(iso3, {})
    candidates |= {aliases.get(candidate, candidate) for candidate in list(candidates)}
    city_name = normalize(city["ascii_name"] or city["name"])
    override = COORDINATE_SUBDIVISION_OVERRIDES.get((iso3, city["ascii_name"] or city["name"]))
    if override:
        exact = next((row for row in subdivisions if normalize(row["Subdivision"]) == normalize(override)), None)
        if exact:
            return exact, "manual_subdivision_override", geospatial_name

    regional_aliases = REGIONAL_SUBDIVISION_ALIASES.get(iso3, {})
    for candidate in list(candidates):
        regional_name = regional_aliases.get(candidate)
        if regional_name:
            exact = next((row for row in subdivisions if normalize(row["Subdivision"]) == normalize(regional_name)), None)
            if exact:
                return exact, "geospatial_regional_match", geospatial_name
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
        method = "geospatial_subdivision_match" if located_feature else "admin_subdivision_match"
        return best[1], method, geospatial_name
    return None, "national_fallback", geospatial_name


def city_adjustment(
    city: dict[str, str],
    selected_rank: int,
    is_capital: bool,
    anchor_hdi: float,
    national_hdi: float,
    projection_year: int,
) -> float:
    population = max(0, int(city["population"] or 0))
    size_signal = max(0.0, min(1.0, math.log10(max(50_000, population) / 50_000) / math.log10(200)))
    rank_signal = max(0.0, min(1.0, 1.0 - (selected_rank - 1) / 6))
    development_gap = max(0.0, min(1.0, (0.88 - national_hdi) / 0.48))
    raw_premium = 0.007 + 0.012 * size_signal + 0.009 * float(is_capital) + 0.004 * rank_signal + 0.009 * development_gap
    if projection_year == 2050:
        raw_premium *= 0.72 + 0.08 * development_gap
    headroom = max(0.001, 0.992 - anchor_hdi)
    return headroom * (1 - math.exp(-raw_premium / headroom))


def city_hdi_from_anchor(
    city: dict[str, str],
    selected_rank: int,
    is_capital: bool,
    anchor_hdi: float,
    national_hdi: float,
    projection_year: int,
) -> tuple[float, float]:
    premium = city_adjustment(city, selected_rank, is_capital, anchor_hdi, national_hdi, projection_year)
    population = max(0, int(city["population"] or 0))
    size_signal = max(0.0, min(1.0, math.log10(max(50_000, population) / 50_000) / math.log10(200)))
    development_gap = max(0.0, min(1.0, (0.88 - national_hdi) / 0.48))
    floor_gap = 0.005 + 0.007 * size_signal + 0.006 * float(is_capital) + 0.003 * development_gap
    if projection_year == 2050:
        floor_gap *= 0.76
    metropolitan_floor = national_hdi + floor_gap
    maximum_regional_uplift = 0.052 + 0.014 * float(is_capital) + 0.008 * size_signal
    city_hdi = max(anchor_hdi + premium, min(metropolitan_floor, anchor_hdi + maximum_regional_uplift))
    city_hdi = min(0.992, max(0.350, city_hdi))
    return city_hdi, city_hdi - anchor_hdi


def main() -> None:
    if not GEONAMES_ZIP.exists() or not ADMIN1_PATH.exists():
        raise FileNotFoundError("Download cities15000.zip and admin1CodesASCII.txt from GeoNames before running this builder.")

    national_rows = load_csv(HDI_PATH)
    subdivision_rows = load_csv(SUBDIVISION_PATH)
    admin_names = load_admin_names()
    adm1_features = load_adm1_features()
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
            subdivision, source_method, geospatial_adm1 = subdivision_match(city, iso3, admin_names, country_subdivisions, adm1_features)
            if subdivision:
                base_2025 = float(subdivision["Subdivision_HDI_2025_Reconciled"])
                base_2050 = float(subdivision["Subdivision_HDI_2050_Projected"])
                subdivision_name = subdivision["Subdivision"]
            else:
                base_2025 = national_2025
                base_2050 = national_2050
                subdivision_name = admin_names.get((alpha2, city["admin1_code"]), "")

            city_2025, adjustment_2025 = city_hdi_from_anchor(city, city_rank, is_capital, base_2025, national_2025, 2025)
            city_2050, adjustment_2050 = city_hdi_from_anchor(city, city_rank, is_capital, base_2050, national_2050, 2050)
            confidence = "higher" if source_method.startswith(("geospatial", "manual")) else "moderate" if subdivision else "modeled"
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
                "Geospatial_ADM1": geospatial_adm1,
                "Subdivision_HDI_2025_Anchor": round(base_2025, 6),
                "Subdivision_HDI_2050_Anchor": round(base_2050, 6),
                "City_HDI_2025": round(city_2025, 6),
                "City_HDI_2050": round(city_2050, 6),
                "City_HDI_Change": round(city_2050 - city_2025, 6),
                "National_HDI_2025": round(national_2025, 6),
                "National_HDI_2050": round(national_2050, 6),
                "Urban_Premium_2025": round(city_2025 - base_2025, 6),
                "Urban_Premium_2050": round(city_2050 - base_2050, 6),
                "City_vs_National_2025": round(city_2025 - national_2025, 6),
                "City_vs_National_2050": round(city_2050 - national_2050, 6),
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
