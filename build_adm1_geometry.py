from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SUBDIVISION_CSV = PROJECT_ROOT / "data" / "output" / "subdivision_hdi_2025_2050.csv"
NATIONAL_HDI_CSV = PROJECT_ROOT / "data" / "output" / "hdi_2050_rankings.csv"
OUTPUT_DIR = PROJECT_ROOT / "web" / "assets" / "geo" / "adm1"
WORLD_OUTPUT = OUTPUT_DIR / "world_subdivisions.geojson"
SIMPLIFY_TOLERANCE_DEGREES = 0.0
WORLD_SIMPLIFY_TOLERANCE_DEGREES = 0.035
COORDINATE_PRECISION = 4
FALLBACK_SOURCES = {
    "PSE": {
        "source": "geoBoundaries gbOpen ADM2 (governorates)",
        "license": "Creative Commons Attribution 4.0 (CC BY 4.0)",
        "source_url": "https://www.geoboundaries.org/api/current/gbOpen/PSE/ADM2/",
    },
    "SSD": {
        "source": "geoBoundaries gbOpen ADM1",
        "license": "Creative Commons Attribution 3.0 IGO (CC BY 3.0 IGO)",
        "source_url": "https://www.geoboundaries.org/api/current/gbOpen/SSD/ADM1/",
    },
}


def point_segment_distance_sq(point: list[float], start: list[float], end: list[float]) -> float:
    px, py = point[:2]
    ax, ay = start[:2]
    bx, by = end[:2]
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return (px - ax) ** 2 + (py - ay) ** 2
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    qx, qy = ax + t * dx, ay + t * dy
    return (px - qx) ** 2 + (py - qy) ** 2


def simplify_line(points: list[list[float]], tolerance: float) -> list[list[float]]:
    if len(points) <= 2:
        return points
    tolerance_sq = tolerance * tolerance
    keep = {0, len(points) - 1}
    stack = [(0, len(points) - 1)]
    while stack:
        start_index, end_index = stack.pop()
        start, end = points[start_index], points[end_index]
        furthest_index = -1
        furthest_distance = tolerance_sq
        for index in range(start_index + 1, end_index):
            distance = point_segment_distance_sq(points[index], start, end)
            if distance > furthest_distance:
                furthest_distance = distance
                furthest_index = index
        if furthest_index >= 0:
            keep.add(furthest_index)
            stack.append((start_index, furthest_index))
            stack.append((furthest_index, end_index))
    return [points[index] for index in sorted(keep)]


def simplify_ring(ring: list[list[float]]) -> list[list[float]]:
    rounded: list[list[float]] = []
    for point in ring:
        candidate = [round(value, COORDINATE_PRECISION) for value in point[:2]]
        if not rounded or candidate != rounded[-1]:
            rounded.append(candidate)
    if rounded[0] != rounded[-1]:
        rounded.append(rounded[0])
    return rounded


def simplify_world_ring(ring: list[list[float]]) -> list[list[float]]:
    rounded = simplify_ring(ring)
    simplified = simplify_line(rounded, WORLD_SIMPLIFY_TOLERANCE_DEGREES)
    if len(simplified) < 4:
        simplified = rounded
    if simplified[0] != simplified[-1]:
        simplified.append(simplified[0])
    return simplified


def simplify_world_geometry(geometry: dict) -> dict:
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])
    if geometry_type == "Polygon":
        simplified = [simplify_world_ring(ring) for ring in coordinates]
    elif geometry_type == "MultiPolygon":
        simplified = [[simplify_world_ring(ring) for ring in polygon] for polygon in coordinates]
    else:
        simplified = coordinates
    return {"type": geometry_type, "coordinates": simplified}


def simplify_geometry(geometry: dict) -> dict:
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])
    if geometry_type == "Polygon":
        simplified = [simplify_ring(ring) for ring in coordinates]
    elif geometry_type == "MultiPolygon":
        simplified = [[simplify_ring(ring) for ring in polygon] for polygon in coordinates]
    else:
        simplified = coordinates
    return {"type": geometry_type, "coordinates": simplified}


def dataset_iso3_codes() -> set[str]:
    with SUBDIVISION_CSV.open(newline="", encoding="utf-8-sig") as handle:
        subdivision_codes = {row["ISO3"].strip() for row in csv.DictReader(handle) if row.get("ISO3")}
    with NATIONAL_HDI_CSV.open(newline="", encoding="utf-8-sig") as handle:
        national_codes = {row["ISO3"].strip() for row in csv.DictReader(handle) if row.get("ISO3")}
    return subdivision_codes | national_codes


def compact_properties(properties: dict) -> dict:
    keys = (
        "name",
        "name_en",
        "name_alt",
        "name_local",
        "region",
        "region_sub",
        "geonunit",
        "adm0_a3",
        "iso_3166_2",
        "adm1_code",
        "type_en",
    )
    return {key: properties.get(key) for key in keys if properties.get(key) not in (None, "")}


def build(source_path: Path) -> None:
    with source_path.open(encoding="utf-8") as handle:
        source = json.load(handle)

    wanted_iso3 = dataset_iso3_codes()
    by_country: dict[str, list[dict]] = defaultdict(list)
    source_feature_count = len(source.get("features", []))
    for feature in source.get("features", []):
        properties = feature.get("properties", {})
        iso3 = properties.get("adm0_a3")
        if iso3 not in wanted_iso3 or not feature.get("geometry"):
            continue
        by_country[iso3].append(
            {
                "type": "Feature",
                "properties": compact_properties(properties),
                "geometry": simplify_geometry(feature["geometry"]),
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    index = {
        "source": "Natural Earth ne_10m_admin_1_states_provinces",
        "source_url": "https://github.com/nvkelso/natural-earth-vector",
        "license": "Public domain",
        "simplify_tolerance_degrees": SIMPLIFY_TOLERANCE_DEGREES,
        "coordinate_precision": COORDINATE_PRECISION,
        "source_feature_count": source_feature_count,
        "countries": {},
    }
    for iso3, features in sorted(by_country.items()):
        payload = {"type": "FeatureCollection", "features": features}
        destination = OUTPUT_DIR / f"{iso3}.geojson"
        destination.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        index["countries"][iso3] = {"features": len(features), "file": destination.name}

    for iso3, metadata in FALLBACK_SOURCES.items():
        destination = OUTPUT_DIR / f"{iso3}.geojson"
        if iso3 not in wanted_iso3 or not destination.exists():
            continue
        with destination.open(encoding="utf-8") as handle:
            fallback_payload = json.load(handle)
        for feature in fallback_payload.get("features", []):
            properties = feature.setdefault("properties", {})
            shape_name = properties.get("shapeName")
            if shape_name:
                properties.setdefault("name", shape_name)
                properties.setdefault("name_en", shape_name)
            properties.setdefault("adm0_a3", iso3)
        destination.write_text(
            json.dumps(fallback_payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        index["countries"][iso3] = {
            "features": len(fallback_payload.get("features", [])),
            "file": destination.name,
            **metadata,
        }

    available_iso3 = set(index["countries"])
    missing = sorted(wanted_iso3 - available_iso3)
    index["dataset_country_count"] = len(wanted_iso3)
    index["geometry_country_count"] = len(available_iso3)
    index["missing_dataset_iso3"] = missing
    (OUTPUT_DIR / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    world_features = []
    for iso3 in sorted(available_iso3):
        with (OUTPUT_DIR / f"{iso3}.geojson").open(encoding="utf-8") as handle:
            country_payload = json.load(handle)
        for feature in country_payload.get("features", []):
            properties = compact_properties(feature.get("properties", {}))
            properties["adm0_a3"] = iso3
            world_features.append({
                "type": "Feature",
                "properties": properties,
                # Closed-ring RDP simplification can invert small multipart regions
                # on a spherical projection. The country assets are already rounded
                # and topology-checked, so reuse that geometry for the world map.
                "geometry": feature["geometry"],
            })
    WORLD_OUTPUT.write_text(
        json.dumps({"type": "FeatureCollection", "features": world_features}, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    total_features = sum(item["features"] for item in index["countries"].values())
    total_bytes = sum(path.stat().st_size for path in OUTPUT_DIR.glob("*.geojson") if path != WORLD_OUTPUT)
    print(f"Natural Earth source features: {source_feature_count:,}")
    print(f"Generated countries: {len(available_iso3):,}/{len(wanted_iso3):,}")
    print(f"Generated ADM1 features: {total_features:,}")
    print(f"Generated geometry size: {total_bytes / 1_000_000:.2f} MB")
    print(f"World overview geometry: {WORLD_OUTPUT.stat().st_size / 1_000_000:.2f} MB")
    print(f"Missing ISO3: {', '.join(missing) if missing else 'none'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact country-level ADM1 GeoJSON assets for the HDI dashboard.")
    parser.add_argument("source", type=Path, help="Path to Natural Earth ne_10m_admin_1_states_provinces.geojson")
    args = parser.parse_args()
    build(args.source.resolve())


if __name__ == "__main__":
    main()
