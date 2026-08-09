"""UNDP HDI country universe helpers."""

from __future__ import annotations

import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pycountry


UNDP_HDI_TABLE_PATH = Path(__file__).resolve().parent / "HDR25_Statistical_Annex_HDI_Table.xlsx"

UNDP_NAME_TO_ISO3_OVERRIDES = {
    "Andorra": "AND",
    "Bahamas": "BHS",
    "Bolivia (Plurinational State of)": "BOL",
    "Brunei Darussalam": "BRN",
    "Cabo Verde": "CPV",
    "Congo": "COG",
    "Congo (Democratic Republic of the)": "COD",
    "Côte d'Ivoire": "CIV",
    "Czechia": "CZE",
    "Eswatini (Kingdom of)": "SWZ",
    "Hong Kong, China (SAR)": "HKG",
    "Iran (Islamic Republic of)": "IRN",
    "Korea (Republic of)": "KOR",
    "Lao People's Democratic Republic": "LAO",
    "Liechtenstein": "LIE",
    "Micronesia (Federated States of)": "FSM",
    "Moldova (Republic of)": "MDA",
    "Nauru": "NRU",
    "Palestine, State of": "PSE",
    "Russian Federation": "RUS",
    "Saint Kitts and Nevis": "KNA",
    "Sao Tome and Principe": "STP",
    "San Marino": "SMR",
    "Tanzania (United Republic of)": "TZA",
    "Türkiye": "TUR",
    "United Kingdom": "GBR",
    "United States": "USA",
    "Venezuela (Bolivarian Republic of)": "VEN",
    "Viet Nam": "VNM",
}


def _read_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    return [
        "".join(t.text or "" for t in si.iter(ns + "t"))
        for si in root.findall(ns + "si")
    ]


def _cell_values(row: ET.Element, shared_strings: list[str]) -> dict[str, str]:
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    values = {}
    for cell in row.findall(ns + "c"):
        ref = cell.attrib.get("r", "")
        col = "".join(ch for ch in ref if ch.isalpha())
        value_node = cell.find(ns + "v")
        value = ""
        if value_node is not None:
            value = value_node.text or ""
            if cell.attrib.get("t") == "s":
                value = shared_strings[int(value)]
        values[col] = value
    return values


def _iso3_for_undp_name(name: str) -> str:
    if name in UNDP_NAME_TO_ISO3_OVERRIDES:
        return UNDP_NAME_TO_ISO3_OVERRIDES[name]
    country = pycountry.countries.lookup(name)
    return country.alpha_3


def load_undp_hdi_country_rows(path: Path = UNDP_HDI_TABLE_PATH) -> list[dict]:
    """Load the 193 ranked country/territory rows from UNDP HDR 2025 table 1."""
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(path) as zf:
        shared_strings = _read_shared_strings(zf)
        sheet = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))

    rows = []
    for row in sheet.findall(".//" + ns + "row"):
        values = _cell_values(row, shared_strings)
        rank = values.get("A", "")
        name = values.get("B", "")
        if not rank.isdigit() or not name:
            continue
        rows.append({
            "undp_rank_2023": int(rank),
            "iso3": _iso3_for_undp_name(name),
            "undp_country": name,
            "hdi_2023": float(values["C"]),
            "life_exp_2023": float(values["E"]),
            "expected_school_2023": float(values["G"]),
            "mean_school_2023": float(values["I"]),
            "gni_ppp_2023": float(values["K"]),
        })
    return rows


UNDP_HDI_COUNTRIES_193 = tuple(row["iso3"] for row in load_undp_hdi_country_rows())
