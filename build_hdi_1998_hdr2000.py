"""Extract report-vintage 1998 HDI values from UNDP's Human Development Report 2000."""

from __future__ import annotations

import csv
import re
import unicodedata
import urllib.request
from pathlib import Path

import pycountry
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent
SOURCE_URL = "https://digitallibrary.un.org/record/430778/files/hdr2000en.pdf"
SOURCE_PAGE = "https://hdr.undp.org/content/human-development-report-2000"
SOURCE_PDF = ROOT / "data" / "raw" / "hdr2000en.pdf"
CURRENT_HDI = ROOT / "data" / "output" / "hdi_2050_rankings.csv"
OUTPUT = ROOT / "data" / "output" / "hdi_1998_hdr2000.csv"

# PDF page indexes and the first country row on each physical page of Table 1.
TABLE_PAGES = [
    (175, 1, "Canada"),
    (176, 51, "Dominica"),
    (177, 101, "Tunisia"),
    (178, 151, "Nigeria"),
]

ISO3_OVERRIDES = {
    "cape verde": "CPV",
    "congo": "COG",
    "congo dem rep of the": "COD",
    "cote d ivoire": "CIV",
    "cote divoire": "CIV",
    "gambia": "GMB",
    "hong kong china sar": "HKG",
    "iran islamic rep of": "IRN",
    "korea rep of": "KOR",
    "lao people s dem rep": "LAO",
    "lao peoples dem rep": "LAO",
    "libyan arab jamahiriya": "LBY",
    "macedonia tfyr": "MKD",
    "moldova rep of": "MDA",
    "russian federation": "RUS",
    "samoa western": "WSM",
    "sao tome and principe": "STP",
    "swaziland": "SWZ",
    "syrian arab republic": "SYR",
    "tanzania u rep of": "TZA",
    "turkey": "TUR",
    "united kingdom": "GBR",
    "united states": "USA",
    "venezuela": "VEN",
    "viet nam": "VNM",
}


def normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKD", value or "").encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def ensure_source_pdf() -> None:
    if SOURCE_PDF.exists():
        return
    SOURCE_PDF.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {SOURCE_URL}")
    urllib.request.urlretrieve(SOURCE_URL, SOURCE_PDF)


def extract_table_rows() -> list[dict[str, object]]:
    reader = PdfReader(SOURCE_PDF)
    rows: list[dict[str, object]] = []

    for page_index, first_rank, first_country in TABLE_PAGES:
        text = reader.pages[page_index].extract_text() or ""
        marker = f"{first_rank} {first_country}"
        marker_offset = text.find(marker)
        if marker_offset < 0:
            raise RuntimeError(f"Could not find Table 1 marker {marker!r} on PDF page index {page_index}")

        text = "\n" + text[marker_offset:]
        last_rank = min(first_rank + 49, 174)
        matches = [
            match
            for match in re.finditer(r"(?m)^\s*(\d{1,3})\s+", text)
            if first_rank <= int(match.group(1)) <= last_rank
        ]

        for position, match in enumerate(matches):
            end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
            raw_row = " ".join(text[match.start():end].split())
            country_match = re.match(r"^(\d+)\s+(.+?)\s+(\d{2}\.\d)\b", raw_row)
            hdi_match = re.search(r"\b(0\.\d{3})\b", raw_row)
            if not country_match or not hdi_match:
                raise RuntimeError(f"Could not parse HDR 2000 row: {raw_row[:240]}")
            rows.append(
                {
                    "Rank_1998_HDR2000": int(country_match.group(1)),
                    "Country_HDR2000": country_match.group(2),
                    "HDI_1998_HDR2000": float(hdi_match.group(1)),
                }
            )

    ranks = [int(row["Rank_1998_HDR2000"]) for row in rows]
    if ranks != list(range(1, 175)):
        raise RuntimeError(f"Expected ranks 1-174, found {len(rows)} rows")
    return rows


def iso3_for_source_name(name: str) -> str:
    normalized = normalize_name(name)
    if normalized in ISO3_OVERRIDES:
        return ISO3_OVERRIDES[normalized]
    try:
        return pycountry.countries.lookup(name).alpha_3
    except LookupError as error:
        raise RuntimeError(f"No ISO3 match for HDR 2000 country {name!r}") from error


def build_output() -> None:
    ensure_source_pdf()
    historical_rows = extract_table_rows()
    historical_by_iso3 = {iso3_for_source_name(str(row["Country_HDR2000"])): row for row in historical_rows}

    with CURRENT_HDI.open(encoding="utf-8-sig", newline="") as handle:
        current_rows = list(csv.DictReader(handle))

    fieldnames = [
        "ISO3",
        "Country",
        "Country_HDR2000",
        "Rank_1998_HDR2000",
        "HDI_1998_HDR2000",
        "Coverage_Status",
        "Data_Year",
        "Report_Year",
        "Methodology_Vintage",
        "Source_Report",
        "Source_Page",
        "Source_PDF",
    ]
    output_rows = []
    for current in current_rows:
        historical = historical_by_iso3.get(current["ISO3"])
        output_rows.append(
            {
                "ISO3": current["ISO3"],
                "Country": current["Country"],
                "Country_HDR2000": historical["Country_HDR2000"] if historical else "",
                "Rank_1998_HDR2000": historical["Rank_1998_HDR2000"] if historical else "",
                "HDI_1998_HDR2000": f"{historical['HDI_1998_HDR2000']:.3f}" if historical else "",
                "Coverage_Status": "Published in HDR 2000 Table 1" if historical else "Not ranked in HDR 2000 Table 1",
                "Data_Year": 1998,
                "Report_Year": 2000,
                "Methodology_Vintage": "HDR 2000 report-vintage methodology",
                "Source_Report": "Human Development Report 2000: Human Rights and Human Development",
                "Source_Page": SOURCE_PAGE,
                "Source_PDF": SOURCE_URL,
            }
        )

    published_count = sum(bool(row["HDI_1998_HDR2000"]) for row in output_rows)
    if len(output_rows) != 193 or published_count != 174:
        raise RuntimeError(f"Expected 193 current countries and 174 published values; found {len(output_rows)} and {published_count}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {OUTPUT} ({published_count} published values, {len(output_rows) - published_count} coverage gaps)")


if __name__ == "__main__":
    build_output()
