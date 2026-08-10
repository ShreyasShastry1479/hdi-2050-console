"""Build an interactive HTML infographic of 2050 population and identity context.

Reads ``data/output/demographic_context_2050.csv`` (the public research
table: per-group shares, TFR, migration intensity, policy
openness) and produces a clean, self-contained Plotly dashboard:

1. **World treemap** -- every recorded population-identity group sized by 2050 population,
   nested World > Region > Country > Identity group. Area = people. Three
   colour modes: dominance (largest-group share), change direction
   (growing/declining), and driver (immigration / fertility / mixed-identity
   recognition / identity-category transition / ageing). Click any tile to open a country drawer with its
   full 2024 vs 2050 breakdown and metadata (TFR, migration intensity).
   Search by country *or* by recorded identity group (e.g. "Pashtun") to highlight every
   country where that group lives.

2. **What-if scenarios** -- migration level (Zero/Low/Medium/High) and
   fertility-convergence (trend continuation / baseline / replacement)
   sliders that re-project the whole mosaic on the fly. Variants are
   precomputed by the evidence-based model, so no browser-side re-modelling.

3. **Biggest shifts** -- ranked horizontal bars: largest baseline-category
   declines and non-reference-category gains, 2024 vs 2050.

4. **Data stories** -- narrative cards with real figures (aging West,
   Sub-Saharan Africa's boom, East Asia, Gulf expatriate economies).

5. **Data table** -- every country's top group with 2050 share and change.

6. **About the model** -- sources and mechanics.

Output: ``data/output/ethnic_demographics_2050.html`` (self-contained).
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.undp_hdi import UNDP_HDI_COUNTRIES_193  # noqa: E402

OUT_CSV = Path("data/output/demographic_context_2050.csv")
OUT_RELIGION_CSV = Path("data/output/religious_composition_2050_model.csv")
OUT_HTML = Path("data/output/ethnic_demographics_2050.html")
POP_WEIGHTS = Path("data/population_weights_2024_2050.csv")

# World Bank-style region groups (ISO3 -> region label)
REGION_OF = {
    # Sub-Saharan Africa
    "AGO": "Sub-Saharan Africa", "BDI": "Sub-Saharan Africa",
    "BEN": "Sub-Saharan Africa", "BFA": "Sub-Saharan Africa",
    "BWA": "Sub-Saharan Africa", "CAF": "Sub-Saharan Africa",
    "CIV": "Sub-Saharan Africa", "CMR": "Sub-Saharan Africa",
    "COD": "Sub-Saharan Africa", "COG": "Sub-Saharan Africa",
    "COM": "Sub-Saharan Africa", "CPV": "Sub-Saharan Africa",
    "DJI": "Sub-Saharan Africa", "ERI": "Sub-Saharan Africa",
    "ETH": "Sub-Saharan Africa", "GAB": "Sub-Saharan Africa",
    "GHA": "Sub-Saharan Africa", "GIN": "Sub-Saharan Africa",
    "GMB": "Sub-Saharan Africa", "GNB": "Sub-Saharan Africa",
    "GNQ": "Sub-Saharan Africa", "KEN": "Sub-Saharan Africa",
    "LBR": "Sub-Saharan Africa", "LSO": "Sub-Saharan Africa",
    "MDG": "Sub-Saharan Africa", "MLI": "Sub-Saharan Africa",
    "MOZ": "Sub-Saharan Africa", "MRT": "Sub-Saharan Africa",
    "MUS": "Sub-Saharan Africa", "MWI": "Sub-Saharan Africa",
    "NAM": "Sub-Saharan Africa", "NER": "Sub-Saharan Africa",
    "NGA": "Sub-Saharan Africa", "RWA": "Sub-Saharan Africa",
    "SDN": "Sub-Saharan Africa", "SEN": "Sub-Saharan Africa",
    "SLE": "Sub-Saharan Africa", "SOM": "Sub-Saharan Africa",
    "SSD": "Sub-Saharan Africa", "STP": "Sub-Saharan Africa",
    "SWZ": "Sub-Saharan Africa", "SYC": "Sub-Saharan Africa",
    "TCD": "Sub-Saharan Africa", "TGO": "Sub-Saharan Africa",
    "TZA": "Sub-Saharan Africa", "UGA": "Sub-Saharan Africa",
    "ZAF": "Sub-Saharan Africa", "ZMB": "Sub-Saharan Africa",
    "ZWE": "Sub-Saharan Africa",
    # Middle East & North Africa
    "DZA": "Middle East & N. Africa", "EGY": "Middle East & N. Africa",
    "ISR": "Middle East & N. Africa", "JOR": "Middle East & N. Africa",
    "LBN": "Middle East & N. Africa", "LBY": "Middle East & N. Africa",
    "MAR": "Middle East & N. Africa", "TUN": "Middle East & N. Africa",
    "YEM": "Middle East & N. Africa", "PSE": "Middle East & N. Africa",
    "SAU": "Middle East & N. Africa", "ARE": "Middle East & N. Africa",
    "QAT": "Middle East & N. Africa", "KWT": "Middle East & N. Africa",
    "BHR": "Middle East & N. Africa", "OMN": "Middle East & N. Africa",
    "IRN": "Middle East & N. Africa", "IRQ": "Middle East & N. Africa",
    "SYR": "Middle East & N. Africa",
    # South Asia
    "AFG": "South Asia", "BGD": "South Asia", "BTN": "South Asia",
    "IND": "South Asia", "LKA": "South Asia", "MDV": "South Asia",
    "NPL": "South Asia", "PAK": "South Asia",
    # East Asia & Pacific
    "AUS": "East Asia & Pacific", "BRN": "East Asia & Pacific",
    "CHN": "East Asia & Pacific", "FJI": "East Asia & Pacific",
    "FSM": "East Asia & Pacific", "HKG": "East Asia & Pacific",
    "IDN": "East Asia & Pacific", "JPN": "East Asia & Pacific",
    "KHM": "East Asia & Pacific", "KIR": "East Asia & Pacific",
    "KOR": "East Asia & Pacific", "LAO": "East Asia & Pacific",
    "MHL": "East Asia & Pacific", "MMR": "East Asia & Pacific",
    "MNG": "East Asia & Pacific", "MYS": "East Asia & Pacific",
    "NRU": "East Asia & Pacific", "NZL": "East Asia & Pacific",
    "PHL": "East Asia & Pacific", "PLW": "East Asia & Pacific",
    "PNG": "East Asia & Pacific", "PRK": "East Asia & Pacific",
    "SGP": "East Asia & Pacific", "SLB": "East Asia & Pacific",
    "THA": "East Asia & Pacific", "TLS": "East Asia & Pacific",
    "TON": "East Asia & Pacific", "TUV": "East Asia & Pacific",
    "VNM": "East Asia & Pacific", "VUT": "East Asia & Pacific",
    "WSM": "East Asia & Pacific",
    # Europe & Central Asia
    "ALB": "Europe & Central Asia", "AND": "Europe & Central Asia",
    "ARM": "Europe & Central Asia", "AUT": "Europe & Central Asia",
    "AZE": "Europe & Central Asia", "BEL": "Europe & Central Asia",
    "BGR": "Europe & Central Asia", "BIH": "Europe & Central Asia",
    "BLR": "Europe & Central Asia", "CHE": "Europe & Central Asia",
    "CYP": "Europe & Central Asia", "CZE": "Europe & Central Asia",
    "DEU": "Europe & Central Asia", "DNK": "Europe & Central Asia",
    "ESP": "Europe & Central Asia", "EST": "Europe & Central Asia",
    "FIN": "Europe & Central Asia", "FRA": "Europe & Central Asia",
    "GBR": "Europe & Central Asia", "GEO": "Europe & Central Asia",
    "GRC": "Europe & Central Asia", "HRV": "Europe & Central Asia",
    "HUN": "Europe & Central Asia", "IRL": "Europe & Central Asia",
    "ISL": "Europe & Central Asia", "ITA": "Europe & Central Asia",
    "KAZ": "Europe & Central Asia", "KGZ": "Europe & Central Asia",
    "LIE": "Europe & Central Asia", "LTU": "Europe & Central Asia",
    "LUX": "Europe & Central Asia", "LVA": "Europe & Central Asia",
    "MCO": "Europe & Central Asia", "MDA": "Europe & Central Asia",
    "MKD": "Europe & Central Asia", "MLT": "Europe & Central Asia",
    "MNE": "Europe & Central Asia", "NLD": "Europe & Central Asia",
    "NOR": "Europe & Central Asia", "POL": "Europe & Central Asia",
    "PRT": "Europe & Central Asia", "ROU": "Europe & Central Asia",
    "RUS": "Europe & Central Asia", "SMR": "Europe & Central Asia",
    "SRB": "Europe & Central Asia", "SVK": "Europe & Central Asia",
    "SVN": "Europe & Central Asia", "SWE": "Europe & Central Asia",
    "TJK": "Europe & Central Asia", "TKM": "Europe & Central Asia",
    "TUR": "Europe & Central Asia", "UKR": "Europe & Central Asia",
    "UZB": "Europe & Central Asia",
    # Latin America & Caribbean
    "ARG": "Latin America & Carib.", "ATG": "Latin America & Carib.",
    "BHS": "Latin America & Carib.", "BLZ": "Latin America & Carib.",
    "BOL": "Latin America & Carib.", "BRA": "Latin America & Carib.",
    "BRB": "Latin America & Carib.", "CHL": "Latin America & Carib.",
    "COL": "Latin America & Carib.", "CRI": "Latin America & Carib.",
    "CUB": "Latin America & Carib.", "DMA": "Latin America & Carib.",
    "DOM": "Latin America & Carib.", "ECU": "Latin America & Carib.",
    "GRD": "Latin America & Carib.", "GTM": "Latin America & Carib.",
    "GUY": "Latin America & Carib.", "HND": "Latin America & Carib.",
    "HTI": "Latin America & Carib.", "JAM": "Latin America & Carib.",
    "KNA": "Latin America & Carib.", "LCA": "Latin America & Carib.",
    "MEX": "Latin America & Carib.", "NIC": "Latin America & Carib.",
    "PAN": "Latin America & Carib.", "PER": "Latin America & Carib.",
    "PRY": "Latin America & Carib.", "SLV": "Latin America & Carib.",
    "SUR": "Latin America & Carib.", "TTO": "Latin America & Carib.",
    "URY": "Latin America & Carib.", "VCT": "Latin America & Carib.",
    "VEN": "Latin America & Carib.",
    # North America
    "CAN": "North America", "USA": "North America",
}

REGION_ORDER = [
    "East Asia & Pacific", "South Asia", "Europe & Central Asia",
    "Middle East & N. Africa", "Sub-Saharan Africa",
    "Latin America & Carib.", "North America",
]

# Sequential colour ramp: light (diverse) -> dark navy (homogeneous).
_DOMINANCE_STOPS = [
    (15, "#dce7f0"), (30, "#a9c4d6"), (45, "#6f9cba"),
    (60, "#3f7396"), (75, "#1f4f6f"), (90, "#0c3550"), (100, "#071f30"),
]

_DRIVER_COLORS = {
    "Immigration": "#e76f51",
    "Fertility": "#2a9d8f",
    "Mixed identity": "#e9c46a",
    "Identity transition": "#a3b18a",
    "Ageing": "#457b9d",
    "Religious composition": "#7c3aed",
    "Stable": "#c0c9d5",
}

# Scenario slider maps (mirror the evidence-based model parameters).
MIG_SCALES = {0: 0.0, 1: 0.5, 2: 1.0, 3: 1.5}
FERT_CONV = {0: 0.2, 1: 0.6, 2: 0.9}
BASELINE_KEY = "2:1"


def dominance_color(share: float) -> str:
    """Colour for a country's dominance (largest-group share), %."""
    for threshold, color in reversed(_DOMINANCE_STOPS):
        if share >= threshold:
            return color
    return _DOMINANCE_STOPS[0][1]


def _driver(profile, name) -> str:
    if str(profile).lower() == "religion":
        return "Religious composition"
    if re.search(r"mixed|multiracial|multiple origins", str(name), re.I):
        return "Mixed identity"
    return {
        "migration_linked": "Immigration",
        "higher_fertility_path": "Fertility",
        "lower_fertility_path": "Ageing",
        "identity_category_transition": "Identity transition",
        "largest_baseline_category": "Stable",
    }.get(profile, "Stable")


def load_data() -> pd.DataFrame:
    if not OUT_CSV.exists():
        raise FileNotFoundError(
            f"{OUT_CSV} not found -- run run_ethnicity_2050_ai.py first")
    df = pd.read_csv(OUT_CSV)
    df["Region"] = df["ISO3"].map(REGION_OF).fillna("Other")
    return df


def load_religion_data() -> pd.DataFrame:
    if not OUT_RELIGION_CSV.exists():
        raise FileNotFoundError(
            f"{OUT_RELIGION_CSV} not found -- run run_religion_2050.py first")
    df = pd.read_csv(OUT_RELIGION_CSV)
    df["Region"] = df["ISO3"].map(REGION_OF).fillna("Other")
    return df


def load_population_maps(df: pd.DataFrame) -> tuple[dict, dict]:
    """({iso3: pop_2024}, {iso3: pop_2050}) from UN weights, else df sums."""
    if POP_WEIGHTS.exists():
        w = pd.read_csv(POP_WEIGHTS)
        p24 = dict(zip(w["ISO3"], w["Population_2024"]))
        p50 = dict(zip(w["ISO3"], w["Population_2050"]))
        return p24, p50
    return (df.groupby("ISO3")["Pop_2024"].sum().to_dict(),
            df.groupby("ISO3")["Pop_2050"].sum().to_dict())


def build_treemap_data(df: pd.DataFrame) -> dict:
    """Build the treemap node arrays plus per-group detail arrays."""
    ids, parents, labels, values = [], [], [], []
    colors, driver = [], []
    profile_col = "Projection_Profile" if "Projection_Profile" in df.columns else "Profile"
    regions, levels, isos = [], [], []
    gpos, share24g, pop24g, cpopg, share50g = [], [], [], [], []
    dom_share, dom_change, dom_driver = [], [], []

    world_pop = df.groupby("ISO3")["Pop_2050"].sum().sum()
    ids.append("world"); parents.append(""); labels.append("World")
    values.append(world_pop); colors.append(dominance_color(100))
    driver.append(None); regions.append(""); levels.append(0); isos.append("")
    gpos.append(-1); share24g.append(None); pop24g.append(None)
    cpopg.append(None); share50g.append(None)
    dom_share.append(None); dom_change.append(None); dom_driver.append(None)

    for region in REGION_ORDER:
        rdf = df[df["Region"] == region]
        if rdf.empty:
            continue
        reg_pop = rdf.groupby("ISO3")["Pop_2050"].sum().sum()
        ids.append(f"r::{region}"); parents.append("world")
        labels.append(region); values.append(reg_pop)
        colors.append(dominance_color(60)); driver.append(None)
        regions.append(region); levels.append(1); isos.append("")
        gpos.append(-1); share24g.append(None); pop24g.append(None)
        cpopg.append(None); share50g.append(None)
        dom_share.append(None); dom_change.append(None); dom_driver.append(None)

        for iso, cdf in rdf.groupby("ISO3"):
            country = cdf["Country"].iloc[0]
            cpop = cdf["Pop_2050"].sum()
            dom = cdf.sort_values("Share_2050_pct", ascending=False).iloc[0]
            ids.append(f"c::{iso}"); parents.append(f"r::{region}")
            labels.append(country); values.append(cpop)
            colors.append(dominance_color(dom["Share_2050_pct"]))
            driver.append(None); regions.append(region); levels.append(2)
            isos.append(iso)
            gpos.append(-1); share24g.append(None); pop24g.append(None)
            cpopg.append(None); share50g.append(None)
            dom_share.append(float(dom["Share_2050_pct"]))
            dom_change.append(float(dom["Change_pp"]))
            dom_driver.append(_driver(dom[profile_col], dom["Group"]))

            for _, r in cdf.iterrows():
                gid = f"g::{iso}::{r['Group']}"
                ids.append(gid); parents.append(f"c::{iso}")
                labels.append(r["Group"]); values.append(r["Pop_2050"])
                colors.append(dominance_color(r["Share_2050_pct"]))
                driver.append(_driver(r[profile_col], r["Group"]))
                regions.append(region); levels.append(3); isos.append(iso)
                gpos.append(len(share50g))
                share24g.append(float(r["Share_2024_pct"]))
                pop24g.append(float(r["Pop_2024"]) if pd.notna(r["Pop_2024"]) else None)
                cpopg.append(float(cpop))
                share50g.append(float(r["Share_2050_pct"]))
                dom_share.append(None); dom_change.append(None); dom_driver.append(None)

    return {
        "ids": ids, "parents": parents, "labels": labels, "values": values,
        "colors": colors, "regions": regions, "levels": levels, "isos": isos,
        "gpos": gpos, "share24g": share24g, "pop24g": pop24g,
        "cpopg": cpopg, "share50g": share50g, "driver": driver,
        "domShare": dom_share, "domChange": dom_change, "domDriver": dom_driver,
    }


def compute_scenario_variants(df: pd.DataFrame,
                              grows: list[tuple[str, str]],
                              p24: dict, p50: dict,
                              gpos_of: dict) -> dict:
    """Precompute 2050 shares for every migration x fertility slider combo.

    Returns ``{ "m:f": [share_pct...] }`` aligned to the full treemap node
    arrays (indexed by ``gpos`` so ``SCEN[key][D.gpos[i]]`` resolves for any
    group node). The model is re-run server-side so the browser only swaps
    arrays. ``2:1`` (Medium migration x baseline fertility) matches the
    central projection.
    """
    import data.ethnicity_model as model
    scen = {}
    for mi, ms in MIG_SCALES.items():
        for fi, fc in FERT_CONV.items():
            key = f"{mi}:{fi}"
            arr = [None] * (max(gpos_of.values()) + 1 if gpos_of else 0)
            for iso, cdf in df.groupby("ISO3"):
                proj = model.project_ethnic_composition(
                    iso, migration_scale=ms, fertility_convergence=fc,
                    pop_2024=p24.get(iso), pop_2050=p50.get(iso))
                for gname, share in proj.items():
                    p = gpos_of.get((iso, gname))
                    if p is not None:
                        arr[p] = share * 100.0
            scen[key] = arr
    return scen


def country_meta(df: pd.DataFrame) -> dict:
    """Per-country metadata for the click drawer."""
    meta = {}
    for iso, g in df.groupby("ISO3"):
        r = g.iloc[0]
        meta[iso] = {
            "name": r["Country"],
            "region": REGION_OF.get(iso, "Other"),
            "pop2050": int(g["Pop_2050"].sum()),
            "tfr24": float(r.get("Nat_TFR_2024", 2.1)),
            "tfr50": float(r.get("Nat_TFR_2050", 2.1)),
            "migInt": float(r.get("Migration_Intensity_2050", 1.0)),
            "skilledSourcePressure": float(r.get("Skilled_Migration_SourcePressure_2050", 0.0)),
            "skilledProgramIntensity": float(r.get("Skilled_Migration_ProgramIntensity_2050", 0.0)),
            "openness": float(r.get("Policy_Openness", 1.0)),
            "policyFeedback": float(r.get("Policy_Feedback_2050", 1.0)),
            "pressure": float(r.get("Demographic_Pressure", 0.0)),
            "identityRecognition": float(r.get("Mixed_Identity_Recognition_Rate", 0.0)),
            "identityTransition": float(r.get("Identity_Category_Transition_Rate", 0.0)),
            "diversity50": float(r.get("Diversity_Index_2050", 0.0)),
            "diversityChange": float(r.get("Diversity_Index_Change", 0.0)),
            "mobilityConvergence": float(r.get("Intergenerational_Mobility_Convergence_2050", 0.0)),
            "regionalConcentration": float(r.get("Subnational_Regional_Concentration_2050", 0.0)),
            "climateMigrationStress": float(r.get("Climate_Migration_Stress_2050", 0.0)),
            "languageAccess": float(r.get("Language_Access_Capacity_2050", 0.0)),
            "languageAccessGap": float(r.get("Language_Access_Gap_2050", 0.0)),
            "serviceDeliveryGap": float(r.get("Inclusive_Service_Delivery_Gap_2050", 0.0)),
            "inclusiveComposition": float(r.get("Inclusive_Mobility_CompositionDiversity_2050", 0.0)),
            "inclusiveIdentityRecognition": float(r.get("Inclusive_Mobility_IdentityFormationMultiplier", 1.0)),
            "inclusiveMixedIdentity": float(r.get("Inclusive_Mobility_MixedIdentityMultiplier", 1.0)),
            "inclusiveIdentityTransition": float(r.get("Inclusive_Mobility_IdentityTransitionRate", 0.0)),
            "inclusiveShift": float(g["Inclusive_Mobility_Delta_vs_Baseline_pp"].abs().mean()
                                      if "Inclusive_Mobility_Delta_vs_Baseline_pp" in g else 0.0),
            "urbanLayer": str(r.get("Urbanization_Layer", "unknown")),
            "urbanAbsorption": float(r.get("Urban_Absorption_Pressure", 0.0)),
        }
    return meta


def _json_safe(data) -> str:
    return json.dumps(data, default=lambda o: float(o))


def top_shifts_figure(df: pd.DataFrame, kind: str) -> go.Figure:
    """Ranked horizontal bars of biggest changes.

    kind = "majority_declines" | "minority_gains"
    """
    if kind == "majority_declines":
        anchor = df[df["Baseline_Reference_Category"]].sort_values("Change_pp").head(10)
        title = "Largest baseline-category declines, 2024 \u2192 2050"
        color = "#e76f51"
        text = [f"{r['Change_pp']:+.1f} pp" for _, r in anchor.iterrows()]
    else:
        non_anchor = df[~df["Baseline_Reference_Category"]].sort_values("Change_pp", ascending=False)
        non_anchor = non_anchor.head(10)
        title = "Largest non-reference-category gains, 2024 \u2192 2050"
        color = "#2a9d8f"
        text = [f"{r['Change_pp']:+.1f} pp" for _, r in non_anchor.iterrows()]
    d = anchor if kind == "majority_declines" else non_anchor
    labels = [f"{r['Country']} \u00b7 {r['Group']}" for _, r in d.iterrows()]
    x = d["Change_pp"].tolist()
    fig = go.Figure(go.Bar(
        x=x, y=labels, orientation="h", marker_color=color,
        text=text, textposition="outside",
        hovertemplate="%{y}<br>%{x:+.1f} pp<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        xaxis=dict(title="Share change (percentage points)",
                   zeroline=True, zerolinecolor="#888", gridcolor="#eef1f5"),
        yaxis=dict(autorange="reversed"),
        height=380, margin=dict(l=10, r=60, t=45, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, Arial", size=12),
    )
    return fig


def build_table_html(df: pd.DataFrame) -> str:
    """HTML table: top group per country with 2050 share and change."""
    rows = []
    top = df.sort_values("Share_2050_pct", ascending=False).groupby("ISO3").first()
    for iso in sorted(top.index):
        r = top.loc[iso]
        chg = r["Change_pp"]
        cls = "pos" if chg > 0 else "neg"
        rows.append(
            f'<tr><td>{r["Country"]}</td><td>{r["Region"]}</td>'
            f'<td>{r["Group"]}</td><td>{r["Share_2050_pct"]:.1f}%</td>'
            f'<td>{r["Pop_2050"]/1e6:.1f} m</td>'
            f'<td class="{cls}">{chg:+.1f} pp</td></tr>')
    header = ("<tr><th>Country</th><th>Region</th><th>Largest group 2050</th>"
              "<th>Share</th><th>Population</th><th>Change vs 2024</th></tr>")
    return f"<table>{header}{''.join(rows)}</table>"


def story_cards(df: pd.DataFrame) -> str:
    """Narrative cards with real figures."""
    cards = []

    # 1) Aging West & migration
    west = df[df["ISO3"].isin(["USA", "CAN", "DEU", "FRA", "GBR"])]
    if not west.empty:
        maj = west[west["Baseline_Reference_Category"]]
        imm = west[west["Projection_Profile"] == "migration_linked"]
        avg_maj_chg = maj["Change_pp"].mean() if not maj.empty else 0.0
        avg_imm_share = (imm.groupby("ISO3")["Share_2050_pct"].sum().mean()
                         if not imm.empty else 0.0)
        cards.append(
            '<div class="story"><h3>The aging West is importing its future</h3>'
            f'<div class="big">{avg_maj_chg:.1f} pp</div>'
            '<p>Average decline of the largest baseline category across the US, Canada, '
            'France, Germany and the UK by 2050, as fertility runs below '
            f'replacement and inward migration fills the labour gap. Migration-'
            f'linked categories reach ~{avg_imm_share:.0f}% of these populations.</p>'
            '</div>')

    # 2) Sub-Saharan Africa's boom
    ssaf = df[df["Region"] == "Sub-Saharan Africa"]
    if not ssaf.empty:
        pop = ssaf.groupby("ISO3")["Pop_2050"].sum().sum()
        top = ssaf.groupby("ISO3").first().sort_values("Pop_2050", ascending=False).iloc[0]
        cards.append(
            '<div class="story"><h3>Sub-Saharan Africa\u2019s demographic boom</h3>'
            f'<div class="big">{pop/1e9:.2f} bn</div>'
            f'<p>Projected 2050 population of the region \u2014 dominated by '
            f'{top["Country"]} ({top["Pop_2050"]/1e6:.0f} m). Rapid natural '
            'increase keeps the mosaic young, with growth concentrated in '
            'higher-fertility population categories rather than migration.</p></div>')

    # 3) East Asia's changing face
    east = df[df["ISO3"].isin(["JPN", "KOR", "CHN", "HKG", "TWN", "SGP"])]
    if not east.empty:
        maj = east[east["Baseline_Reference_Category"]]
        imm = east[east["Projection_Profile"] == "migration_linked"]
        avg_maj_chg = maj["Change_pp"].mean() if not maj.empty else 0.0
        avg_imm_share = (imm.groupby("ISO3")["Share_2050_pct"].sum().mean()
                         if not imm.empty else 0.0)
        cards.append(
            '<div class="story"><h3>East Asia\u2019s changing face</h3>'
            f'<div class="big">{avg_maj_chg:.1f} pp</div>'
            '<p>Average 2024\u21922050 decline of the largest baseline category in '
            'Japan, South Korea, China, Hong Kong, Taiwan and Singapore, '
            'where the world\u2019s lowest fertility collides with rising '
            f'labour migration (migration-linked groups to ~{avg_imm_share:.0f}% of '
            'the population).</p></div>')

    # 4) Gulf expatriate economies
    gulf = df[df["ISO3"].isin(["ARE", "QAT", "KWT", "BHR", "OMN"])]
    if not gulf.empty:
        foreign = gulf[gulf["Projection_Profile"] == "migration_linked"]
        share = (foreign.groupby("ISO3")["Share_2050_pct"].sum().mean()
                 if not foreign.empty else 0.0)
        cards.append(
            '<div class="story"><h3>The Gulf\u2019s expatriate economies</h3>'
            f'<div class="big">{share:.0f}%</div>'
            '<p>Average 2050 share of migration-linked groups across the UAE, '
            'Qatar, Kuwait, Bahrain and Oman \u2014 among the highest in the '
            'world, driven by sustained labour recruitment rather than natural '
            'increase.</p></div>')

    # 5) Late skilled-migration surge
    skilled = df[df["Skilled_Migration_SourcePressure_2050"].fillna(0) > 0]
    destinations = df[df["Skilled_Migration_ProgramIntensity_2050"].fillna(0) > 0]
    if not skilled.empty and not destinations.empty:
        sources = skilled.groupby(["ISO3", "Country"], as_index=False).first()
        top_sources = sources.sort_values(
            "Skilled_Migration_SourcePressure_2050", ascending=False).head(5)
        top_source_names = ", ".join(top_sources["Country"].tolist())
        dest_count = destinations["ISO3"].nunique()
        cards.append(
            '<div class="story"><h3>The 2040s skilled-migration surge</h3>'
            f'<div class="big">{dest_count}</div>'
            '<p>Destination economies are assigned explicit skilled-labor '
            'program intensity, while high-demographic-dividend source '
            f'countries such as {top_source_names} supply larger late-period '
            'diaspora flows into African, South Asian, and expatriate-worker '
            'groups.</p></div>')

    # 6) Mobility and integration screen
    if {"Intergenerational_Mobility_Convergence_2050", "Climate_Migration_Stress_2050"}.issubset(df.columns):
        country = df.groupby(["ISO3", "Country"], as_index=False).first()
        high_mobility = country.sort_values(
            "Intergenerational_Mobility_Convergence_2050", ascending=False).head(1).iloc[0]
        high_stress = country.sort_values(
            "Climate_Migration_Stress_2050", ascending=False).head(1).iloc[0]
        cards.append(
            '<div class="story"><h3>Mobility separates growth from convergence</h3>'
            f'<div class="big">{high_mobility["Intergenerational_Mobility_Convergence_2050"]:.2f}</div>'
            f'<p>{high_mobility["Country"]} has the strongest mobility-convergence screen, while '
            f'{high_stress["Country"]} shows the highest climate-migration stress. These scores flag '
            'where demographic change is more or less likely to convert into education, income, and '
            'digital-adaptation gains.</p></div>')

    # 7) Inclusive mobility and identity-recognition scenario
    if "Inclusive_Mobility_Delta_vs_Baseline_pp" in df.columns:
        country = df.groupby(["ISO3", "Country"], as_index=False).agg({
            "Inclusive_Mobility_Delta_vs_Baseline_pp": lambda s: s.abs().mean(),
            "Inclusive_Mobility_CompositionDiversity_2050": "first",
        })
        top = country.sort_values("Inclusive_Mobility_Delta_vs_Baseline_pp", ascending=False).head(1).iloc[0]
        cards.append(
            '<div class="story"><h3>Inclusive mobility and identity recognition</h3>'
            f'<div class="big">{top["Inclusive_Mobility_Delta_vs_Baseline_pp"]:.2f} pp</div>'
            f'<p>{top["Country"]} has the largest average group-share movement versus baseline under the '
            'inclusive-mobility scenario, where broader identity recognition and intergenerational mobility '
            'change recorded categories. The scenario is descriptive and does not treat any composition as '
            'a preferred development outcome.</p></div>')

    return '<div class="stories">' + "".join(cards) + "</div>"


METHODOLOGY_HTML = """
<details id="methodology">
  <summary>About the model \u2014 data sources &amp; mechanics</summary>
  <div class="meth">
    <h4>Data sources</h4>
    <p>Baseline population-identity composition is assembled from national censuses and
    surveys (US Census Bureau ACS, Statistics Canada, UK ONS, INSEE, Destatis,
    ABS and other statistical offices), Pew Research Center estimates, and
    academic sources, harmonised to a 2024 baseline. Total fertility rates
    and population projections come from the UN World Population Prospects
    2024 / World Bank WDI. 2050 populations are UN medium-variant.</p>
    <h4>How the projection works</h4>
    <ul>
      <li><b>Fertility differential</b> \u2014 each group grows at
      ln(TFR<sub>g</sub>/TFR<sub>nat</sub>)/29 yr, converting its fertility
      relative to the national average into an intrinsic growth rate.</li>
      <li><b>Age-structure momentum</b> \u2014 young high-fertility and
      immigrant groups keep producing births for ~2 decades beyond what TFR
      alone implies; per-group overrides capture documented cases (e.g. US
      Black/African-American, Native American, GBR Pakistani/Bangladeshi).</li>
      <li><b>Migration</b> \u2014 each country imports a specific mix of
      groups. Intensity = a static settlement baseline plus a
      demographic-pressure boost (below-replacement fertility + population
      decline) that ramps up to 2050, scaled by policy openness. A global
      migration-impact multiplier of 1.25 raises immigration\u2019s effect on
      the 2050 composition.</li>
      <li><b>Late skilled-migration surge</b> \u2014 after the late 2030s,
      skilled-labor programs add a second migration channel from demographic
      dividend source countries such as Nigeria, Ethiopia, Ghana, Kenya,
      Pakistan, Bangladesh, India, the Philippines, Egypt and Vietnam. The
      effect is destination-group specific and ramps quadratically toward
      2050, so it mostly changes the 2040s rather than the near-term path.</li>
      <li><b>Intergenerational mobility screen</b> \u2014 combines education,
      human-capital absorption, digital infrastructure, policy openness,
      dependency pressure, and the inclusive service-delivery gap to estimate whether
      migration-linked and locally born populations plausibly converge in education
      and income by 2050.</li>
      <li><b>Sub-national concentration screen</b> \u2014 uses urbanization,
      demographic pressure, migration intensity, urban-service pressure, and
      climate exposure to flag countries where national averages may hide state,
      provincial, metropolitan, or rural-heartland divergence.</li>
      <li><b>Climate-migration stress</b> \u2014 combines climate risk,
      climate/resource drag, demographic pressure, urban absorption pressure,
      and skilled-source pressure to identify countries where displacement
      could strain services and city infrastructure.</li>
      <li><b>Language-access vectors</b> \u2014 pair an access-capacity score
      with an access-gap score using education, digital infrastructure,
      multilingual public-service availability, urban pressure, and policy
      openness. Identity composition is not used as a penalty.</li>
      <li><b>Inclusive mobility scenario</b> \u2014 exports alternate 2050
      recorded shares under broader mixed-identity recognition, more fluid
      self-identification categories, and stronger intergenerational mobility.
      It describes possible statistical-category changes and does not define a
      preferred social outcome.</li>
      <li><b>Mixed-identity recognition</b> \u2014 approximates how statistical
      systems may increasingly allow people to report multiple backgrounds.
      It is not interpreted as a development target.</li>
      <li><b>Identity-category transition</b> \u2014 represents possible changes
      in self-identification or census classification over time. It does not
      assume that movement toward a majority category is desirable.</li>
      <li><b>Convergence</b> \u2014 group TFRs pull toward the national TFR
      over time, and the national TFR moves toward its 2050 target.</li>
      <li><b>Renormalisation</b> \u2014 shares are re-scaled to 100% every
      year, and a residual \u201cOther\u201d bucket absorbs the remainder.</li>
      <li><b>Urbanization layer</b> \u2014 the model detail table links each
      country to the HDI console's urbanization proxy so analysts can separate
      rural-heavy pressure from metropolitan-heavy absorption capacity.</li>
      <li><b>Validation fields</b> \u2014 the exported model table includes
      composition indices, effective category counts, policy feedback, and HDI
      component links for cross-table regression diagnostics.</li>
    </ul>
    <h4>What-if scenarios</h4>
    <p>The migration slider scales the net-migration deviation from zero to
    high; the fertility slider changes how quickly group fertility converges
    toward the national average (trend continuation vs replacement). Variants
    are precomputed by the model \u2014 the browser only swaps share arrays.</p>
    <h4>How this connects to HDI</h4>
    <p>The mosaic is not used as an official HDI formula input. Instead, it is
    a contextual layer for interpreting long-run HDI drivers: labor-force
    depth, age structure, migration capacity, policy openness, schooling
    demand, health-system pressure, and human-capital absorption. For
    cross-country research, join this table to
    <code>data/output/hdi_2050_rankings.csv</code> by ISO3 and compare group
    dynamics against HDI gain, health, education, income, demographic
    dividend, dependency pressure, growth prospects, and scenario outcomes.
    The dashboard reports simple univariate R-squared diagnostics as screening
    evidence, not causal proof.</p>
    <h4>Responsible interpretation</h4>
    <p>Identity categories are socially defined, fluid, and inconsistent across
    countries and censuses. The model does not rank groups, assign inherent
    characteristics, or treat diversity or concentration as a direct cause of
    HDI. Development effects operate only through measurable access and policy
    conditions such as education, health services, labor-market mobility,
    institutional inclusion, and language access. These scenarios must not be
    used for profiling, exclusion, or claims about group capability.</p>
  </div>
</details>
"""


def build_html(treemap_data: dict, religion_data: dict, scen_json: str, meta_json: str,
               shifts_html: str, table_html: str, stories_html: str,
               summary: str, methodology: str = METHODOLOGY_HTML) -> str:
    """Assemble a single self-contained HTML page."""
    data_json = _json_safe(treemap_data)
    religion_json = _json_safe(religion_data)
    plotly_js = get_plotlyjs().replace("</script", "<\\/script")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Demographic Mosaic 2050</title>
<script>{plotly_js}</script>
<style>
  :root {{
    --bg: #f5f6f8; --card: #ffffff; --ink: #1a1a2e; --accent: #2a9d8f;
    --line: #e6e9ef;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; font-family: 'Segoe UI', Arial, sans-serif;
         background: var(--bg); color: var(--ink); }}
  header {{ background: linear-gradient(120deg, #0c1e2e, #264653);
           color: #fff; padding: 28px 36px; }}
  header h1 {{ margin: 0; font-size: 26px; font-weight: 600; }}
  header p {{ margin: 6px 0 0; color: #bfd8d1; font-size: 14px; }}
  .wrap {{ max-width: 1500px; margin: 0 auto; padding: 22px 28px; }}
  .card {{ background: var(--card); border-radius: 12px;
          box-shadow: 0 1px 5px rgba(0,0,0,.07); padding: 18px; }}
  .stats {{ display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 18px; }}
  .stat {{ flex: 1; min-width: 170px; background: var(--card);
          border-radius: 12px; padding: 14px 18px;
          box-shadow: 0 1px 5px rgba(0,0,0,.07); }}
  .stat .v {{ font-size: 24px; font-weight: 700; color: var(--accent); }}
  .stat .k {{ font-size: 12px; text-transform: uppercase; letter-spacing: .05em;
             color: #6b7683; margin-top: 4px; }}
  .controls {{ display: flex; gap: 12px; align-items: center;
              flex-wrap: wrap; margin: 14px 0; position: sticky; top: 0;
              z-index: 30; background: rgba(255,255,255,.94);
              border: 1px solid var(--line); border-radius: 12px;
              padding: 12px; backdrop-filter: blur(12px);
              box-shadow: 0 8px 22px rgba(20,30,45,.08); }}
  .controls label {{ font-size: 13px; color: #6b7683; }}
  select, input[type=text] {{ padding: 8px 12px; font-size: 14px;
    border: 1px solid var(--line); border-radius: 8px; background: #fff; }}
  #search {{ min-width: 240px; }}
  .seg {{ display: inline-flex; border: 1px solid var(--line);
         border-radius: 8px; overflow: hidden; }}
  .seg button {{ border: 0; background: #fff; padding: 8px 12px; font-size: 13px;
                cursor: pointer; color: #6b7683; }}
  .seg button.active {{ background: var(--accent); color: #fff; }}
  .seg button + button {{ border-left: 1px solid var(--line); }}
  .hint {{ font-size: 12.5px; color: #6b7683; margin-top: 6px; }}
  #treemap {{ margin-top: 12px; border-radius: 10px; overflow: hidden;
              border: 1px solid var(--line); background: #f8fafc; }}
  .bridge {{ display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(280px, .8fr);
            gap: 14px; margin: 18px 0; }}
  .bridge h2 {{ margin-top: 0; }}
  .bridge p {{ color: #556; line-height: 1.58; font-size: 13.5px; }}
  .bridge-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
                 gap: 10px; }}
  .bridge-card {{ border: 1px solid var(--line); border-radius: 10px;
                 padding: 13px; background: #fbfcfd; min-height: 118px; }}
  .bridge-card h3 {{ margin: 0 0 6px; color: #264653; font-size: 13.5px; }}
  .bridge-card p {{ margin: 0; color: #5b6671; font-size: 12.5px; line-height: 1.5; }}
  .bridge-card .tag {{ display: inline-block; margin-bottom: 7px;
                      padding: 4px 7px; border-radius: 999px;
                      background: #e8f5f3; color: #2a9d8f;
                      font-size: 11px; font-weight: 700; }}
  .data-links {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
                gap: 12px; margin: 12px 0 4px; }}
  .data-link {{ border: 1px solid var(--line); border-radius: 10px;
               padding: 13px; background: #fbfcfd; color: #556;
               font-size: 12.5px; line-height: 1.5; }}
  .data-link code {{ color: #264653; font-size: 12px; word-break: break-word; }}
  #matchinfo {{ font-size: 12.5px; color: #d1495b; margin-top: 6px; }}
  #barview, #cardgrid {{ display: none; }}
  #barview {{ height: 720px; }}
  .country-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                  gap: 12px; max-height: 720px; overflow: auto; padding: 4px; }}
  .country-card {{ border: 1px solid var(--line); border-radius: 12px;
                  padding: 12px; background: #fff; cursor: pointer;
                  transition: transform .15s ease, box-shadow .15s ease; }}
  .country-card:hover {{ transform: translateY(-2px);
                        box-shadow: 0 8px 24px rgba(20,30,45,.10); }}
  .country-card .name {{ font-weight: 700; color: #264653; }}
  .country-card .meta {{ color: #6b7683; font-size: 12px; margin: 3px 0 8px; }}
  .mini-track {{ height: 8px; border-radius: 999px; background: #edf1f4;
                overflow: hidden; margin-top: 8px; }}
  .mini-fill {{ height: 100%; border-radius: inherit; background: var(--accent); }}
  #scen-panel {{ display: none; margin-top: 12px; padding: 14px 16px;
                background: #f0f4f7; border-radius: 10px; }}
  #scen-panel .row {{ display: flex; align-items: center; gap: 14px;
                     flex-wrap: wrap; margin: 6px 0; }}
  #scen-panel label {{ font-size: 13px; color: #334; min-width: 190px; }}
  #scen-panel input[type=range] {{ width: 260px; }}
  #scen-note {{ font-size: 12.5px; color: #2a9d8f; margin-left: 6px; }}
  h2 {{ font-size: 17px; margin: 26px 0 10px; color: var(--accent);
       border-bottom: 2px solid var(--line); padding-bottom: 6px; }}
  .shifts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
  @media (max-width: 1000px) {{ .shifts, .bridge {{ grid-template-columns: 1fr; }} }}
  @media (max-width: 640px) {{ .bridge-grid {{ grid-template-columns: 1fr; }} }}
  #table-wrap {{ overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th, td {{ padding: 6px 10px; text-align: left;
           border-bottom: 1px solid var(--line); }}
  th {{ position: sticky; top: 0; background: #fff; color: #6b7683; }}
  tr:hover td {{ background: #f2f7f6; }}
  .pos {{ color: #2a9d8f; font-weight: 600; }} .neg {{ color: #e76f51; font-weight: 600; }}
  .stories {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
             gap: 14px; }}
  .story {{ background: var(--card); border-radius: 12px; padding: 16px 18px;
           box-shadow: 0 1px 5px rgba(0,0,0,.07); }}
  .story h3 {{ margin: 0 0 6px; font-size: 14.5px; color: #264653; }}
  .story .big {{ font-size: 26px; font-weight: 700; color: var(--accent); }}
  .story p {{ font-size: 13px; color: #556; line-height: 1.5; margin: 6px 0 0; }}
  details#methodology {{ margin-top: 26px; background: var(--card);
       border-radius: 12px; box-shadow: 0 1px 5px rgba(0,0,0,.07);
       padding: 16px 18px; }}
  details#methodology summary {{ font-size: 15px; font-weight: 600;
       color: #264653; cursor: pointer; }}
  .meth {{ font-size: 13px; color: #445; line-height: 1.55; margin-top: 10px; }}
  .meth h4 {{ margin: 14px 0 4px; font-size: 13.5px; color: var(--accent); }}
  .meth ul {{ margin: 6px 0 0 18px; padding: 0; }}
  footer {{ text-align: center; color: #8a94a0; font-size: 12px; padding: 26px; }}
  #overlay {{ display: none; position: fixed; inset: 0; background: rgba(10,15,25,.45);
             z-index: 40; }}
  #drawer {{ position: fixed; top: 0; right: 0; bottom: 0; width: 440px;
            max-width: 92vw; background: #fff; z-index: 50;
            transform: translateX(105%); transition: transform .25s ease;
            box-shadow: -6px 0 24px rgba(0,0,0,.18); overflow-y: auto; }}
  #drawer.open {{ transform: translateX(0); }}
  #drawer .head {{ position: sticky; top: 0; background: #0c1e2e; color: #fff;
                  padding: 16px 18px; }}
  #drawer .head h3 {{ margin: 0; font-size: 18px; }}
  #drawer .head .sub {{ font-size: 12px; color: #bfd8d1; }}
  #close-drawer {{ position: absolute; top: 10px; right: 12px; border: 0;
                  background: rgba(255,255,255,.15); color: #fff; width: 30px;
                  height: 30px; border-radius: 50%; font-size: 16px;
                  cursor: pointer; }}
  .chips {{ display: flex; gap: 8px; flex-wrap: wrap; padding: 12px 18px; }}
  .chip {{ background: #eef3f6; border-radius: 8px; padding: 6px 10px;
          font-size: 12px; }}
  .chip b {{ display: block; font-size: 14px; color: #264653; }}
  #drawer .grp {{ padding: 0 18px 14px; }}
  #dw-timeline {{ height: 260px; margin: 10px 18px 14px; }}
  .grow {{ margin-bottom: 10px; }}
  .grow .nm {{ font-size: 13px; font-weight: 600; }}
  .grow .dv {{ font-size: 11px; color: #6b7683; }}
  .bars {{ display: flex; gap: 6px; align-items: center; margin-top: 3px; }}
  .bar24 {{ background: #cfd8e0; height: 12px; border-radius: 3px; }}
  .bar50 {{ height: 12px; border-radius: 3px; }}
  .grow .t {{ font-size: 12px; color: #445; margin-left: 6px; }}
</style>
</head>
<body>
<header>
  <h1>The World's Demographic Mosaic in 2050</h1>
  <p>Population-identity and religious-composition layers sized by projected 2050 population &bull;
     scenario-style fertility + migration model &bull; 2024 vs 2050 &bull;
     click any tile for details</p>
</header>
<div class="wrap">
  {summary}
  <section class="bridge">
    <div class="card">
      <h2>HDI interpretation bridge</h2>
      <p>This mosaic is designed to sit beside the HDI 2050 Projection Console as a demographic context layer. The treemap describes recorded population-identity and religious categories, while the HDI dashboard explains health, education, income, technology, governance, and scenario outcomes.</p>
      <p>Projected group-share movements should be read as signals for labor-force depth, age-structure pressure, migration capacity, integration policy, schooling demand, religious pluralism, health-system load, and human-capital absorption. These are not direct HDI components, but they help explain why two countries with similar income or education baselines can follow different development paths by 2050.</p>
      <div class="data-links">
        <div class="data-link"><b>HDI projection table</b><br><code>data/output/hdi_2050_rankings.csv</code><br>Use for 2025 baseline HDI, 2050 HDI, indices, uncertainty, and development drivers.</div>
        <div class="data-link"><b>Population and identity context table</b><br><code>data/output/demographic_context_2050.csv</code><br>Use for group shares, fertility convergence, migration intensity, policy access, and demographic pressure.</div>
        <div class="data-link"><b>Religious composition table</b><br><code>data/output/religious_composition_2050_model.csv</code><br>Use for projected 2024 and 2050 religion shares, absolute population counts, and country-level pluralism context.</div>
      </div>
    </div>
    <div class="bridge-grid">
      <div class="bridge-card">
        <span class="tag">HDI driver</span>
        <h3>Migration and skilled-labor supply</h3>
        <p>Inward migration can soften ageing and support income growth, but only if education, institutions, and integration capacity convert it into productive human capital.</p>
      </div>
      <div class="bridge-card">
        <span class="tag">Scenario lens</span>
        <h3>Youth bulges and catch-up</h3>
        <p>Young populations can accelerate catch-up in the 2040s, but high service demand means schooling, health access, and job creation must scale fast enough.</p>
      </div>
      <div class="bridge-card">
        <span class="tag">Stress test</span>
        <h3>Demographic pressure</h3>
        <p>Rapid population growth, low policy capacity, or sharp composition shifts can make smooth HDI gains less plausible under stagnation or climate/resource stress scenarios.</p>
      </div>
      <div class="bridge-card">
        <span class="tag">Research workflow</span>
        <h3>Cross-reference analysis</h3>
        <p>Join this table to the HDI ranking table by ISO3 to test whether migration intensity, policy openness, or fertility convergence helps explain projected HDI gains.</p>
      </div>
      <div class="bridge-card">
        <span class="tag">Religious futures</span>
        <h3>North America and Europe</h3>
        <p>The religion layer now leans into secularization, rising unaffiliated populations, and growing Muslim population shares, with North America's Christian share declining and Western Europe moving toward roughly one-tenth Muslim by 2050.</p>
      </div>
      <div class="bridge-card">
        <span class="tag">Religious futures</span>
        <h3>Sub-Saharan Africa</h3>
        <p>High fertility and young age structures make the region a core 2050 growth engine for both Christianity and Islam, while traditional/folk categories decline as identities consolidate.</p>
      </div>
      <div class="bridge-card">
        <span class="tag">Religious futures</span>
        <h3>Asia-Pacific</h3>
        <p>India is calibrated to remain about three-quarters Hindu while becoming the largest Muslim-population country by scale. Buddhist-heavy countries are dampened by ageing and low fertility.</p>
      </div>
      <div class="bridge-card">
        <span class="tag">Religious futures</span>
        <h3>MENA</h3>
        <p>The region remains overwhelmingly Muslim in the 2050 layer, while smaller Christian and other religious communities are modeled as more sensitive to migration, conflict displacement, and urbanization stress.</p>
      </div>
    </div>
  </section>
  <h2>The world, at a glance</h2>
  <div class="card">
    <div class="controls">
      <label for="region">Region</label>
      <select id="region">
        <option value="all">Whole world</option>
        <option value="East Asia & Pacific">East Asia &amp; Pacific</option>
        <option value="South Asia">South Asia</option>
        <option value="Europe & Central Asia">Europe &amp; Central Asia</option>
        <option value="Middle East & N. Africa">Middle East &amp; N. Africa</option>
        <option value="Sub-Saharan Africa">Sub-Saharan Africa</option>
        <option value="Latin America & Carib.">Latin America &amp; Carib.</option>
        <option value="North America">North America</option>
      </select>
      <label for="search">Search country or group</label>
      <input id="search" type="text" placeholder="e.g. Japan, Pashtun, Bengali...">
      <button id="clear" type="button">Clear</button>
      <div class="seg" id="layermode">
        <button data-layer="ethnic" class="active">Population identity</button>
        <button data-layer="religion">Religious</button>
      </div>
      <div class="seg" id="colormode">
        <button data-mode="dominance" class="active">Dominance</button>
        <button data-mode="change">Change</button>
        <button data-mode="driver">Driver</button>
      </div>
      <div class="seg" id="viewmode">
        <button data-view="treemap" class="active">Treemap</button>
        <button data-view="bar">Bar chart</button>
        <button data-view="cards">Cards</button>
      </div>
      <button id="scen-toggle" type="button">What-if scenarios</button>
    </div>
    <div id="scen-panel">
      <div class="row">
        <label>Migration level: <b id="mig-label">Medium</b></label>
        <input id="mig" type="range" min="0" max="3" step="1" value="2">
      </div>
      <div class="row">
        <label>Fertility convergence: <b id="fert-label">Baseline</b></label>
        <input id="fert" type="range" min="0" max="2" step="1" value="1">
      </div>
      <div class="row">
        <button id="scen-reset" type="button">Reset to central projection</button>
        <span id="scen-note"></span>
      </div>
    </div>
    <div id="treemap" style="height:720px"></div>
    <div id="barview"></div>
    <div id="cardgrid" class="country-grid"></div>
    <div id="matchinfo"></div>
    <div class="hint">
      <b>Area</b> = population in 2050. <b>Colour</b>: Dominance = share held
      by the largest group; Change = teal growing / red declining;
      Driver = immigration, fertility, mixed-identity recognition, identity-category transition, ageing, or religious composition.
      <b>Click a tile</b> to open the country breakdown. <b>Double-click</b>
      drills deeper in the treemap.
    </div>
  </div>
  <h2>Biggest shifts, 2024 &rarr; 2050</h2>
  <div class="shifts">{shifts_html}</div>
  <h2>Data stories</h2>
  {stories_html}
  <h2>Data table</h2>
  <div class="card" id="table-wrap">{table_html}</div>
  {methodology}
</div>
<footer>Generated from the population and identity context model
  &bull; data/output/demographic_context_2050.csv</footer>
<div id="overlay"></div>
<aside id="drawer">
  <button id="close-drawer" type="button">&times;</button>
  <div class="head"><h3 id="dw-name"></h3><div class="sub" id="dw-sub"></div></div>
  <div class="chips" id="dw-chips"></div>
  <div id="dw-timeline"></div>
  <div class="grp" id="dw-groups"></div>
</aside>
<script>
  const DSETS = {{
    ethnic: {data_json},
    religion: {religion_json}
  }};
  let D = DSETS.ethnic;
  const SCEN = {scen_json};
  const META = {meta_json};
  const MIG_LABELS = ['Zero', 'Low', 'Medium', 'High'];
  const FERT_LABELS = ['Trend continuation', 'Baseline', 'Replacement'];
  const DRIVER_COLORS = {{
    Immigration: '#e76f51', Fertility: '#2a9d8f', 'Mixed identity': '#e9c46a',
    'Identity transition': '#a3b18a', Ageing: '#457b9d',
    'Religious composition': '#7c3aed', Stable: '#c0c9d5'
  }};
  const RELIGION_COLORS = {{
    'Catholic Christianity': '#2563eb',
    'Protestant Christianity': '#38bdf8',
    'Orthodox Christianity': '#1d4ed8',
    'Other Christianity': '#93c5fd',
    'African Independent / Syncretic Christianity': '#06b6d4',
    'Sunni Islam': '#16a34a',
    'Shia Islam': '#0f766e',
    'Ibadi / Other Islam': '#84cc16',
    'Hinduism': '#f97316',
    'Theravada Buddhism': '#f59e0b',
    'Mahayana Buddhism': '#eab308',
    'Vajrayana / Other Buddhism': '#ca8a04',
    'Sikhism': '#d97706',
    'Bahai': '#8b5cf6',
    'Zoroastrianism': '#dc2626',
    'Folk / Traditional': '#a16207',
    'Indigenous / Syncretic traditions': '#7c2d12',
    'Judaism': '#6366f1',
    'Unaffiliated': '#64748b',
    'Other religions': '#ec4899'
  }};

  const state = {{ layer: 'ethnic', region: 'all', q: '', colorMode: 'dominance', view: 'treemap', mig: 2, fert: 1 }};

  const layout = {{
    margin: {{ l: 2, r: 2, t: 6, b: 2 }},
    paper_bgcolor: 'rgba(0,0,0,0)',
    font: {{ family: 'Segoe UI, Arial', size: 12 }},
    uniformtext: {{ minsize: 12, mode: 'hide' }},
  }};

  function fmtPop(v) {{
    if (v == null || !isFinite(v)) return '';
    if (v >= 1e9) return (v/1e9).toFixed(2) + ' bn';
    if (v >= 1e6) return (v/1e6).toFixed(1) + ' m';
    if (v >= 1e3) return (v/1e3).toFixed(0) + ' k';
    return String(Math.round(v));
  }}
  function fmtInt(v) {{ return (v == null) ? 'n/a' : Math.round(v).toLocaleString(); }}
  function scenKey() {{ return state.mig + ':' + state.fert; }}
  function share50For(i) {{
    const g = D.gpos[i];
    if (g < 0) return null;
    if (state.layer !== 'ethnic') return D.share50g[g];
    if (state.mig === 2 && state.fert === 1) return D.share50g[g];
    const arr = SCEN[scenKey()];
    return (arr && isFinite(arr[g])) ? arr[g] : D.share50g[g];
  }}

  function dominanceColor(s) {{
    const stops = [[15,'#dce7f0'],[30,'#a9c4d6'],[45,'#6f9cba'],
                   [60,'#3f7396'],[75,'#1f4f6f'],[90,'#0c3550'],[100,'#071f30']];
    for (let k = stops.length - 1; k >= 0; k--) if (s >= stops[k][0]) return stops[k][1];
    return stops[0][1];
  }}
  function lerp(a, b, t) {{ return Math.round(a + (b - a) * t); }}
  function changeColor(pp) {{
    const v = Math.max(-1, Math.min(1, pp / 6));
    if (v < 0) {{ const f = 1 + v;
      return 'rgb(' + lerp(214,245,f) + ',' + lerp(69,240,f) + ',' + lerp(69,234,f) + ')'; }}
    const f = 1 - v;
    return 'rgb(' + lerp(31,245,f) + ',' + lerp(138,240,f) + ',' + lerp(112,234,f) + ')';
  }}
  function religionColor(label) {{
    return RELIGION_COLORS[label] || '#94a3b8';
  }}
  function dominantReligionForCountry(i) {{
    const countryId = D.ids[i];
    let bestLabel = null, bestShare = -1;
    for (let j = 0; j < D.levels.length; j++) {{
      if (D.levels[j] !== 3 || D.parents[j] !== countryId) continue;
      const s = share50For(j);
      if (s > bestShare) {{
        bestShare = s;
        bestLabel = D.labels[j];
      }}
    }}
    return bestLabel;
  }}
  function dominantReligionChangeForCountry(i) {{
    const countryId = D.ids[i];
    let bestShare = -1, bestChange = 0;
    for (let j = 0; j < D.levels.length; j++) {{
      if (D.levels[j] !== 3 || D.parents[j] !== countryId) continue;
      const g = D.gpos[j];
      const s = share50For(j);
      if (s > bestShare) {{
        bestShare = s;
        bestChange = s - D.share24g[g];
      }}
    }}
    return bestChange;
  }}
  function colorFor(i) {{
    const lv = D.levels[i];
    if (state.layer === 'religion') {{
      if (state.colorMode === 'change') {{
        if (lv === 3) {{
          const s = share50For(i);
          return changeColor(s - D.share24g[D.gpos[i]]);
        }}
        if (lv === 2) return changeColor(dominantReligionChangeForCountry(i));
        return '#e8ebf0';
      }}
      if (state.colorMode === 'dominance') {{
        if (lv === 0) return dominanceColor(100);
        if (lv === 1) return dominanceColor(60);
        if (lv === 2) return dominanceColor(D.domShare[i]);
        return dominanceColor(share50For(i));
      }}
      if (lv === 0) return '#0f172a';
      if (lv === 1) return '#dbe4ee';
      if (lv === 2) return religionColor(dominantReligionForCountry(i));
      if (lv === 3) return religionColor(D.labels[i]);
    }}
    if (state.colorMode === 'dominance') {{
      if (lv === 0) return dominanceColor(100);
      if (lv === 1) return dominanceColor(60);
      if (lv === 2) return dominanceColor(D.domShare[i]);
      return dominanceColor(share50For(i));
    }}
    if (state.colorMode === 'change') {{
      if (lv === 3) {{ const s = share50For(i); return changeColor(s - D.share24g[D.gpos[i]]); }}
      if (lv === 2) return changeColor(D.domChange[i]);
      return '#e8ebf0';
    }}
    if (lv === 2) return DRIVER_COLORS[D.domDriver[i]] || '#c0c9d5';
    if (lv === 3) return DRIVER_COLORS[D.driver[i]] || '#c0c9d5';
    return '#e8ebf0';
  }}
  function nodeValue(i) {{
    const g = D.gpos[i];
    return (g >= 0) ? D.cpopg[g] * share50For(i) / 100 : D.values[i];
  }}
  function shortLabel(label, value, rootValue) {{
    const share = rootValue > 0 ? value / rootValue : 0;
    if (share < 0.0025) return '';
    if (share < 0.006) return label.length > 12 ? label.slice(0, 10) + '…' : label;
    if (share < 0.012) return label.length > 18 ? label.slice(0, 16) + '…' : label;
    return label;
  }}
  function compactLabel(label, maxLen) {{
    if (!label) return '';
    return label.length > maxLen ? label.slice(0, Math.max(4, maxLen - 1)) + '...' : label;
  }}
  function tileText(i, c, value, rootValue) {{
    const lv = D.levels[i];
    const share = rootValue > 0 ? value / rootValue : 0;
    const label = D.labels[i];
    if (lv === 0) return '<b>World</b><br>' + fmtPop(c[4]);
    if (lv === 1) return share >= 0.055 ? '<b>' + compactLabel(label, 22) + '</b><br>' + fmtPop(c[4]) : '';
    if (lv === 2) {{
      if (share < 0.006) return '';
      if (state.layer === 'religion') {{
        if (share < 0.012) return '';
        const maxLen = share > 0.075 ? 22 : share > 0.035 ? 16 : 10;
        return '<b>' + compactLabel(label, maxLen) + '</b><br>' + fmtPop(c[4]);
      }}
      const maxLen = share > 0.06 ? 24 : share > 0.025 ? 18 : 12;
      return '<b>' + compactLabel(label, maxLen) + '</b><br>' + fmtPop(c[4]);
    }}
    if (state.layer === 'religion') {{
      if (share < 0.028) return '';
      const groupShare = c[2];
      if (share < 0.055) return compactLabel(label, 12);
      const maxLen = share > 0.09 ? 24 : 16;
      return compactLabel(label, maxLen) + '<br>' + groupShare.toFixed(1) + '%';
    }}
    if (share < 0.018) return '';
    const groupShare = c[2];
    const maxLen = share > 0.05 ? 24 : 14;
    return compactLabel(label, maxLen) + '<br>' + groupShare.toFixed(1) + '%';
  }}
  function customFor(i) {{
    const lv = D.levels[i];
    if (lv === 3) {{ const g = D.gpos[i]; const s = share50For(i);
      return [D.labels[i], D.share24g[g], s, s - D.share24g[g],
              nodeValue(i), D.pop24g[g], D.driver[i]]; }}
    if (lv === 2) return [D.labels[i], null, null, D.domShare[i], D.values[i], null, D.domDriver[i]];
    return [D.labels[i], null, null, null, D.values[i], null, null];
  }}
  function hoverFor(i, c) {{
    const lv = D.levels[i];
    let ht = '<b>' + D.labels[i] + '</b><br>2050 population: ' + fmtPop(c[4]);
    if (lv === 3) {{
      ht += '<br>2024: ' + c[1].toFixed(1) + '% \u2192 2050: ' + c[2].toFixed(1) +
            '% (' + (c[3] >= 0 ? '+' : '') + c[3].toFixed(1) + ' pp)';
      if (c[5] != null && c[4] != null)
        ht += '<br>People: ' + fmtInt(c[5]) + ' \u2192 ' + fmtInt(c[4]) +
              ' (' + (c[4] - c[5] >= 0 ? '+' : '') + fmtInt(c[4] - c[5]) + ')';
      ht += '<br>Driver: ' + c[6];
    }} else if (lv === 2) {{
      ht += '<br>Largest group holds ' + c[3].toFixed(1) + '% in 2050';
    }}
    return ht;
  }}

  function visibleNodeSet() {{
    const region = state.region, q = state.q;
    const rootIdx = (region !== 'all') ? D.ids.indexOf('r::' + region) : D.ids.indexOf('world');

    const include = new Set();
    const idxOfParent = (id) => id ? D.ids.indexOf(id) : -1;
    for (let i = 0; i < D.levels.length; i++) {{
      if (D.levels[i] < 2) continue;
      if (region !== 'all' && D.regions[i] !== region) continue;
      if (q && !D.labels[i].toLowerCase().includes(q)) continue;
      include.add(i);
      if (D.levels[i] === 2) {{
        for (let j = 0; j < D.levels.length; j++) {{
          if (D.levels[j] === 3 && D.parents[j] === D.ids[i]) include.add(j);
        }}
      }} else if (D.levels[i] === 3) {{
        include.add(idxOfParent(D.parents[i]));
      }}
    }}
    if (include.size === 0) include.add(rootIdx);

    const nodeSet = new Set();
    for (const i of include) {{
      let k = i;
      while (k !== -1 && !nodeSet.has(k)) {{
        nodeSet.add(k);
        if (k === rootIdx) break;
        k = D.parents[k] ? D.ids.indexOf(D.parents[k]) : -1;
      }}
    }}
    nodeSet.add(rootIdx);

    // Groups matched by the search query -> highlight their tiles + countries.
    const matchedGroups = new Set(), matchedCountries = new Set();
    if (q) {{
      for (let i = 0; i < D.levels.length; i++) {{
        if (D.levels[i] === 3 && D.labels[i].toLowerCase().includes(q)) {{
          matchedGroups.add(i);
          const pIdx = idxOfParent(D.parents[i]);
          if (pIdx >= 0) matchedCountries.add(D.isos[pIdx]);
        }}
      }}
    }}
    document.getElementById('matchinfo').textContent = (q && matchedGroups.size > 0)
      ? matchedCountries.size + ' countr' + (matchedCountries.size === 1 ? 'y' : 'ies') +
        ' contain a group matching \u201c' + q + '\u201d (outlined)'
      : '';
    return {{ rootIdx, nodeSet, matchedGroups, matchedCountries }};
  }}

  function renderTreemap() {{
    const {{ rootIdx, nodeSet, matchedGroups, matchedCountries }} = visibleNodeSet();

    const ids = [], parents = [], labels = [], values = [], colors = [];
    const custom = [], hovertext = [], texts = [], lineW = [], lineC = [];
    const rootValue = nodeValue(rootIdx);
    for (const i of nodeSet) {{
      ids.push(D.ids[i]);
      const par = D.parents[i];
      const parIdx = par ? D.ids.indexOf(par) : -1;
      parents.push(nodeSet.has(parIdx) && i !== rootIdx ? par : '');
      const value = nodeValue(i);
      labels.push(D.labels[i]);
      values.push(value);
      const c = customFor(i);
      custom.push(c);
      colors.push(colorFor(i));
      hovertext.push(hoverFor(i, c));
      texts.push(tileText(i, c, value, rootValue));
      const hit = state.q && ((D.levels[i] === 3 && matchedGroups.has(i)) ||
                        (D.levels[i] === 2 && matchedCountries.has(D.isos[i])));
      lineW.push(hit ? 2.6 : (D.levels[i] <= 2 ? 0.9 : 0.25));
      lineC.push(hit ? '#d1495b' : (D.levels[i] <= 2 ? 'rgba(255,255,255,0.55)' : 'rgba(255,255,255,0.16)'));
    }}

    // Plotly treemap with branchvalues 'total' fails (renders blank) when any
    // node's children sum exceeds the node value. Group values are derived as
    // pop * share / 100, so float rounding can make a country's children sum
    // slightly larger than the country itself. Recompute every parent value as
    // the sum of its children, bottom-up, so the tree stays consistent.
    const posOf = new Map();
    for (let k = 0; k < ids.length; k++) posOf.set(ids[k], k);
    const isParent = new Array(ids.length).fill(false);
    for (let k = 0; k < ids.length; k++) {{
      const p = parents[k];
      if (p && posOf.has(p)) isParent[posOf.get(p)] = true;
    }}
    for (let pass = 0; pass < 4; pass++) {{
      const cs = new Array(ids.length).fill(0);
      for (let k = 0; k < ids.length; k++) {{
        const p = parents[k];
        if (p && posOf.has(p)) cs[posOf.get(p)] += values[k];
      }}
      for (let k = 0; k < ids.length; k++) if (isParent[k]) values[k] = cs[k];
    }}

    const trace = {{
      type: 'treemap', ids, parents, labels, values,
      branchvalues: 'total',
      marker: {{ colors, line: {{ color: lineC, width: lineW }} }},
      customdata: custom,
      hovertext, hoverinfo: 'text',
      text: texts,
      textinfo: 'text',
      textfont: {{ size: 12, color: '#f8fafc' }},
      tiling: {{ packing: 'squarify', pad: 2 }},
      pathbar: {{ visible: true, thickness: 22, textfont: {{ size: 12 }} }},
    }};
    const maxdepth = (state.region !== 'all' || state.q) ? 3 : 2;
    return Plotly.react('treemap', [trace], Object.assign({{}}, layout, {{ maxdepth }}));
  }}

  function visibleGroupRows(limit = 35) {{
    const region = state.region, q = state.q;
    const rows = [];
    for (let i = 0; i < D.levels.length; i++) {{
      if (D.levels[i] !== 3) continue;
      if (region !== 'all' && D.regions[i] !== region) continue;
      const countryIdx = D.ids.indexOf(D.parents[i]);
      const country = countryIdx >= 0 ? D.labels[countryIdx] : D.isos[i];
      const groupMatch = D.labels[i].toLowerCase().includes(q);
      const countryMatch = country.toLowerCase().includes(q);
      if (q && !groupMatch && !countryMatch) continue;
      const g = D.gpos[i], s50 = share50For(i), s24 = D.share24g[g];
      rows.push({{ i, iso: D.isos[i], country, group: D.labels[i], region: D.regions[i],
        pop: nodeValue(i), s24, s50, change: s50 - s24, driver: D.driver[i] }});
    }}
    return rows.sort((a, b) => b.pop - a.pop).slice(0, limit);
  }}

  function renderBarView() {{
    const rows = visibleGroupRows(30).reverse();
    const trace = {{
      type: 'bar', orientation: 'h',
      y: rows.map(r => r.country + ' · ' + r.group),
      x: rows.map(r => r.pop),
      marker: {{ color: rows.map(r => state.layer === 'religion' && state.colorMode !== 'change' ? religionColor(r.group) : (r.change >= 0 ? '#2a9d8f' : '#e76f51')) }},
      customdata: rows.map(r => [r.country, r.group, r.s24, r.s50, r.change, r.driver]),
      hovertemplate: '<b>%{{customdata[0]}}</b><br>%{{customdata[1]}}<br>2050 population: %{{x:,.0f}}<br>Share: %{{customdata[2]:.1f}}% → %{{customdata[3]:.1f}}% (%{{customdata[4]:+.1f}} pp)<br>Driver: %{{customdata[5]}}<extra></extra>'
    }};
    return Plotly.react('barview', [trace], {{
      margin: {{ l: 210, r: 30, t: 20, b: 40 }},
      xaxis: {{ title: '2050 population', gridcolor: '#eef1f5' }},
      yaxis: {{ automargin: true }},
      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
      font: {{ family: 'Segoe UI, Arial', size: 12 }}
    }}).then(() => {{
      const div = document.getElementById('barview');
      div.removeAllListeners && div.removeAllListeners('plotly_click');
      div.on && div.on('plotly_click', function(ev) {{
        const idx = ev.points && ev.points[0] ? ev.points[0].pointNumber : null;
        const row = idx != null ? rows[idx] : null;
        if (row) openDrawer(row.iso);
      }});
    }});
  }}

  function renderCardGrid() {{
    const rows = visibleGroupRows(60);
    document.getElementById('cardgrid').innerHTML = rows.map(r => {{
      const cls = r.change >= 0 ? 'pos' : 'neg';
      const width = Math.max(4, Math.min(100, r.s50));
      const fill = state.layer === 'religion' && state.colorMode !== 'change' ? religionColor(r.group) : (r.change >= 0 ? '#2a9d8f' : '#e76f51');
      return '<div class="country-card" data-iso="' + r.iso + '">' +
        '<div class="name">' + r.country + '</div>' +
        '<div class="meta">' + r.group + ' · ' + r.driver + '</div>' +
        '<div><b>' + fmtPop(r.pop) + '</b> in 2050</div>' +
        '<div class="meta">' + r.s24.toFixed(1) + '% → ' + r.s50.toFixed(1) +
          '% <span class="' + cls + '">' + (r.change >= 0 ? '+' : '') + r.change.toFixed(1) + ' pp</span></div>' +
        '<div class="mini-track"><div class="mini-fill" style="width:' + width + '%;background:' +
          fill + '"></div></div>' +
      '</div>';
    }}).join('');
    document.querySelectorAll('.country-card').forEach(card => {{
      card.addEventListener('click', () => openDrawer(card.dataset.iso));
    }});
    return Promise.resolve();
  }}

  function render() {{
    D = DSETS[state.layer] || DSETS.ethnic;
    document.getElementById('treemap').style.display = state.view === 'treemap' ? 'block' : 'none';
    document.getElementById('barview').style.display = state.view === 'bar' ? 'block' : 'none';
    document.getElementById('cardgrid').style.display = state.view === 'cards' ? 'grid' : 'none';
    if (state.view === 'bar') return renderBarView();
    if (state.view === 'cards') return renderCardGrid();
    return renderTreemap();
  }}

  // ---- drawer -------------------------------------------------------------
  function activeCountryGroups() {{
    const out = {{}};
    for (let i = 0; i < D.levels.length; i++) {{
      if (D.levels[i] !== 3) continue;
      const iso = D.isos[i];
      if (!out[iso]) out[iso] = [];
      out[iso].push({{ name: D.labels[i], g: D.gpos[i], s24: D.share24g[D.gpos[i]],
        driver: D.driver[i] }});
    }}
    return out;
  }}
  function openDrawer(iso) {{
    const m = META[iso];
    if (!m) return;
    document.getElementById('dw-name').textContent = m.name;
    document.getElementById('dw-sub').textContent = m.region;
    document.getElementById('dw-chips').innerHTML =
      '<div class="chip">2050 population<b>' + fmtPop(m.pop2050) + '</b></div>' +
      '<div class="chip">TFR 2024 \u2192 2050<b>' + m.tfr24.toFixed(1) + ' \u2192 ' +
        m.tfr50.toFixed(1) + '</b></div>' +
      '<div class="chip">Migration intensity 2050<b>' + m.migInt.toFixed(2) + '</b></div>' +
      '<div class="chip">Skilled source pressure<b>' + m.skilledSourcePressure.toFixed(2) + '</b></div>' +
      '<div class="chip">Skilled program intensity<b>' + m.skilledProgramIntensity.toFixed(2) + '</b></div>' +
      '<div class="chip">Policy openness<b>' + m.openness.toFixed(1) + '</b></div>' +
      '<div class="chip">Policy feedback<b>' + m.policyFeedback.toFixed(2) + '</b></div>' +
      '<div class="chip">Demographic pressure<b>' + (m.pressure * 100).toFixed(0) + '%</b></div>' +
      '<div class="chip">Mixed-identity recognition<b>' + m.identityRecognition.toFixed(3) + '</b></div>' +
      '<div class="chip">Identity-category transition<b>' + m.identityTransition.toFixed(4) + '</b></div>' +
      '<div class="chip">Composition diversity 2050<b>' + m.diversity50.toFixed(3) + '</b></div>' +
      '<div class="chip">Urban layer<b>' + m.urbanLayer.replaceAll('-', ' ') + '</b></div>' +
      '<div class="chip">Urban absorption pressure<b>' + m.urbanAbsorption.toFixed(2) + '</b></div>' +
      '<div class="chip">Mobility convergence<b>' + m.mobilityConvergence.toFixed(2) + '</b></div>' +
      '<div class="chip">Regional concentration<b>' + m.regionalConcentration.toFixed(2) + '</b></div>' +
      '<div class="chip">Climate migration stress<b>' + m.climateMigrationStress.toFixed(2) + '</b></div>' +
      '<div class="chip">Language access capacity<b>' + m.languageAccess.toFixed(2) + '</b></div>' +
      '<div class="chip">Language access gap<b>' + m.languageAccessGap.toFixed(2) + '</b></div>' +
      '<div class="chip">Inclusive service-delivery gap<b>' + m.serviceDeliveryGap.toFixed(2) + '</b></div>' +
      '<div class="chip">Inclusive-mobility shift<b>' + m.inclusiveShift.toFixed(2) + ' pp</b></div>' +
      '<div class="chip">Inclusive composition index<b>' + m.inclusiveComposition.toFixed(3) + '</b></div>' +
      '<div class="chip">Identity-recognition multiplier<b>' + m.inclusiveIdentityRecognition.toFixed(2) + 'x</b></div>' +
      '<div class="chip">Mixed-identity multiplier<b>' + m.inclusiveMixedIdentity.toFixed(2) + 'x</b></div>';
    const grp = activeCountryGroups()[iso] || [];
    grp.sort((a, b) => shareForEntry(b) - shareForEntry(a));
    const maxS = Math.max.apply(null, grp.map(shareForEntry));
    const rows = grp.map((e) => {{
      const s50 = shareForEntry(e);
      const chg = s50 - e.s24;
      const cls = chg >= 0 ? 'pos' : 'neg';
      const w24 = Math.max(2, e.s24 / maxS * 100);
      const w50 = Math.max(2, s50 / maxS * 100);
      const fill = state.layer === 'religion' ? religionColor(e.name) : (cls === 'pos' ? '#2a9d8f' : '#e76f51');
      return '<div class="grow">' +
        '<span class="nm">' + e.name + '</span> ' +
        '<span class="dv">(' + e.driver + ')</span>' +
        '<div class="bars">' +
          '<div class="bar24" style="width:' + w24 + 'px"></div>' +
          '<div class="bar50" style="width:' + w50 + 'px;background:' +
            fill + '"></div>' +
          '<span class="t">' + e.s24.toFixed(1) + '% \u2192 ' + s50.toFixed(1) +
            '% <b class="' + cls + '">(' + (chg >= 0 ? '+' : '') + chg.toFixed(1) +
            ' pp)</b></span>' +
        '</div></div>';
    }});
    document.getElementById('dw-groups').innerHTML = rows.join('');
    renderTimeline(iso, grp.slice(0, 6));
    document.getElementById('overlay').style.display = 'block';
    document.getElementById('drawer').classList.add('open');
  }}
  function renderTimeline(iso, groups) {{
    const years = [2024, 2030, 2040, 2050];
    const traces = groups.map((e) => {{
      const s50 = shareForEntry(e);
      return {{
        type: 'scatter', mode: 'lines+markers',
        name: e.name.length > 22 ? e.name.slice(0, 20) + '…' : e.name,
        x: years,
        y: years.map(y => e.s24 + (s50 - e.s24) * ((y - 2024) / 26)),
        hovertemplate: e.name + '<br>%{{x}}: %{{y:.1f}}%<extra></extra>',
        line: {{ width: 2, color: state.layer === 'religion' ? religionColor(e.name) : undefined }}
      }};
    }});
    Plotly.react('dw-timeline', traces, {{
      title: {{ text: 'Projected composition trajectory', font: {{ size: 13 }} }},
      margin: {{ l: 42, r: 12, t: 34, b: 34 }},
      yaxis: {{ title: 'Share', ticksuffix: '%', gridcolor: '#eef1f5' }},
      xaxis: {{ tickmode: 'array', tickvals: years }},
      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
      font: {{ family: 'Segoe UI, Arial', size: 11 }},
      showlegend: true,
      legend: {{ orientation: 'h', y: -0.25 }}
    }});
  }}
  function shareForEntry(e) {{
    if (state.layer !== 'ethnic') return D.share50g[e.g];
    if (state.mig === 2 && state.fert === 1) return D.share50g[e.g];
    const arr = SCEN[scenKey()];
    return (arr && isFinite(arr[e.g])) ? arr[e.g] : D.share50g[e.g];
  }}
  function closeDrawer() {{
    document.getElementById('overlay').style.display = 'none';
    document.getElementById('drawer').classList.remove('open');
  }}
  const treemapDiv = document.getElementById('treemap');
  function bindClicks() {{
    treemapDiv.on('plotly_click', function (ev) {{
      const pt = ev.points && ev.points[0];
      if (!pt || !pt.data.ids) return;
      const id = pt.data.ids[pt.pointNumber];
      let iso = null;
      if (id && id.indexOf('c::') === 0) iso = id.slice(3);
      else if (id && id.indexOf('g::') === 0) iso = id.split('::')[1];
      if (iso && META[iso]) openDrawer(iso);
    }});
  }}
  document.getElementById('close-drawer').addEventListener('click', closeDrawer);
  document.getElementById('overlay').addEventListener('click', closeDrawer);

  // ---- controls -----------------------------------------------------------
  document.getElementById('region').addEventListener('change', (e) => {{
    state.region = e.target.value; render();
  }});
  document.getElementById('search').addEventListener('input', (e) => {{
    state.q = e.target.value.trim().toLowerCase(); render();
  }});
  document.getElementById('clear').addEventListener('click', () => {{
    document.getElementById('search').value = ''; state.q = ''; render();
  }});
  document.querySelectorAll('#layermode button').forEach((b) => {{
    b.addEventListener('click', () => {{
      document.querySelectorAll('#layermode button').forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      state.layer = b.dataset.layer;
      if (state.layer === 'religion' && state.colorMode === 'dominance') {{
        state.colorMode = 'driver';
        document.querySelectorAll('#colormode button').forEach(x => {{
          x.classList.toggle('active', x.dataset.mode === 'driver');
        }});
      }}
      state.q = '';
      document.getElementById('search').value = '';
      document.getElementById('search').placeholder = state.layer === 'religion'
        ? 'e.g. India, Islam, Christianity...'
        : 'e.g. Japan, Pashtun, Bengali...';
      document.getElementById('scen-panel').style.display = state.layer === 'ethnic'
        ? document.getElementById('scen-panel').style.display
        : 'none';
      render().then(() => {{ if (state.view === 'treemap') bindClicks(); }});
    }});
  }});
  document.querySelectorAll('#colormode button').forEach((b) => {{
    b.addEventListener('click', () => {{
      document.querySelectorAll('#colormode button').forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      state.colorMode = b.dataset.mode; render();
    }});
  }});
  document.querySelectorAll('#viewmode button').forEach((b) => {{
    b.addEventListener('click', () => {{
      document.querySelectorAll('#viewmode button').forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      state.view = b.dataset.view; render().then(() => {{ if (state.view === 'treemap') bindClicks(); }});
    }});
  }});
  const scenPanel = document.getElementById('scen-panel');
  document.getElementById('scen-toggle').addEventListener('click', () => {{
    scenPanel.style.display = scenPanel.style.display === 'none' ? 'block' : 'none';
  }});
  function updateScen() {{
    if (state.layer !== 'ethnic') {{
      document.getElementById('scen-note').textContent = 'Religion layer uses the central 2050 projection.';
      return render();
    }}
    document.getElementById('mig-label').textContent = MIG_LABELS[state.mig];
    document.getElementById('fert-label').textContent = FERT_LABELS[state.fert];
    const base = (state.mig === 2 && state.fert === 1);
    document.getElementById('scen-note').textContent = base
      ? 'Central projection' : MIG_LABELS[state.mig] + ' migration \u00d7 ' +
        FERT_LABELS[state.fert] + ' fertility';
    return render();
  }}
  document.getElementById('mig').addEventListener('input', (e) => {{
    state.mig = parseInt(e.target.value, 10); updateScen();
  }});
  document.getElementById('fert').addEventListener('input', (e) => {{
    state.fert = parseInt(e.target.value, 10); updateScen();
  }});
  document.getElementById('scen-reset').addEventListener('click', () => {{
    document.getElementById('mig').value = 2;
    document.getElementById('fert').value = 1;
    state.mig = 2; state.fert = 1; updateScen();
  }});
  updateScen().then(bindClicks);
</script>
</body>
</html>
"""


def main():
    df = load_data()
    religion_df = load_religion_data()
    missing = set(UNDP_HDI_COUNTRIES_193) - set(df["ISO3"])
    if missing:
        raise RuntimeError(f"Missing countries: {sorted(missing)}")
    missing_religion = set(UNDP_HDI_COUNTRIES_193) - set(religion_df["ISO3"])
    if missing_religion:
        raise RuntimeError(f"Missing religion countries: {sorted(missing_religion)}")

    n_countries = df["ISO3"].nunique()
    n_groups = len(df)
    n_religion_rows = len(religion_df)
    median_chg = df["Change_pp"].abs().median()
    median_religion_chg = religion_df["Change_pp"].abs().median()
    world_2050 = df.groupby("ISO3")["Pop_2050"].sum().sum()
    summary = f"""
    <div class="stats">
      <div class="stat"><div class="v">{n_countries}</div>
        <div class="k">Countries</div></div>
      <div class="stat"><div class="v">{n_groups:,}</div>
        <div class="k">Identity-group rows tracked</div></div>
      <div class="stat"><div class="v">{n_religion_rows:,}</div>
        <div class="k">Religious rows tracked</div></div>
      <div class="stat"><div class="v">{median_chg:.2f} pp</div>
        <div class="k">Median group change 2024&rarr;2050</div></div>
      <div class="stat"><div class="v">{median_religion_chg:.2f} pp</div>
        <div class="k">Median religion shift</div></div>
      <div class="stat"><div class="v">{world_2050/1e9:.1f} bn</div>
        <div class="k">Combined 2050 population (listed)</div></div>
    </div>"""

    print("Building treemap data...")
    treemap_data = build_treemap_data(df)
    religion_treemap_data = build_treemap_data(religion_df)

    print("Precomputing scenario variants (12 projections x 193 countries)...")
    p24, p50 = load_population_maps(df)
    grows = [(iso, grp) for i, (iso, grp) in
             enumerate(zip(treemap_data["isos"], treemap_data["labels"]))
             if treemap_data["levels"][i] == 3]
    gpos_of = {(iso, grp): gp
               for (iso, grp), gp in
               zip(grows, [treemap_data["gpos"][i] for i in range(len(treemap_data["levels"]))
                           if treemap_data["levels"][i] == 3])}
    scen = compute_scenario_variants(df, grows, p24, p50, gpos_of)
    scen_json = _json_safe(scen)
    meta_json = _json_safe(country_meta(df))

    shifts_div = (
        f'<div class="card">{top_shifts_figure(df, "majority_declines").to_html(full_html=False, include_plotlyjs=False)}</div>'
        f'<div class="card">{top_shifts_figure(df, "minority_gains").to_html(full_html=False, include_plotlyjs=False)}</div>'
    )
    table_html = build_table_html(df)
    stories_html = story_cards(df)

    html = build_html(treemap_data, religion_treemap_data, scen_json, meta_json, shifts_div,
                      table_html, stories_html, summary)
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Dashboard written: {OUT_HTML}")
    print(f"  treemap nodes: {len(treemap_data['ids'])} "
          f"({n_countries} countries, {n_groups} groups)")
    print(f"  scenario variants: {len(scen)}")


if __name__ == "__main__":
    main()
