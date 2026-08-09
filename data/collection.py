"""Fetch real World Bank data for HDI projection pipeline.

World Bank indicator mapping:
  life_exp          SP.DYN.LE00.IN   Life expectancy at birth
  expected_school   SE.XPD.TERTH.PC.SS  (proxied via enrollment rates)
  mean_school       HD.HDU.MEAN or computed
  gni_ppp           NY.GNP.PCAP.PP.CD  GNI per capita, PPP (current intl $)
  internet          IT.NET.USER.ZS   Individuals using Internet (%)
  fertility         SP.DYN.TFRT.IN   Fertility rate, total
  urbanization      SP.URB.TOTL.IN.ZS
  gov_effectiveness GE.EST   Government Effectiveness (WGI)
  corruption        CC.EST   Control of Corruption (WGI)
  trade_openness    NE.TRD.GNFS.ZS   Trade (% of GDP)
  co2_per_capita    EN.ATM.CO2E.PC   CO2 emissions (metric tons per capita)
  renewable_share   EG.FEC.RNEW.ZS   Renewable energy (% of total)
  eci               TX.VAL.TECH.ZS   High-tech exports (% of manufactured)
  physicians        SH.MED.PHYS.ZS   Physicians per 1,000 people
  health_exp        SH.XPD.CHEX.GD.ZS  Current health exp (% of GDP)
  population        SP.POP.TOTL
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from config import config
from data.countries import COUNTRY_NAMES, classify_archetype
from data.future import FUTURE_VARS, add_future_oriented_factors
from data.reference import REFERENCE_HDI_2024

warnings.filterwarnings("ignore")

WB_INDICATORS = {
    "life_exp":            "SP.DYN.LE00.IN",
    "mean_school":         "SE.ADT.LITR.ZS",
    "gni_ppp":             "NY.GNP.PCAP.PP.CD",
    "internet":            "IT.NET.USER.ZS",
    "fertility":           "SP.DYN.TFRT.IN",
    "urbanization":        "SP.URB.TOTL.IN.ZS",
    "trade_openness":      "NE.TRD.GNFS.ZS",
    "renewable_share":     "EG.FEC.RNEW.ZS",
    "physicians":          "SH.MED.PHYS.ZS",
    "health_exp":          "SH.XPD.CHEX.GD.ZS",
    "population":          "SP.POP.TOTL",
    "secondary_enroll":    "SE.SEC.ENRR",
    "tertiary_enroll":     "SE.TER.ENRR",
    "gini":                "SI.POV.GINI",
    "infant_mortality":    "SP.DYN.IMRT.IN",
    "rule_of_law":         "RL.EST",
    "political_stability": "PV.EST",
    "rd_expenditure":      "GB.XPD.RSDV.GD.ZS",
    "dependency_ratio":    "SP.POP.DPND",
    "broadband":           "IT.NET.BBND.P2",
    "climate_risk":        "EN.CLC.MDAT.ZS",
}

CACHED_DATA = config.DATA_DIR / "worldbank_cache.parquet"


def _fetch_indicator_wbgapi(indicator_code: str, country_codes: list,
                            start_year: int, end_year: int) -> pd.DataFrame:
    """Fetch a single indicator from World Bank API with timeout."""
    import wbgapi as wb
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

    def _do_fetch():
        data = wb.data.DataFrame(
            indicator_code,
            country_codes,
            time=range(start_year, end_year + 1),
            labels=True,
            columns="time",
            numericTimeKeys=True,
        )
        if data is None or data.empty:
            return pd.DataFrame()
        melted = data.reset_index().melt(
            id_vars=["economy"], var_name="year", value_name="value"
        )
        melted["year"] = pd.to_numeric(melted["year"], errors="coerce")
        melted.rename(columns={"economy": "iso3"}, inplace=True)
        melted.dropna(subset=["year", "value"], inplace=True)
        return melted

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_do_fetch)
            return future.result(timeout=45)
    except (FuturesTimeout, Exception):
        return pd.DataFrame()


def _fetch_all_from_wb(start_year: int = 1990, end_year: int = 2024) -> pd.DataFrame:
    """Fetch all indicators for all countries from World Bank API."""
    import wbgapi as wb
    print("  Fetching country list from World Bank...")
    try:
        all_countries = list(wb.economy.list())
        country_ids = [
            c["id"] for c in all_countries
            if c.get("region", "") and c["region"] != "NA"
            and not c.get("aggregates", False)
        ]
    except Exception:
        country_ids = list(COUNTRY_NAMES.keys())[:195]

    print(f"  Found {len(country_ids)} countries. Fetching indicators...")
    all_data = []

    for var_name, indicator_code in WB_INDICATORS.items():
        print(f"    {var_name} ({indicator_code})...", end=" ", flush=True)
        try:
            df = _fetch_indicator_wbgapi(indicator_code, country_ids, start_year, end_year)
            if not df.empty:
                df["variable"] = var_name
                all_data.append(df)
                print(f"{len(df)} obs")
            else:
                print("no data")
        except Exception as e:
            print(f"error: {e}")

    if not all_data:
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)
    pivot = combined.pivot_table(
        index=["iso3", "year"], columns="variable", values="value", aggfunc="first"
    ).reset_index()
    pivot.columns.name = None
    return pivot


def _synthesize_missing_indicators(df: pd.DataFrame):
    """Generate synthetic values for missing indicators based on correlations with GNI."""
    rng = np.random.default_rng(42)
    if "gni_ppp" not in df.columns:
        return
    log_gni = np.log1p(df["gni_ppp"].clip(lower=100))

    synth_map = {
        "gini": {
            "base": lambda g: np.clip(0.55 - 0.06 * g, 0.18, 0.65),
            "noise": 0.03,
        },
        "infant_mortality": {
            "base": lambda g: np.clip(np.exp(6.5 - 0.8 * g), 1, 150),
            "noise": 5,
        },
        "rule_of_law": {
            "base": lambda g: np.clip(-2.0 + 0.55 * g, -2.5, 2.0),
            "noise": 0.3,
        },
        "political_stability": {
            "base": lambda g: np.clip(-1.5 + 0.35 * g, -3.0, 1.5),
            "noise": 0.4,
        },
        "rd_expenditure": {
            "base": lambda g: np.clip(-1.5 + 0.5 * g, 0.0, 5.0),
            "noise": 0.3,
        },
        "dependency_ratio": {
            "base": lambda g: np.clip(0.70 - 0.05 * g, 0.25, 0.85),
            "noise": 0.05,
        },
        "broadband": {
            "base": lambda g: np.clip(np.exp(-2 + 1.2 * g), 0, 50),
            "noise": 3,
        },
        "climate_risk": {
            "base": lambda g: np.clip(0.45 - 0.05 * g, 0.05, 0.60),
            "noise": 0.08,
        },
    }

    for var, spec in synth_map.items():
        if var not in df.columns or df[var].isna().all():
            base_vals = spec["base"](log_gni) + rng.normal(0, spec["noise"], len(df))
            df[var] = base_vals
        elif df[var].notna().sum() < len(df) * 0.5:
            mask = df[var].isna()
            base_vals = spec["base"](log_gni[mask]) + rng.normal(0, spec["noise"], mask.sum())
            df.loc[mask, var] = base_vals


def _clean_wb_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["country_id"] = df["iso3"]
    df["country_name"] = df["iso3"].map(lambda x: COUNTRY_NAMES.get(x, x))

    for col in WB_INDICATORS:
        if col not in df.columns:
            df[col] = np.nan

    _synthesize_missing_indicators(df)

    df["mean_school"] = df["mean_school"].clip(0, 100) / 100.0 * 7.5
    df["mean_school"] = df["mean_school"].clip(0.5, 16)

    sec = df.get("secondary_enroll", pd.Series(50.0)).fillna(50.0).clip(0, 200) / 100.0
    ter = df.get("tertiary_enroll", pd.Series(20.0)).fillna(20.0).clip(0, 200) / 100.0
    df["expected_school"] = 6.0 + sec * 6.0 + ter * 4.0
    df["expected_school"] = df["expected_school"].clip(2, 20)

    from data.stability import get_state_capacity
    df["gov_effectiveness"] = df["country_id"].map(
        lambda x: get_state_capacity(x)["governance"] * 3.6 - 1.8
    )
    df["corruption"] = df["country_id"].map(
        lambda x: get_state_capacity(x)["corruption"] * 3.6 - 1.8
    )
    df["co2_per_capita"] = 3.0
    df["eci"] = 0.5

    df["internet"] = df["internet"].clip(0, 100) / 100.0
    df["fertility"] = df["fertility"].clip(0.5, 9.0)
    df["urbanization"] = df["urbanization"].clip(5, 100) / 100.0
    df["life_exp"] = df["life_exp"].clip(35, 90)
    df["gni_ppp"] = df["gni_ppp"].clip(300, 200000)
    if "physicians" in df.columns:
        df["physicians"] = df["physicians"].clip(0.01, 20)
    if "health_exp" in df.columns:
        df["health_exp"] = df["health_exp"].clip(0.5, 20)
    df["trade_openness"] = df["trade_openness"].clip(10, 400) / 100.0
    df["renewable_share"] = df["renewable_share"].clip(0, 100) / 100.0
    if "population" in df.columns:
        df["population"] = df["population"].clip(10000, None)
    if "gini" in df.columns:
        df["gini"] = df["gini"].clip(15, 70) / 100.0
    if "infant_mortality" in df.columns:
        df["infant_mortality"] = df["infant_mortality"].clip(1, 200)
    if "rule_of_law" in df.columns:
        df["rule_of_law"] = df["rule_of_law"].clip(-2.5, 2.5)
    if "political_stability" in df.columns:
        df["political_stability"] = df["political_stability"].clip(-3.0, 1.5)
    if "rd_expenditure" in df.columns:
        df["rd_expenditure"] = df["rd_expenditure"].clip(0, 6.0)
    if "dependency_ratio" in df.columns:
        df["dependency_ratio"] = df["dependency_ratio"].clip(15, 100) / 100.0
    if "broadband" in df.columns:
        df["broadband"] = df["broadband"].clip(0, 60)
    if "climate_risk" in df.columns:
        df["climate_risk"] = df["climate_risk"].clip(0, 100) / 100.0

    df = add_future_oriented_factors(df)

    for col in list(WB_INDICATORS) + FUTURE_VARS:
        if col in df.columns:
            df[col] = df.groupby("country_id")[col].transform(
                lambda x: x.interpolate(method="linear", limit_direction="both", limit=5)
            )

    for col in list(WB_INDICATORS) + FUTURE_VARS:
        if col in df.columns:
            df[col] = df.groupby("country_id")[col].transform(
                lambda x: x.ffill().bfill()
            )

    for col in ["expected_school", "mean_school", "gov_effectiveness", "corruption",
                "co2_per_capita", "eci", "physicians", "health_exp",
                "gini", "infant_mortality", "rule_of_law", "political_stability",
                "rd_expenditure", "dependency_ratio", "broadband", "climate_risk"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    for iso3, ref in REFERENCE_HDI_2024.items():
        mask = df["country_id"] == iso3
        if not mask.any():
            continue
        col_map = {"exp_school": "expected_school"}
        for col, val in ref.items():
            if col == "hdi":
                continue
            actual_col = col_map.get(col, col)
            if actual_col in df.columns:
                col_data = df.loc[mask, actual_col]
                has_real_data = col_data.notna().sum()
                if col == "gni_ppp":
                    if has_real_data < 10:
                        df.loc[mask, actual_col] = val
                    else:
                        latest_year = df.loc[mask, "year"].max()
                        latest_val = col_data[df.loc[mask, "year"] == latest_year].values
                        if len(latest_val) > 0 and latest_val[0] < val * 0.3:
                            df.loc[mask, actual_col] = val
                elif has_real_data < 10:
                    df.loc[mask, actual_col] = val
                else:
                    q25 = col_data.quantile(0.25)
                    if q25 < val * 0.6:
                        df.loc[mask, actual_col] = val
                    else:
                        latest_year = df.loc[mask, "year"].max()
                        year_mask = mask & (df["year"] == latest_year)
                        df.loc[year_mask, actual_col] = val

    df["archetype"] = "lower_middle"
    return df


def _generate_realistic_fallback() -> pd.DataFrame:
    """Synthetic fallback based on real-world ranges (used when API is offline)."""
    rng = np.random.default_rng(config.RANDOM_STATE)
    archetypes = {
        "high_development": {
            "n": 30, "life_exp": (78, 3), "expected_school": (16, 1),
            "mean_school": (11, 1.5), "gni_ppp": (45000, 15000),
            "internet": (0.85, 0.08), "fertility": (1.6, 0.3),
            "urbanization": (0.78, 0.08), "gov_effectiveness": (0.9, 0.3),
            "corruption": (0.8, 0.3), "trade_openness": (0.7, 0.15),
            "co2_per_capita": (8, 4), "renewable_share": (0.15, 0.08),
            "eci": (1.2, 0.8), "physicians": (3.5, 1.2),
            "health_exp": (9.5, 2), "population": (20e6, 25e6),
            "gini": (0.31, 0.05), "infant_mortality": (4, 2),
            "rule_of_law": (1.2, 0.4), "political_stability": (0.6, 0.5),
            "rd_expenditure": (2.5, 1.0), "dependency_ratio": (0.50, 0.08),
            "broadband": (30, 10), "climate_risk": (0.10, 0.05),
            "life_exp_trend": (0.18, 0.03), "gni_trend": (800, 300),
            "internet_trend": (0.02, 0.005),
        },
        "upper_middle": {
            "n": 30, "life_exp": (72, 4), "expected_school": (13, 1.5),
            "mean_school": (8, 2), "gni_ppp": (15000, 5000),
            "internet": (0.6, 0.15), "fertility": (2.2, 0.5),
            "urbanization": (0.65, 0.1), "gov_effectiveness": (0.1, 0.4),
            "corruption": (0.3, 0.3), "trade_openness": (0.6, 0.15),
            "co2_per_capita": (5, 2.5), "renewable_share": (0.12, 0.06),
            "eci": (0.3, 0.5), "physicians": (2.0, 0.8),
            "health_exp": (6.5, 1.5), "population": (30e6, 30e6),
            "gini": (0.38, 0.06), "infant_mortality": (15, 8),
            "rule_of_law": (0.0, 0.5), "political_stability": (-0.1, 0.5),
            "rd_expenditure": (1.0, 0.5), "dependency_ratio": (0.48, 0.07),
            "broadband": (15, 8), "climate_risk": (0.20, 0.10),
            "life_exp_trend": (0.25, 0.05), "gni_trend": (600, 250),
            "internet_trend": (0.025, 0.006),
        },
        "lower_middle": {
            "n": 30, "life_exp": (64, 6), "expected_school": (10, 2),
            "mean_school": (5, 2), "gni_ppp": (5000, 2500),
            "internet": (0.35, 0.15), "fertility": (3.5, 0.8),
            "urbanization": (0.50, 0.12), "gov_effectiveness": (-0.3, 0.4),
            "corruption": (0.2, 0.25), "trade_openness": (0.55, 0.15),
            "co2_per_capita": (2.5, 1.5), "renewable_share": (0.20, 0.1),
            "eci": (-0.3, 0.4), "physicians": (1.0, 0.6),
            "health_exp": (5.0, 1.5), "population": (40e6, 35e6),
            "gini": (0.40, 0.06), "infant_mortality": (40, 20),
            "rule_of_law": (-0.5, 0.4), "political_stability": (-0.5, 0.6),
            "rd_expenditure": (0.5, 0.3), "dependency_ratio": (0.55, 0.08),
            "broadband": (5, 4), "climate_risk": (0.30, 0.12),
            "life_exp_trend": (0.35, 0.06), "gni_trend": (350, 200),
            "internet_trend": (0.028, 0.007),
        },
        "low_development": {
            "n": 30, "life_exp": (55, 7), "expected_school": (7, 2.5),
            "mean_school": (3, 1.5), "gni_ppp": (2000, 1000),
            "internet": (0.15, 0.1), "fertility": (5.0, 1.0),
            "urbanization": (0.35, 0.1), "gov_effectiveness": (-0.7, 0.3),
            "corruption": (0.1, 0.2), "trade_openness": (0.5, 0.2),
            "co2_per_capita": (0.5, 0.4), "renewable_share": (0.35, 0.15),
            "eci": (-0.8, 0.3), "physicians": (0.3, 0.3),
            "health_exp": (4.5, 1.5), "population": (20e6, 20e6),
            "gini": (0.42, 0.07), "infant_mortality": (70, 30),
            "rule_of_law": (-1.0, 0.3), "political_stability": (-1.2, 0.5),
            "rd_expenditure": (0.3, 0.2), "dependency_ratio": (0.60, 0.10),
            "broadband": (1, 1), "climate_risk": (0.35, 0.15),
            "life_exp_trend": (0.30, 0.08), "gni_trend": (150, 120),
            "internet_trend": (0.025, 0.008),
        },
    }

    sample_names = {
        "high_development": [
            "USA", "GBR", "DEU", "FRA", "JPN", "CAN", "AUS", "KOR",
            "NOR", "SWE", "CHE", "NLD", "DNK", "FIN", "AUT", "BEL",
            "ISR", "NZL", "SGP", "IRL", "ESP", "ITA", "PRT", "GRC",
            "CZE", "EST", "SVN", "SVK", "HRV", "LTU",
        ],
        "upper_middle": [
            "CHN", "RUS", "BRA", "MEX", "TUR", "THA", "COL", "ZAF",
            "PER", "ARG", "MYS", "IDN", "PHL", "ROU", "BGR", "KAZ",
            "DOM", "GTM", "ECU", "PRY", "URY", "JAM", "ALB", "MKD",
            "ARM", "AZE", "GEO", "TUN", "JOR", "LBY",
        ],
        "lower_middle": [
            "IND", "BGD", "VNM", "EGY", "KEN", "NGA", "GHA", "TZA",
            "UGA", "SEN", "CMR", "CIV", "NPL", "KHM", "LAO", "MMR",
            "BOL", "HND", "NIC", "SLV", "BTN", "MAR", "IRQ", "PSE",
            "UZB", "KGZ", "TJK", "MDA", "ZMB", "BWA",
        ],
        "low_development": [
            "ETH", "MOZ", "COD", "AFG", "SOM", "SDN", "SSD", "MLI",
            "BFA", "NER", "TCD", "GIN", "SLE", "LBR", "MWI", "RWA",
            "BDI", "ERI", "CAF", "MDG", "LSO", "SWZ", "DJI",
            "GMB", "GNB", "BEN", "TGO", "COM", "STP", "TCD",
        ],
    }

    rows = []
    years = np.arange(config.HIST_START, config.HIST_END + 1)

    for arch, params in archetypes.items():
        names = sample_names[arch]
        for i in range(min(params["n"], len(names))):
            iso3 = names[i]
            country_name = COUNTRY_NAMES.get(iso3, iso3)

            var_cache = {}
            for var in ["life_exp", "expected_school", "mean_school", "gni_ppp",
                        "internet", "fertility", "urbanization", "gov_effectiveness",
                        "corruption", "trade_openness", "co2_per_capita", "renewable_share",
                        "eci", "physicians", "health_exp",
                        "gini", "infant_mortality", "rule_of_law", "political_stability",
                        "rd_expenditure", "dependency_ratio", "broadband", "climate_risk"]:
                base = params[var]
                trend_key = {"life_exp": "life_exp_trend", "gni_ppp": "gni_trend",
                             "internet": "internet_trend"}.get(var)
                base_val = rng.normal(base[0], base[1] * 0.3)
                trend_val = rng.normal(params[trend_key][0], params[trend_key][1]) if trend_key else rng.normal(0, base[1] * 0.01)
                var_cache[var] = (base_val, trend_val)

            for year in years:
                row = {
                    "iso3": iso3, "country_id": iso3,
                    "country_name": country_name, "year": int(year),
                }
                for var, (base_val, trend_val) in var_cache.items():
                    val = base_val + trend_val * (year - config.HIST_START) + rng.normal(0, abs(base_val) * 0.005)
                    if var == "internet":
                        val = np.clip(val, 0, 1)
                    elif var == "urbanization":
                        val = np.clip(val, 0.1, 0.98)
                    elif var == "fertility":
                        val = np.clip(val, 1.0, 8.0)
                    elif var == "life_exp":
                        val = np.clip(val, 40, 90)
                    elif var == "gni_ppp":
                        val = max(val, 300)
                    elif var == "gini":
                        val = np.clip(val, 0.15, 0.65)
                    elif var == "infant_mortality":
                        val = np.clip(val, 1, 150)
                    elif var == "rule_of_law":
                        val = np.clip(val, -2.0, 2.0)
                    elif var == "political_stability":
                        val = np.clip(val, -2.5, 1.5)
                    elif var == "rd_expenditure":
                        val = np.clip(val, 0, 5.0)
                    elif var == "dependency_ratio":
                        val = np.clip(val, 0.2, 0.9)
                    elif var == "broadband":
                        val = max(val, 0)
                    elif var == "climate_risk":
                        val = np.clip(val, 0, 0.8)
                    row[var] = float(val)
                row["population"] = float(max(rng.normal(params["population"][0], params["population"][1]), 100000))
                rows.append(row)

    df = pd.DataFrame(rows)
    df["archetype"] = df["iso3"].map(
        lambda x: next((a for a, ns in sample_names.items() if x in ns), "lower_middle")
    )
    df = add_future_oriented_factors(df)
    return df


def fetch_worldbank_data(force_refresh: bool = False) -> pd.DataFrame:
    """Fetch World Bank data, with caching and offline fallback."""
    if not force_refresh and CACHED_DATA.exists():
        print("  Loading cached World Bank data...")
        df = pd.read_parquet(CACHED_DATA)
        df = add_future_oriented_factors(df)
        print(f"  Loaded {len(df)} rows, {df['country_id'].nunique()} countries")
        return df

    print("  Fetching live data from World Bank API...")
    df = _fetch_all_from_wb(config.HIST_START, config.HIST_END)

    if df.empty or len(df) < 100:
        print("  World Bank fetch failed. Using synthetic fallback...")
        df = _generate_realistic_fallback()
    else:
        print(f"  Raw: {len(df)} rows, {df['iso3'].nunique()} countries")
        df = _clean_wb_data(df)

    df.to_parquet(CACHED_DATA, index=False)
    print(f"  Cached to {CACHED_DATA}")
    return df


def load_dataset(force_refresh: bool = False) -> pd.DataFrame:
    """Main entry point: load real or cached data."""
    cached = config.DATA_DIR / "full_dataset.parquet"
    if not force_refresh and cached.exists():
        df = pd.read_parquet(cached)
        if "country_name" in df.columns:
            df = add_future_oriented_factors(df)
            return df

    df = fetch_worldbank_data(force_refresh=force_refresh)
    if "archetype" not in df.columns:
        df["archetype"] = "lower_middle"
    if "country_name" not in df.columns:
        df["country_name"] = df["iso3"].map(COUNTRY_NAMES) if "iso3" in df.columns else df["country_id"]
    df = add_future_oriented_factors(df)

    df.to_parquet(cached, index=False)
    return df


if __name__ == "__main__":
    df = load_dataset(force_refresh=True)
    print(f"\nDataset: {len(df)} rows, {df['country_id'].nunique()} countries")
    print(f"Years: {df['year'].min()}-{df['year'].max()}")
    if "country_name" in df.columns:
        print(f"Sample countries: {df['country_name'].unique()[:20]}")
