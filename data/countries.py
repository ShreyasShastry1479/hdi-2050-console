"""ISO3 country code to name mapping for all recognized countries."""
import pycountry


def get_country_name_map() -> dict:
    """Returns dict: ISO3 -> (name, iso2, region, income_group)."""
    mapping = {}
    for c in pycountry.countries:
        try:
            mapping[c.alpha_3] = {
                "name": c.name,
                "iso2": c.alpha_2,
            }
        except AttributeError:
            continue

    WB_INCOME_GROUPS = {
        "LIC": "low_income",
        "LMC": "lower_middle_income",
        "LMY": "lower_middle_income",
        "UMC": "upper_middle_income",
        "HIC": "high_income",
    }
    WB_REGIONS = {
        "SSF": "Sub-Saharan Africa",
        "SAS": "South Asia",
        "ECS": "Europe & Central Asia",
        "LCN": "Latin America & Caribbean",
        "MEA": "Middle East & North Africa",
        "EAS": "East Asia & Pacific",
        "NAC": "North America",
    }

    WB_TO_ISO3 = {
        "ZAF": "ZAF", "CHN": "CHN", "IND": "IND", "BRA": "BRA",
        "RUS": "RUS", "JPN": "JPN", "DEU": "DEU", "GBR": "GBR",
        "FRA": "FRA", "USA": "USA", "KOR": "KOR", "ITA": "ITA",
        "CAN": "CAN", "AUS": "AUS", "ESP": "ESP", "MEX": "MEX",
        "IDN": "IDN", "TUR": "TUR", "SAU": "SAU", "ARG": "ARG",
        "POL": "POL", "NGA": "NGA", "PAK": "PAK", "EGY": "EGY",
        "VNM": "VNM", "PHL": "PHL", "BGD": "BGD", "IRN": "IRN",
        "THA": "THA", "COL": "COL", "MYS": "MYS", "PER": "PER",
        "KEN": "KEN", "ZWE": "ZWE", "ETH": "ETH", "GHA": "GHA",
        "TZA": "TZA", "UGA": "UGA", "MOZ": "MOZ", "SEN": "SEN",
        "CMR": "CMR", "CIV": "CIV", "MDG": "MDG", "AGO": "AGO",
        "MML": "MMR", "NPL": "NPL", "LKA": "LKA", "KHM": "KHM",
        "RWA": "RWA", "BWA": "BWA", "NAM": "NAM", "GAB": "GAB",
        "BOL": "BOL", "ECU": "ECU", "PRY": "PRY", "URY": "URY",
        "DOM": "DOM", "GTM": "GTM", "HND": "HND", "SLV": "SLV",
        "NIC": "NIC", "CRI": "CRI", "PAN": "PAN", "CUB": "CUB",
        "JAM": "JAM", "TTO": "TTO", "GUY": "GUY", "SUR": "SUR",
        "BLZ": "BLZ", "HTI": "HTI", "ALB": "ALB", "BIH": "BIH",
        "BGR": "BGR", "HRV": "HRV", "CZE": "CZE", "EST": "EST",
        "GEO": "GEO", "HUN": "HUN", "KAZ": "KAZ", "KGZ": "KGZ",
        "LVA": "LVA", "LTU": "LTU", "MKD": "MKD", "MDA": "MDA",
        "MNE": "MNE", "ROU": "ROU", "SRB": "SRB", "SVK": "SVK",
        "SVN": "SVN", "TJK": "TJK", "UKR": "UKR", "UZB": "UZB",
        "ARM": "ARM", "AZE": "AZE", "BLR": "BLR", "CYP": "CYP",
        "ISL": "ISL", "IRL": "IRL", "NOR": "NOR", "SWE": "SWE",
        "DNK": "DNK", "FIN": "FIN", "CHE": "CHE", "AUT": "AUT",
        "BEL": "BEL", "NLD": "NLD", "LUX": "LUX", "PRT": "PRT",
        "GRC": "GRC", "MLT": "MLT", "ISR": "ISR", "ARE": "ARE",
        "QAT": "QAT", "KWT": "KWT", "BHR": "BHR", "OMN": "OMN",
        "JOR": "JOR", "LBN": "LBN", "IRQ": "IRQ", "YEM": "YEM",
        "SYR": "SYR", "AFG": "AFG", "MMR": "MMR", "PRK": "PRK",
        "MNG": "MNG", "LAO": "LAO", "BRN": "BRN", "TLS": "TLS",
        "FJI": "FJI", "PNG": "PNG", "SLB": "SLB", "VUT": "VUT",
        "WSM": "WSM", "TON": "TON", "KIR": "KIR", "MHL": "MHL",
        "FSM": "FSM", "PLW": "PLW", "NRU": "NRU", "TUV": "TUV",
        "MNP": "MNP", "GUM": "GUM", "ASM": "ASM", "PYF": "PYF",
        "NCL": "NCL", "SLV": "SLV", "BHS": "BHS", "BRB": "BRB",
        "BMU": "BMU", "CYM": "CYM", "VGB": "VGB", "AIA": "AIA",
        "MSR": "MSR", "TCA": "TCA", "KNA": "KNA", "ATG": "ATG",
        "DMA": "DMA", "LCA": "LCA", "VCT": "VCT", "GRD": "GRD",
        "CPV": "CPV", "STP": "STP", "GNQ": "GNQ", "DJI": "DJI",
        "COM": "COM", "MUS": "MUS", "SYC": "SYC", "MDG": "MDG",
        "MWI": "MWI", "ZMB": "ZMB", "COD": "COD", "COG": "COG",
        "CAF": "CAF", "TCD": "TCD", "SDN": "SDN", "SSD": "SSD",
        "ERI": "ERI", "SOM": "SOM", "DZA": "DZA", "MAR": "MAR",
        "TUN": "TUN", "LBY": "LBY", "MRT": "MRT", "MLI": "MLI",
        "BFA": "BFA", "NER": "NER", "GIN": "GIN", "SLE": "SLE",
        "LBR": "LBR", "GMB": "GMB", "GNB": "GNB", "BEN": "BEN",
        "TGO": "TGO", "SWZ": "SWZ", "LSO": "LSO",
    }

    return mapping


COUNTRY_NAMES = {
    "AFG": "Afghanistan", "ALB": "Albania", "DZA": "Algeria",
    "AGO": "Angola", "ARG": "Argentina", "ARM": "Armenia",
    "AUS": "Australia", "AUT": "Austria", "AZE": "Azerbaijan",
    "BGD": "Bangladesh", "BLR": "Belarus", "BEL": "Belgium",
    "BEN": "Benin", "BOL": "Bolivia", "BIH": "Bosnia & Herzegovina",
    "BWA": "Botswana", "BRA": "Brazil", "BGR": "Bulgaria",
    "BFA": "Burkina Faso", "BDI": "Burundi", "KHM": "Cambodia",
    "CMR": "Cameroon", "CAN": "Canada", "CPV": "Cape Verde",
    "CAF": "Central African Rep.", "TCD": "Chad", "CHL": "Chile",
    "CHN": "China", "COL": "Colombia", "COD": "Congo (DRC)",
    "COG": "Congo", "CRI": "Costa Rica", "CIV": "Cote d'Ivoire",
    "HRV": "Croatia", "CUB": "Cuba", "CZE": "Czechia",
    "DNK": "Denmark", "DJI": "Djibouti", "DOM": "Dominican Rep.",
    "ECU": "Ecuador", "EGY": "Egypt", "SLV": "El Salvador",
    "GNQ": "Equatorial Guinea", "ERI": "Eritrea", "EST": "Estonia",
    "SWZ": "Eswatini", "ETH": "Ethiopia", "FIN": "Finland",
    "FRA": "France", "GAB": "Gabon", "GMB": "Gambia",
    "GEO": "Georgia", "DEU": "Germany", "GHA": "Ghana",
    "GRC": "Greece", "GTM": "Guatemala", "GIN": "Guinea",
    "GNB": "Guinea-Bissau", "GUY": "Guyana", "HTI": "Haiti",
    "HND": "Honduras", "HUN": "Hungary", "ISL": "Iceland",
    "IND": "India", "IDN": "Indonesia", "IRN": "Iran",
    "IRQ": "Iraq", "IRL": "Ireland", "ISR": "Israel",
    "ITA": "Italy", "JAM": "Jamaica", "JPN": "Japan",
    "JOR": "Jordan", "KAZ": "Kazakhstan", "KEN": "Kenya",
    "KIR": "Kiribati", "KOR": "South Korea", "KWT": "Kuwait",
    "KGZ": "Kyrgyzstan", "LAO": "Laos", "LVA": "Latvia",
    "LBN": "Lebanon", "LSO": "Lesotho", "LBR": "Liberia",
    "LBY": "Libya", "LTU": "Lithuania", "MDG": "Madagascar",
    "MWI": "Malawi", "MYS": "Malaysia", "MLI": "Mali",
    "MRT": "Mauritania", "MUS": "Mauritius", "MEX": "Mexico",
    "MDA": "Moldova", "MNG": "Mongolia", "MNE": "Montenegro",
    "MAR": "Morocco", "MOZ": "Mozambique", "MMR": "Myanmar",
    "NAM": "Namibia", "NPL": "Nepal", "NLD": "Netherlands",
    "NZL": "New Zealand", "NIC": "Nicaragua", "NER": "Niger",
    "NGA": "Nigeria", "MKD": "North Macedonia", "NOR": "Norway",
    "OMN": "Oman", "PAK": "Pakistan", "PAN": "Panama",
    "PNG": "Papua New Guinea", "PRY": "Paraguay", "PER": "Peru",
    "PHL": "Philippines", "POL": "Poland", "PRT": "Portugal",
    "QAT": "Qatar", "ROU": "Romania", "RUS": "Russia",
    "RWA": "Rwanda", "SAU": "Saudi Arabia", "SEN": "Senegal",
    "SRB": "Serbia", "SLE": "Sierra Leone", "SGP": "Singapore",
    "SVK": "Slovakia", "SVN": "Slovenia", "SOM": "Somalia",
    "ZAF": "South Africa", "ESP": "Spain", "LKA": "Sri Lanka",
    "SDN": "Sudan", "SUR": "Suriname", "SWE": "Sweden",
    "CHE": "Switzerland", "SYR": "Syria", "TWN": "Taiwan",
    "TJK": "Tajikistan", "TZA": "Tanzania", "THA": "Thailand",
    "TLS": "Timor-Leste", "TGO": "Togo", "TUN": "Tunisia",
    "TUR": "Turkey", "TKM": "Turkmenistan", "UGA": "Uganda",
    "UKR": "Ukraine", "ARE": "UAE", "GBR": "UK",
    "USA": "USA", "URY": "Uruguay", "UZB": "Uzbekistan",
    "VUT": "Vanuatu", "VEN": "Venezuela", "VNM": "Vietnam",
    "YEM": "Yemen", "ZMB": "Zambia", "ZWE": "Zimbabwe",
    "SSD": "South Sudan", "PSE": "Palestine", "PRK": "North Korea",
    "CUW": "Curacao", "IMN": "Isle of Man", "FRO": "Faroe Islands",
    "GRL": "Greenland", "HKG": "Hong Kong", "MAC": "Macau",
    "NRU": "Nauru", "TUV": "Tuvalu", "MHL": "Marshall Islands",
    "FSM": "Micronesia", "PLW": "Palau", "TON": "Tonga",
    "WSM": "Samoa", "AND": "Andorra", "LIE": "Liechtenstein",
    "MCO": "Monaco", "SMR": "San Marino",
    "CYP": "Cyprus", "BHR": "Bahrain", "TCA": "Turks & Caicos Islands",
    "LUX": "Luxembourg", "CYM": "Cayman Islands",
    "MDV": "Maldives", "PRI": "Puerto Rico", "GIB": "Gibraltar",
    "BMU": "Bermuda", "MLT": "Malta", "BRB": "Barbados",
    "SXM": "Sint Maarten", "TTO": "Trinidad & Tobago",
    "ABW": "Aruba", "BRN": "Brunei", "FJI": "Fiji",
    "SYC": "Seychelles", "BLZ": "Belize",
    "VCT": "St. Vincent & Grenadines", "XKX": "Kosovo",
    "VGB": "British Virgin Islands", "BHS": "Bahamas",
    "VIR": "US Virgin Islands", "LCA": "Saint Lucia",
    "BTN": "Bhutan", "ATG": "Antigua & Barbuda", "GRD": "Grenada",
    "MAF": "Saint Martin", "GUM": "Guam", "CHI": "Channel Islands",
    "NCL": "New Caledonia", "MNP": "Northern Mariana Islands",
    "DMA": "Dominica", "PYF": "French Polynesia",
    "KNA": "Saint Kitts & Nevis", "STP": "Sao Tome & Principe",
    "ASM": "American Samoa", "COM": "Comoros",
    "SLB": "Solomon Islands", "BDI": "Burundi",
    "COD": "Congo (DRC)", "COG": "Congo",
}


def get_name(iso3: str) -> str:
    return COUNTRY_NAMES.get(iso3, iso3)


INCOME_GROUP_LABELS = {
    "low_income": "Low Income",
    "lower_middle_income": "Lower Middle Income",
    "upper_middle_income": "Upper Middle Income",
    "high_income": "High Income",
}

REGION_LABELS = {
    "EAS": "East Asia & Pacific",
    "ECS": "Europe & Central Asia",
    "LCN": "Latin America & Caribbean",
    "MEA": "Middle East & North Africa",
    "NAC": "North America",
    "SAS": "South Asia",
    "SSF": "Sub-Saharan Africa",
}


def classify_archetype(income_group: str) -> str:
    mapping = {
        "high_income": "high_development",
        "upper_middle_income": "upper_middle",
        "lower_middle_income": "lower_middle",
        "low_income": "low_development",
    }
    return mapping.get(income_group, "lower_middle")
