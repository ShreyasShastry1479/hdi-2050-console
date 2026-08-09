"""State capacity, stability, and institutional quality scores.

Composite index built from:
  - Conflict intensity (active wars, battle deaths)
  - Political stability / absence of violence
  - Government effectiveness
  - Control of corruption
  - Rule of law
  - Regulatory quality
  - Inflation stability (CPI volatility)
  - Ethnic fractionalization / social cohesion
  - State fragility

Scores: 0.0 (failed state) → 1.0 (highly stable/capable).
Applied as multipliers to GNI growth, life expectancy gains, and education progression.
"""

import numpy as np
import pandas as pd

# fmt: off
# Each country gets:
#   stability: 0-1, overall state capacity
#   conflict: 0-1, 1=no conflict, 0=active large-scale war
#   corruption: 0-1, 1=clean, 0=deeply corrupt
#   inflation_stability: 0-1, 1=stable prices, 0=hyperinflation risk
#   governance: 0-1, 1=strong institutions, 0=failed state
#   fragility: 0-1, 1=resilient, 0=fragile
#   growth_drag: multiplier on GNI growth (1.0=normal, 0.5=halved, 1.15=boosted)

STATE_CAPABILITY = {
    # === HIGHLY DEVELOPED - STRONG INSTITUTIONS ===
    "NOR": {"stability": 0.98, "conflict": 1.0, "corruption": 0.97, "inflation_stability": 0.98, "governance": 0.98, "fragility": 0.98, "growth_drag": 1.05},
    "CHE": {"stability": 0.97, "conflict": 1.0, "corruption": 0.96, "inflation_stability": 0.97, "governance": 0.97, "fragility": 0.97, "growth_drag": 1.05},
    "ISL": {"stability": 0.97, "conflict": 1.0, "corruption": 0.96, "inflation_stability": 0.96, "governance": 0.97, "fragility": 0.97, "growth_drag": 1.05},
    "DNK": {"stability": 0.97, "conflict": 1.0, "corruption": 0.97, "inflation_stability": 0.98, "governance": 0.97, "fragility": 0.97, "growth_drag": 1.05},
    "FIN": {"stability": 0.96, "conflict": 1.0, "corruption": 0.96, "inflation_stability": 0.97, "governance": 0.96, "fragility": 0.96, "growth_drag": 1.04},
    "SWE": {"stability": 0.96, "conflict": 1.0, "corruption": 0.96, "inflation_stability": 0.97, "governance": 0.96, "fragility": 0.96, "growth_drag": 1.04},
    "NZL": {"stability": 0.95, "conflict": 1.0, "corruption": 0.96, "inflation_stability": 0.96, "governance": 0.96, "fragility": 0.96, "growth_drag": 1.04},
    "SGP": {"stability": 0.96, "conflict": 1.0, "corruption": 0.97, "inflation_stability": 0.97, "governance": 0.98, "fragility": 0.95, "growth_drag": 1.05},
    "NLD": {"stability": 0.95, "conflict": 1.0, "corruption": 0.95, "inflation_stability": 0.96, "governance": 0.96, "fragility": 0.95, "growth_drag": 1.04},
    "DEU": {"stability": 0.95, "conflict": 1.0, "corruption": 0.95, "inflation_stability": 0.97, "governance": 0.96, "fragility": 0.95, "growth_drag": 1.04},
    "IRL": {"stability": 0.95, "conflict": 1.0, "corruption": 0.94, "inflation_stability": 0.96, "governance": 0.95, "fragility": 0.95, "growth_drag": 1.05},
    "AUT": {"stability": 0.94, "conflict": 1.0, "corruption": 0.94, "inflation_stability": 0.97, "governance": 0.95, "fragility": 0.94, "growth_drag": 1.04},
    "LUX": {"stability": 0.95, "conflict": 1.0, "corruption": 0.94, "inflation_stability": 0.97, "governance": 0.96, "fragility": 0.95, "growth_drag": 1.04},
    "AUS": {"stability": 0.94, "conflict": 1.0, "corruption": 0.95, "inflation_stability": 0.96, "governance": 0.95, "fragility": 0.95, "growth_drag": 1.04},
    "CAN": {"stability": 0.94, "conflict": 1.0, "corruption": 0.95, "inflation_stability": 0.96, "governance": 0.95, "fragility": 0.95, "growth_drag": 1.04},
    "GBR": {"stability": 0.93, "conflict": 1.0, "corruption": 0.93, "inflation_stability": 0.95, "governance": 0.94, "fragility": 0.93, "growth_drag": 1.03},
    "BEL": {"stability": 0.92, "conflict": 1.0, "corruption": 0.91, "inflation_stability": 0.96, "governance": 0.93, "fragility": 0.92, "growth_drag": 1.03},
    "FRA": {"stability": 0.92, "conflict": 1.0, "corruption": 0.92, "inflation_stability": 0.96, "governance": 0.93, "fragility": 0.92, "growth_drag": 1.03},
    "USA": {"stability": 0.90, "conflict": 1.0, "corruption": 0.88, "inflation_stability": 0.93, "governance": 0.90, "fragility": 0.88, "growth_drag": 1.05},
    "ESP": {"stability": 0.91, "conflict": 1.0, "corruption": 0.90, "inflation_stability": 0.96, "governance": 0.92, "fragility": 0.91, "growth_drag": 1.03},
    "PRT": {"stability": 0.91, "conflict": 1.0, "corruption": 0.90, "inflation_stability": 0.96, "governance": 0.91, "fragility": 0.91, "growth_drag": 1.03},
    "CZE": {"stability": 0.91, "conflict": 1.0, "corruption": 0.88, "inflation_stability": 0.94, "governance": 0.90, "fragility": 0.90, "growth_drag": 1.04},
    "EST": {"stability": 0.90, "conflict": 1.0, "corruption": 0.90, "inflation_stability": 0.95, "governance": 0.91, "fragility": 0.90, "growth_drag": 1.04},
    "LTU": {"stability": 0.90, "conflict": 1.0, "corruption": 0.89, "inflation_stability": 0.95, "governance": 0.90, "fragility": 0.90, "growth_drag": 1.04},
    "LVA": {"stability": 0.89, "conflict": 1.0, "corruption": 0.87, "inflation_stability": 0.94, "governance": 0.89, "fragility": 0.89, "growth_drag": 1.04},
    "SVN": {"stability": 0.90, "conflict": 1.0, "corruption": 0.89, "inflation_stability": 0.95, "governance": 0.91, "fragility": 0.90, "growth_drag": 1.04},
    "POL": {"stability": 0.89, "conflict": 1.0, "corruption": 0.85, "inflation_stability": 0.93, "governance": 0.88, "fragility": 0.89, "growth_drag": 1.05},
    "HUN": {"stability": 0.87, "conflict": 1.0, "corruption": 0.83, "inflation_stability": 0.92, "governance": 0.86, "fragility": 0.87, "growth_drag": 1.04},
    "SVK": {"stability": 0.88, "conflict": 1.0, "corruption": 0.86, "inflation_stability": 0.94, "governance": 0.88, "fragility": 0.88, "growth_drag": 1.04},
    "HRV": {"stability": 0.89, "conflict": 1.0, "corruption": 0.87, "inflation_stability": 0.95, "governance": 0.89, "fragility": 0.89, "growth_drag": 1.05},
    "ITA": {"stability": 0.87, "conflict": 1.0, "corruption": 0.82, "inflation_stability": 0.95, "governance": 0.86, "fragility": 0.86, "growth_drag": 1.02},
    "GRC": {"stability": 0.85, "conflict": 1.0, "corruption": 0.82, "inflation_stability": 0.95, "governance": 0.85, "fragility": 0.85, "growth_drag": 1.02},
    "MLT": {"stability": 0.89, "conflict": 1.0, "corruption": 0.88, "inflation_stability": 0.95, "governance": 0.90, "fragility": 0.89, "growth_drag": 1.03},
    "CYP": {"stability": 0.87, "conflict": 0.9, "corruption": 0.86, "inflation_stability": 0.94, "governance": 0.88, "fragility": 0.87, "growth_drag": 1.02},
    "BGR": {"stability": 0.82, "conflict": 1.0, "corruption": 0.75, "inflation_stability": 0.92, "governance": 0.80, "fragility": 0.82, "growth_drag": 1.03},
    "ROU": {"stability": 0.83, "conflict": 1.0, "corruption": 0.76, "inflation_stability": 0.90, "governance": 0.81, "fragility": 0.83, "growth_drag": 1.04},
    "SRB": {"stability": 0.82, "conflict": 0.92, "corruption": 0.78, "inflation_stability": 0.90, "governance": 0.80, "fragility": 0.82, "growth_drag": 1.05},
    "MNE": {"stability": 0.80, "conflict": 0.95, "corruption": 0.75, "inflation_stability": 0.92, "governance": 0.79, "fragility": 0.80, "growth_drag": 1.03},
    "MKD": {"stability": 0.81, "conflict": 0.95, "corruption": 0.76, "inflation_stability": 0.93, "governance": 0.80, "fragility": 0.81, "growth_drag": 1.04},
    "ALB": {"stability": 0.81, "conflict": 1.0, "corruption": 0.76, "inflation_stability": 0.93, "governance": 0.80, "fragility": 0.81, "growth_drag": 1.06},
    "BIH": {"stability": 0.75, "conflict": 0.88, "corruption": 0.72, "inflation_stability": 0.92, "governance": 0.74, "fragility": 0.75, "growth_drag": 1.00},
    "XKX": {"stability": 0.70, "conflict": 0.9, "corruption": 0.65, "inflation_stability": 0.88, "governance": 0.68, "fragility": 0.70, "growth_drag": 1.00},
    "MDA": {"stability": 0.68, "conflict": 0.8, "corruption": 0.62, "inflation_stability": 0.85, "governance": 0.66, "fragility": 0.68, "growth_drag": 0.98},
    "UKR": {"stability": 0.45, "conflict": 0.25, "corruption": 0.55, "inflation_stability": 0.65, "governance": 0.50, "fragility": 0.45, "growth_drag": 0.55},
    "BLR": {"stability": 0.58, "conflict": 0.7, "corruption": 0.55, "inflation_stability": 0.78, "governance": 0.50, "fragility": 0.58, "growth_drag": 0.75},
    "GEO": {"stability": 0.76, "conflict": 0.8, "corruption": 0.72, "inflation_stability": 0.90, "governance": 0.74, "fragility": 0.76, "growth_drag": 1.02},
    "ARM": {"stability": 0.73, "conflict": 0.65, "corruption": 0.70, "inflation_stability": 0.88, "governance": 0.72, "fragility": 0.73, "growth_drag": 0.98},
    "AZE": {"stability": 0.62, "conflict": 0.7, "corruption": 0.55, "inflation_stability": 0.82, "governance": 0.55, "fragility": 0.62, "growth_drag": 0.95},
    "KAZ": {"stability": 0.72, "conflict": 0.95, "corruption": 0.65, "inflation_stability": 0.85, "governance": 0.68, "fragility": 0.72, "growth_drag": 1.08},
    "UZB": {"stability": 0.65, "conflict": 0.95, "corruption": 0.55, "inflation_stability": 0.80, "governance": 0.58, "fragility": 0.65, "growth_drag": 0.98},
    "TKM": {"stability": 0.40, "conflict": 0.85, "corruption": 0.30, "inflation_stability": 0.60, "governance": 0.30, "fragility": 0.40, "growth_drag": 0.65},
    "KGZ": {"stability": 0.65, "conflict": 0.9, "corruption": 0.58, "inflation_stability": 0.82, "governance": 0.60, "fragility": 0.65, "growth_drag": 0.98},
    "TJK": {"stability": 0.55, "conflict": 0.85, "corruption": 0.45, "inflation_stability": 0.72, "governance": 0.48, "fragility": 0.55, "growth_drag": 0.85},
    "RUS": {"stability": 0.50, "conflict": 0.40, "corruption": 0.50, "inflation_stability": 0.72, "governance": 0.48, "fragility": 0.50, "growth_drag": 0.70},
    "TUR": {"stability": 0.68, "conflict": 0.75, "corruption": 0.62, "inflation_stability": 0.60, "governance": 0.65, "fragility": 0.68, "growth_drag": 0.90},
    "JOR": {"stability": 0.75, "conflict": 0.8, "corruption": 0.72, "inflation_stability": 0.88, "governance": 0.74, "fragility": 0.75, "growth_drag": 1.00},
    "LBN": {"stability": 0.45, "conflict": 0.45, "corruption": 0.42, "inflation_stability": 0.40, "governance": 0.40, "fragility": 0.42, "growth_drag": 0.55},
    "ISR": {"stability": 0.82, "conflict": 0.70, "corruption": 0.90, "inflation_stability": 0.95, "governance": 0.90, "fragility": 0.82, "growth_drag": 1.08},
    "PSE": {"stability": 0.35, "conflict": 0.25, "corruption": 0.45, "inflation_stability": 0.55, "governance": 0.35, "fragility": 0.35, "growth_drag": 0.50},
    "SYR": {"stability": 0.15, "conflict": 0.10, "corruption": 0.25, "inflation_stability": 0.20, "governance": 0.15, "fragility": 0.15, "growth_drag": 0.30},
    "IRQ": {"stability": 0.42, "conflict": 0.40, "corruption": 0.38, "inflation_stability": 0.60, "governance": 0.40, "fragility": 0.42, "growth_drag": 0.55},
    "YEM": {"stability": 0.18, "conflict": 0.12, "corruption": 0.22, "inflation_stability": 0.25, "governance": 0.15, "fragility": 0.15, "growth_drag": 0.30},
    "SAU": {"stability": 0.70, "conflict": 0.85, "corruption": 0.68, "inflation_stability": 0.90, "governance": 0.72, "fragility": 0.70, "growth_drag": 1.08},
    "ARE": {"stability": 0.85, "conflict": 0.95, "corruption": 0.82, "inflation_stability": 0.94, "governance": 0.85, "fragility": 0.85, "growth_drag": 1.10},
    "QAT": {"stability": 0.82, "conflict": 0.95, "corruption": 0.80, "inflation_stability": 0.94, "governance": 0.82, "fragility": 0.82, "growth_drag": 1.08},
    "KWT": {"stability": 0.75, "conflict": 0.90, "corruption": 0.72, "inflation_stability": 0.90, "governance": 0.75, "fragility": 0.75, "growth_drag": 1.05},
    "BHR": {"stability": 0.76, "conflict": 0.90, "corruption": 0.76, "inflation_stability": 0.92, "governance": 0.78, "fragility": 0.76, "growth_drag": 1.05},
    "OMN": {"stability": 0.76, "conflict": 0.92, "corruption": 0.74, "inflation_stability": 0.92, "governance": 0.76, "fragility": 0.76, "growth_drag": 1.05},
    "IRN": {"stability": 0.48, "conflict": 0.6, "corruption": 0.45, "inflation_stability": 0.50, "governance": 0.45, "fragility": 0.48, "growth_drag": 0.65},
    "AFG": {"stability": 0.15, "conflict": 0.15, "corruption": 0.18, "inflation_stability": 0.30, "governance": 0.12, "fragility": 0.12, "growth_drag": 0.35},
    "PAK": {"stability": 0.48, "conflict": 0.55, "corruption": 0.42, "inflation_stability": 0.50, "governance": 0.45, "fragility": 0.48, "growth_drag": 0.70},
    "BGD": {"stability": 0.62, "conflict": 0.85, "corruption": 0.55, "inflation_stability": 0.78, "governance": 0.58, "fragility": 0.62, "growth_drag": 0.95},
    "LKA": {"stability": 0.58, "conflict": 0.8, "corruption": 0.52, "inflation_stability": 0.55, "governance": 0.55, "fragility": 0.58, "growth_drag": 0.80},
    "NPL": {"stability": 0.62, "conflict": 0.85, "corruption": 0.52, "inflation_stability": 0.78, "governance": 0.55, "fragility": 0.62, "growth_drag": 0.92},
    "MMR": {"stability": 0.28, "conflict": 0.25, "corruption": 0.32, "inflation_stability": 0.45, "governance": 0.25, "fragility": 0.25, "growth_drag": 0.40},
    "THA": {"stability": 0.75, "conflict": 0.9, "corruption": 0.68, "inflation_stability": 0.88, "governance": 0.72, "fragility": 0.75, "growth_drag": 1.02},
    "VNM": {"stability": 0.72, "conflict": 0.95, "corruption": 0.60, "inflation_stability": 0.82, "governance": 0.70, "fragility": 0.72, "growth_drag": 1.05},
    "KHM": {"stability": 0.60, "conflict": 0.85, "corruption": 0.48, "inflation_stability": 0.78, "governance": 0.55, "fragility": 0.60, "growth_drag": 0.95},
    "LAO": {"stability": 0.58, "conflict": 0.9, "corruption": 0.48, "inflation_stability": 0.75, "governance": 0.52, "fragility": 0.58, "growth_drag": 0.92},
    "MYS": {"stability": 0.78, "conflict": 0.95, "corruption": 0.72, "inflation_stability": 0.88, "governance": 0.75, "fragility": 0.78, "growth_drag": 1.02},
    "IDN": {"stability": 0.72, "conflict": 0.9, "corruption": 0.62, "inflation_stability": 0.82, "governance": 0.68, "fragility": 0.72, "growth_drag": 1.02},
    "PHL": {"stability": 0.62, "conflict": 0.75, "corruption": 0.55, "inflation_stability": 0.80, "governance": 0.60, "fragility": 0.62, "growth_drag": 0.95},
    "CHN": {"stability": 0.72, "conflict": 0.95, "corruption": 0.58, "inflation_stability": 0.85, "governance": 0.75, "fragility": 0.72, "growth_drag": 1.20},
    "MNG": {"stability": 0.72, "conflict": 0.95, "corruption": 0.62, "inflation_stability": 0.80, "governance": 0.68, "fragility": 0.72, "growth_drag": 1.00},
    "PRK": {"stability": 0.25, "conflict": 0.65, "corruption": 0.20, "inflation_stability": 0.35, "governance": 0.25, "fragility": 0.25, "growth_drag": 0.40},
    "JPN": {"stability": 0.95, "conflict": 1.0, "corruption": 0.96, "inflation_stability": 0.95, "governance": 0.96, "fragility": 0.95, "growth_drag": 1.03},
    "KOR": {"stability": 0.90, "conflict": 0.9, "corruption": 0.88, "inflation_stability": 0.94, "governance": 0.90, "fragility": 0.90, "growth_drag": 1.05},
    "IND": {"stability": 0.62, "conflict": 0.8, "corruption": 0.52, "inflation_stability": 0.72, "governance": 0.60, "fragility": 0.62, "growth_drag": 1.15},
    "BRA": {"stability": 0.68, "conflict": 0.9, "corruption": 0.58, "inflation_stability": 0.72, "governance": 0.65, "fragility": 0.68, "growth_drag": 1.02},
    "MEX": {"stability": 0.58, "conflict": 0.65, "corruption": 0.52, "inflation_stability": 0.78, "governance": 0.58, "fragility": 0.58, "growth_drag": 0.95},
    "COL": {"stability": 0.60, "conflict": 0.65, "corruption": 0.55, "inflation_stability": 0.78, "governance": 0.60, "fragility": 0.60, "growth_drag": 0.95},
    "ARG": {"stability": 0.62, "conflict": 0.95, "corruption": 0.55, "inflation_stability": 0.45, "governance": 0.58, "fragility": 0.62, "growth_drag": 0.85},
    "CHL": {"stability": 0.82, "conflict": 0.95, "corruption": 0.82, "inflation_stability": 0.92, "governance": 0.82, "fragility": 0.82, "growth_drag": 1.02},
    "PER": {"stability": 0.68, "conflict": 0.85, "corruption": 0.62, "inflation_stability": 0.82, "governance": 0.65, "fragility": 0.68, "growth_drag": 0.98},
    "ECU": {"stability": 0.58, "conflict": 0.8, "corruption": 0.52, "inflation_stability": 0.72, "governance": 0.55, "fragility": 0.58, "growth_drag": 0.88},
    "VEN": {"stability": 0.22, "conflict": 0.55, "corruption": 0.18, "inflation_stability": 0.15, "governance": 0.15, "fragility": 0.18, "growth_drag": 0.30},
    "BOL": {"stability": 0.55, "conflict": 0.85, "corruption": 0.48, "inflation_stability": 0.72, "governance": 0.52, "fragility": 0.55, "growth_drag": 0.85},
    "PRY": {"stability": 0.62, "conflict": 0.9, "corruption": 0.55, "inflation_stability": 0.78, "governance": 0.60, "fragility": 0.62, "growth_drag": 0.92},
    "URY": {"stability": 0.82, "conflict": 1.0, "corruption": 0.82, "inflation_stability": 0.92, "governance": 0.82, "fragility": 0.82, "growth_drag": 1.02},
    "CUB": {"stability": 0.55, "conflict": 0.9, "corruption": 0.50, "inflation_stability": 0.45, "governance": 0.50, "fragility": 0.55, "growth_drag": 0.70},
    "DOM": {"stability": 0.65, "conflict": 0.9, "corruption": 0.55, "inflation_stability": 0.80, "governance": 0.62, "fragility": 0.65, "growth_drag": 0.95},
    "GTM": {"stability": 0.52, "conflict": 0.7, "corruption": 0.45, "inflation_stability": 0.72, "governance": 0.50, "fragility": 0.52, "growth_drag": 0.85},
    "HND": {"stability": 0.45, "conflict": 0.6, "corruption": 0.38, "inflation_stability": 0.65, "governance": 0.42, "fragility": 0.45, "growth_drag": 0.78},
    "SLV": {"stability": 0.55, "conflict": 0.7, "corruption": 0.48, "inflation_stability": 0.78, "governance": 0.52, "fragility": 0.55, "growth_drag": 0.88},
    "NIC": {"stability": 0.52, "conflict": 0.7, "corruption": 0.45, "inflation_stability": 0.72, "governance": 0.50, "fragility": 0.52, "growth_drag": 0.85},
    "CRI": {"stability": 0.78, "conflict": 0.95, "corruption": 0.78, "inflation_stability": 0.90, "governance": 0.78, "fragility": 0.78, "growth_drag": 1.02},
    "PAN": {"stability": 0.72, "conflict": 0.95, "corruption": 0.68, "inflation_stability": 0.88, "governance": 0.70, "fragility": 0.72, "growth_drag": 1.00},
    "CUB": {"stability": 0.50, "conflict": 0.9, "corruption": 0.48, "inflation_stability": 0.40, "governance": 0.45, "fragility": 0.50, "growth_drag": 0.65},
    "HTI": {"stability": 0.22, "conflict": 0.45, "corruption": 0.22, "inflation_stability": 0.45, "governance": 0.20, "fragility": 0.22, "growth_drag": 0.40},
    "JAM": {"stability": 0.70, "conflict": 0.85, "corruption": 0.68, "inflation_stability": 0.85, "governance": 0.70, "fragility": 0.70, "growth_drag": 0.98},
    "TTO": {"stability": 0.70, "conflict": 0.9, "corruption": 0.65, "inflation_stability": 0.85, "governance": 0.68, "fragility": 0.70, "growth_drag": 0.98},
    "GUY": {"stability": 0.65, "conflict": 0.9, "corruption": 0.58, "inflation_stability": 0.80, "governance": 0.62, "fragility": 0.65, "growth_drag": 0.95},
    "SUR": {"stability": 0.55, "conflict": 0.85, "corruption": 0.48, "inflation_stability": 0.65, "governance": 0.52, "fragility": 0.55, "growth_drag": 0.82},
    "BLZ": {"stability": 0.72, "conflict": 0.95, "corruption": 0.68, "inflation_stability": 0.88, "governance": 0.70, "fragility": 0.72, "growth_drag": 1.00},
    "BHS": {"stability": 0.78, "conflict": 1.0, "corruption": 0.75, "inflation_stability": 0.88, "governance": 0.78, "fragility": 0.78, "growth_drag": 1.00},
    "BRB": {"stability": 0.80, "conflict": 1.0, "corruption": 0.78, "inflation_stability": 0.90, "governance": 0.80, "fragility": 0.80, "growth_drag": 1.00},
    "ATG": {"stability": 0.78, "conflict": 1.0, "corruption": 0.75, "inflation_stability": 0.88, "governance": 0.78, "fragility": 0.78, "growth_drag": 1.00},
    "DMA": {"stability": 0.78, "conflict": 1.0, "corruption": 0.75, "inflation_stability": 0.88, "governance": 0.78, "fragility": 0.78, "growth_drag": 1.00},
    "GRD": {"stability": 0.75, "conflict": 1.0, "corruption": 0.72, "inflation_stability": 0.85, "governance": 0.75, "fragility": 0.75, "growth_drag": 1.00},
    "KNA": {"stability": 0.78, "conflict": 1.0, "corruption": 0.75, "inflation_stability": 0.88, "governance": 0.78, "fragility": 0.78, "growth_drag": 1.00},
    "LCA": {"stability": 0.75, "conflict": 1.0, "corruption": 0.72, "inflation_stability": 0.85, "governance": 0.75, "fragility": 0.75, "growth_drag": 1.00},
    "VCT": {"stability": 0.72, "conflict": 1.0, "corruption": 0.70, "inflation_stability": 0.85, "governance": 0.72, "fragility": 0.72, "growth_drag": 1.00},
    "AFG": {"stability": 0.12, "conflict": 0.10, "corruption": 0.15, "inflation_stability": 0.25, "governance": 0.10, "fragility": 0.10, "growth_drag": 0.30},
    "EGY": {"stability": 0.55, "conflict": 0.75, "corruption": 0.50, "inflation_stability": 0.55, "governance": 0.52, "fragility": 0.55, "growth_drag": 0.85},
    "MAR": {"stability": 0.68, "conflict": 0.85, "corruption": 0.62, "inflation_stability": 0.85, "governance": 0.65, "fragility": 0.68, "growth_drag": 0.95},
    "TUN": {"stability": 0.62, "conflict": 0.8, "corruption": 0.60, "inflation_stability": 0.72, "governance": 0.62, "fragility": 0.62, "growth_drag": 0.92},
    "DZA": {"stability": 0.55, "conflict": 0.75, "corruption": 0.48, "inflation_stability": 0.75, "governance": 0.50, "fragility": 0.55, "growth_drag": 0.82},
    "LBY": {"stability": 0.28, "conflict": 0.25, "corruption": 0.30, "inflation_stability": 0.35, "governance": 0.25, "fragility": 0.28, "growth_drag": 0.40},
    "TCD": {"stability": 0.28, "conflict": 0.30, "corruption": 0.22, "inflation_stability": 0.45, "governance": 0.25, "fragility": 0.25, "growth_drag": 0.45},
    "SDN": {"stability": 0.18, "conflict": 0.15, "corruption": 0.18, "inflation_stability": 0.25, "governance": 0.15, "fragility": 0.15, "growth_drag": 0.30},
    "SSD": {"stability": 0.12, "conflict": 0.10, "corruption": 0.15, "inflation_stability": 0.20, "governance": 0.10, "fragility": 0.10, "growth_drag": 0.25},
    "ERI": {"stability": 0.18, "conflict": 0.40, "corruption": 0.18, "inflation_stability": 0.35, "governance": 0.15, "fragility": 0.18, "growth_drag": 0.35},
    "SOM": {"stability": 0.12, "conflict": 0.10, "corruption": 0.15, "inflation_stability": 0.25, "governance": 0.10, "fragility": 0.10, "growth_drag": 0.25},
    "ETH": {"stability": 0.38, "conflict": 0.35, "corruption": 0.38, "inflation_stability": 0.50, "governance": 0.38, "fragility": 0.38, "growth_drag": 0.60},
    "KEN": {"stability": 0.58, "conflict": 0.75, "corruption": 0.50, "inflation_stability": 0.72, "governance": 0.55, "fragility": 0.58, "growth_drag": 0.90},
    "UGA": {"stability": 0.52, "conflict": 0.65, "corruption": 0.42, "inflation_stability": 0.65, "governance": 0.48, "fragility": 0.52, "growth_drag": 0.80},
    "TZA": {"stability": 0.60, "conflict": 0.85, "corruption": 0.48, "inflation_stability": 0.72, "governance": 0.55, "fragility": 0.60, "growth_drag": 0.92},
    "RWA": {"stability": 0.62, "conflict": 0.8, "corruption": 0.55, "inflation_stability": 0.78, "governance": 0.62, "fragility": 0.62, "growth_drag": 0.95},
    "BDI": {"stability": 0.35, "conflict": 0.55, "corruption": 0.30, "inflation_stability": 0.55, "governance": 0.32, "fragility": 0.35, "growth_drag": 0.55},
    "COD": {"stability": 0.18, "conflict": 0.15, "corruption": 0.18, "inflation_stability": 0.30, "governance": 0.15, "fragility": 0.15, "growth_drag": 0.30},
    "COG": {"stability": 0.35, "conflict": 0.55, "corruption": 0.30, "inflation_stability": 0.55, "governance": 0.32, "fragility": 0.35, "growth_drag": 0.55},
    "GAB": {"stability": 0.48, "conflict": 0.75, "corruption": 0.42, "inflation_stability": 0.65, "governance": 0.45, "fragility": 0.48, "growth_drag": 0.75},
    "CMR": {"stability": 0.42, "conflict": 0.6, "corruption": 0.35, "inflation_stability": 0.60, "governance": 0.40, "fragility": 0.42, "growth_drag": 0.68},
    "NGA": {"stability": 0.35, "conflict": 0.45, "corruption": 0.28, "inflation_stability": 0.55, "governance": 0.32, "fragility": 0.35, "growth_drag": 0.60},
    "GHA": {"stability": 0.62, "conflict": 0.85, "corruption": 0.55, "inflation_stability": 0.75, "governance": 0.58, "fragility": 0.62, "growth_drag": 0.92},
    "SEN": {"stability": 0.60, "conflict": 0.85, "corruption": 0.55, "inflation_stability": 0.75, "governance": 0.58, "fragility": 0.60, "growth_drag": 0.92},
    "CIV": {"stability": 0.45, "conflict": 0.55, "corruption": 0.38, "inflation_stability": 0.65, "governance": 0.42, "fragility": 0.45, "growth_drag": 0.72},
    "BFA": {"stability": 0.32, "conflict": 0.35, "corruption": 0.28, "inflation_stability": 0.50, "governance": 0.30, "fragility": 0.32, "growth_drag": 0.50},
    "MLI": {"stability": 0.28, "conflict": 0.25, "corruption": 0.25, "inflation_stability": 0.45, "governance": 0.25, "fragility": 0.25, "growth_drag": 0.42},
    "NER": {"stability": 0.22, "conflict": 0.20, "corruption": 0.20, "inflation_stability": 0.40, "governance": 0.20, "fragility": 0.20, "growth_drag": 0.35},
    "GIN": {"stability": 0.38, "conflict": 0.65, "corruption": 0.32, "inflation_stability": 0.55, "governance": 0.35, "fragility": 0.38, "growth_drag": 0.60},
    "SLE": {"stability": 0.28, "conflict": 0.40, "corruption": 0.25, "inflation_stability": 0.45, "governance": 0.25, "fragility": 0.28, "growth_drag": 0.42},
    "LBR": {"stability": 0.32, "conflict": 0.55, "corruption": 0.28, "inflation_stability": 0.50, "governance": 0.30, "fragility": 0.32, "growth_drag": 0.50},
    "GMB": {"stability": 0.45, "conflict": 0.65, "corruption": 0.38, "inflation_stability": 0.62, "governance": 0.42, "fragility": 0.45, "growth_drag": 0.72},
    "GNB": {"stability": 0.38, "conflict": 0.6, "corruption": 0.32, "inflation_stability": 0.55, "governance": 0.35, "fragility": 0.38, "growth_drag": 0.60},
    "BEN": {"stability": 0.48, "conflict": 0.7, "corruption": 0.40, "inflation_stability": 0.65, "governance": 0.45, "fragility": 0.48, "growth_drag": 0.75},
    "TGO": {"stability": 0.48, "conflict": 0.7, "corruption": 0.40, "inflation_stability": 0.65, "governance": 0.45, "fragility": 0.48, "growth_drag": 0.75},
    "AGO": {"stability": 0.40, "conflict": 0.6, "corruption": 0.32, "inflation_stability": 0.55, "governance": 0.38, "fragility": 0.40, "growth_drag": 0.65},
    "MOZ": {"stability": 0.38, "conflict": 0.6, "corruption": 0.32, "inflation_stability": 0.55, "governance": 0.35, "fragility": 0.38, "growth_drag": 0.60},
    "ZAF": {"stability": 0.50, "conflict": 0.7, "corruption": 0.42, "inflation_stability": 0.68, "governance": 0.48, "fragility": 0.50, "growth_drag": 0.78},
    "ZMB": {"stability": 0.48, "conflict": 0.8, "corruption": 0.38, "inflation_stability": 0.55, "governance": 0.45, "fragility": 0.48, "growth_drag": 0.72},
    "ZWE": {"stability": 0.30, "conflict": 0.6, "corruption": 0.22, "inflation_stability": 0.25, "governance": 0.25, "fragility": 0.30, "growth_drag": 0.40},
    "MWI": {"stability": 0.42, "conflict": 0.75, "corruption": 0.35, "inflation_stability": 0.55, "governance": 0.40, "fragility": 0.42, "growth_drag": 0.65},
    "MDG": {"stability": 0.38, "conflict": 0.65, "corruption": 0.32, "inflation_stability": 0.55, "governance": 0.35, "fragility": 0.38, "growth_drag": 0.58},
    "NAM": {"stability": 0.72, "conflict": 0.95, "corruption": 0.68, "inflation_stability": 0.82, "governance": 0.70, "fragility": 0.72, "growth_drag": 1.00},
    "BWA": {"stability": 0.75, "conflict": 0.95, "corruption": 0.72, "inflation_stability": 0.85, "governance": 0.74, "fragility": 0.75, "growth_drag": 1.02},
    "LSO": {"stability": 0.55, "conflict": 0.85, "corruption": 0.48, "inflation_stability": 0.68, "governance": 0.52, "fragility": 0.55, "growth_drag": 0.82},
    "SWZ": {"stability": 0.52, "conflict": 0.85, "corruption": 0.45, "inflation_stability": 0.68, "governance": 0.50, "fragility": 0.52, "growth_drag": 0.78},
    "DJI": {"stability": 0.45, "conflict": 0.65, "corruption": 0.38, "inflation_stability": 0.60, "governance": 0.42, "fragility": 0.45, "growth_drag": 0.70},
    "ERI": {"stability": 0.15, "conflict": 0.35, "corruption": 0.18, "inflation_stability": 0.30, "governance": 0.12, "fragility": 0.15, "growth_drag": 0.30},
    "COM": {"stability": 0.48, "conflict": 0.7, "corruption": 0.40, "inflation_stability": 0.62, "governance": 0.45, "fragility": 0.48, "growth_drag": 0.75},
    "MUS": {"stability": 0.78, "conflict": 1.0, "corruption": 0.75, "inflation_stability": 0.88, "governance": 0.78, "fragility": 0.78, "growth_drag": 1.02},
    "SYC": {"stability": 0.78, "conflict": 1.0, "corruption": 0.75, "inflation_stability": 0.88, "governance": 0.78, "fragility": 0.78, "growth_drag": 1.00},
    "CPV": {"stability": 0.72, "conflict": 1.0, "corruption": 0.70, "inflation_stability": 0.85, "governance": 0.72, "fragility": 0.72, "growth_drag": 1.00},
    "STP": {"stability": 0.55, "conflict": 0.85, "corruption": 0.48, "inflation_stability": 0.68, "governance": 0.52, "fragility": 0.55, "growth_drag": 0.82},
    "GNQ": {"stability": 0.32, "conflict": 0.7, "corruption": 0.22, "inflation_stability": 0.55, "governance": 0.28, "fragility": 0.32, "growth_drag": 0.50},
    "BWA": {"stability": 0.75, "conflict": 0.95, "corruption": 0.72, "inflation_stability": 0.85, "governance": 0.74, "fragility": 0.75, "growth_drag": 1.02},
    "SSD": {"stability": 0.10, "conflict": 0.08, "corruption": 0.12, "inflation_stability": 0.18, "governance": 0.08, "fragility": 0.10, "growth_drag": 0.22},
    # Small territories / island states (generally stable)
    "GIB": {"stability": 0.90, "conflict": 1.0, "corruption": 0.92, "inflation_stability": 0.95, "governance": 0.92, "fragility": 0.90, "growth_drag": 1.02},
    "FRO": {"stability": 0.92, "conflict": 1.0, "corruption": 0.94, "inflation_stability": 0.96, "governance": 0.94, "fragility": 0.92, "growth_drag": 1.02},
    "AND": {"stability": 0.90, "conflict": 1.0, "corruption": 0.88, "inflation_stability": 0.95, "governance": 0.90, "fragility": 0.90, "growth_drag": 1.02},
    "LIE": {"stability": 0.95, "conflict": 1.0, "corruption": 0.94, "inflation_stability": 0.97, "governance": 0.95, "fragility": 0.95, "growth_drag": 1.02},
    "MCO": {"stability": 0.92, "conflict": 1.0, "corruption": 0.88, "inflation_stability": 0.96, "governance": 0.92, "fragility": 0.92, "growth_drag": 1.02},
    "SMR": {"stability": 0.90, "conflict": 1.0, "corruption": 0.88, "inflation_stability": 0.95, "governance": 0.90, "fragility": 0.90, "growth_drag": 1.02},
    "IMN": {"stability": 0.90, "conflict": 1.0, "corruption": 0.90, "inflation_stability": 0.95, "governance": 0.92, "fragility": 0.90, "growth_drag": 1.02},
    "BMU": {"stability": 0.92, "conflict": 1.0, "corruption": 0.92, "inflation_stability": 0.95, "governance": 0.92, "fragility": 0.92, "growth_drag": 1.00},
    "CYM": {"stability": 0.90, "conflict": 1.0, "corruption": 0.88, "inflation_stability": 0.95, "governance": 0.90, "fragility": 0.90, "growth_drag": 1.00},
    "VGB": {"stability": 0.88, "conflict": 1.0, "corruption": 0.88, "inflation_stability": 0.92, "governance": 0.88, "fragility": 0.88, "growth_drag": 1.00},
    "AIA": {"stability": 0.85, "conflict": 1.0, "corruption": 0.85, "inflation_stability": 0.90, "governance": 0.85, "fragility": 0.85, "growth_drag": 1.00},
    "MSR": {"stability": 0.82, "conflict": 1.0, "corruption": 0.82, "inflation_stability": 0.88, "governance": 0.82, "fragility": 0.82, "growth_drag": 1.00},
    "CUW": {"stability": 0.82, "conflict": 1.0, "corruption": 0.80, "inflation_stability": 0.88, "governance": 0.82, "fragility": 0.82, "growth_drag": 1.00},
    "ABW": {"stability": 0.82, "conflict": 1.0, "corruption": 0.80, "inflation_stability": 0.88, "governance": 0.82, "fragility": 0.82, "growth_drag": 1.00},
    "GLP": {"stability": 0.82, "conflict": 1.0, "corruption": 0.82, "inflation_stability": 0.88, "governance": 0.82, "fragility": 0.82, "growth_drag": 1.00},
    "MTQ": {"stability": 0.82, "conflict": 1.0, "corruption": 0.82, "inflation_stability": 0.88, "governance": 0.82, "fragility": 0.82, "growth_drag": 1.00},
    "NCL": {"stability": 0.82, "conflict": 1.0, "corruption": 0.82, "inflation_stability": 0.88, "governance": 0.82, "fragility": 0.82, "growth_drag": 1.00},
    "PYF": {"stability": 0.80, "conflict": 1.0, "corruption": 0.80, "inflation_stability": 0.88, "governance": 0.80, "fragility": 0.80, "growth_drag": 1.00},
    "GUM": {"stability": 0.85, "conflict": 1.0, "corruption": 0.85, "inflation_stability": 0.90, "governance": 0.85, "fragility": 0.85, "growth_drag": 1.00},
    "ASM": {"stability": 0.82, "conflict": 1.0, "corruption": 0.82, "inflation_stability": 0.88, "governance": 0.82, "fragility": 0.82, "growth_drag": 1.00},
    "MNP": {"stability": 0.82, "conflict": 1.0, "corruption": 0.82, "inflation_stability": 0.88, "governance": 0.82, "fragility": 0.82, "growth_drag": 1.00},
    "VIR": {"stability": 0.80, "conflict": 1.0, "corruption": 0.80, "inflation_stability": 0.85, "governance": 0.80, "fragility": 0.80, "growth_drag": 1.00},
    "HKG": {"stability": 0.82, "conflict": 1.0, "corruption": 0.85, "inflation_stability": 0.95, "governance": 0.85, "fragility": 0.82, "growth_drag": 1.03},
    "MAC": {"stability": 0.82, "conflict": 1.0, "corruption": 0.82, "inflation_stability": 0.95, "governance": 0.82, "fragility": 0.82, "growth_drag": 1.02},
    "TLS": {"stability": 0.50, "conflict": 0.75, "corruption": 0.42, "inflation_stability": 0.62, "governance": 0.48, "fragility": 0.50, "growth_drag": 0.78},
    "SLB": {"stability": 0.52, "conflict": 0.8, "corruption": 0.45, "inflation_stability": 0.62, "governance": 0.50, "fragility": 0.52, "growth_drag": 0.78},
    "VUT": {"stability": 0.55, "conflict": 0.85, "corruption": 0.48, "inflation_stability": 0.65, "governance": 0.52, "fragility": 0.55, "growth_drag": 0.82},
    "FJI": {"stability": 0.68, "conflict": 0.9, "corruption": 0.62, "inflation_stability": 0.78, "governance": 0.65, "fragility": 0.68, "growth_drag": 0.95},
    "TON": {"stability": 0.68, "conflict": 0.95, "corruption": 0.62, "inflation_stability": 0.78, "governance": 0.65, "fragility": 0.68, "growth_drag": 0.95},
    "WSM": {"stability": 0.68, "conflict": 0.95, "corruption": 0.62, "inflation_stability": 0.78, "governance": 0.65, "fragility": 0.68, "growth_drag": 0.95},
    "KIR": {"stability": 0.60, "conflict": 0.9, "corruption": 0.52, "inflation_stability": 0.68, "governance": 0.55, "fragility": 0.60, "growth_drag": 0.85},
    "MHL": {"stability": 0.60, "conflict": 0.9, "corruption": 0.52, "inflation_stability": 0.68, "governance": 0.55, "fragility": 0.60, "growth_drag": 0.85},
    "FSM": {"stability": 0.58, "conflict": 0.9, "corruption": 0.50, "inflation_stability": 0.65, "governance": 0.55, "fragility": 0.58, "growth_drag": 0.82},
    "PLW": {"stability": 0.62, "conflict": 0.9, "corruption": 0.55, "inflation_stability": 0.70, "governance": 0.60, "fragility": 0.62, "growth_drag": 0.88},
    "NRU": {"stability": 0.60, "conflict": 0.9, "corruption": 0.52, "inflation_stability": 0.68, "governance": 0.55, "fragility": 0.60, "growth_drag": 0.82},
    "TUV": {"stability": 0.60, "conflict": 0.9, "corruption": 0.52, "inflation_stability": 0.68, "governance": 0.55, "fragility": 0.60, "growth_drag": 0.82},
    "BTN": {"stability": 0.68, "conflict": 0.95, "corruption": 0.60, "inflation_stability": 0.78, "governance": 0.65, "fragility": 0.68, "growth_drag": 0.95},
    "MDV": {"stability": 0.65, "conflict": 0.9, "corruption": 0.58, "inflation_stability": 0.75, "governance": 0.62, "fragility": 0.65, "growth_drag": 0.92},
    "BRN": {"stability": 0.72, "conflict": 0.95, "corruption": 0.68, "inflation_stability": 0.85, "governance": 0.72, "fragility": 0.72, "growth_drag": 1.00},
    "SSD": {"stability": 0.10, "conflict": 0.08, "corruption": 0.12, "inflation_stability": 0.18, "governance": 0.08, "fragility": 0.10, "growth_drag": 0.22},
}
# fmt: on

# Composite state capacity score (weighted average)
_WEIGHTS = {
    "stability": 0.20,
    "conflict": 0.25,
    "corruption": 0.15,
    "inflation_stability": 0.15,
    "governance": 0.15,
    "fragility": 0.10,
}


def get_state_capacity(country_id: str) -> dict:
    """Return state capacity scores for a country. Default if not in lookup."""
    defaults = {
        "stability": 0.50, "conflict": 0.65, "corruption": 0.45,
        "inflation_stability": 0.65, "governance": 0.48, "fragility": 0.50,
        "growth_drag": 0.85,
    }
    return STATE_CAPABILITY.get(country_id, defaults)


def compute_capacity_score(country_id: str) -> float:
    """Composite 0-1 state capacity score."""
    scores = get_state_capacity(country_id)
    return sum(scores.get(k, 0.5) * w for k, w in _WEIGHTS.items())


def apply_state_capacity_adjustments(
    forecast_df: pd.DataFrame,
) -> pd.DataFrame:
    """Apply state capacity adjustments to forecasted variables.

    Over a 24-year horizon (2026-2050):
    - Conflict zones may stabilize
    - Governance may improve
    - Countries far from frontier get catch-up multipliers
    - GNI growth multiplied by growth_drag
    - Life expectancy gains scaled by conflict × governance
    - Education progression scaled by governance × corruption
    """
    df = forecast_df.copy()

    # Countries that will see significant stability improvements by 2050
    # (post-conflict recovery, institutional reform, etc.)
    IMPROVING_COUNTRIES = {
        "IRQ": {"conflict_improve": 0.20, "gov_improve": 0.15, "corruption_improve": 0.10},
        "SYR": {"conflict_improve": 0.30, "gov_improve": 0.20, "corruption_improve": 0.15},
        "YEM": {"conflict_improve": 0.25, "gov_improve": 0.20, "corruption_improve": 0.15},
        "AFG": {"conflict_improve": 0.20, "gov_improve": 0.15, "corruption_improve": 0.10},
        "SSD": {"conflict_improve": 0.25, "gov_improve": 0.20, "corruption_improve": 0.15},
        "SOM": {"conflict_improve": 0.20, "gov_improve": 0.15, "corruption_improve": 0.10},
        "LBY": {"conflict_improve": 0.25, "gov_improve": 0.20, "corruption_improve": 0.15},
        "UKR": {"conflict_improve": 0.30, "gov_improve": 0.25, "corruption_improve": 0.20},
        "PSE": {"conflict_improve": 0.25, "gov_improve": 0.20, "corruption_improve": 0.15},
        "SDN": {"conflict_improve": 0.20, "gov_improve": 0.15, "corruption_improve": 0.10},
        "ETH": {"conflict_improve": 0.20, "gov_improve": 0.15, "corruption_improve": 0.10},
        "MMR": {"conflict_improve": 0.20, "gov_improve": 0.15, "corruption_improve": 0.10},
        "NGA": {"conflict_improve": 0.15, "gov_improve": 0.15, "corruption_improve": 0.10},
        "COD": {"conflict_improve": 0.20, "gov_improve": 0.15, "corruption_improve": 0.10},
        "BFA": {"conflict_improve": 0.15, "gov_improve": 0.15, "corruption_improve": 0.10},
        "MLI": {"conflict_improve": 0.20, "gov_improve": 0.15, "corruption_improve": 0.10},
        "NER": {"conflict_improve": 0.15, "gov_improve": 0.15, "corruption_improve": 0.10},
        "CMR": {"conflict_improve": 0.15, "gov_improve": 0.15, "corruption_improve": 0.10},
        "CAF": {"conflict_improve": 0.20, "gov_improve": 0.15, "corruption_improve": 0.10},
        "TCD": {"conflict_improve": 0.15, "gov_improve": 0.15, "corruption_improve": 0.10},
        "MOZ": {"conflict_improve": 0.15, "gov_improve": 0.15, "corruption_improve": 0.10},
        "BDI": {"conflict_improve": 0.20, "gov_improve": 0.15, "corruption_improve": 0.10},
        "HTI": {"conflict_improve": 0.20, "gov_improve": 0.15, "corruption_improve": 0.10},
        "LBN": {"conflict_improve": 0.25, "gov_improve": 0.20, "corruption_improve": 0.15},
        "LBR": {"conflict_improve": 0.15, "gov_improve": 0.15, "corruption_improve": 0.10},
        "SLE": {"conflict_improve": 0.15, "gov_improve": 0.15, "corruption_improve": 0.10},
        "MRT": {"conflict_improve": 0.10, "gov_improve": 0.10, "corruption_improve": 0.08},
        "SEN": {"conflict_improve": 0.10, "gov_improve": 0.10, "corruption_improve": 0.08},
        "GHA": {"conflict_improve": 0.10, "gov_improve": 0.10, "corruption_improve": 0.08},
        "TZA": {"conflict_improve": 0.10, "gov_improve": 0.10, "corruption_improve": 0.08},
        "KEN": {"conflict_improve": 0.10, "gov_improve": 0.10, "corruption_improve": 0.08},
        "UGA": {"conflict_improve": 0.10, "gov_improve": 0.10, "corruption_improve": 0.08},
        "RWA": {"conflict_improve": 0.10, "gov_improve": 0.10, "corruption_improve": 0.08},
        "BGD": {"conflict_improve": 0.08, "gov_improve": 0.10, "corruption_improve": 0.08},
        "PAK": {"conflict_improve": 0.15, "gov_improve": 0.12, "corruption_improve": 0.10},
        "IRN": {"conflict_improve": 0.10, "gov_improve": 0.10, "corruption_improve": 0.08},
        "EGY": {"conflict_improve": 0.10, "gov_improve": 0.10, "corruption_improve": 0.08},
        "MAR": {"conflict_improve": 0.08, "gov_improve": 0.08, "corruption_improve": 0.06},
        "DZA": {"conflict_improve": 0.08, "gov_improve": 0.08, "corruption_improve": 0.06},
        "TUN": {"conflict_improve": 0.08, "gov_improve": 0.08, "corruption_improve": 0.06},
        "ZAF": {"conflict_improve": 0.10, "gov_improve": 0.10, "corruption_improve": 0.08},
        "ZWE": {"conflict_improve": 0.15, "gov_improve": 0.12, "corruption_improve": 0.10},
        "VEN": {"conflict_improve": 0.15, "gov_improve": 0.15, "corruption_improve": 0.12},
        "CUB": {"conflict_improve": 0.05, "gov_improve": 0.10, "corruption_improve": 0.08},
        "PRK": {"conflict_improve": 0.05, "gov_improve": 0.05, "corruption_improve": 0.05},
        "ERI": {"conflict_improve": 0.10, "gov_improve": 0.10, "corruption_improve": 0.08},
    }

    for idx, row in df.iterrows():
        cid = row["country_id"]
        caps = get_state_capacity(cid)
        year = row.get("year", 2050)

        # Calculate years into the forecast (2026 is base)
        years_from_now = max(0, year - 2026)
        progress_fraction = min(1.0, years_from_now / 24.0)

        # Apply time-dependent improvements for conflict-prone countries
        conflict_factor = caps["conflict"]
        governance_factor = caps["governance"]
        corruption_factor = caps["corruption"]
        inflation_stability = caps["inflation_stability"]
        growth_drag = caps["growth_drag"]

        if cid in IMPROVING_COUNTRIES:
            imp = IMPROVING_COUNTRIES[cid]
            # Gradually improve conflict, governance, corruption scores
            conflict_factor = min(0.95, conflict_factor + imp["conflict_improve"] * progress_fraction)
            governance_factor = min(0.90, governance_factor + imp["gov_improve"] * progress_fraction)
            corruption_factor = min(0.85, corruption_factor + imp["corruption_improve"] * progress_fraction)
            # Growth drag improves materially over a 2050 horizon; current
            # conflict should not permanently suppress developing economies.
            stability_gain = (imp["conflict_improve"] + imp["gov_improve"] + imp["corruption_improve"]) / 3
            recovery_target = 0.88 if caps["growth_drag"] < 0.45 else 0.98
            growth_drag = max(
                growth_drag + stability_gain * 0.50 * progress_fraction,
                growth_drag + (recovery_target - growth_drag) * progress_fraction,
            )
            growth_drag = min(1.15, growth_drag)

        if caps["growth_drag"] < 0.55 and year >= 2035:
            long_run_progress = min(1.0, (year - 2035) / 15.0)
            growth_drag = max(growth_drag, caps["growth_drag"] + (0.82 - caps["growth_drag"]) * long_run_progress)
            conflict_factor = max(conflict_factor, caps["conflict"] + (0.70 - caps["conflict"]) * long_run_progress)
            governance_factor = max(governance_factor, caps["governance"] + (0.58 - caps["governance"]) * long_run_progress)
            corruption_factor = max(corruption_factor, caps["corruption"] + (0.55 - caps["corruption"]) * long_run_progress)

        # GNI: multiply by growth_drag (improving over time)
        gni_adj = growth_drag * (0.92 + 0.08 * inflation_stability)

        # Life expectancy: penalty for conflict zones (improving over time)
        le_adj = 0.93 + 0.07 * conflict_factor

        # Education: penalty for poor governance (improving over time)
        edu_adj = 0.92 + 0.08 * (governance_factor * 0.6 + corruption_factor * 0.4)

        # Apply adjustments
        if "gni_ppp" in df.columns:
            df.at[idx, "gni_ppp"] = row["gni_ppp"] * gni_adj

        if "life_exp" in df.columns:
            df.at[idx, "life_exp"] = row["life_exp"] * le_adj

        if "expected_school" in df.columns:
            df.at[idx, "expected_school"] = row["expected_school"] * edu_adj

        if "mean_school" in df.columns:
            df.at[idx, "mean_school"] = row["mean_school"] * edu_adj

    # Development catch-up multipliers for countries far from the frontier
    # These represent structural advantages of latecomer development
    CATCHUP_MULTIPLIERS = {
        # Sub-Saharan Africa - rapid urbanization, tech leapfrogging
        "ETH": {"gni_mult": 1.15, "le_add": 2.0, "edu_add": 0.8},
        "NGA": {"gni_mult": 1.12, "le_add": 1.5, "edu_add": 0.6},
        "KEN": {"gni_mult": 1.10, "le_add": 1.5, "edu_add": 0.5},
        "TZA": {"gni_mult": 1.10, "le_add": 1.5, "edu_add": 0.5},
        "GHA": {"gni_mult": 1.10, "le_add": 1.5, "edu_add": 0.5},
        "RWA": {"gni_mult": 1.12, "le_add": 2.0, "edu_add": 0.6},
        "SEN": {"gni_mult": 1.08, "le_add": 1.5, "edu_add": 0.5},
        "CIV": {"gni_mult": 1.08, "le_add": 1.5, "edu_add": 0.5},
        "CMR": {"gni_mult": 1.08, "le_add": 1.5, "edu_add": 0.5},
        "AGO": {"gni_mult": 1.10, "le_add": 1.5, "edu_add": 0.5},
        "MOZ": {"gni_mult": 1.10, "le_add": 2.0, "edu_add": 0.6},
        "ZMB": {"gni_mult": 1.08, "le_add": 1.5, "edu_add": 0.5},
        "ZWE": {"gni_mult": 1.08, "le_add": 1.5, "edu_add": 0.5},
        "BWA": {"gni_mult": 1.06, "le_add": 1.0, "edu_add": 0.4},
        "NAM": {"gni_mult": 1.06, "le_add": 1.0, "edu_add": 0.4},
        # South Asia
        "IND": {"gni_mult": 1.15, "le_add": 2.5, "edu_add": 1.0},
        "BGD": {"gni_mult": 1.12, "le_add": 2.0, "edu_add": 0.8},
        "PAK": {"gni_mult": 1.10, "le_add": 1.5, "edu_add": 0.6},
        "NPL": {"gni_mult": 1.08, "le_add": 1.5, "edu_add": 0.5},
        "LKA": {"gni_mult": 1.06, "le_add": 1.0, "edu_add": 0.4},
        # Southeast Asia
        "IDN": {"gni_mult": 1.10, "le_add": 1.5, "edu_add": 0.6},
        "VNM": {"gni_mult": 1.12, "le_add": 1.5, "edu_add": 0.6},
        "PHL": {"gni_mult": 1.10, "le_add": 1.5, "edu_add": 0.6},
        "KHM": {"gni_mult": 1.10, "le_add": 1.5, "edu_add": 0.5},
        "LAO": {"gni_mult": 1.08, "le_add": 1.5, "edu_add": 0.5},
        "MMR": {"gni_mult": 1.08, "le_add": 1.5, "edu_add": 0.5},
        # Central Asia
        "KAZ": {"gni_mult": 1.08, "le_add": 1.0, "edu_add": 0.5},
        "UZB": {"gni_mult": 1.10, "le_add": 1.5, "edu_add": 0.6},
        "KGZ": {"gni_mult": 1.08, "le_add": 1.5, "edu_add": 0.5},
        "TJK": {"gni_mult": 1.08, "le_add": 1.5, "edu_add": 0.5},
        # Middle East & North Africa
        "EGY": {"gni_mult": 1.08, "le_add": 1.0, "edu_add": 0.5},
        "MAR": {"gni_mult": 1.08, "le_add": 1.0, "edu_add": 0.5},
        "TUN": {"gni_mult": 1.06, "le_add": 1.0, "edu_add": 0.4},
        "DZA": {"gni_mult": 1.06, "le_add": 1.0, "edu_add": 0.4},
        "JOR": {"gni_mult": 1.06, "le_add": 1.0, "edu_add": 0.4},
        "IRQ": {"gni_mult": 1.10, "le_add": 2.0, "edu_add": 0.6},
        # Latin America
        "BRA": {"gni_mult": 1.06, "le_add": 1.0, "edu_add": 0.4},
        "MEX": {"gni_mult": 1.06, "le_add": 1.0, "edu_add": 0.4},
        "COL": {"gni_mult": 1.06, "le_add": 1.0, "edu_add": 0.4},
        "PER": {"gni_mult": 1.06, "le_add": 1.0, "edu_add": 0.4},
        "BOL": {"gni_mult": 1.06, "le_add": 1.0, "edu_add": 0.4},
        "ECU": {"gni_mult": 1.06, "le_add": 1.0, "edu_add": 0.4},
        "DOM": {"gni_mult": 1.06, "le_add": 1.0, "edu_add": 0.4},
        "GTM": {"gni_mult": 1.06, "le_add": 1.0, "edu_add": 0.4},
        "HND": {"gni_mult": 1.06, "le_add": 1.0, "edu_add": 0.4},
        "NIC": {"gni_mult": 1.06, "le_add": 1.0, "edu_add": 0.4},
        "SLV": {"gni_mult": 1.06, "le_add": 1.0, "edu_add": 0.4},
        "PRY": {"gni_mult": 1.06, "le_add": 1.0, "edu_add": 0.4},
        "CRI": {"gni_mult": 1.04, "le_add": 0.5, "edu_add": 0.3},
        "PAN": {"gni_mult": 1.04, "le_add": 0.5, "edu_add": 0.3},
        # Europe catch-up
        "SRB": {"gni_mult": 1.06, "le_add": 0.8, "edu_add": 0.4},
        "BIH": {"gni_mult": 1.08, "le_add": 1.0, "edu_add": 0.5},
        "MKD": {"gni_mult": 1.08, "le_add": 1.0, "edu_add": 0.5},
        "ALB": {"gni_mult": 1.06, "le_add": 0.8, "edu_add": 0.4},
        "MNE": {"gni_mult": 1.06, "le_add": 0.8, "edu_add": 0.4},
        "MDA": {"gni_mult": 1.08, "le_add": 1.0, "edu_add": 0.5},
        "GEO": {"gni_mult": 1.08, "le_add": 1.0, "edu_add": 0.5},
        "ARM": {"gni_mult": 1.06, "le_add": 0.8, "edu_add": 0.4},
        "AZE": {"gni_mult": 1.06, "le_add": 0.8, "edu_add": 0.4},
        "UKR": {"gni_mult": 1.10, "le_add": 1.5, "edu_add": 0.6},
        # China - already fast
        "CHN": {"gni_mult": 1.08, "le_add": 1.0, "edu_add": 0.5},
        # Turkey
        "TUR": {"gni_mult": 1.06, "le_add": 0.8, "edu_add": 0.4},
        # Russia
        "RUS": {"gni_mult": 1.04, "le_add": 0.8, "edu_add": 0.4},
    }

    # Apply catch-up multipliers
    for idx, row in df.iterrows():
        cid = row["country_id"]
        if cid in CATCHUP_MULTIPLIERS:
            catchup = CATCHUP_MULTIPLIERS[cid]
            if "gni_ppp" in df.columns:
                df.at[idx, "gni_ppp"] = row["gni_ppp"] * catchup["gni_mult"]
            if "life_exp" in df.columns:
                df.at[idx, "life_exp"] = min(88.0, row["life_exp"] + catchup["le_add"])
            if "mean_school" in df.columns:
                df.at[idx, "mean_school"] = min(15.0, row["mean_school"] + catchup["edu_add"])
            if "expected_school" in df.columns:
                df.at[idx, "expected_school"] = min(18.0, row["expected_school"] + catchup["edu_add"] * 0.6)

    # Country-specific education boosts for rapidly developing economies
    EDUCATION_BOOSTS = {
        "CHN": {"mean_school_add": 1.5, "exp_school_add": 0.8, "life_exp_add": 1.5},
        "IND": {"mean_school_add": 1.2, "exp_school_add": 0.6, "life_exp_add": 1.0},
        "BGD": {"mean_school_add": 0.8, "exp_school_add": 0.4, "life_exp_add": 0.5},
        "VNM": {"mean_school_add": 0.8, "exp_school_add": 0.5, "life_exp_add": 0.8},
        "IDN": {"mean_school_add": 0.6, "exp_school_add": 0.4, "life_exp_add": 0.5},
        "PHL": {"mean_school_add": 0.5, "exp_school_add": 0.3, "life_exp_add": 0.5},
        "THA": {"mean_school_add": 0.5, "exp_school_add": 0.3, "life_exp_add": 0.5},
        "MYS": {"mean_school_add": 0.4, "exp_school_add": 0.3, "life_exp_add": 0.5},
        "TUR": {"mean_school_add": 0.5, "exp_school_add": 0.3, "life_exp_add": 0.5},
        "POL": {"mean_school_add": 0.4, "exp_school_add": 0.2, "life_exp_add": 0.5},
        "KOR": {"mean_school_add": 0.3, "exp_school_add": 0.2, "life_exp_add": 0.3},
        "ISR": {"mean_school_add": 0.5, "exp_school_add": 0.3, "life_exp_add": 0.5},
        "KAZ": {"mean_school_add": 0.8, "exp_school_add": 0.5, "life_exp_add": 0.8},
        "ARE": {"mean_school_add": 0.6, "exp_school_add": 0.4, "life_exp_add": 0.8},
        "SAU": {"mean_school_add": 0.8, "exp_school_add": 0.5, "life_exp_add": 1.0},
        "QAT": {"mean_school_add": 0.5, "exp_school_add": 0.3, "life_exp_add": 0.8},
        "KWT": {"mean_school_add": 0.4, "exp_school_add": 0.3, "life_exp_add": 0.6},
        "BHR": {"mean_school_add": 0.4, "exp_school_add": 0.3, "life_exp_add": 0.6},
        "OMN": {"mean_school_add": 0.5, "exp_school_add": 0.3, "life_exp_add": 0.7},
        "SRB": {"mean_school_add": 0.4, "exp_school_add": 0.2, "life_exp_add": 0.5},
        "HRV": {"mean_school_add": 0.3, "exp_school_add": 0.2, "life_exp_add": 0.4},
        "BIH": {"mean_school_add": 0.4, "exp_school_add": 0.2, "life_exp_add": 0.5},
        "MKD": {"mean_school_add": 0.4, "exp_school_add": 0.2, "life_exp_add": 0.5},
        "MNE": {"mean_school_add": 0.3, "exp_school_add": 0.2, "life_exp_add": 0.4},
        "ALB": {"mean_school_add": 0.4, "exp_school_add": 0.2, "life_exp_add": 0.5},
    }
    for idx, row in df.iterrows():
        cid = row["country_id"]
        if cid in EDUCATION_BOOSTS:
            boost = EDUCATION_BOOSTS[cid]
            if "mean_school" in df.columns:
                df.at[idx, "mean_school"] = min(14.0, row["mean_school"] + boost["mean_school_add"])
            if "expected_school" in df.columns:
                df.at[idx, "expected_school"] = min(18.0, row["expected_school"] + boost["exp_school_add"])
            if "life_exp" in df.columns:
                df.at[idx, "life_exp"] = min(88.0, row["life_exp"] + boost["life_exp_add"])

    return df


# Institutional efficiency multipliers (0.70 - 1.15)
# How efficiently does a country convert economic growth into human development?
# High corruption/poor governance → lower multiplier (GDP growth doesn't reach people)
# Strong institutions → higher multiplier (GDP growth translates to HDI growth)
INSTITUTIONAL_EFFICIENCY = {
    # Very high efficiency (1.10-1.15) - strong institutions, low corruption
    "SGP": 1.15, "DNK": 1.12, "NOR": 1.12, "FIN": 1.12, "ISL": 1.12,
    "SWE": 1.12, "CHE": 1.12, "NLD": 1.11, "DEU": 1.11, "NZL": 1.11,
    "AUS": 1.10, "CAN": 1.10, "GBR": 1.10, "IRL": 1.10, "AUT": 1.10,
    "LUX": 1.10, "BEL": 1.10, "FRA": 1.10, "JPN": 1.10, "KOR": 1.10,

    # High efficiency (1.05-1.09)
    "USA": 1.08, "ESP": 1.07, "PRT": 1.07, "CZE": 1.07, "EST": 1.07,
    "LTU": 1.07, "LVA": 1.06, "SVN": 1.07, "MLT": 1.07, "CYP": 1.06,
    "ISR": 1.08, "CHL": 1.06, "URY": 1.06, "POL": 1.06, "HRV": 1.06,
    "SVK": 1.06, "HUN": 1.05, "BGR": 1.05, "ROU": 1.05, "ITA": 1.05,
    "GRC": 1.04, "QAT": 1.05, "ARE": 1.08, "SAU": 1.02, "KWT": 1.02,
    "BHR": 1.03, "OMN": 1.02,

    # Medium efficiency (0.95-1.04)
    "MYS": 1.02, "THA": 1.01, "COL": 0.98, "PER": 0.97, "BRA": 0.98,
    "MEX": 0.97, "CRI": 1.01, "PAN": 1.00, "CHN": 1.05, "RUS": 0.92,
    "TUR": 0.95, "JOR": 1.00, "MAR": 0.96, "TUN": 0.97, "EGY": 0.93,
    "KAZ": 1.00, "GEO": 0.98, "ARM": 0.97, "SRB": 0.97, "MNE": 0.96,
    "MKD": 0.96, "ALB": 0.95, "BIH": 0.93, "NAM": 1.00, "BWA": 1.01,
    "MUS": 1.02, "FJI": 0.98, "DOM": 0.96, "JAM": 0.97, "TTO": 0.97,
    "PRY": 0.95, "ECU": 0.94, "BLZ": 0.96, "GTM": 0.93, "SLV": 0.94,
    "HND": 0.91, "NIC": 0.92, "BOL": 0.92, "IDN": 0.98, "PHL": 0.96,
    "VNM": 0.97, "LKA": 0.93, "BGD": 0.92, "IND": 0.93, "UKR": 0.88,
    "MDA": 0.92, "KGZ": 0.93, "UZB": 0.90, "TJK": 0.88, "AZE": 0.90,
    "IRN": 0.85, "LBN": 0.85, "DZA": 0.88, "BGR": 1.05,

    # Low efficiency (0.80-0.94) - weak institutions, corruption drag
    "PAK": 0.85, "NPL": 0.88, "BGD": 0.90, "KEN": 0.90, "GHA": 0.92,
    "SEN": 0.91, "TZA": 0.89, "RWA": 0.95, "UGA": 0.88, "ETH": 0.87,
    "NGA": 0.80, "CMR": 0.82, "CIV": 0.83, "AGO": 0.78, "MOZ": 0.80,
    "ZMB": 0.85, "ZWE": 0.78, "MWI": 0.82, "MDG": 0.80, "BFA": 0.78,
    "MLI": 0.78, "NER": 0.75, "TCD": 0.75, "GIN": 0.80, "SLE": 0.78,
    "LBR": 0.78, "BEN": 0.85, "TGO": 0.85, "GMB": 0.83, "GNB": 0.82,
    "MRT": 0.85, "COM": 0.85, "DJI": 0.85, "SWZ": 0.88, "LSO": 0.85,
    "NAM": 0.95, "BWA": 1.00, "GAB": 0.85, "COG": 0.78, "GNQ": 0.75,
    "STP": 0.88, "CPV": 0.95, "SYC": 0.95, "MUS": 1.00,

    # Very low efficiency (0.70-0.79) - fragile/conflict states
    "IRQ": 0.78, "SYR": 0.70, "YEM": 0.72, "AFG": 0.72, "SSD": 0.70,
    "SOM": 0.70, "LBY": 0.75, "SDN": 0.72, "ERI": 0.72, "COD": 0.72,
    "CAF": 0.72, "SSD": 0.70, "HTI": 0.75, "VEN": 0.72, "CUB": 0.80,
    "PRK": 0.70, "BLR": 0.82, "MMR": 0.75, "TLS": 0.85,
    "PSE": 0.78,

    # Microstates & territories
    "AND": 1.05, "LIE": 1.10, "MCO": 1.08, "SMR": 1.05, "IMN": 1.08,
    "GIB": 1.05, "FRO": 1.08, "BMU": 1.05, "CYM": 1.05, "VGB": 1.02,
    "CUW": 1.02, "ABW": 1.02, "TCA": 1.02, "AIA": 1.00, "MSR": 1.00,
    "HKG": 1.08, "MAC": 1.02, "GUM": 1.02, "ASM": 1.00, "MNP": 1.00,
    "VIR": 1.00, "NCL": 1.02, "PYF": 1.00, "GLP": 1.02, "MTQ": 1.02,
    "WLF": 1.00, "BLM": 1.00, "MAF": 1.00, "SPM": 1.00, "KNA": 1.00,
    "LCA": 1.00, "VCT": 1.00, "DMA": 1.00, "GRD": 1.00, "ATG": 1.00,
    "BRB": 1.00, "BHS": 1.00,

    # Oceania
    "PLW": 0.95, "FJI": 0.98, "TON": 0.95, "WSM": 0.95, "KIR": 0.90,
    "MHL": 0.90, "FSM": 0.90, "NRU": 0.92, "TUV": 0.90, "SLB": 0.88,
    "VUT": 0.88, "PNG": 0.82,

    # Additional European
    "LUX": 1.10,

    # China special case
    "CHN": 1.05,

    # India special case
    "IND": 0.93,
}


def get_institutional_efficiency(country_id: str) -> float:
    """Return institutional efficiency multiplier (0.70-1.15).

    This multiplier determines how efficiently a country converts
    economic growth (GDP) into human development (HDI).

    High multiplier (1.10-1.15): Strong rule of law, low corruption,
        effective public services → GDP growth reaches people

    Low multiplier (0.70-0.80): Weak institutions, corruption,
        elite capture → GDP growth doesn't improve lives
    """
    return INSTITUTIONAL_EFFICIENCY.get(country_id, 0.90)


def get_governance_multiplier(country_id: str, year: int = 2050) -> float:
    """Time-varying governance multiplier that improves over 24 years.

    For conflict/improving countries, governance improves toward 2050.
    For stable countries, stays roughly constant.
    """
    base = get_institutional_efficiency(country_id)

    IMPROVING = {
        "IRQ": 0.08, "SYR": 0.12, "YEM": 0.10, "AFG": 0.08,
        "SSD": 0.10, "SOM": 0.08, "LBY": 0.10, "UKR": 0.12,
        "SDN": 0.08, "ETH": 0.08, "MMR": 0.08, "NGA": 0.06,
        "COD": 0.08, "VEN": 0.08, "CUB": 0.05, "PRK": 0.03,
        "HTI": 0.08, "LBN": 0.10, "ZWE": 0.08, "ERI": 0.05,
        "BGD": 0.04, "PAK": 0.06, "IND": 0.04,
    }

    if year <= 2024:
        return base
    if year >= 2050:
        improvement = IMPROVING.get(country_id, 0.02)
        return min(base + improvement, 1.15)

    t = (year - 2024) / (2050 - 2024)
    improvement = IMPROVING.get(country_id, 0.02)
    return min(base + improvement * t, 1.15)
