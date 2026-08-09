import numpy as np
import pandas as pd
from config import config
from data.future import FUTURE_VARS


SCENARIO_MODIFIERS = {
    "baseline": {
        "description": "IMF/UN/World Bank trend continuation",
        "gni_growth_modifier": 1.0,
        "education_acceleration": 1.0,
        "health_improvement": 1.0,
        "governance_change": 0.0,
        "internet_adoption_modifier": 1.0,
        "fertility_modifier": 1.0,
        "urbanization_modifier": 1.0,
        "renewable_acceleration": 1.0,
        "future_tech_acceleration": 1.0,
        "shock_probability": 0.02,
    },
    "high_growth": {
        "description": "AI adoption, manufacturing expansion, education reforms",
        "gni_growth_modifier": 1.5,
        "education_acceleration": 1.8,
        "health_improvement": 1.3,
        "governance_change": 0.15,
        "internet_adoption_modifier": 1.4,
        "fertility_modifier": 0.85,
        "urbanization_modifier": 1.2,
        "renewable_acceleration": 1.5,
        "future_tech_acceleration": 1.45,
        "shock_probability": 0.01,
    },
    "low_growth": {
        "description": "Recessions, conflict, climate impacts",
        "gni_growth_modifier": 0.5,
        "education_acceleration": 0.6,
        "health_improvement": 0.7,
        "governance_change": -0.10,
        "internet_adoption_modifier": 0.7,
        "fertility_modifier": 1.1,
        "urbanization_modifier": 0.8,
        "renewable_acceleration": 0.7,
        "future_tech_acceleration": 0.75,
        "shock_probability": 0.08,
    },
    "green_transition": {
        "description": "Renewable investment, improved health, higher education spending",
        "gni_growth_modifier": 1.1,
        "education_acceleration": 1.5,
        "health_improvement": 1.6,
        "governance_change": 0.10,
        "internet_adoption_modifier": 1.2,
        "fertility_modifier": 0.90,
        "urbanization_modifier": 1.05,
        "renewable_acceleration": 2.5,
        "future_tech_acceleration": 1.25,
        "shock_probability": 0.03,
    },
}


class ScenarioEngine:
    def __init__(self):
        self.scenarios = SCENARIO_MODIFIERS

    def apply_scenario(
        self,
        base_forecasts: pd.DataFrame,
        scenario_name: str,
    ) -> pd.DataFrame:
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        mod = self.scenarios[scenario_name]
        df = base_forecasts.copy()
        df["scenario"] = scenario_name

        years_ahead = df["year"] - config.HIST_END
        ramp = np.clip(years_ahead / 10, 0, 1)

        if "gni_ppp" in df.columns:
            gni_mod = 1.0 + (mod["gni_growth_modifier"] - 1.0) * ramp
            df["gni_ppp"] = df["gni_ppp"] * gni_mod

        if "expected_school" in df.columns:
            edu_mod = 1.0 + (mod["education_acceleration"] - 1.0) * ramp
            df["expected_school"] = df["expected_school"] * edu_mod
        if "mean_school" in df.columns:
            edu_mod = 1.0 + (mod["education_acceleration"] - 1.0) * ramp * 0.5
            df["mean_school"] = df["mean_school"] * edu_mod

        if "life_exp" in df.columns:
            health_mod = 1.0 + (mod["health_improvement"] - 1.0) * ramp * 0.1
            df["life_exp"] = df["life_exp"] * health_mod

        if "gov_effectiveness" in df.columns:
            df["gov_effectiveness"] = df["gov_effectiveness"] + mod["governance_change"] * ramp
        if "corruption" in df.columns:
            df["corruption"] = df["corruption"] + mod["governance_change"] * ramp * 0.5

        if "internet" in df.columns:
            net_mod = 1.0 + (mod["internet_adoption_modifier"] - 1.0) * ramp
            df["internet"] = np.clip(df["internet"] * net_mod, 0, 1)

        if "fertility" in df.columns:
            df["fertility"] = df["fertility"] * (1.0 + (mod["fertility_modifier"] - 1.0) * ramp)
        if "urbanization" in df.columns:
            urb_mod = 1.0 + (mod["urbanization_modifier"] - 1.0) * ramp * 0.1
            df["urbanization"] = np.clip(df["urbanization"] * urb_mod, 0, 1)

        if "renewable_share" in df.columns:
            ren_mod = 1.0 + (mod["renewable_acceleration"] - 1.0) * ramp
            df["renewable_share"] = np.clip(df["renewable_share"] * ren_mod, 0, 1)

        future_mod = 1.0 + (mod["future_tech_acceleration"] - 1.0) * ramp
        for col in FUTURE_VARS:
            if col in df.columns:
                df[col] = np.clip(df[col] * future_mod, 0, 1)

        shock_prob = mod["shock_probability"]
        shock_mask = np.random.random(len(df)) < shock_prob
        if shock_mask.any():
            df.loc[shock_mask, "gni_ppp"] *= np.random.uniform(0.85, 1.0, shock_mask.sum())

        return df

    def generate_all_scenarios(
        self,
        base_forecasts: pd.DataFrame,
    ) -> dict:
        return {
            name: self.apply_scenario(base_forecasts, name)
            for name in self.scenarios
        }
