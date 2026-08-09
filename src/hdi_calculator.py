import numpy as np
import pandas as pd


class HDICalculator:
    """Official UNDP Human Development Index calculation.

    HDI = (I_health * I_education * I_income)^(1/3)

    where each dimension index is a normalized geometric mean of its indicators.
    """

    GOALS = {
        "life_expectancy": {"min": 20, "max": 88.0},
        "expected_years_schooling": {"min": 0, "max": 18},
        "mean_years_schooling": {"min": 0, "max": 16},
        "gni_per_capita_ppp": {"min": 100, "max": 105000},
    }

    @staticmethod
    def _dimension_index(value: float, dim_min: float, dim_max: float) -> float:
        if dim_max == dim_min:
            return 0.0
        return max(0.0, min(1.0, (value - dim_min) / (dim_max - dim_min)))

    def health_index(self, life_expectancy: pd.Series) -> pd.Series:
        g = self.GOALS["life_expectancy"]
        return self._dimension_index_series(life_expectancy, g["min"], g["max"])

    def education_index(
        self,
        expected_years: pd.Series,
        mean_years: pd.Series,
    ) -> pd.Series:
        e_min = self.GOALS["expected_years_schooling"]["min"]
        e_max = self.GOALS["expected_years_schooling"]["max"]
        m_min = self.GOALS["mean_years_schooling"]["min"]
        m_max = self.GOALS["mean_years_schooling"]["max"]

        ei = self._dimension_index_series(expected_years, e_min, e_max)
        mi = self._dimension_index_series(mean_years, m_min, m_max)
        return np.sqrt(ei * mi)

    def income_index(self, gni_ppp: pd.Series) -> pd.Series:
        g = self.GOALS["gni_per_capita_ppp"]
        ln_val = np.log(np.maximum(gni_ppp, 1.0))
        ln_min = np.log(max(g["min"], 1.0))
        ln_max = np.log(max(g["max"], 2.0))
        return self._dimension_index_series(ln_val, ln_min, ln_max)

    def compute_hdi(
        self,
        life_expectancy: pd.Series,
        expected_years: pd.Series,
        mean_years: pd.Series,
        gni_ppp: pd.Series,
    ) -> pd.Series:
        h = self.health_index(life_expectancy)
        e = self.education_index(expected_years, mean_years)
        i = self.income_index(gni_ppp)
        return (h * e * i) ** (1.0 / 3.0)

    def compute_all_components(
        self,
        life_expectancy: pd.Series,
        expected_years: pd.Series,
        mean_years: pd.Series,
        gni_ppp: pd.Series,
    ) -> pd.DataFrame:
        return pd.DataFrame({
            "health_index": self.health_index(life_expectancy),
            "education_index": self.education_index(expected_years, mean_years),
            "income_index": self.income_index(gni_ppp),
            "hdi": self.compute_hdi(life_expectancy, expected_years, mean_years, gni_ppp),
        })

    @staticmethod
    def _dimension_index_series(value: pd.Series, dim_min: float, dim_max: float) -> pd.Series:
        if dim_max == dim_min:
            return pd.Series(0.0, index=value.index)
        idx = (value - dim_min) / (dim_max - dim_min)
        return idx.clip(0.0, 1.0)


hdi_calc = HDICalculator()
