import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

from config import config


def _clean(X, y=None):
    X_c = X.fillna(0).replace([np.inf, -np.inf], 0)
    if y is not None:
        y_c = y.fillna(y.median()).replace([np.inf, -np.inf], y.median())
        return X_c, y_c
    return X_c


class HDIComponentEnsemble:
    def __init__(self, component_name: str, lightweight: bool = False):
        self.component_name = component_name
        self.lightweight = lightweight
        self.models = {}
        self.weights = {}
        self.quantile_models = {}
        self.feature_names = None
        self.cv_score = None

    def _build_models(self):
        if self.lightweight:
            return {
                "lgb": lgb.LGBMRegressor(
                    n_estimators=80, max_depth=5, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.7,
                    random_state=config.RANDOM_STATE, n_jobs=-1, verbose=-1,
                ),
                "ridge": Ridge(alpha=10.0),
            }
        return {
            "rf": RandomForestRegressor(
                n_estimators=150, max_depth=8, min_samples_leaf=15,
                max_features=0.5, random_state=config.RANDOM_STATE, n_jobs=-1,
            ),
            "xgb": xgb.XGBRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1,
                reg_lambda=1.0, min_child_weight=5, random_state=config.RANDOM_STATE,
                n_jobs=-1, verbosity=0,
            ),
            "lgb": lgb.LGBMRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1,
                reg_lambda=1.0, min_child_samples=10, random_state=config.RANDOM_STATE,
                n_jobs=-1, verbose=-1,
            ),
            "cat": CatBoostRegressor(
                iterations=200, depth=6, learning_rate=0.05,
                l2_leaf_reg=3.0, random_seed=config.RANDOM_STATE, verbose=0,
            ),
        }

    def _optimize_weights(self, X, y):
        """Optimize weights using held-out validation (no model retraining)."""
        n_models = len(self.models)
        if self.lightweight or n_models <= 2:
            self.weights = {name: 1.0 / n_models for name in self.models}
            return

        split = int(len(X) * 0.8)
        X_tr, X_val = X.iloc[:split], X.iloc[split:]
        y_val = y.iloc[split:]

        preds = np.column_stack([m.predict(X_val) for m in self.models.values()])
        y_v = y_val.values

        model_names = list(self.models.keys())
        best_weights = None
        best_score = np.inf
        for _ in range(500):
            w = np.random.dirichlet(np.ones(n_models))
            pred = preds @ w
            score = mean_squared_error(y_v, pred)
            if score < best_score:
                best_score = score
                best_weights = w

        self.weights = {name: float(w) for name, w in zip(model_names, best_weights)}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HDIComponentEnsemble":
        self.feature_names = X.columns.tolist()
        X_c, y_c = _clean(X, y)

        self.models = self._build_models()
        for name, model in self.models.items():
            model.fit(X_c, y_c)

        self._optimize_weights(X_c, y_c)

        if not self.lightweight:
            for q in [0.05, 0.95]:
                qmodel = GradientBoostingRegressor(
                    loss="quantile", alpha=q, n_estimators=150, max_depth=5,
                    learning_rate=0.05, min_samples_leaf=10, random_state=config.RANDOM_STATE,
                )
                qmodel.fit(X_c, y_c)
                self.quantile_models[q] = qmodel

        self.cv_score = self._quick_score(X_c, y_c)
        return self

    def _quick_score(self, X, y):
        split = int(len(X) * 0.8)
        X_tr, X_val = X.iloc[:split], X.iloc[split:]
        y_tr, y_val = y.iloc[:split], y.iloc[split:]
        temp = self._build_models()
        for name, model in temp.items():
            model.fit(X_tr, y_tr)
        preds = np.column_stack([m.predict(X_val) for m in temp.values()])
        total_w = sum(self.weights.values()) or 1.0
        w = np.array([self.weights[n] / total_w for n in temp.keys()])
        pred = preds @ w
        return {"mae_mean": float(mean_absolute_error(y_val, pred)), "mae_std": 0.0}

    def predict(self, X: pd.DataFrame) -> dict:
        X_c = _clean(X)
        preds = {}
        total_w = sum(self.weights.values()) or 1.0
        weighted = np.zeros(len(X_c))

        for name, model in self.models.items():
            p = model.predict(X_c)
            preds[name] = p
            weighted += self.weights.get(name, 0) / total_w * p

        preds["mean"] = weighted
        if self.quantile_models:
            preds["lower"] = self.quantile_models[0.05].predict(X_c)
            preds["upper"] = self.quantile_models[0.95].predict(X_c)
        else:
            preds["lower"] = weighted * 0.97
            preds["upper"] = weighted * 1.03
        return preds

    def get_feature_importances(self) -> pd.Series:
        imp = np.zeros(len(self.feature_names))
        total_w = sum(self.weights.values()) or 1.0
        for name, model in self.models.items():
            if hasattr(model, "feature_importances_"):
                imp += (self.weights.get(name, 0) / total_w) * model.feature_importances_
            elif hasattr(model, "coef_"):
                imp += (self.weights.get(name, 0) / total_w) * np.abs(model.coef_)
        return pd.Series(imp, index=self.feature_names).sort_values(ascending=False)


class HDIPredictionPipeline:
    def __init__(self, lightweight: bool = False):
        self.lightweight = lightweight
        self.components = {}
        self.component_map = {
            "life_expectancy": "life_exp",
            "expected_years_schooling": "expected_school",
            "mean_years_schooling": "mean_school",
            "gni_per_capita_ppp": "gni_ppp",
        }

    def fit(self, X: pd.DataFrame, y_dict: dict) -> "HDIPredictionPipeline":
        for comp_name, target_col in self.component_map.items():
            self.components[comp_name] = HDIComponentEnsemble(comp_name, self.lightweight)
            self.components[comp_name].fit(X, y_dict[target_col])
            score = self.components[comp_name].cv_score
            print(f"    {comp_name}: CV MAE = {score['mae_mean']:.6f}")
        return self

    def predict(self, X: pd.DataFrame, with_intervals: bool = False) -> dict:
        result = {}
        for cn, m in self.components.items():
            preds = m.predict(X)
            result[cn] = preds["mean"]
            if with_intervals:
                result[f"{cn}_lower"] = preds["lower"]
                result[f"{cn}_upper"] = preds["upper"]
        return result

    def get_feature_importances(self) -> dict:
        return {cn: m.get_feature_importances() for cn, m in self.components.items()}
