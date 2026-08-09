import numpy as np
import pandas as pd
import shap


class ModelExplainer:
    def __init__(self, pipeline, feature_names: list):
        self.pipeline = pipeline
        self.feature_names = feature_names
        self.explainers = {}
        self.shap_values = {}

    def compute_shap_values(self, X: pd.DataFrame, sample_size: int = 300):
        X_clean = X.fillna(0).replace([np.inf, -np.inf], 0)
        if len(X_clean) > sample_size:
            X_sample = X_clean.sample(n=sample_size, random_state=42)
        else:
            X_sample = X_clean

        for comp_name, ensemble in self.pipeline.components.items():
            best_tree_model = None
            best_name = None
            for name, model in ensemble.models.items():
                if hasattr(model, "feature_importances_"):
                    best_tree_model = model
                    best_name = name
                    break

            if best_tree_model is None:
                continue

            try:
                explainer = shap.TreeExplainer(best_tree_model)
                self.explainers[comp_name] = explainer
                self.shap_values[comp_name] = explainer.shap_values(X_sample)
            except Exception:
                pass

        return self.shap_values

    def get_top_features(self, comp_name: str, top_n: int = 10) -> pd.DataFrame:
        if comp_name not in self.shap_values:
            return pd.DataFrame()
        sv = self.shap_values[comp_name]
        if isinstance(sv, list):
            sv = sv[0]
        importance = np.abs(sv).mean(axis=0)
        return pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False).head(top_n)

    def explain_country(self, X: pd.DataFrame, country_idx: int, comp_name: str) -> pd.DataFrame:
        if comp_name not in self.explainers:
            return pd.DataFrame()
        X_clean = X.fillna(0).replace([np.inf, -np.inf], 0)
        sv = self.explainers[comp_name].shap_values(X_clean.iloc[[country_idx]])
        if isinstance(sv, list):
            sv = sv[0]
        sv = sv[0]
        return pd.DataFrame({
            "feature": self.feature_names,
            "shap_value": sv,
            "feature_value": X_clean.iloc[country_idx].values,
        }).sort_values("shap_value", key=abs, ascending=False)

    def get_global_summary(self) -> dict:
        return {cn: self.get_top_features(cn, top_n=15) for cn in self.shap_values}
