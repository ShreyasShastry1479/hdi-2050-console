"""Build compact, browser-readable ML research artifacts for the HDI console.

The exported randomized-tree ensemble is a surrogate of the existing 2050 scenario table. It
supports local counterfactual sensitivity analysis; it is not presented as an
independently validated forecast of 2050 outcomes.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
import shap
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
NATIONAL_CSV = ROOT / "data" / "output" / "hdi_2050_rankings.csv"
SUBDIVISION_CSV = ROOT / "data" / "output" / "subdivision_hdi_2025_2050.csv"
OUTPUT_JSON = ROOT / "data" / "output" / "ml_research_artifacts.json"

FEATURES = [
    "HDI_Baseline",
    "HealthIndex_2025",
    "EducationIndex_2025",
    "IncomeIndex_2025",
    "ai_adoption_index",
    "DigitalInfraDevelopment",
    "green_energy_investment",
    "HealthExp_2024",
    "RenewableShare_2024",
    "InstEfficiency",
    "DemographicDividend",
    "HumanCapitalAbsorption",
    "ResourceDrag",
    "GrowthVolatility",
    "FutureReadiness",
    "IndustrializationHDIAcceleration",
    "ClimateRisk_2024",
    "Gini_2024",
    "GrowthProspectScore",
    "DevelopmentMomentumScore",
    "RecoveryPotential2050",
    "LowGrowthProspectDrag",
    "TrajectoryHDISpeedMultiplier",
    "EconomicDiversification",
    "DevelopingCatchupReadiness",
    "WorkforceMomentum",
    "DependencyPressure",
]

FEATURE_LABELS = {
    "HDI_Baseline": "2025 HDI anchor",
    "HealthIndex_2025": "Health capacity",
    "EducationIndex_2025": "Education capacity",
    "IncomeIndex_2025": "Income capacity",
    "ai_adoption_index": "AI adoption readiness",
    "DigitalInfraDevelopment": "Digital infrastructure",
    "green_energy_investment": "Green investment readiness",
    "HealthExp_2024": "Health expenditure",
    "RenewableShare_2024": "Renewable-energy share",
    "InstEfficiency": "Institutional efficiency",
    "DemographicDividend": "Demographic dividend",
    "HumanCapitalAbsorption": "Human-capital absorption",
    "ResourceDrag": "Resource dependence drag",
    "GrowthVolatility": "Growth volatility",
    "FutureReadiness": "Future readiness",
    "IndustrializationHDIAcceleration": "Industrialization acceleration",
    "ClimateRisk_2024": "Climate vulnerability",
    "Gini_2024": "Income inequality",
    "GrowthProspectScore": "Growth prospects",
    "DevelopmentMomentumScore": "Development momentum",
    "RecoveryPotential2050": "Recovery potential",
    "LowGrowthProspectDrag": "Low-growth prospect drag",
    "TrajectoryHDISpeedMultiplier": "Trajectory speed",
    "EconomicDiversification": "Economic diversification",
    "DevelopingCatchupReadiness": "Catch-up readiness",
    "WorkforceMomentum": "Workforce momentum",
    "DependencyPressure": "Dependency pressure",
}


def finite(value: object, digits: int = 8) -> float | None:
    number = float(value)
    return round(number, digits) if np.isfinite(number) else None


def export_tree(estimator: object) -> dict[str, list]:
    tree = estimator.tree_
    return {
        "left": tree.children_left.astype(int).tolist(),
        "right": tree.children_right.astype(int).tolist(),
        "feature": tree.feature.astype(int).tolist(),
        "threshold": np.round(tree.threshold, 8).tolist(),
        "value": np.round(tree.value[:, 0, 0], 8).tolist(),
    }


def cluster_label(frame: pd.DataFrame) -> str:
    hdi = frame["Subdivision_HDI_2050_Projected"].median()
    gain = frame["Subdivision_HDI_Change_2025_to_2050"].median()
    growth = frame["Subdivision_Population_Growth_2024_to_2050"].median()
    if hdi >= 0.90 and gain <= 0.035:
        return "High-HDI mature regions"
    if hdi >= 0.82 and gain > 0.055:
        return "Advanced convergence regions"
    if gain >= 0.10 and growth >= 0.20:
        return "Rapid catch-up growth hubs"
    if gain >= 0.08:
        return "Fast human-development convergers"
    if growth >= 0.35:
        return "High-population-growth regions"
    if hdi < 0.60:
        return "Low-HDI constraint regions"
    return "Moderate transition regions"


def main() -> None:
    national = pd.read_csv(NATIONAL_CSV)
    subdivisions = pd.read_csv(SUBDIVISION_CSV)

    medians = national[FEATURES].median(numeric_only=True)
    x = national[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(medians)
    y = pd.to_numeric(national["HDI_2050_Gain"], errors="coerce").fillna(0.0)

    model_args = dict(
        n_estimators=72,
        max_depth=9,
        min_samples_leaf=1,
        max_features=0.90,
        random_state=42,
        n_jobs=1,
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = cross_val_predict(ExtraTreesRegressor(**model_args), x, y, cv=cv, n_jobs=1)
    model = ExtraTreesRegressor(**model_args).fit(x, y)
    fitted = model.predict(x)

    explainer = shap.TreeExplainer(model)
    shap_values = np.asarray(explainer.shap_values(x), dtype=float)
    expected_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])

    anomaly_fields = FEATURES + ["HDI_2050_Gain"]
    anomaly_frame = national[anomaly_fields].apply(pd.to_numeric, errors="coerce")
    anomaly_frame = anomaly_frame.fillna(anomaly_frame.median(numeric_only=True))
    anomaly_scaled = StandardScaler().fit_transform(anomaly_frame)
    national_detector = IsolationForest(
        n_estimators=240,
        contamination=0.08,
        random_state=42,
        n_jobs=-1,
    ).fit(anomaly_scaled)
    national_raw = -national_detector.score_samples(anomaly_scaled)
    national_pct = pd.Series(national_raw).rank(method="average", pct=True).to_numpy()

    national_records: dict[str, dict] = {}
    for index, row in national.iterrows():
        iso3 = str(row["ISO3"])
        national_records[iso3] = {
            "country": str(row["Country"]),
            "baseline_hdi": finite(row["HDI_Baseline"]),
            "projected_hdi": finite(row["HDI_2050"]),
            "published_gain": finite(row["HDI_2050_Gain"]),
            "surrogate_gain": finite(fitted[index]),
            "features": {feature: finite(x.iloc[index][feature]) for feature in FEATURES},
            "shap": {feature: finite(shap_values[index, position]) for position, feature in enumerate(FEATURES)},
            "anomaly_score": finite(national_raw[index], 6),
            "anomaly_percentile": finite(national_pct[index], 4),
            "anomaly_flag": bool(national_pct[index] >= 0.92),
        }

    joined = subdivisions.merge(
        national[
            [
                "ISO3",
                "DigitalInfraDevelopment",
                "DemographicDividend",
                "HumanCapitalAbsorption",
                "ResourceDrag",
                "DevelopmentMomentumScore",
                "FutureReadiness",
            ]
        ],
        on="ISO3",
        how="left",
        validate="many_to_one",
    )
    joined["Relative_HDI_2025"] = (
        joined["Subdivision_HDI_2025_Reconciled"] - joined["National_HDI_2025_Target"]
    )
    joined["Log_Population_2050"] = np.log1p(joined["Subdivision_Population_2050_Est"].clip(lower=0))

    subdivision_features = [
        "Subdivision_HDI_2025_Reconciled",
        "Subdivision_HDI_2050_Projected",
        "Subdivision_HDI_Change_2025_to_2050",
        "Relative_HDI_2025",
        "Subdivision_Population_Growth_2024_to_2050",
        "Subdivision_Weight_2050",
        "Log_Population_2050",
        "DigitalInfraDevelopment",
        "DemographicDividend",
        "HumanCapitalAbsorption",
        "ResourceDrag",
        "DevelopmentMomentumScore",
        "FutureReadiness",
    ]
    sub_x = joined[subdivision_features].apply(pd.to_numeric, errors="coerce")
    sub_x = sub_x.fillna(sub_x.median(numeric_only=True))
    scaler = StandardScaler()
    sub_scaled = scaler.fit_transform(sub_x)

    candidate_clusters = range(6, 13)
    cluster_trials: list[tuple[float, int, KMeans, np.ndarray]] = []
    for count in candidate_clusters:
        trial = KMeans(n_clusters=count, n_init=20, random_state=42)
        labels = trial.fit_predict(sub_scaled)
        score = silhouette_score(sub_scaled, labels, sample_size=min(1200, len(joined)), random_state=42)
        cluster_trials.append((score, count, trial, labels))
    silhouette, cluster_count, cluster_model, cluster_ids = max(cluster_trials, key=lambda item: item[0])
    joined["ML_Cluster"] = cluster_ids

    sub_detector = IsolationForest(
        n_estimators=280,
        contamination=0.06,
        random_state=42,
        n_jobs=-1,
    ).fit(sub_scaled)
    sub_raw = -sub_detector.score_samples(sub_scaled)
    sub_pct = pd.Series(sub_raw).rank(method="average", pct=True).to_numpy()
    joined["ML_Anomaly_Score"] = sub_raw
    joined["ML_Anomaly_Percentile"] = sub_pct
    joined["ML_Peer_Gain_Z"] = joined.groupby("ML_Cluster")["Subdivision_HDI_Change_2025_to_2050"].transform(
        lambda values: (values - values.mean()) / max(values.std(ddof=0), 1e-9)
    )

    neighbor_model = NearestNeighbors(n_neighbors=min(50, len(joined)), metric="euclidean").fit(sub_scaled)
    neighbor_indices = neighbor_model.kneighbors(sub_scaled, return_distance=False)
    keys = [f"{row.ISO3}::{row.Subdivision}" for row in joined.itertuples()]
    twin_keys: list[list[str]] = []
    for index, candidates in enumerate(neighbor_indices):
        own_iso3 = str(joined.iloc[index]["ISO3"])
        matches = [keys[item] for item in candidates if item != index and str(joined.iloc[item]["ISO3"]) != own_iso3][:5]
        if len(matches) < 5:
            matches.extend([keys[item] for item in candidates if item != index and keys[item] not in matches][: 5 - len(matches)])
        twin_keys.append(matches)

    clusters: list[dict] = []
    for cluster_id in range(cluster_count):
        cluster_frame = joined[joined["ML_Cluster"] == cluster_id]
        clusters.append(
            {
                "id": int(cluster_id),
                "label": cluster_label(cluster_frame),
                "size": int(len(cluster_frame)),
                "median_hdi_2025": finite(cluster_frame["Subdivision_HDI_2025_Reconciled"].median()),
                "median_hdi_2050": finite(cluster_frame["Subdivision_HDI_2050_Projected"].median()),
                "median_gain": finite(cluster_frame["Subdivision_HDI_Change_2025_to_2050"].median()),
                "median_population_growth": finite(cluster_frame["Subdivision_Population_Growth_2024_to_2050"].median()),
            }
        )

    subdivision_records: dict[str, dict] = {}
    for index, row in joined.iterrows():
        key = keys[index]
        subdivision_records[key] = {
            "country": str(row["Country"]),
            "iso3": str(row["ISO3"]),
            "subdivision": str(row["Subdivision"]),
            "continent": str(row["Continent_Source"]),
            "hdi_2025": finite(row["Subdivision_HDI_2025_Reconciled"]),
            "hdi_2050": finite(row["Subdivision_HDI_2050_Projected"]),
            "gain": finite(row["Subdivision_HDI_Change_2025_to_2050"]),
            "population_2050": int(round(float(row["Subdivision_Population_2050_Est"]))),
            "population_growth": finite(row["Subdivision_Population_Growth_2024_to_2050"]),
            "cluster": int(row["ML_Cluster"]),
            "anomaly_score": finite(row["ML_Anomaly_Score"], 6),
            "anomaly_percentile": finite(row["ML_Anomaly_Percentile"], 4),
            "anomaly_flag": bool(row["ML_Anomaly_Percentile"] >= 0.94),
            "peer_gain_z": finite(row["ML_Peer_Gain_Z"], 4),
            "twins": twin_keys[index],
        }

    artifact = {
        "meta": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "national_rows": int(len(national)),
            "subdivision_rows": int(len(joined)),
            "subdivision_countries": int(joined["ISO3"].nunique()),
            "purpose": "Interactive sensitivity, explainability, anomaly detection, and peer-region discovery",
            "scope_note": "The randomized-tree ensemble is a surrogate of the published baseline scenario, not an independently validated 2050 forecasting model.",
        },
        "surrogate": {
            "model_type": "ExtraTreesRegressor",
            "target": "HDI_2050_Gain",
            "feature_order": FEATURES,
            "feature_labels": FEATURE_LABELS,
            "feature_medians": {feature: finite(medians[feature]) for feature in FEATURES},
            "expected_gain": finite(expected_value),
            "trees": [export_tree(tree) for tree in model.estimators_],
            "validation": {
                "method": "Five-fold shuffled cross-validation against the existing 2050 scenario output",
                "interpretation": "Measures surrogate fidelity, not real-world forecast accuracy",
                "r2": finite(r2_score(y, oof), 5),
                "mae": finite(mean_absolute_error(y, oof), 5),
                "rmse": finite(mean_squared_error(y, oof) ** 0.5, 5),
                "training_r2": finite(r2_score(y, fitted), 5),
            },
            "policy_controls": [
                {"id": "ai", "feature": "ai_adoption_index", "label": "AI adoption readiness", "min": 0, "max": 1, "step": 0.01, "unit": "index"},
                {"id": "digital", "feature": "DigitalInfraDevelopment", "label": "Digital infrastructure", "min": 0, "max": 1, "step": 0.01, "unit": "index"},
                {"id": "green", "feature": "green_energy_investment", "label": "Green investment readiness", "min": 0, "max": 1, "step": 0.01, "unit": "index"},
                {"id": "education", "feature": "EducationIndex_2025", "label": "Education budget uplift", "min": 0, "max": 5, "step": 0.1, "unit": "pp GDP", "transform": "add 0.012 index points per percentage point"},
                {"id": "health", "feature": "HealthExp_2024", "label": "Health expenditure target", "min": 2, "max": 15, "step": 0.1, "unit": "% GDP"},
                {"id": "demographic", "feature": "DemographicDividend", "label": "Demographic dividend realization", "min": 0, "max": 1, "step": 0.01, "unit": "index"},
            ],
        },
        "national": national_records,
        "subdivision_model": {
            "algorithm": "KMeans + standardized Euclidean nearest-neighbor search + Isolation Forest",
            "features": subdivision_features,
            "cluster_count": int(cluster_count),
            "silhouette_score": finite(silhouette, 5),
            "clusters": clusters,
        },
        "subdivisions": subdivision_records,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(artifact, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {OUTPUT_JSON} ({OUTPUT_JSON.stat().st_size / 1024 / 1024:.2f} MiB)")
    print(json.dumps(artifact["surrogate"]["validation"], indent=2))
    print(
        f"National anomalies: {sum(record['anomaly_flag'] for record in national_records.values())}; "
        f"subdivision anomalies: {sum(record['anomaly_flag'] for record in subdivision_records.values())}; "
        f"clusters: {cluster_count} (silhouette {silhouette:.3f})"
    )


if __name__ == "__main__":
    main()
