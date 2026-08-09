import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from config import config


OUTPUT_DIR = config.OUTPUT_DIR


def plot_scenario_trajectories(scenario_results: dict, save: bool = True):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    colors = {
        "baseline": "#2196F3",
        "high_growth": "#4CAF50",
        "low_growth": "#F44336",
        "green_transition": "#FF9800",
    }

    for scenario_name, df in scenario_results.items():
        grouped = df.groupby("year")["predicted_hdi"].mean()
        for ax_idx, (ax, title) in enumerate(zip(axes.flat, [
            "Average HDI Trajectory", "HDI Distribution 2050",
            "Life Expectancy Trajectory", "GNI per Capita Trajectory",
        ])):
            if ax_idx == 0:
                ax.plot(grouped.index, grouped.values, label=scenario_name,
                        color=colors.get(scenario_name, "gray"), linewidth=2)
                ax.set_ylabel("Mean HDI")
            elif ax_idx == 1 and scenario_name == list(scenario_results.keys())[0]:
                latest = {s: d[d["year"] == config.FORECAST_END]["predicted_hdi"]
                          for s, d in scenario_results.items()}
                data = [v.values for v in latest.values()]
                labels = list(latest.keys())
                ax.boxplot(data, tick_labels=labels, patch_artist=True,
                           boxprops=dict(facecolor="lightblue"))
                ax.set_ylabel("HDI 2050")
                ax.tick_params(axis="x", rotation=30)
            elif ax_idx == 2:
                grouped_le = df.groupby("year")["life_exp"].mean()
                ax.plot(grouped_le.index, grouped_le.values, label=scenario_name,
                        color=colors.get(scenario_name, "gray"), linewidth=2)
                ax.set_ylabel("Mean Life Expectancy")
            elif ax_idx == 3:
                grouped_gni = df.groupby("year")["gni_ppp"].mean()
                ax.plot(grouped_gni.index, grouped_gni.values, label=scenario_name,
                        color=colors.get(scenario_name, "gray"), linewidth=2)
                ax.set_ylabel("Mean GNI per capita (PPP)")

    axes[0, 0].set_title("Mean HDI by Scenario (2026-2050)")
    axes[0, 0].legend()
    axes[0, 1].set_title("HDI Distribution in 2050")
    axes[1, 0].set_title("Mean Life Expectancy by Scenario")
    axes[1, 1].set_title("Mean GNI per Capita by Scenario")

    for ax in axes.flat:
        ax.set_xlabel("Year")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(OUTPUT_DIR / "scenario_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_cv_results(cv_summary: dict, save: bool = True):
    targets = list(cv_summary.keys())
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    metrics = ["mae", "rmse", "r2", "mape"]

    for ax, metric in zip(axes, metrics):
        vals = [cv_summary[t][metric]["mean"] for t in targets]
        errs = [cv_summary[t][metric]["std"] for t in targets]
        ax.barh(targets, vals, xerr=errs, capsize=3, color="#2196F3", alpha=0.7)
        ax.set_xlabel(metric.upper())
        ax.set_title(f"CV {metric.upper()}")
        ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    if save:
        plt.savefig(OUTPUT_DIR / "cv_results.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_importance(importances: dict, save: bool = True):
    n = len(importances)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 8))
    if n == 1:
        axes = [axes]

    for ax, (comp_name, imp) in zip(axes, importances.items()):
        top = imp.head(15)
        ax.barh(top.index[::-1], top.values[::-1], color="#4CAF50", alpha=0.7)
        ax.set_title(f"Top Features: {comp_name}")
        ax.set_xlabel("Importance")
        ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    if save:
        plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_hdi_heatmap(results_df: pd.DataFrame, save: bool = True):
    pivot = results_df.pivot_table(
        values="predicted_hdi", index="archetype", columns="year", aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0.3, vmax=0.95)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(range(0, len(pivot.columns), 5))
    ax.set_xticklabels(pivot.columns[::5])
    plt.colorbar(im, ax=ax, label="HDI")
    ax.set_title("HDI Trajectory Heatmap by Development Group")
    ax.set_xlabel("Year")
    plt.tight_layout()
    if save:
        plt.savefig(OUTPUT_DIR / "hdi_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()


def generate_all_plots(
    scenario_results: dict,
    cv_summary: dict = None,
    importances: dict = None,
    results_df: pd.DataFrame = None,
):
    print("Generating scenario trajectory plots...")
    plot_scenario_trajectories(scenario_results)

    if cv_summary:
        print("Generating CV results plots...")
        plot_cv_results(cv_summary)

    if importances:
        print("Generating feature importance plots...")
        plot_feature_importance(importances)

    if results_df is not None:
        print("Generating HDI heatmap...")
        plot_hdi_heatmap(results_df)

    print(f"All plots saved to {OUTPUT_DIR}")
