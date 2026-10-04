"""Activity 2 - JoPACC monthly payment pattern analysis.

This module is the reusable Python version of ``Activity_2.ipynb``.  One
observation is one payment-system month.  The three clustering features are
expressed relative to that system's own historical mean so that systems with
very different absolute scales can be compared without treating structural
zeros as genuine activity.
"""

from __future__ import annotations

import base64
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, Birch, DBSCAN, HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "jopacc_system_month_november_2022_to_june_2026.csv"

FEATURE_COLUMNS = [
    "relative_transaction_level",
    "relative_value_level",
    "relative_average_transaction_value",
]


def _model_factories() -> dict[str, Callable[[], object]]:
    """Return the five fresh estimators compared in the Activity 2 notebook."""
    return {
        "K-means": lambda: KMeans(
            n_clusters=3,
            init="k-means++",
            n_init=20,
            random_state=42,
        ),
        "Hierarchical": lambda: AgglomerativeClustering(
            n_clusters=3,
            linkage="ward",
        ),
        "DBSCAN": lambda: DBSCAN(eps=0.4, min_samples=4),
        "BIRCH": lambda: Birch(n_clusters=3),
        "HDBSCAN": lambda: HDBSCAN(min_cluster_size=5, copy=True),
    }


def validate_and_prepare(
    data_path: str | Path,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Load and validate dates, identifiers, measures, and coverage."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"JoPACC dataset was not found: {data_path}")

    monthly = pd.read_csv(data_path)
    required = {
        "report_month",
        "report_year",
        "report_quarter",
        "system",
        "transactions_total",
        "transaction_value_jod",
    }
    missing_columns = sorted(required.difference(monthly.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    monthly["report_month"] = pd.to_datetime(
        monthly["report_month"], format="%Y-%m", errors="coerce"
    )
    for column in ["report_year", "transactions_total", "transaction_value_jod"]:
        monthly[column] = pd.to_numeric(monthly[column], errors="coerce")

    invalid_dates = int(monthly["report_month"].isna().sum())
    duplicate_rows = int(monthly.duplicated(["system", "report_month"]).sum())
    missing_values = int(monthly.isna().sum().sum())
    nonpositive_transactions = int((monthly["transactions_total"] <= 0).sum())
    negative_values = int((monthly["transaction_value_jod"] < 0).sum())

    if invalid_dates:
        raise ValueError(f"The dataset contains {invalid_dates} invalid report months.")
    if duplicate_rows:
        raise ValueError(
            f"The dataset contains {duplicate_rows} duplicate system-month rows."
        )
    if missing_values:
        raise ValueError(f"The dataset contains {missing_values} missing values.")
    if nonpositive_transactions:
        raise ValueError(
            "Average transaction value cannot be calculated because one or more "
            "rows have zero or negative transaction totals."
        )
    if negative_values:
        raise ValueError(
            f"The dataset contains {negative_values} negative JOD transaction values."
        )

    expected_quarter = "Q" + monthly["report_month"].dt.quarter.astype(str)
    incorrect_year = int(
        (monthly["report_year"] != monthly["report_month"].dt.year).sum()
    )
    incorrect_quarter = int((monthly["report_quarter"] != expected_quarter).sum())
    if incorrect_year or incorrect_quarter:
        raise ValueError(
            "Date metadata is inconsistent: "
            f"{incorrect_year} year errors and {incorrect_quarter} quarter errors."
        )

    monthly["report_year"] = monthly["report_year"].astype(int)
    monthly = monthly.sort_values(["system", "report_month"]).reset_index(drop=True)

    numeric_columns = monthly.select_dtypes(include="number").columns
    structural_rows = []
    for system, system_data in monthly.groupby("system", sort=True):
        zero_columns = [
            column for column in numeric_columns if (system_data[column] == 0).all()
        ]
        structural_rows.append(
            {
                "system": system,
                "structural_zero_count": len(zero_columns),
                "structural_zero_columns": ", ".join(zero_columns),
            }
        )

    coverage = (
        monthly.groupby("system", sort=True)
        .agg(
            observations=("report_month", "size"),
            first_month=("report_month", "min"),
            last_month=("report_month", "max"),
        )
        .reset_index()
    )
    validation = {
        "rows": len(monthly),
        "columns": len(monthly.columns),
        "systems": int(monthly["system"].nunique()),
        "months": int(monthly["report_month"].nunique()),
        "first_month": monthly["report_month"].min(),
        "last_month": monthly["report_month"].max(),
        "missing_values": missing_values,
        "invalid_dates": invalid_dates,
        "duplicate_system_months": duplicate_rows,
        "incorrect_years": incorrect_year,
        "incorrect_quarters": incorrect_quarter,
        "nonpositive_transactions": nonpositive_transactions,
        "negative_transaction_values": negative_values,
        "coverage": coverage,
    }
    return monthly, validation, pd.DataFrame(structural_rows)


def engineer_features(monthly: pd.DataFrame) -> pd.DataFrame:
    """Create the three comparable features justified in Activity 2."""
    prepared = monthly.copy()
    prepared["average_transaction_value"] = (
        prepared["transaction_value_jod"] / prepared["transactions_total"]
    )
    prepared["relative_transaction_level"] = prepared["transactions_total"] / (
        prepared.groupby("system")["transactions_total"].transform("mean")
    )
    prepared["relative_value_level"] = prepared["transaction_value_jod"] / (
        prepared.groupby("system")["transaction_value_jod"].transform("mean")
    )
    prepared["relative_average_transaction_value"] = prepared[
        "average_transaction_value"
    ] / prepared.groupby("system")["average_transaction_value"].transform("mean")
    return prepared


def _internal_scores(features: np.ndarray, labels: np.ndarray) -> dict:
    """Evaluate non-noise observations while also reporting noise coverage."""
    labels = np.asarray(labels)
    included = labels >= 0
    included_labels = labels[included]
    cluster_count = len(np.unique(included_labels)) if included.any() else 0
    scores = {
        "clusters": cluster_count,
        "noise_pct": float((~included).mean() * 100),
        "silhouette": np.nan,
        "calinski_harabasz": np.nan,
        "davies_bouldin": np.nan,
    }
    if cluster_count >= 2 and included.sum() > cluster_count:
        included_features = features[included]
        scores.update(
            {
                "silhouette": silhouette_score(included_features, included_labels),
                "calinski_harabasz": calinski_harabasz_score(
                    included_features, included_labels
                ),
                "davies_bouldin": davies_bouldin_score(
                    included_features, included_labels
                ),
            }
        )
    return scores


def _perturbation_stability(
    features: np.ndarray,
    factory: Callable[[], object],
    baseline_labels: np.ndarray,
    repeats: int = 20,
    noise_standard_deviation: float = 0.02,
) -> float:
    """Mean ARI after small reproducible changes to the PCA coordinates."""
    rng = np.random.default_rng(42)
    scores = []
    for _ in range(repeats):
        perturbed = features + rng.normal(
            0.0, noise_standard_deviation, size=features.shape
        )
        labels = factory().fit_predict(perturbed)
        scores.append(adjusted_rand_score(baseline_labels, labels))
    return float(np.mean(scores))


def compare_models(
    pca_features: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, np.ndarray]]:
    """Fit and compare the five clustering algorithms from the notebook."""
    fitted_models: dict[str, object] = {}
    model_labels: dict[str, np.ndarray] = {}
    rows = []
    for name, factory in _model_factories().items():
        model = factory()
        labels = model.fit_predict(pca_features)
        fitted_models[name] = model
        model_labels[name] = labels
        scores = _internal_scores(pca_features, labels)
        scores["stability_ari"] = _perturbation_stability(
            pca_features, factory, labels
        )
        rows.append({"algorithm": name, **scores})
    return pd.DataFrame(rows), fitted_models, model_labels


def _descriptive_label(profile: pd.Series) -> str:
    """Convert a numerical cluster profile into an analyst-friendly label."""
    activity = profile["activity_score"]
    average_value = profile["relative_average_transaction_value"]
    if activity >= 1.25 and average_value < 0.9:
        return "High activity with smaller transactions"
    if activity <= 0.75 and average_value > 1.1:
        return "Low activity with larger transactions"
    return "Normal activity"


def _apply_final_model(
    monthly: pd.DataFrame,
    pca_features: np.ndarray,
    kmeans: KMeans,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, int]]:
    """Apply stable cluster numbering, labels, and unusualness thresholds."""
    result = monthly.copy()
    raw_labels = np.asarray(kmeans.labels_)

    raw_profiles = result.assign(raw_cluster=raw_labels).groupby("raw_cluster")[
        FEATURE_COLUMNS
    ].mean()
    raw_profiles["activity_score"] = raw_profiles[
        ["relative_transaction_level", "relative_value_level"]
    ].mean(axis=1)

    ordered_raw = raw_profiles["activity_score"].sort_values(ascending=False).index
    cluster_map = {int(raw): rank for rank, raw in enumerate(ordered_raw, start=1)}
    result["cluster"] = pd.Series(raw_labels).map(cluster_map).astype(int)

    result[["PC1", "PC2"]] = pca_features
    distances = kmeans.transform(pca_features)
    result["distance_to_center"] = distances[np.arange(len(result)), raw_labels]
    result["distance_percentile_within_cluster"] = (
        result.groupby("cluster")["distance_to_center"].rank(pct=True) * 100
    )
    thresholds = result.groupby("cluster")["distance_to_center"].quantile(0.95)
    result["unusual_threshold"] = result["cluster"].map(thresholds)
    result["state"] = np.where(
        result["distance_to_center"] > result["unusual_threshold"],
        "unusual",
        "normal",
    )

    profiles = result.groupby("cluster")[FEATURE_COLUMNS].mean()
    profiles["activity_score"] = profiles[
        ["relative_transaction_level", "relative_value_level"]
    ].mean(axis=1)
    profiles["observations"] = result.groupby("cluster").size()
    profiles["pattern"] = profiles.apply(_descriptive_label, axis=1)
    result["pattern"] = result["cluster"].map(profiles["pattern"].to_dict())
    result["unusual_reason"] = np.where(
        result["state"].eq("unusual"),
        "Further from the centre of its pattern than the 95th percentile of that pattern",
        "Inside the 95th percentile distance from the centre of its pattern",
    )
    return result, profiles.reset_index(), cluster_map


def build_analysis(data_path: str | Path = DEFAULT_DATA_PATH) -> dict:
    """Run the complete final clustering pipeline from Activity 2."""
    monthly, validation, structural_zeros = validate_and_prepare(data_path)
    monthly = engineer_features(monthly)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(monthly[FEATURE_COLUMNS])
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(scaled_features)

    comparison, fitted_models, model_labels = compare_models(pca_features)
    result, profiles, cluster_map = _apply_final_model(
        monthly, pca_features, fitted_models["K-means"]
    )

    loadings = pd.DataFrame(
        pca.components_.T,
        index=FEATURE_COLUMNS,
        columns=["PC1", "PC2"],
    )
    pca_summary = pd.DataFrame(
        {
            "component": ["PC1", "PC2"],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(
                pca.explained_variance_ratio_
            ),
        }
    )
    return {
        "data": result,
        "validation": validation,
        "structural_zeros": structural_zeros,
        "features": monthly[FEATURE_COLUMNS],
        "scaled_features": scaled_features,
        "pca_features": pca_features,
        "pca": pca,
        "pca_summary": pca_summary,
        "loadings": loadings,
        "comparison": comparison,
        "profiles": profiles,
        "cluster_map": cluster_map,
        "model": fitted_models["K-means"],
        "model_labels": model_labels,
        "scaler": scaler,
    }


@lru_cache(maxsize=2)
def get_analysis(data_path: str = str(DEFAULT_DATA_PATH)) -> dict:
    """Cache the fitted historical model for repeated Flask requests."""
    return build_analysis(Path(data_path).resolve())


def find_similar_months(
    analysis: dict, row_index: int, count: int = 5
) -> pd.DataFrame:
    """Return nearest system-month observations in the two-dimensional PCA space."""
    data = analysis["data"]
    position = data.index.get_loc(row_index)
    coordinates = analysis["pca_features"]
    distances = np.linalg.norm(coordinates - coordinates[position], axis=1)
    nearest_positions = np.argsort(distances)[1 : count + 1]
    columns = [
        "report_month",
        "system",
        "cluster",
        "pattern",
        "state",
        "transactions_total",
        "transaction_value_jod",
    ]
    similar = data.iloc[nearest_positions][columns].copy()
    similar["similarity_distance"] = distances[nearest_positions]
    return similar


def _figure_data_uri(figure: plt.Figure) -> str:
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
    plt.close(figure)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def create_trend_chart(filtered: pd.DataFrame) -> str | None:
    """Plot historical transaction and value trends for the selected systems."""
    if filtered.empty:
        return None

    figure, axes = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True)
    for system, group in filtered.sort_values("report_month").groupby("system"):
        axes[0].plot(
            group["report_month"],
            group["transactions_total"] / 1_000_000,
            marker="o",
            markersize=3,
            label=system,
        )
        axes[1].plot(
            group["report_month"],
            group["transaction_value_jod"] / 1_000_000_000,
            marker="o",
            markersize=3,
            label=system,
        )
    axes[0].set_ylabel("Transactions (millions)")
    axes[1].set_ylabel("Total value (JOD billions)")
    axes[1].set_xlabel("Month")
    axes[0].set_title("How the records we selected changed over time")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncol=3)
    figure.autofmt_xdate()
    figure.tight_layout()
    return _figure_data_uri(figure)


def create_pca_chart(data: pd.DataFrame, highlighted_indices: list[int]) -> str:
    """Visualise interpreted patterns, unusual rows, and current selections."""
    colours = {1: "#c2410c", 2: "#2563eb", 3: "#15803d"}
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    for cluster, group in data.groupby("cluster"):
        axis.scatter(
            group["PC1"],
            group["PC2"],
            s=34,
            alpha=0.62,
            color=colours[int(cluster)],
            label=group["pattern"].iloc[0],
        )
    unusual = data[data["state"].eq("unusual")]
    axis.scatter(
        unusual["PC1"],
        unusual["PC2"],
        s=85,
        facecolors="none",
        edgecolors="#111827",
        linewidths=1.4,
        label="Unusual inside its pattern",
    )
    if highlighted_indices:
        selected = data.loc[highlighted_indices]
        axis.scatter(
            selected["PC1"],
            selected["PC2"],
            s=110,
            color="#facc15",
            edgecolors="#111827",
            linewidths=1.4,
            marker="*",
            label="What we selected",
            zorder=5,
        )
    axis.set_title("The monthly payment patterns shown in PCA space")
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8, loc="best")
    return _figure_data_uri(figure)


def export_clustered_data(
    output_path: str | Path,
    data_path: str | Path = DEFAULT_DATA_PATH,
) -> Path:
    """Export the notebook's final interpreted system-month dataset."""
    output_path = Path(output_path)
    exported = build_analysis(data_path)["data"].copy()
    exported["report_month"] = exported["report_month"].dt.strftime("%Y-%m")
    exported.to_csv(output_path, index=False)
    return output_path.resolve()


if __name__ == "__main__":
    analysis = build_analysis()
    print("Validation")
    print(
        {
            key: value
            for key, value in analysis["validation"].items()
            if key != "coverage"
        }
    )
    print("\nPCA")
    print(analysis["pca_summary"].round(4).to_string(index=False))
    print("\nModel comparison")
    print(analysis["comparison"].round(4).to_string(index=False))
    print("\nCluster profiles")
    print(analysis["profiles"].round(4).to_string(index=False))
    print("\nState counts")
    print(analysis["data"]["state"].value_counts().to_string())
