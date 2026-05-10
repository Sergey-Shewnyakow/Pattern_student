import pandas as pd

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

from src.cluster_features import prepare_cluster_matrix


def _calculate_clustering_metrics(x_scaled, labels):
    """
    Считает метрики качества кластеризации.

    Метрики считаются только если реально получилось больше одного кластера.
    """
    unique_labels = set(labels)

    if len(unique_labels) <= 1:
        return {
            "silhouette_score": None,
            "calinski_harabasz_score": None,
            "davies_bouldin_score": None,
        }

    return {
        "silhouette_score": silhouette_score(x_scaled, labels),
        "calinski_harabasz_score": calinski_harabasz_score(x_scaled, labels),
        "davies_bouldin_score": davies_bouldin_score(x_scaled, labels),
    }


def run_agglomerative(
    features_df: pd.DataFrame,
    n_clusters: int = 4,
    linkage: str = "ward",
):
    """
    Запускает Agglomerative Clustering.

    Метод использует тот же набор признаков, что и KMeans:
    - типы учебной активности;
    - общий уровень активности;
    - регулярность.

    Это позволяет сравнивать KMeans и Agglomerative между собой.
    """
    numeric_df, x_scaled, scaler = prepare_cluster_matrix(features_df)

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage,
    )

    labels = model.fit_predict(x_scaled)

    result_df = features_df.copy()
    result_df["cluster"] = labels

    metrics = _calculate_clustering_metrics(
        x_scaled=x_scaled,
        labels=labels,
    )

    cluster_profiles = (
        result_df.groupby("cluster")
        .mean(numeric_only=True)
        .reset_index()
    )

    return {
        "result_df": result_df,
        "metrics": metrics,
        "cluster_profiles": cluster_profiles,
        "model": model,
        "scaler": scaler,
        "used_features": list(numeric_df.columns),
        "linkage": linkage,
    }


def evaluate_agglomerative_range(
    features_df: pd.DataFrame,
    k_min: int = 2,
    k_max: int = 8,
    linkage: str = "ward",
):
    """
    Считает метрики Agglomerative Clustering для разного числа кластеров.
    """
    numeric_df, x_scaled, _ = prepare_cluster_matrix(features_df)

    rows = []

    for k in range(k_min, k_max + 1):
        if k >= len(features_df):
            continue

        model = AgglomerativeClustering(
            n_clusters=k,
            linkage=linkage,
        )

        labels = model.fit_predict(x_scaled)

        metrics = _calculate_clustering_metrics(
            x_scaled=x_scaled,
            labels=labels,
        )

        row = {
            "k": k,
            "linkage": linkage,
            "silhouette_score": metrics["silhouette_score"],
            "calinski_harabasz_score": metrics["calinski_harabasz_score"],
            "davies_bouldin_score": metrics["davies_bouldin_score"],
            "used_features": ", ".join(numeric_df.columns),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def compare_agglomerative_linkages(
    features_df: pd.DataFrame,
    n_clusters: int = 4,
):
    """
    Сравнивает разные варианты linkage.

    ward обычно является основным вариантом для числовых стандартизированных признаков.
    average и complete можно использовать как проверочные варианты.
    """
    linkages = ["ward", "average", "complete"]

    rows = []

    for linkage in linkages:
        result = run_agglomerative(
            features_df=features_df,
            n_clusters=n_clusters,
            linkage=linkage,
        )

        metrics = result["metrics"]

        rows.append(
            {
                "linkage": linkage,
                "n_clusters": n_clusters,
                "silhouette_score": metrics["silhouette_score"],
                "calinski_harabasz_score": metrics["calinski_harabasz_score"],
                "davies_bouldin_score": metrics["davies_bouldin_score"],
                "used_features": ", ".join(result["used_features"]),
            }
        )

    return pd.DataFrame(rows)