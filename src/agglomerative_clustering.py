import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)


def prepare_features_for_agglomerative(features_df: pd.DataFrame):
    numeric_df = features_df.select_dtypes(include="number").copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df)

    return numeric_df, X_scaled, scaler


def _safe_cluster_metrics(X_scaled, labels):
    unique_labels = pd.Series(labels).nunique()

    if unique_labels < 2:
        return {
            "silhouette_score": None,
            "calinski_harabasz_score": None,
            "davies_bouldin_score": None,
        }

    return {
        "silhouette_score": silhouette_score(X_scaled, labels),
        "calinski_harabasz_score": calinski_harabasz_score(X_scaled, labels),
        "davies_bouldin_score": davies_bouldin_score(X_scaled, labels),
    }


def run_agglomerative(
    features_df: pd.DataFrame,
    n_clusters: int | None = 3,
    linkage: str = "ward",
    metric: str = "euclidean",
    distance_threshold: float | None = None,
):
    """
    Запускает Agglomerative Clustering:
    - либо с фиксированным числом кластеров
    - либо с distance_threshold
    """
    numeric_df, X_scaled, scaler = prepare_features_for_agglomerative(features_df)

    if linkage == "ward":
        metric = "euclidean"

    if distance_threshold is not None:
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            compute_full_tree=True,
            linkage=linkage,
            metric=metric,
            compute_distances=True,
        )
    else:
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            metric=metric,
            compute_distances=True
        )

    labels = model.fit_predict(X_scaled)

    result_df = features_df.copy()
    result_df["cluster"] = labels

    metrics = _safe_cluster_metrics(X_scaled, labels)
    metrics["cluster_count"] = int(pd.Series(labels).nunique())

    if distance_threshold is not None:
        metrics["distance_threshold"] = float(distance_threshold)
    else:
        metrics["n_clusters"] = int(n_clusters)

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
    }


def evaluate_agglomerative_range(
    features_df: pd.DataFrame,
    k_min: int = 2,
    k_max: int = 6,
    linkage: str = "ward",
    metric: str = "euclidean"
):
    """
    Оценка Agglomerative для диапазона числа кластеров.
    """
    numeric_df, X_scaled, _ = prepare_features_for_agglomerative(features_df)

    if linkage == "ward":
        metric = "euclidean"

    rows = []

    for k in range(k_min, k_max + 1):
        model = AgglomerativeClustering(
            n_clusters=k,
            linkage=linkage,
            metric=metric
        )
        labels = model.fit_predict(X_scaled)

        metric_values = _safe_cluster_metrics(X_scaled, labels)

        row = {
            "k": k,
            "silhouette_score": metric_values["silhouette_score"],
            "calinski_harabasz_score": metric_values["calinski_harabasz_score"],
            "davies_bouldin_score": metric_values["davies_bouldin_score"],
            "linkage": linkage,
            "metric": metric,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def evaluate_agglomerative_thresholds(
    features_df: pd.DataFrame,
    threshold_values: list[float],
    linkage: str = "ward",
    metric: str = "euclidean"
):
    """
    Оценка Agglomerative для диапазона distance_threshold.
    """
    numeric_df, X_scaled, _ = prepare_features_for_agglomerative(features_df)

    if linkage == "ward":
        metric = "euclidean"

    rows = []

    for threshold in threshold_values:
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            compute_full_tree=True,
            linkage=linkage,
            metric=metric,
            compute_distances=True,
        )
        labels = model.fit_predict(X_scaled)

        metric_values = _safe_cluster_metrics(X_scaled, labels)

        row = {
            "distance_threshold": threshold,
            "cluster_count": int(pd.Series(labels).nunique()),
            "silhouette_score": metric_values["silhouette_score"],
            "calinski_harabasz_score": metric_values["calinski_harabasz_score"],
            "davies_bouldin_score": metric_values["davies_bouldin_score"],
            "linkage": linkage,
            "metric": metric,
        }
        rows.append(row)

    return pd.DataFrame(rows)