import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)


def prepare_features_for_hdbscan(features_df: pd.DataFrame):
    """
    Подготавливает данные для HDBSCAN:
    - оставляет только числовые признаки
    - масштабирует данные
    """
    numeric_df = features_df.select_dtypes(include="number").copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df)

    return numeric_df, X_scaled, scaler


def _compute_metrics_without_noise(X_scaled, labels):
    """
    Считает метрики на точках без шума (-1), если кластеров достаточно.
    """
    valid_mask = labels != -1
    X_valid = X_scaled[valid_mask]
    labels_valid = labels[valid_mask]

    unique_clusters = set(labels_valid)

    if len(X_valid) < 2 or len(unique_clusters) < 2:
        return {
            "silhouette_score": None,
            "calinski_harabasz_score": None,
            "davies_bouldin_score": None,
        }

    return {
        "silhouette_score": silhouette_score(X_valid, labels_valid),
        "calinski_harabasz_score": calinski_harabasz_score(X_valid, labels_valid),
        "davies_bouldin_score": davies_bouldin_score(X_valid, labels_valid),
    }


def run_hdbscan(
    features_df: pd.DataFrame,
    min_cluster_size: int = 5,
    min_samples: int | None = None,
    cluster_selection_method: str = "eom",
    metric: str = "euclidean"
):
    """
    Запускает HDBSCAN на таблице признаков студентов.
    """
    numeric_df, X_scaled, scaler = prepare_features_for_hdbscan(features_df)

    model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
        metric=metric
    )

    labels = model.fit_predict(X_scaled)

    result_df = features_df.copy()
    result_df["cluster"] = labels

    metrics = _compute_metrics_without_noise(X_scaled, labels)

    cluster_profiles = (
        result_df[result_df["cluster"] != -1]
        .groupby("cluster")
        .mean(numeric_only=True)
        .reset_index()
    )

    noise_count = int((labels == -1).sum())
    cluster_count = len(set(labels)) - (1 if -1 in labels else 0)

    metrics["noise_count"] = noise_count
    metrics["cluster_count"] = cluster_count

    return {
        "result_df": result_df,
        "metrics": metrics,
        "cluster_profiles": cluster_profiles,
        "model": model,
        "scaler": scaler,
    }


def evaluate_hdbscan_range(
    features_df: pd.DataFrame,
    min_cluster_size_values: list[int],
    min_samples: int | None = None,
    cluster_selection_method: str = "eom",
    metric: str = "euclidean"
):
    """
    Оценивает HDBSCAN для нескольких значений min_cluster_size.
    """
    numeric_df, X_scaled, _ = prepare_features_for_hdbscan(features_df)

    rows = []

    for min_cluster_size in min_cluster_size_values:
        model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method=cluster_selection_method,
            metric=metric
        )

        labels = model.fit_predict(X_scaled)
        cluster_count = len(set(labels)) - (1 if -1 in labels else 0)
        noise_count = int((labels == -1).sum())

        metric_values = _compute_metrics_without_noise(X_scaled, labels)

        rows.append({
            "min_cluster_size": min_cluster_size,
            "cluster_count": cluster_count,
            "noise_count": noise_count,
            "silhouette_score": metric_values["silhouette_score"],
            "calinski_harabasz_score": metric_values["calinski_harabasz_score"],
            "davies_bouldin_score": metric_values["davies_bouldin_score"],
            "metric": metric,
            "cluster_selection_method": cluster_selection_method,
        })

    return pd.DataFrame(rows)