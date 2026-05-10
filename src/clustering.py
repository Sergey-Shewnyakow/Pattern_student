import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

from src.cluster_features import prepare_cluster_matrix


def run_kmeans(
    features_df: pd.DataFrame,
    n_clusters: int = 4,
    random_state: int = 42,
):
    numeric_df, x_scaled, scaler = prepare_cluster_matrix(features_df)

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )

    labels = model.fit_predict(x_scaled)

    result_df = features_df.copy()
    result_df["cluster"] = labels

    if n_clusters > 1 and len(set(labels)) > 1:
        metrics = {
            "silhouette_score": silhouette_score(x_scaled, labels),
            "calinski_harabasz_score": calinski_harabasz_score(x_scaled, labels),
            "davies_bouldin_score": davies_bouldin_score(x_scaled, labels),
        }
    else:
        metrics = {
            "silhouette_score": None,
            "calinski_harabasz_score": None,
            "davies_bouldin_score": None,
        }

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
    }


def evaluate_kmeans_range(
    features_df: pd.DataFrame,
    k_min: int = 2,
    k_max: int = 8,
):
    numeric_df, x_scaled, _ = prepare_cluster_matrix(features_df)

    rows = []

    for k in range(k_min, k_max + 1):
        if k >= len(features_df):
            continue

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
        )

        labels = model.fit_predict(x_scaled)

        if len(set(labels)) > 1:
            silhouette = silhouette_score(x_scaled, labels)
            calinski = calinski_harabasz_score(x_scaled, labels)
            davies = davies_bouldin_score(x_scaled, labels)
        else:
            silhouette = None
            calinski = None
            davies = None

        rows.append(
            {
                "k": k,
                "silhouette_score": silhouette,
                "calinski_harabasz_score": calinski,
                "davies_bouldin_score": davies,
                "inertia": model.inertia_,
                "used_features": ", ".join(numeric_df.columns),
            }
        )

    return pd.DataFrame(rows)