import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)


def prepare_features_for_clustering(features_df: pd.DataFrame):
    """
    Подготавливает данные для кластеризации:
    - убирает student_id
    - оставляет только числовые признаки
    - масштабирует данные
    """
    numeric_df = features_df.select_dtypes(include="number").copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df)

    return numeric_df, X_scaled, scaler


def run_kmeans(features_df: pd.DataFrame, n_clusters: int = 3, random_state: int = 42):
    """
    Запускает KMeans на таблице признаков студентов.
    Возвращает:
    - DataFrame с кластерами
    - метрики качества
    - средние профили кластеров
    """
    numeric_df, X_scaled, scaler = prepare_features_for_clustering(features_df)

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X_scaled)

    result_df = features_df.copy()
    result_df["cluster"] = labels

    metrics = {
        "silhouette_score": silhouette_score(X_scaled, labels),
        "calinski_harabasz_score": calinski_harabasz_score(X_scaled, labels),
        "davies_bouldin_score": davies_bouldin_score(X_scaled, labels),
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
    }

def evaluate_kmeans_range(features_df: pd.DataFrame, k_min: int = 2, k_max: int = 8):
    """
    Считает метрики KMeans для диапазона числа кластеров.
    """
    numeric_df, X_scaled, _ = prepare_features_for_clustering(features_df)

    rows = []

    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)

        row = {
            "k": k,
            "silhouette_score": silhouette_score(X_scaled, labels),
            "calinski_harabasz_score": calinski_harabasz_score(X_scaled, labels),
            "davies_bouldin_score": davies_bouldin_score(X_scaled, labels),
            "inertia": model.inertia_,
        }
        rows.append(row)

    return pd.DataFrame(rows)