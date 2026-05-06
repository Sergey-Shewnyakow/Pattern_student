import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)


def prepare_features_for_gmm(features_df: pd.DataFrame):
    """
    Подготавливает данные для GMM:
    - оставляет только числовые признаки
    - масштабирует данные
    """
    numeric_df = features_df.select_dtypes(include="number").copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df)

    return numeric_df, X_scaled, scaler


def run_gmm(
    features_df: pd.DataFrame,
    n_components: int = 3,
    covariance_type: str = "full",
    random_state: int = 42
):
    """
    Запускает Gaussian Mixture Model на таблице признаков студентов.
    """
    numeric_df, X_scaled, scaler = prepare_features_for_gmm(features_df)

    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state
    )

    labels = model.fit_predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    max_probability = probabilities.max(axis=1)

    result_df = features_df.copy()
    result_df["cluster"] = labels
    result_df["gmm_confidence"] = max_probability

    metrics = {
        "silhouette_score": silhouette_score(X_scaled, labels),
        "calinski_harabasz_score": calinski_harabasz_score(X_scaled, labels),
        "davies_bouldin_score": davies_bouldin_score(X_scaled, labels),
        "bic": model.bic(X_scaled),
        "aic": model.aic(X_scaled),
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


def evaluate_gmm_range(
    features_df: pd.DataFrame,
    k_min: int = 2,
    k_max: int = 6,
    covariance_type: str = "full",
    random_state: int = 42
):
    """
    Считает метрики GMM для диапазона числа компонент.
    """
    numeric_df, X_scaled, _ = prepare_features_for_gmm(features_df)

    rows = []

    for k in range(k_min, k_max + 1):
        model = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            random_state=random_state
        )

        labels = model.fit_predict(X_scaled)

        row = {
            "k": k,
            "silhouette_score": silhouette_score(X_scaled, labels),
            "calinski_harabasz_score": calinski_harabasz_score(X_scaled, labels),
            "davies_bouldin_score": davies_bouldin_score(X_scaled, labels),
            "bic": model.bic(X_scaled),
            "aic": model.aic(X_scaled),
            "covariance_type": covariance_type,
        }
        rows.append(row)

    return pd.DataFrame(rows)