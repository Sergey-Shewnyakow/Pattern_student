import pandas as pd

from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

from src.cluster_features import prepare_cluster_matrix


def _calculate_clustering_metrics(x_scaled, labels, model=None):
    """
    Считает метрики качества кластеризации.

    Для GMM дополнительно можно считать AIC и BIC,
    если передана обученная модель.
    """
    unique_labels = set(labels)

    metrics = {}

    if len(unique_labels) <= 1:
        metrics["silhouette_score"] = None
        metrics["calinski_harabasz_score"] = None
        metrics["davies_bouldin_score"] = None
    else:
        metrics["silhouette_score"] = silhouette_score(x_scaled, labels)
        metrics["calinski_harabasz_score"] = calinski_harabasz_score(x_scaled, labels)
        metrics["davies_bouldin_score"] = davies_bouldin_score(x_scaled, labels)

    if model is not None:
        metrics["aic"] = model.aic(x_scaled)
        metrics["bic"] = model.bic(x_scaled)

    return metrics


def run_gmm(
    features_df: pd.DataFrame,
    n_components: int = 4,
    covariance_type: str = "full",
    random_state: int = 42,
):
    """
    Запускает Gaussian Mixture Model.

    GMM использует тот же набор признаков, что KMeans и Agglomerative:
    - типы учебной активности;
    - общий уровень активности;
    - регулярность поведения.

    В отличие от KMeans, GMM даёт не только номер кластера,
    но и вероятность принадлежности пользователя к выбранному кластеру.
    """
    numeric_df, x_scaled, scaler = prepare_cluster_matrix(features_df)

    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
    )

    labels = model.fit_predict(x_scaled)
    probabilities = model.predict_proba(x_scaled)

    result_df = features_df.copy()
    result_df["cluster"] = labels
    result_df["cluster_probability"] = probabilities.max(axis=1)

    # Вероятности принадлежности к каждому компоненту GMM
    for component_idx in range(n_components):
        result_df[f"gmm_probability_cluster_{component_idx}"] = probabilities[
            :,
            component_idx,
        ]

    metrics = _calculate_clustering_metrics(
        x_scaled=x_scaled,
        labels=labels,
        model=model,
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
        "n_components": n_components,
        "covariance_type": covariance_type,
    }


def evaluate_gmm_range(
    features_df: pd.DataFrame,
    k_min: int = 2,
    k_max: int = 8,
    covariance_type: str = "full",
    random_state: int = 42,
):
    """
    Считает метрики GMM для разного числа компонент.

    Для GMM особенно важны:
    - AIC;
    - BIC.

    Чем ниже AIC/BIC, тем лучше модель с учётом сложности.
    """
    numeric_df, x_scaled, _ = prepare_cluster_matrix(features_df)

    rows = []

    for k in range(k_min, k_max + 1):
        if k >= len(features_df):
            continue

        model = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            random_state=random_state,
        )

        labels = model.fit_predict(x_scaled)

        metrics = _calculate_clustering_metrics(
            x_scaled=x_scaled,
            labels=labels,
            model=model,
        )

        row = {
            "k": k,
            "covariance_type": covariance_type,
            "silhouette_score": metrics["silhouette_score"],
            "calinski_harabasz_score": metrics["calinski_harabasz_score"],
            "davies_bouldin_score": metrics["davies_bouldin_score"],
            "aic": metrics["aic"],
            "bic": metrics["bic"],
            "used_features": ", ".join(numeric_df.columns),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def compare_gmm_covariance_types(
    features_df: pd.DataFrame,
    n_components: int = 4,
    random_state: int = 42,
):
    """
    Сравнивает разные типы covariance_type для GMM.

    full — самый гибкий вариант;
    diag — проще и устойчивее;
    tied — общая ковариационная матрица;
    spherical — самый простой вариант.
    """
    covariance_types = ["full", "diag", "tied", "spherical"]

    rows = []

    for covariance_type in covariance_types:
        result = run_gmm(
            features_df=features_df,
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state,
        )

        metrics = result["metrics"]

        rows.append(
            {
                "covariance_type": covariance_type,
                "n_components": n_components,
                "silhouette_score": metrics["silhouette_score"],
                "calinski_harabasz_score": metrics["calinski_harabasz_score"],
                "davies_bouldin_score": metrics["davies_bouldin_score"],
                "aic": metrics["aic"],
                "bic": metrics["bic"],
                "used_features": ", ".join(result["used_features"]),
            }
        )

    return pd.DataFrame(rows)