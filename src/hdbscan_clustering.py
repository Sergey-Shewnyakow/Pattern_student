import pandas as pd

from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

from src.cluster_features import prepare_cluster_matrix


def _calculate_hdbscan_metrics(x_scaled, labels):
    """
    Считает метрики HDBSCAN.

    Важно:
    HDBSCAN может пометить часть объектов как шум: cluster = -1.
    Метрики качества считаем только по объектам, которые попали в реальные кластеры.
    """
    labels_series = pd.Series(labels)

    non_noise_mask = labels_series != -1
    non_noise_labels = labels_series[non_noise_mask]

    noise_count = int((labels_series == -1).sum())
    clusters_count = int(len(set(labels)) - (1 if -1 in labels else 0))

    if non_noise_mask.sum() <= 1 or len(set(non_noise_labels)) <= 1:
        return {
            "silhouette_score": None,
            "calinski_harabasz_score": None,
            "davies_bouldin_score": None,
            "clusters_count": clusters_count,
            "noise_count": noise_count,
            "noise_share": noise_count / len(labels) if len(labels) > 0 else 0,
        }

    x_non_noise = x_scaled[non_noise_mask.values]

    return {
        "silhouette_score": silhouette_score(x_non_noise, non_noise_labels),
        "calinski_harabasz_score": calinski_harabasz_score(
            x_non_noise,
            non_noise_labels,
        ),
        "davies_bouldin_score": davies_bouldin_score(
            x_non_noise,
            non_noise_labels,
        ),
        "clusters_count": clusters_count,
        "noise_count": noise_count,
        "noise_share": noise_count / len(labels) if len(labels) > 0 else 0,
    }


def run_hdbscan(
    features_df: pd.DataFrame,
    min_cluster_size: int = 10,
    min_samples: int | None = None,
):
    """
    Запускает HDBSCAN.

    HDBSCAN использует тот же набор признаков, что KMeans, Agglomerative и GMM:
    - типы учебной активности;
    - общий уровень активности;
    - регулярность поведения.

    Особенность HDBSCAN:
    - не нужно заранее задавать число кластеров;
    - метод может выделять шумовые объекты: cluster = -1.
    """
    try:
        import hdbscan
    except ImportError as exc:
        raise ImportError(
            "Библиотека hdbscan не установлена. "
            "Установите её командой: pip install hdbscan"
        ) from exc

    numeric_df, x_scaled, scaler = prepare_cluster_matrix(features_df)

    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        prediction_data=True,
    )

    labels = model.fit_predict(x_scaled)

    result_df = features_df.copy()
    result_df["cluster"] = labels

    if hasattr(model, "probabilities_"):
        result_df["cluster_probability"] = model.probabilities_
    else:
        result_df["cluster_probability"] = 0.0

    result_df["is_noise"] = result_df["cluster"] == -1

    metrics = _calculate_hdbscan_metrics(
        x_scaled=x_scaled,
        labels=labels,
    )

    # Профили считаем только по реальным кластерам, без шума
    clustered_df = result_df[result_df["cluster"] != -1].copy()

    if clustered_df.empty:
        cluster_profiles = pd.DataFrame()
    else:
        cluster_profiles = (
            clustered_df.groupby("cluster")
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
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
    }


def evaluate_hdbscan_params(
    features_df: pd.DataFrame,
    min_cluster_size_values: list[int],
    min_samples_values: list[int | None],
):
    """
    Перебирает параметры HDBSCAN.

    Основные параметры:
    - min_cluster_size: минимальный размер кластера;
    - min_samples: насколько строго метод считает точки шумом.

    Чем больше min_cluster_size и min_samples, тем больше объектов может уйти в шум.
    """
    rows = []

    for min_cluster_size in min_cluster_size_values:
        for min_samples in min_samples_values:
            result = run_hdbscan(
                features_df=features_df,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
            )

            metrics = result["metrics"]

            rows.append(
                {
                    "min_cluster_size": min_cluster_size,
                    "min_samples": min_samples,
                    "clusters_count": metrics["clusters_count"],
                    "noise_count": metrics["noise_count"],
                    "noise_share": metrics["noise_share"],
                    "silhouette_score": metrics["silhouette_score"],
                    "calinski_harabasz_score": metrics["calinski_harabasz_score"],
                    "davies_bouldin_score": metrics["davies_bouldin_score"],
                    "used_features": ", ".join(result["used_features"]),
                }
            )

    return pd.DataFrame(rows)