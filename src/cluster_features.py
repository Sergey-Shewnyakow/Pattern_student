import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


CLUSTER_FEATURES = [
    # -----------------------------
    # 1. Типы учебной активности
    # -----------------------------
    "video_share",
    "lecture_share",
    "practice_share",
    "test_share",
    "page_share",
    "study_material_share",
    "control_activity_share",

    # -----------------------------
    # 2. Факт использования ресурсов
    # -----------------------------
    "used_video",
    "used_lecture",
    "used_practice",
    "used_test",
    "used_page",
    "material_diversity_count",
    "full_course_activity",
    "practice_test_without_materials",

    # -----------------------------
    # 3. Общий уровень активности
    # -----------------------------
    "total_events",
    "active_days",
    "active_weeks",
    "sessions_count",
    "unique_activities",

    # -----------------------------
    # 4. Регулярность
    # -----------------------------
    "weekly_regularity_cv",
    "long_pauses_over_3d",
]


def get_available_cluster_features(features_df: pd.DataFrame) -> list[str]:
    """
    Возвращает только те признаки, которые реально есть в таблице.
    """
    return [col for col in CLUSTER_FEATURES if col in features_df.columns]


def prepare_cluster_matrix(features_df: pd.DataFrame):
    """
    Общая подготовка признаков для всех методов кластеризации.

    Используется один и тот же набор признаков:
    - типы учебной активности;
    - общий уровень активности;
    - регулярность.

    Это нужно, чтобы KMeans, GMM, Agglomerative и HDBSCAN
    были сравнимы между собой.
    """
    selected_features = get_available_cluster_features(features_df)

    if selected_features:
        numeric_df = features_df[selected_features].copy()
    else:
        numeric_df = features_df.select_dtypes(include="number").copy()

    numeric_df = numeric_df.replace([np.inf, -np.inf], 0)
    numeric_df = numeric_df.fillna(0)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(numeric_df)

    return numeric_df, x_scaled, scaler