import pandas as pd


def _safe_mean(df: pd.DataFrame, col: str, default: float = 0.0) -> float:
    if col not in df.columns:
        return default
    value = df[col].mean()
    if pd.isna(value):
        return default
    return float(value)


def _level(cluster_value: float, global_value: float) -> str:
    """
    Определяет уровень признака относительно общей выборки.
    """
    if abs(global_value) < 1e-9:
        if abs(cluster_value) < 1e-9:
            return "medium"
        return "high"

    ratio = cluster_value / global_value

    if ratio >= 1.3:
        return "high"
    if ratio <= 0.7:
        return "low"
    return "medium"


def suggest_cluster_name(cluster_row: pd.Series, global_means: dict) -> tuple[str, str]:
    """
    Возвращает:
    - краткое название кластера
    - краткое обоснование
    """

    total_events = float(cluster_row.get("total_events", 0))
    active_days = float(cluster_row.get("active_days", 0))
    sessions_count = float(cluster_row.get("sessions_count", 0))
    avg_events_per_session = float(cluster_row.get("avg_events_per_session", 0))
    weekly_regularity_cv = float(cluster_row.get("weekly_regularity_cv", 0))
    night_activity_ratio = float(cluster_row.get("night_activity_ratio", 0))
    long_pauses_over_1d = float(cluster_row.get("long_pauses_over_1d", 0))

    activity_level = _level(total_events, global_means.get("total_events", 0))
    days_level = _level(active_days, global_means.get("active_days", 0))
    sessions_level = _level(sessions_count, global_means.get("sessions_count", 0))
    intensity_level = _level(avg_events_per_session, global_means.get("avg_events_per_session", 0))
    regularity_level = _level(weekly_regularity_cv, global_means.get("weekly_regularity_cv", 0))
    night_level = _level(night_activity_ratio, global_means.get("night_activity_ratio", 0))
    pauses_level = _level(long_pauses_over_1d, global_means.get("long_pauses_over_1d", 0))

    # 1. Систематические
    if (
        activity_level == "high"
        and days_level == "high"
        and sessions_level == "high"
        and regularity_level == "low"
    ):
        return (
            "Систематически активные",
            "Высокая общая активность, много активных дней и сессий, при этом поведение относительно регулярное."
        )

    # 2. Эпизодические
    if (
        activity_level == "low"
        and days_level == "low"
        and sessions_level == "low"
    ):
        return (
            "Эпизодические",
            "Низкая активность, мало активных дней и небольшое число сессий."
        )

    # 3. Рывковые интенсивные
    if (
        activity_level == "high"
        and intensity_level == "high"
        and (days_level in ["low", "medium"])
        and regularity_level == "high"
    ):
        return (
            "Рывково-интенсивные",
            "Высокая активность концентрируется в отдельных интенсивных сессиях, при этом поведение менее регулярное."
        )

    # 4. Нерегулярные / хаотичные
    if (
        regularity_level == "high"
        and pauses_level == "high"
    ):
        return (
            "Нерегулярные",
            "Для кластера характерны большие паузы и выраженная неравномерность активности."
        )

    # 5. Ночные активные
    if night_level == "high" and activity_level in ["medium", "high"]:
        return (
            "Ночно-активные",
            "Кластер отличается повышенной долей ночной активности."
        )

    # 6. Умеренно активные
    if (
        activity_level == "medium"
        and days_level == "medium"
        and sessions_level == "medium"
    ):
        return (
            "Умеренно активные",
            "Показатели активности и вовлечённости близки к средним по выборке."
        )

    # 7. Если кластер активный, но без яркого паттерна
    if activity_level == "high":
        return (
            "Активные",
            "Кластер демонстрирует активность выше средней, но без ярко выраженного специализированного паттерна."
        )

    # 8. Если кластер слабый по активности
    if activity_level == "low":
        return (
            "Слабоактивные",
            "Для кластера характерна активность ниже средней."
        )

    return (
        "Смешанный профиль",
        "Кластер сочетает признаки нескольких поведенческих стратегий и не имеет одного ярко выраженного паттерна."
    )


def build_cluster_names(result_df: pd.DataFrame, cluster_profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Формирует таблицу с названиями и описаниями кластеров.
    """
    global_means = {
        col: _safe_mean(result_df, col)
        for col in result_df.select_dtypes(include="number").columns
        if col != "cluster"
    }

    rows = []

    cluster_sizes = (
        result_df.groupby("cluster")
        .size()
        .reset_index(name="cluster_size")
    )

    profiles_with_size = cluster_profiles.merge(cluster_sizes, on="cluster", how="left")

    for _, row in profiles_with_size.iterrows():
        cluster_id = int(row["cluster"])
        suggested_name, description = suggest_cluster_name(row, global_means)

        rows.append({
            "cluster": cluster_id,
            "cluster_size": int(row["cluster_size"]),
            "suggested_name": suggested_name,
            "description": description,
        })

    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)