import pandas as pd


def _safe_mean(df: pd.DataFrame, col: str, default: float = 0.0) -> float:
    if col not in df.columns:
        return default
    value = df[col].mean()
    if pd.isna(value):
        return default
    return float(value)


def _level(cluster_value: float, global_value: float) -> str:
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


def suggest_pm_cluster_name(cluster_row: pd.Series, global_means: dict) -> tuple[str, str]:
    trace_length = float(cluster_row.get("trace_length", 0))
    trace_unique_activities = float(cluster_row.get("trace_unique_activities", 0))
    trace_duration_hours = float(cluster_row.get("trace_duration_hours", 0))
    backtrack_count = float(cluster_row.get("backtrack_count", 0))
    rework_ratio = float(cluster_row.get("rework_ratio", 0))
    linearity_score = float(cluster_row.get("linearity_score", 0))
    path_complexity = float(cluster_row.get("path_complexity", 0))
    variant_frequency = float(cluster_row.get("variant_frequency", 0))

    length_level = _level(trace_length, global_means.get("trace_length", 0))
    unique_level = _level(trace_unique_activities, global_means.get("trace_unique_activities", 0))
    duration_level = _level(trace_duration_hours, global_means.get("trace_duration_hours", 0))
    backtrack_level = _level(backtrack_count, global_means.get("backtrack_count", 0))
    rework_level = _level(rework_ratio, global_means.get("rework_ratio", 0))
    linearity_level = _level(linearity_score, global_means.get("linearity_score", 0))
    complexity_level = _level(path_complexity, global_means.get("path_complexity", 0))
    variant_level = _level(variant_frequency, global_means.get("variant_frequency", 0))

    if (
        length_level == "high"
        and complexity_level == "high"
        and backtrack_level == "high"
        and rework_level == "high"
    ):
        return (
            "Сложные возвратные траектории",
            "Для кластера характерны длинные траектории, высокая сложность, возвраты и повторное прохождение шагов."
        )

    if (
        linearity_level == "high"
        and backtrack_level == "low"
        and rework_level == "low"
    ):
        return (
            "Линейные траектории",
            "Кластер характеризуется относительно прямолинейным прохождением без выраженных возвратов и повторов."
        )

    if (
        variant_level == "high"
        and complexity_level in ["low", "medium"]
    ):
        return (
            "Типовые траектории",
            "Студенты этого кластера следуют более распространённым и типичным вариантам прохождения."
        )

    if (
        variant_level == "low"
        and complexity_level == "high"
    ):
        return (
            "Нетипичные траектории",
            "Кластер включает менее типовые варианты прохождения с повышенной сложностью."
        )

    if (
        length_level == "low"
        and unique_level == "low"
        and duration_level == "low"
    ):
        return (
            "Короткие траектории",
            "Для кластера характерны короткие и упрощённые траектории поведения."
        )

    if (
        duration_level == "high"
        and length_level in ["medium", "high"]
        and linearity_level == "low"
    ):
        return (
            "Растянутые нелинейные траектории",
            "Прохождение растянуто во времени и сопровождается нелинейными переходами."
        )

    return (
        "Смешанный process-профиль",
        "Кластер сочетает признаки нескольких процессных стратегий и не имеет одного ярко выраженного паттерна."
    )


def build_pm_cluster_names(result_df: pd.DataFrame, cluster_profiles: pd.DataFrame) -> pd.DataFrame:
    global_means = {
        col: _safe_mean(result_df, col)
        for col in result_df.select_dtypes(include="number").columns
        if col != "cluster"
    }

    cluster_sizes = (
        result_df.groupby("cluster")
        .size()
        .reset_index(name="cluster_size")
    )

    profiles_with_size = cluster_profiles.merge(cluster_sizes, on="cluster", how="left")

    rows = []
    for _, row in profiles_with_size.iterrows():
        cluster_id = int(row["cluster"])
        suggested_name, description = suggest_pm_cluster_name(row, global_means)

        rows.append({
            "cluster": cluster_id,
            "cluster_size": int(row["cluster_size"]),
            "suggested_name": suggested_name,
            "description": description,
        })

    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)