import pandas as pd

from src.event_classification import add_event_type_columns


def _join_unique_values(values, limit: int = 8) -> str:
    unique_values = []

    for value in values:
        value_s = str(value).strip()

        if not value_s:
            continue

        if value_s not in unique_values:
            unique_values.append(value_s)

        if len(unique_values) >= limit:
            break

    return "; ".join(unique_values)


def _calculate_auto_event_count_threshold(user_event_counts: pd.Series) -> int:
    """
    Автоматически рассчитывает порог сверхбольшого количества событий.

    Используется IQR-метод:
    threshold = Q3 + 1.5 * IQR
    """
    if user_event_counts.empty:
        return 0

    q1 = user_event_counts.quantile(0.25)
    q3 = user_event_counts.quantile(0.75)
    iqr = q3 - q1

    threshold = q3 + 1.5 * iqr

    return int(round(threshold))


def detect_role_action_anomalies(
    features_df: pd.DataFrame,
    log_df: pd.DataFrame,
    include_context_dependent: bool = False,
) -> pd.DataFrame:
    """
    Выявляет пользователей, которые похожи на преподавателей/администраторов.

    Isolation Forest НЕ используется.

    Пользователь исключается, если:
    1. у него есть жёсткие административные события;
    2. у него автоматически выявлено сверхбольшое количество событий.
    """
    result = features_df.copy()

    if "student_id" not in result.columns:
        raise ValueError("В features_df отсутствует колонка student_id")

    result["student_id"] = result["student_id"].astype(str)

    # Удаляем старые служебные колонки, чтобы merge не создавал total_events_x / total_events_y
    columns_to_drop = [
        "staff_action_anomaly",
        "event_count_anomaly",
        "final_anomaly",
        "staff_action_count",
        "admin_action_count",
        "total_events",
        "event_count_threshold",
        "event_count_q1",
        "event_count_q3",
        "event_count_iqr",
        "staff_action_examples",
        "staff_action_reasons",
        "anomaly_reason",
        "exclude_manual",
        "exclude_final",
    ]

    result = result.drop(
        columns=[col for col in columns_to_drop if col in result.columns],
        errors="ignore",
    )

    empty_columns = {
        "staff_action_anomaly": False,
        "event_count_anomaly": False,
        "final_anomaly": False,
        "staff_action_count": 0,
        "admin_action_count": 0,
        "total_events": 0,
        "event_count_threshold": 0,
        "event_count_q1": 0,
        "event_count_q3": 0,
        "event_count_iqr": 0,
        "staff_action_examples": "",
        "staff_action_reasons": "",
        "anomaly_reason": "",
    }

    if log_df is None or log_df.empty:
        for col, value in empty_columns.items():
            result[col] = value
        return result

    events_df = log_df.copy()

    if "student_id" not in events_df.columns:
        raise ValueError("В log_df отсутствует колонка student_id")

    if "role_event_type" not in events_df.columns:
        events_df = add_event_type_columns(events_df)

    events_df["student_id"] = events_df["student_id"].astype(str)

    # -----------------------------
    # 1. Количество всех событий пользователя
    # -----------------------------
    user_event_counts = (
        events_df.groupby("student_id")
        .size()
        .rename("total_events")
        .reset_index()
    )

    counts_series = user_event_counts["total_events"]

    q1 = counts_series.quantile(0.25)
    q3 = counts_series.quantile(0.75)
    iqr = q3 - q1

    threshold = _calculate_auto_event_count_threshold(counts_series)

    result = result.merge(user_event_counts, on="student_id", how="left")

    result["total_events"] = result["total_events"].fillna(0).astype(int)
    result["event_count_threshold"] = threshold
    result["event_count_q1"] = round(float(q1), 2)
    result["event_count_q3"] = round(float(q3), 2)
    result["event_count_iqr"] = round(float(iqr), 2)

    result["event_count_anomaly"] = result["total_events"] > threshold

    # -----------------------------
    # 2. Жёсткие административные события
    # -----------------------------
    anomaly_events_df = events_df[
        events_df["role_event_type"] == "admin_action"
    ].copy()

    if anomaly_events_df.empty:
        result["staff_action_count"] = 0
        result["admin_action_count"] = 0
        result["staff_action_examples"] = ""
        result["staff_action_reasons"] = ""
        result["staff_action_anomaly"] = False
    else:
        counts = (
            anomaly_events_df.groupby("student_id")
            .size()
            .rename("staff_action_count")
            .reset_index()
        )

        examples = (
            anomaly_events_df.groupby("student_id")["activity"]
            .apply(lambda values: _join_unique_values(values, limit=8))
            .rename("staff_action_examples")
            .reset_index()
        )

        reasons = (
            anomaly_events_df.groupby("student_id")["role_event_reason"]
            .apply(lambda values: _join_unique_values(values, limit=8))
            .rename("staff_action_reasons")
            .reset_index()
        )

        result = result.merge(counts, on="student_id", how="left")
        result = result.merge(examples, on="student_id", how="left")
        result = result.merge(reasons, on="student_id", how="left")

        result["staff_action_count"] = result["staff_action_count"].fillna(0).astype(int)
        result["admin_action_count"] = result["staff_action_count"]

        result["staff_action_examples"] = result["staff_action_examples"].fillna("")
        result["staff_action_reasons"] = result["staff_action_reasons"].fillna("")

        result["staff_action_anomaly"] = result["staff_action_count"] > 0

    # На всякий случай, если административных действий нет у части пользователей
    if "staff_action_count" not in result.columns:
        result["staff_action_count"] = 0

    if "admin_action_count" not in result.columns:
        result["admin_action_count"] = 0

    if "staff_action_examples" not in result.columns:
        result["staff_action_examples"] = ""

    if "staff_action_reasons" not in result.columns:
        result["staff_action_reasons"] = ""

    if "staff_action_anomaly" not in result.columns:
        result["staff_action_anomaly"] = False

    result["staff_action_count"] = result["staff_action_count"].fillna(0).astype(int)
    result["admin_action_count"] = result["admin_action_count"].fillna(0).astype(int)
    result["staff_action_examples"] = result["staff_action_examples"].fillna("")
    result["staff_action_reasons"] = result["staff_action_reasons"].fillna("")
    result["staff_action_anomaly"] = result["staff_action_anomaly"].fillna(False)

    # -----------------------------
    # 3. Итоговая аномалия
    # -----------------------------
    result["final_anomaly"] = (
        result["staff_action_anomaly"]
        | result["event_count_anomaly"]
    )

    def build_reason(row):
        reasons = []

        if row["staff_action_anomaly"]:
            reasons.append(
                "Найдены административные действия. "
                f"Количество: {row['staff_action_count']}. "
                f"Примеры: {row['staff_action_examples']}"
            )

        if row["event_count_anomaly"]:
            reasons.append(
                "Сверхбольшое количество событий. "
                f"У пользователя: {row['total_events']}, "
                f"автоматический порог: {row['event_count_threshold']}. "
                f"Метод: Q3 + 1.5 × IQR"
            )

        return " | ".join(reasons)

    result["anomaly_reason"] = result.apply(
        lambda row: build_reason(row) if row["final_anomaly"] else "",
        axis=1,
    )

    return result


# Совместимость со старыми импортами.
def detect_combined_anomalies(
    features_df: pd.DataFrame,
    log_df: pd.DataFrame,
    contamination: float = 0.05,
    random_state: int = 42,
    count_context_dependent_as_forbidden: bool = False,
) -> pd.DataFrame:
    return detect_role_action_anomalies(
        features_df=features_df,
        log_df=log_df,
    )