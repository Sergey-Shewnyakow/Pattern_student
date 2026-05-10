import pandas as pd
import numpy as np


VIDEO_KEYWORDS = [
    "видеолекция",
    "видео",
    "video",
]

LECTURE_KEYWORDS = [
    "лекция",
]

PRACTICE_KEYWORDS = [
    "практическая",
    "практика",
    "практическое",
    "практико-ориентированный",
    "задание",
    "работа представлена",
    "представлен ответ",
    "файл был загружен",
    "ответ в виде файла",
]

TEST_KEYWORDS = [
    "тест",
    "попытка теста",
    "тестирование",
]


def _safe_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().lower()


def _row_text(row: pd.Series) -> str:
    """
    Собирает текст из доступных колонок, чтобы искать тип активности
    не только в activity, но и в component/context.
    """
    parts = []

    for col in ["activity", "component", "context", "event_name", "description"]:
        if col in row.index:
            parts.append(_safe_text(row.get(col)))

    return " ".join(parts)


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def classify_material_type(row: pd.Series) -> str:
    """
    Определяет тип учебной активности по реальной структуре Moodle-лога.

    Для текущего курса:
    - Страница: Видеолекция... -> video
    - Компонент Лекция -> lecture
    - Задание / Ответ в виде файла -> practice
    - Тест -> test
    - Остальные страницы -> page
    - Форум / комментарии -> communication
    - Остальное -> other
    """
    text = _row_text(row)

    component = _safe_text(row.get("component"))
    context = _safe_text(row.get("context"))

    # Видеолекции в твоём логе находятся в контексте:
    # "Страница: Видеолекция..."
    if component == "страница" and "видеолекция" in context:
        return "video"

    if _contains_any(text, VIDEO_KEYWORDS):
        return "video"

    # Интерактивные лекционные материалы Moodle
    if component == "лекция":
        return "lecture"

    if context.startswith("лекция:"):
        return "lecture"

    # Практические задания и отправка файлов
    if component in ["задание", "ответ в виде файла"]:
        return "practice"

    if _contains_any(text, PRACTICE_KEYWORDS):
        return "practice"

    # Тесты
    if component == "тест":
        return "test"

    if context.startswith("тест:"):
        return "test"

    if _contains_any(text, TEST_KEYWORDS):
        return "test"

    # Обычные страницы курса: правила, введение и т.п.
    if component == "страница":
        return "page"

    # Коммуникации
    if component in ["форум", "комментарии к ответу"]:
        return "communication"

    return "other"


def build_material_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Строит признаки по реальным типам активности в курсе:
    - video;
    - lecture;
    - practice;
    - test;
    - page;
    - communication;
    - other.
    """
    temp = df.copy()

    temp["material_type"] = temp.apply(classify_material_type, axis=1)

    material_counts = (
        temp.groupby(["student_id", "material_type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for col in [
        "video",
        "lecture",
        "practice",
        "test",
        "page",
        "communication",
        "other",
    ]:
        if col not in material_counts.columns:
            material_counts[col] = 0

    material_counts = material_counts.rename(
        columns={
            "video": "video_events",
            "lecture": "lecture_events",
            "practice": "practice_events",
            "test": "test_events",
            "page": "page_events",
            "communication": "communication_events",
            "other": "other_material_events",
        }
    )

    material_counts["study_material_events"] = (
        material_counts["video_events"]
        + material_counts["lecture_events"]
        + material_counts["page_events"]
    )

    material_counts["control_activity_events"] = (
        material_counts["practice_events"]
        + material_counts["test_events"]
    )

    material_counts["learning_related_events"] = (
        material_counts["video_events"]
        + material_counts["lecture_events"]
        + material_counts["practice_events"]
        + material_counts["test_events"]
        + material_counts["page_events"]
    )

    denominator = material_counts["learning_related_events"].replace(0, np.nan)

    material_counts["video_share"] = (
        material_counts["video_events"] / denominator
    ).fillna(0)

    material_counts["lecture_share"] = (
        material_counts["lecture_events"] / denominator
    ).fillna(0)

    material_counts["practice_share"] = (
        material_counts["practice_events"] / denominator
    ).fillna(0)

    material_counts["test_share"] = (
        material_counts["test_events"] / denominator
    ).fillna(0)

    material_counts["page_share"] = (
        material_counts["page_events"] / denominator
    ).fillna(0)

    material_counts["control_activity_share"] = (
        material_counts["control_activity_events"] / denominator
    ).fillna(0)

    material_counts["study_material_share"] = (
        material_counts["study_material_events"] / denominator
    ).fillna(0)

    material_counts["used_video"] = (
        material_counts["video_events"] > 0
    ).astype(int)

    material_counts["used_lecture"] = (
        material_counts["lecture_events"] > 0
    ).astype(int)

    material_counts["used_practice"] = (
        material_counts["practice_events"] > 0
    ).astype(int)

    material_counts["used_test"] = (
        material_counts["test_events"] > 0
    ).astype(int)

    material_counts["used_page"] = (
        material_counts["page_events"] > 0
    ).astype(int)

    material_counts["used_study_materials"] = (
        material_counts["study_material_events"] > 0
    ).astype(int)

    material_counts["used_control_activities"] = (
        material_counts["control_activity_events"] > 0
    ).astype(int)

    material_counts["material_diversity_count"] = (
        material_counts["used_video"]
        + material_counts["used_lecture"]
        + material_counts["used_page"]
    )

    material_counts["full_course_activity"] = (
        (material_counts["used_video"] == 1)
        & (material_counts["used_lecture"] == 1)
        & (material_counts["used_practice"] == 1)
        & (material_counts["used_test"] == 1)
    ).astype(int)

    material_counts["practice_test_without_materials"] = (
        (material_counts["control_activity_events"] > 0)
        & (material_counts["study_material_events"] == 0)
    ).astype(int)

    return material_counts


def build_student_features(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()

    temp["date"] = temp["timestamp"].dt.date
    temp["hour"] = temp["timestamp"].dt.hour
    temp["weekday"] = temp["timestamp"].dt.weekday
    temp["week"] = temp["timestamp"].dt.isocalendar().week.astype(int)

    # -----------------------------
    # Базовые признаки
    # -----------------------------
    agg_dict = {
        "total_events": ("activity", "count"),
        "active_days": ("date", "nunique"),
        "active_weeks": ("week", "nunique"),
        "unique_activities": ("activity", "nunique"),
    }

    if "context" in temp.columns:
        agg_dict["unique_contexts"] = ("context", "nunique")

    if "component" in temp.columns:
        agg_dict["unique_components"] = ("component", "nunique")

    base_df = temp.groupby("student_id").agg(**agg_dict).reset_index()

    # -----------------------------
    # Признаки по сессиям
    # -----------------------------
    session_stats = (
        temp.groupby(["student_id", "session_id"])
        .agg(
            session_start=("timestamp", "min"),
            session_end=("timestamp", "max"),
            events_in_session=("activity", "count"),
        )
        .reset_index()
    )

    session_stats["session_duration_min"] = (
        session_stats["session_end"] - session_stats["session_start"]
    ).dt.total_seconds() / 60

    session_df = (
        session_stats.groupby("student_id")
        .agg(
            sessions_count=("session_id", "nunique"),
            avg_session_length_min=("session_duration_min", "mean"),
            median_session_length_min=("session_duration_min", "median"),
            max_session_length_min=("session_duration_min", "max"),
            avg_events_per_session=("events_in_session", "mean"),
            max_events_per_session=("events_in_session", "max"),
        )
        .reset_index()
    )

    # -----------------------------
    # Интервалы между событиями
    # -----------------------------
    temp["prev_timestamp"] = temp.groupby("student_id")["timestamp"].shift(1)

    temp["gap_between_events_min"] = (
        (temp["timestamp"] - temp["prev_timestamp"]).dt.total_seconds() / 60
    )

    gap_df = (
        temp.groupby("student_id")
        .agg(
            avg_gap_between_events_min=("gap_between_events_min", "mean"),
            median_gap_between_events_min=("gap_between_events_min", "median"),
            max_gap_between_events_min=("gap_between_events_min", "max"),
        )
        .reset_index()
    )

    # -----------------------------
    # Регулярность по неделям
    # -----------------------------
    weekly_activity = (
        temp.groupby(["student_id", "week"])
        .size()
        .reset_index(name="events_per_week")
    )

    regularity_df = (
        weekly_activity.groupby("student_id")
        .agg(
            avg_events_per_week=("events_per_week", "mean"),
            std_events_per_week=("events_per_week", "std"),
        )
        .reset_index()
    )

    regularity_df["std_events_per_week"] = regularity_df[
        "std_events_per_week"
    ].fillna(0)

    regularity_df["weekly_regularity_cv"] = (
        regularity_df["std_events_per_week"]
        / regularity_df["avg_events_per_week"]
    ).replace([np.inf, -np.inf], 0).fillna(0)

    # -----------------------------
    # Ночная / выходная активность
    # -----------------------------
    behavior_df = (
        temp.groupby("student_id")
        .apply(
            lambda x: pd.Series(
                {
                    "night_activity_ratio": ((x["hour"] >= 0) & (x["hour"] < 6)).mean(),
                    "weekend_activity_ratio": (x["weekday"] >= 5).mean(),
                }
            )
        )
        .reset_index()
    )

    # -----------------------------
    # Повторы действий
    # -----------------------------
    activity_counts = (
        temp.groupby(["student_id", "activity"])
        .size()
        .reset_index(name="activity_count")
    )

    repeat_df = (
        activity_counts.groupby("student_id")
        .agg(
            repeated_activities_count=("activity_count", lambda x: (x > 1).sum()),
            avg_activity_repeats=("activity_count", "mean"),
            max_activity_repeats=("activity_count", "max"),
        )
        .reset_index()
    )

    # -----------------------------
    # Длинные паузы между активностями
    # -----------------------------
    pause_df = (
        temp.groupby("student_id")
        .apply(
            lambda x: pd.Series(
                {
                    "long_pauses_over_1d": (x["gap_between_events_min"] > 1440).sum(),
                    "long_pauses_over_3d": (x["gap_between_events_min"] > 4320).sum(),
                }
            )
        )
        .reset_index()
    )

    # -----------------------------
    # Признаки по реальным типам действий курса
    # -----------------------------
    material_df = build_material_features(temp)

    # -----------------------------
    # Объединение
    # -----------------------------
    features_df = base_df.merge(session_df, on="student_id", how="left")
    features_df = features_df.merge(gap_df, on="student_id", how="left")
    features_df = features_df.merge(regularity_df, on="student_id", how="left")
    features_df = features_df.merge(behavior_df, on="student_id", how="left")
    features_df = features_df.merge(repeat_df, on="student_id", how="left")
    features_df = features_df.merge(pause_df, on="student_id", how="left")
    features_df = features_df.merge(material_df, on="student_id", how="left")

    numeric_cols = features_df.select_dtypes(include="number").columns
    features_df[numeric_cols] = features_df[numeric_cols].fillna(0)

    return features_df