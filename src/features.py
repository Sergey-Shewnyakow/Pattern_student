import pandas as pd
import numpy as np


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

    regularity_df["std_events_per_week"] = regularity_df["std_events_per_week"].fillna(0)
    regularity_df["weekly_regularity_cv"] = (
        regularity_df["std_events_per_week"] / regularity_df["avg_events_per_week"]
    ).replace([np.inf, -np.inf], 0).fillna(0)

    # -----------------------------
    # Ночная / выходная активность
    # -----------------------------
    behavior_df = (
        temp.groupby("student_id")
        .apply(
            lambda x: pd.Series({
                "night_activity_ratio": ((x["hour"] >= 0) & (x["hour"] < 6)).mean(),
                "weekend_activity_ratio": (x["weekday"] >= 5).mean(),
            })
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
            lambda x: pd.Series({
                "long_pauses_over_1d": (x["gap_between_events_min"] > 1440).sum(),
                "long_pauses_over_3d": (x["gap_between_events_min"] > 4320).sum(),
            })
        )
        .reset_index()
    )

    # -----------------------------
    # Объединение
    # -----------------------------
    features_df = base_df.merge(session_df, on="student_id", how="left")
    features_df = features_df.merge(gap_df, on="student_id", how="left")
    features_df = features_df.merge(regularity_df, on="student_id", how="left")
    features_df = features_df.merge(behavior_df, on="student_id", how="left")
    features_df = features_df.merge(repeat_df, on="student_id", how="left")
    features_df = features_df.merge(pause_df, on="student_id", how="left")

    numeric_cols = features_df.select_dtypes(include="number").columns
    features_df[numeric_cols] = features_df[numeric_cols].fillna(0)

    return features_df