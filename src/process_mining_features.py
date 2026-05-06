import pandas as pd


def _count_backtracks(activity_list: list[str]) -> int:
    """
    Считает число возвратов к уже встречавшейся активности
    после перехода на другую активность.
    """
    backtracks = 0
    seen = set()

    for i, activity in enumerate(activity_list):
        if i == 0:
            seen.add(activity)
            continue

        prev_activity = activity_list[i - 1]

        if activity in seen and activity != prev_activity:
            backtracks += 1

        seen.add(activity)

    return backtracks


def _count_self_loops(activity_list: list[str]) -> int:
    """
    Считает число повторов вида A -> A.
    """
    if len(activity_list) < 2:
        return 0

    count = 0
    for i in range(1, len(activity_list)):
        if activity_list[i] == activity_list[i - 1]:
            count += 1
    return count


def _extract_transitions(activity_list: list[str]) -> list[str]:
    """
    Возвращает список переходов вида A -> B.
    """
    if len(activity_list) < 2:
        return []

    transitions = []
    for i in range(1, len(activity_list)):
        transitions.append(f"{activity_list[i - 1]} -> {activity_list[i]}")
    return transitions


def build_trace_preview(df: pd.DataFrame, max_students: int = 50) -> pd.DataFrame:
    """
    Формирует предпросмотр траекторий студентов.
    """
    trace_df = (
        df.groupby("student_id")
        .agg(
            trace=("activity", lambda x: " | ".join(x.astype(str).tolist())),
            trace_length=("activity", "count"),
            start_time=("timestamp", "min"),
            end_time=("timestamp", "max"),
        )
        .reset_index()
        .sort_values("trace_length", ascending=False)
        .head(max_students)
    )

    return trace_df


def build_variant_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Формирует таблицу вариантов траекторий.
    """
    variant_df = (
        df.groupby("student_id")
        .agg(
            trace_tuple=("activity", lambda x: tuple(x.astype(str).tolist()))
        )
        .reset_index()
    )

    variant_freq = (
        variant_df.groupby("trace_tuple")
        .size()
        .reset_index(name="variant_frequency")
        .sort_values("variant_frequency", ascending=False)
        .reset_index(drop=True)
    )

    variant_freq["variant_id"] = ["V" + str(i + 1) for i in range(len(variant_freq))]

    variant_df = variant_df.merge(variant_freq, on="trace_tuple", how="left")
    variant_df["trace"] = variant_df["trace_tuple"].apply(lambda x: " | ".join(x))

    return variant_df[["student_id", "variant_id", "variant_frequency", "trace"]]


def build_process_mining_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Строит process mining-признаки на уровне студента.
    Возвращает:
    - pm_features_df
    - variants_df
    """
    temp = df.copy()
    temp = temp.sort_values(["student_id", "timestamp"]).reset_index(drop=True)

    temp["next_activity"] = temp.groupby("student_id")["activity"].shift(-1)
    temp["next_timestamp"] = temp.groupby("student_id")["timestamp"].shift(-1)

    temp["time_to_next_step_min"] = (
        (temp["next_timestamp"] - temp["timestamp"]).dt.total_seconds() / 60
    )

    trace_rows = []

    for student_id, group in temp.groupby("student_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)

        activity_list = group["activity"].astype(str).tolist()
        timestamps = group["timestamp"].tolist()

        transitions = _extract_transitions(activity_list)
        unique_transitions = set(transitions)

        trace_length = len(activity_list)
        trace_unique_activities = len(set(activity_list))
        trace_duration_hours = 0.0

        if len(timestamps) >= 2:
            trace_duration_hours = (
                (timestamps[-1] - timestamps[0]).total_seconds() / 3600
            )

        self_loop_count = _count_self_loops(activity_list)
        backtrack_count = _count_backtracks(activity_list)

        repeated_steps = trace_length - trace_unique_activities
        rework_ratio = repeated_steps / trace_length if trace_length > 0 else 0.0

        unique_transitions_count = len(unique_transitions)
        transition_count = len(transitions)

        linearity_score = 0.0
        if transition_count > 0:
            linearity_score = unique_transitions_count / transition_count

        avg_time_between_steps = group["time_to_next_step_min"].dropna().mean()
        median_time_between_steps = group["time_to_next_step_min"].dropna().median()
        max_time_between_steps = group["time_to_next_step_min"].dropna().max()

        if pd.isna(avg_time_between_steps):
            avg_time_between_steps = 0.0
        if pd.isna(median_time_between_steps):
            median_time_between_steps = 0.0
        if pd.isna(max_time_between_steps):
            max_time_between_steps = 0.0

        first_activity = activity_list[0] if trace_length > 0 else ""
        last_activity = activity_list[-1] if trace_length > 0 else ""

        path_complexity = (
            trace_unique_activities
            + self_loop_count
            + backtrack_count
            + (1 - linearity_score if transition_count > 0 else 0)
        )

        trace_rows.append({
            "student_id": student_id,
            "trace_length": trace_length,
            "trace_unique_activities": trace_unique_activities,
            "trace_duration_hours": trace_duration_hours,
            "transition_count": transition_count,
            "unique_transitions": unique_transitions_count,
            "self_loop_count": self_loop_count,
            "backtrack_count": backtrack_count,
            "rework_ratio": rework_ratio,
            "avg_time_between_steps": avg_time_between_steps,
            "median_time_between_steps": median_time_between_steps,
            "max_time_between_steps": max_time_between_steps,
            "linearity_score": linearity_score,
            "path_complexity": path_complexity,
            "first_activity": first_activity,
            "last_activity": last_activity,
        })

    pm_features_df = pd.DataFrame(trace_rows)

    variants_df = build_variant_table(temp)

    pm_features_df = pm_features_df.merge(
        variants_df[["student_id", "variant_id", "variant_frequency"]],
        on="student_id",
        how="left"
    )

    max_variant_frequency = pm_features_df["variant_frequency"].max()
    if pd.isna(max_variant_frequency):
        max_variant_frequency = 0

    pm_features_df["is_top_variant"] = (
        pm_features_df["variant_frequency"] == max_variant_frequency
    ).astype(int)

    return pm_features_df, variants_df