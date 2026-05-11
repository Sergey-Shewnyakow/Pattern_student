import pandas as pd


def calculate_process_metrics(
    event_log: pd.DataFrame,
    case_id_col: str = "student_id",
    activity_col: str = "process_activity",
    timestamp_col: str = "timestamp",
) -> dict:
    """
    Считает основные метрики process mining.
    """
    if event_log is None or event_log.empty:
        return {
            "case_count": 0,
            "events_count": 0,
            "unique_activities_count": 0,
            "variants_count": 0,
            "most_common_variant_count": 0,
            "most_common_variant_share": 0,
            "avg_trace_length": 0,
            "median_trace_length": 0,
            "avg_unique_activities": 0,
            "avg_rework_count": 0,
            "avg_self_loops_count": 0,
            "avg_linearity": 0,
            "avg_complexity": 0,
        }

    traces = []

    for case_id, group in event_log.sort_values(
        [case_id_col, timestamp_col]
    ).groupby(case_id_col):
        activities = group[activity_col].astype(str).tolist()

        if not activities:
            continue

        trace_length = len(activities)
        unique_activities = len(set(activities))

        transitions = list(zip(activities[:-1], activities[1:]))

        self_loops = sum(
            1 for source, target in transitions if source == target
        )

        rework_count = trace_length - unique_activities

        linearity = unique_activities / trace_length if trace_length > 0 else 0

        complexity = len(set(transitions)) + unique_activities

        traces.append(
            {
                "case_id": case_id,
                "variant": tuple(activities),
                "trace_length": trace_length,
                "unique_activities": unique_activities,
                "self_loops": self_loops,
                "rework_count": rework_count,
                "linearity": linearity,
                "complexity": complexity,
            }
        )

    traces_df = pd.DataFrame(traces)

    if traces_df.empty:
        return {
            "case_count": 0,
            "events_count": len(event_log),
            "unique_activities_count": event_log[activity_col].nunique(),
            "variants_count": 0,
            "most_common_variant_count": 0,
            "most_common_variant_share": 0,
            "avg_trace_length": 0,
            "median_trace_length": 0,
            "avg_unique_activities": 0,
            "avg_rework_count": 0,
            "avg_self_loops_count": 0,
            "avg_linearity": 0,
            "avg_complexity": 0,
        }

    variant_counts = traces_df["variant"].value_counts()

    most_common_variant_count = int(variant_counts.iloc[0])

    case_count = int(traces_df["case_id"].nunique())

    return {
        "case_count": case_count,
        "events_count": int(len(event_log)),
        "unique_activities_count": int(event_log[activity_col].nunique()),
        "variants_count": int(traces_df["variant"].nunique()),
        "most_common_variant_count": most_common_variant_count,
        "most_common_variant_share": (
            most_common_variant_count / case_count if case_count > 0 else 0
        ),
        "avg_trace_length": float(traces_df["trace_length"].mean()),
        "median_trace_length": float(traces_df["trace_length"].median()),
        "avg_unique_activities": float(traces_df["unique_activities"].mean()),
        "avg_rework_count": float(traces_df["rework_count"].mean()),
        "avg_self_loops_count": float(traces_df["self_loops"].mean()),
        "avg_linearity": float(traces_df["linearity"].mean()),
        "avg_complexity": float(traces_df["complexity"].mean()),
    }


def calculate_directly_follows(
    event_log: pd.DataFrame,
    case_id_col: str = "student_id",
    activity_col: str = "process_activity",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Считает directly-follows переходы A -> B.
    """
    if event_log is None or event_log.empty:
        return pd.DataFrame(columns=["source", "target", "count", "share"])

    df = event_log.sort_values([case_id_col, timestamp_col]).copy()

    df["next_activity"] = df.groupby(case_id_col)[activity_col].shift(-1)

    transitions_df = df[df["next_activity"].notna()].copy()

    transitions_df = (
        transitions_df.groupby([activity_col, "next_activity"])
        .size()
        .reset_index(name="count")
        .rename(
            columns={
                activity_col: "source",
                "next_activity": "target",
            }
        )
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    total_count = transitions_df["count"].sum()

    transitions_df["share"] = (
        transitions_df["count"] / total_count if total_count > 0 else 0
    )

    return transitions_df


def calculate_activity_frequencies(
    event_log: pd.DataFrame,
    activity_col: str = "process_activity",
) -> pd.DataFrame:
    """
    Частоты действий.
    """
    if event_log is None or event_log.empty:
        return pd.DataFrame(columns=["activity", "count", "share"])

    freq_df = (
        event_log[activity_col]
        .value_counts()
        .reset_index()
    )

    freq_df.columns = ["activity", "count"]

    total = freq_df["count"].sum()

    freq_df["share"] = freq_df["count"] / total if total > 0 else 0

    return freq_df


def calculate_variants(
    event_log: pd.DataFrame,
    case_id_col: str = "student_id",
    activity_col: str = "process_activity",
    timestamp_col: str = "timestamp",
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Считает наиболее частые варианты траекторий.
    """
    if event_log is None or event_log.empty:
        return pd.DataFrame(columns=["variant", "count", "share"])

    variants = []

    for _, group in event_log.sort_values(
        [case_id_col, timestamp_col]
    ).groupby(case_id_col):
        activities = group[activity_col].astype(str).tolist()
        variants.append(" → ".join(activities))

    variants_df = pd.Series(variants).value_counts().reset_index()

    variants_df.columns = ["variant", "count"]

    total_cases = variants_df["count"].sum()

    variants_df["share"] = (
        variants_df["count"] / total_cases if total_cases > 0 else 0
    )

    return variants_df.head(top_n)


def calculate_transition_matrix(
    transitions_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Строит матрицу переходов source x target.
    """
    if transitions_df is None or transitions_df.empty:
        return pd.DataFrame()

    matrix_df = transitions_df.pivot_table(
        index="source",
        columns="target",
        values="count",
        fill_value=0,
        aggfunc="sum",
    )

    return matrix_df


def prepare_step_sankey_data(
    event_log: pd.DataFrame,
    max_steps: int = 6,
    top_actions_per_step: int = 8,
    case_id_col: str = "student_id",
    activity_col: str = "process_activity",
    timestamp_col: str = "timestamp",
) -> dict:
    """
    Готовит данные для Sankey по первым N шагам траектории.

    Редкие действия на каждом шаге объединяются в "Прочее".
    """
    if event_log is None or event_log.empty:
        return {
            "labels": [],
            "sources": [],
            "targets": [],
            "values": [],
        }

    rows = []

    for case_id, group in event_log.sort_values(
        [case_id_col, timestamp_col]
    ).groupby(case_id_col):
        activities = group[activity_col].astype(str).tolist()[:max_steps]

        for step_index, activity in enumerate(activities):
            rows.append(
                {
                    "case_id": case_id,
                    "step": step_index + 1,
                    "activity": activity,
                }
            )

    steps_df = pd.DataFrame(rows)

    if steps_df.empty:
        return {
            "labels": [],
            "sources": [],
            "targets": [],
            "values": [],
        }

    # Оставляем top-N действий на каждом шаге, остальное -> Прочее
    cleaned_steps = []

    for step, group in steps_df.groupby("step"):
        top_actions = (
            group["activity"]
            .value_counts()
            .head(top_actions_per_step)
            .index
            .tolist()
        )

        temp = group.copy()
        temp["activity_clean"] = temp["activity"].where(
            temp["activity"].isin(top_actions),
            "Прочее",
        )

        cleaned_steps.append(temp)

    steps_df = pd.concat(cleaned_steps, ignore_index=True)

    step_wide = steps_df.pivot_table(
        index="case_id",
        columns="step",
        values="activity_clean",
        aggfunc="first",
    )

    labels = []
    label_to_index = {}

    def get_label_index(step: int, activity: str) -> int:
        label = f"{step}. {activity}"

        if label not in label_to_index:
            label_to_index[label] = len(labels)
            labels.append(label)

        return label_to_index[label]

    sources = []
    targets = []
    values = []

    link_counts = {}

    for _, row in step_wide.iterrows():
        for step in range(1, max_steps):
            if step not in row.index or step + 1 not in row.index:
                continue

            source_activity = row.get(step)
            target_activity = row.get(step + 1)

            if pd.isna(source_activity) or pd.isna(target_activity):
                continue

            source_idx = get_label_index(step, source_activity)
            target_idx = get_label_index(step + 1, target_activity)

            key = (source_idx, target_idx)

            link_counts[key] = link_counts.get(key, 0) + 1

    for (source_idx, target_idx), count in link_counts.items():
        sources.append(source_idx)
        targets.append(target_idx)
        values.append(count)

    return {
        "labels": labels,
        "sources": sources,
        "targets": targets,
        "values": values,
    }


def compare_process_metrics_by_cluster(
    event_log: pd.DataFrame,
    clustered_students_df: pd.DataFrame,
    cluster_col: str = "cluster",
    cluster_name_col: str = "suggested_name",
) -> pd.DataFrame:
    """
    Считает process mining метрики по каждому кластеру.
    """
    if event_log is None or event_log.empty:
        return pd.DataFrame()

    if clustered_students_df is None or clustered_students_df.empty:
        return pd.DataFrame()

    students_df = clustered_students_df.copy()
    students_df["student_id"] = students_df["student_id"].astype(str)

    event_log = event_log.copy()
    event_log["student_id"] = event_log["student_id"].astype(str)

    merged_log = event_log.merge(
        students_df[["student_id", cluster_col, cluster_name_col]],
        on="student_id",
        how="inner",
    )

    rows = []

    for cluster_id, group in merged_log.groupby(cluster_col):
        metrics = calculate_process_metrics(group)

        cluster_name_values = group[cluster_name_col].dropna().unique()

        cluster_name = (
            cluster_name_values[0]
            if len(cluster_name_values) > 0
            else str(cluster_id)
        )

        row = {
            "cluster": cluster_id,
            "cluster_name": cluster_name,
        }

        row.update(metrics)

        rows.append(row)

    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)