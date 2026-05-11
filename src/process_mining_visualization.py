import math

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_process_map(
    transitions_df: pd.DataFrame,
    activity_freq_df: pd.DataFrame,
    top_transitions: int = 30,
    min_transition_count: int = 1,
):
    """
    Обычная process map / Directly-Follows Graph.

    Узлы — действия.
    Линии — переходы.
    Размер узла — частота действия.
    """
    if transitions_df is None or transitions_df.empty:
        return go.Figure()

    filtered_transitions = transitions_df[
        transitions_df["count"] >= min_transition_count
    ].head(top_transitions).copy()

    if filtered_transitions.empty:
        return go.Figure()

    activities = sorted(
        set(filtered_transitions["source"]).union(
            set(filtered_transitions["target"])
        )
    )

    freq_map = {}

    if activity_freq_df is not None and not activity_freq_df.empty:
        freq_map = dict(
            zip(activity_freq_df["activity"], activity_freq_df["count"])
        )

    positions = {}
    n = len(activities)

    for i, activity in enumerate(activities):
        angle = 2 * math.pi * i / max(n, 1)
        positions[activity] = (
            math.cos(angle),
            math.sin(angle),
        )

    fig = go.Figure()

    max_count = filtered_transitions["count"].max()

    for _, row in filtered_transitions.iterrows():
        source = row["source"]
        target = row["target"]
        count = row["count"]

        x0, y0 = positions[source]
        x1, y1 = positions[target]

        width = 1 + 5 * (count / max_count)

        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(width=width),
                hoverinfo="text",
                text=f"{source} → {target}<br>Переходов: {count}",
                showlegend=False,
            )
        )

    node_x = []
    node_y = []
    node_text = []
    node_size = []

    max_freq = max(freq_map.values()) if freq_map else 1

    for activity in activities:
        x, y = positions[activity]

        freq = freq_map.get(activity, 1)

        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{activity}<br>Событий: {freq}")
        node_size.append(20 + 40 * (freq / max_freq))

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=activities,
            textposition="top center",
            marker=dict(
                size=node_size,
                line=dict(width=1),
            ),
            hovertext=node_text,
            hoverinfo="text",
            showlegend=False,
        )
    )

    fig.update_layout(
        title="Process map: наиболее частые переходы",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=700,
    )

    return fig


def plot_transition_heatmap(
    transition_matrix_df: pd.DataFrame,
    max_activities: int = 20,
):
    if transition_matrix_df is None or transition_matrix_df.empty:
        return go.Figure()

    activity_scores = (
        transition_matrix_df.sum(axis=1)
        + transition_matrix_df.sum(axis=0)
    ).sort_values(ascending=False)

    top_activities = activity_scores.head(max_activities).index.tolist()

    matrix = transition_matrix_df.loc[
        transition_matrix_df.index.intersection(top_activities),
        transition_matrix_df.columns.intersection(top_activities),
    ]

    fig = px.imshow(
        matrix,
        text_auto=True,
        aspect="auto",
        title="Heatmap переходов между действиями",
    )

    fig.update_layout(
        xaxis_title="Следующее действие",
        yaxis_title="Текущее действие",
        height=700,
    )

    return fig


def plot_top_transitions(
    transitions_df: pd.DataFrame,
    top_n: int = 20,
):
    if transitions_df is None or transitions_df.empty:
        return go.Figure()

    top_df = transitions_df.head(top_n).copy()

    top_df["transition"] = (
        top_df["source"].astype(str)
        + " → "
        + top_df["target"].astype(str)
    )

    fig = px.bar(
        top_df.sort_values("count"),
        x="count",
        y="transition",
        orientation="h",
        title=f"Top-{top_n} наиболее частых переходов",
    )

    fig.update_layout(
        xaxis_title="Число переходов",
        yaxis_title="Переход",
        height=max(500, top_n * 25),
    )

    return fig


def plot_dotted_chart(
    event_log: pd.DataFrame,
    selected_students=None,
    max_students: int = 80,
):
    """
    Dotted chart:
    X — время,
    Y — студент,
    цвет — действие.

    Можно показать:
    - всех студентов из выбранной области;
    - конкретных выбранных студентов.
    """
    if event_log is None or event_log.empty:
        return go.Figure()

    df = event_log.copy()
    df["student_id"] = df["student_id"].astype(str)

    if selected_students:
        selected_students = [str(student_id) for student_id in selected_students]

        df = df[df["student_id"].isin(selected_students)].copy()
    else:
        selected_students = (
            df["student_id"]
            .drop_duplicates()
            .head(max_students)
            .tolist()
        )

        df = df[df["student_id"].isin(selected_students)].copy()

    if df.empty:
        return go.Figure()

    hover_columns = [
        col for col in ["component", "activity", "context"]
        if col in df.columns
    ]

    fig = px.scatter(
        df,
        x="timestamp",
        y="student_id",
        color="process_activity",
        hover_data=hover_columns,
        title="Dotted chart: активность студентов во времени",
    )

    fig.update_layout(
        xaxis_title="Время",
        yaxis_title="Студент",
        height=max(600, len(selected_students) * 14),
    )

    return fig