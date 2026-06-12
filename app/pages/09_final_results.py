import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.state import init_session_state
from src.ui_styles import apply_global_styles

try:
    from src.cluster_naming import build_cluster_names
except Exception:
    build_cluster_names = None

try:
    from src.cluster_name_editor import apply_custom_cluster_names
except Exception:
    apply_custom_cluster_names = None


# ============================================================
# Настройка страницы
# ============================================================

st.set_page_config(
    page_title="Final Results",
    layout="wide",
)

init_session_state()
apply_global_styles()

st.title("Итоговые паттерны поведения студентов")




# ============================================================
# Дополнительный CSS для широкого отображения
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 100% !important;
        }

        div[data-testid="stVerticalBlock"] {
            gap: 0.8rem;
        }

        .pattern-card {
            padding: 1.1rem;
            border-radius: 18px;
            border: 1px solid rgba(49, 51, 63, 0.15);
            background: linear-gradient(180deg, #ffffff 0%, #f7f8fa 100%);
            box-shadow: 0 4px 18px rgba(0,0,0,0.05);
            margin-bottom: 0.75rem;
        }

        .pattern-title {
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }

        .pattern-subtitle {
            color: #555;
            font-size: 0.95rem;
            margin-bottom: 0.4rem;
        }

        .pattern-badge {
            display: inline-block;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            background: #eef2ff;
            color: #1f3a8a;
            font-size: 0.82rem;
            font-weight: 600;
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
        }

        .risk-badge {
            display: inline-block;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            background: #fff1f2;
            color: #9f1239;
            font-size: 0.82rem;
            font-weight: 600;
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
        }

        .good-badge {
            display: inline-block;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            background: #ecfdf5;
            color: #065f46;
            font-size: 0.82rem;
            font-weight: 600;
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Базовые функции
# ============================================================

def safe_text(value, default="Нет данных") -> str:
    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except Exception:
        pass

    text = str(value).strip()

    if text == "":
        return default

    return text


def short_text(value, max_len=80) -> str:
    text = safe_text(value, "")

    if len(text) <= max_len:
        return text

    return text[:max_len] + "..."


def get_result_from_session(possible_keys):
    for key in possible_keys:
        value = st.session_state.get(key)

        if value is not None:
            return value

    return None


def is_valid_method_result(result) -> bool:
    if result is None:
        return False

    if not isinstance(result, dict):
        return False

    result_df = result.get("result_df")

    if result_df is None:
        return False

    if not isinstance(result_df, pd.DataFrame):
        return False

    if result_df.empty:
        return False

    required_cols = {"student_id", "cluster"}

    return required_cols.issubset(set(result_df.columns))


def make_student_label(student_id) -> str:
    """
    Подпись студента в selectbox.
    Ничего не сокращаем, чтобы полностью отображалась фамилия / ФИО.
    """
    student_id = str(student_id).strip()

    if student_id == "":
        return "Студент без идентификатора"

    return student_id


def percent_text(value) -> str:
    try:
        if pd.isna(value):
            return "—"
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return "—"


def number_text(value, digits=2) -> str:
    try:
        if pd.isna(value):
            return "—"

        value = float(value)

        if value.is_integer():
            return str(int(value))

        return f"{value:.{digits}f}"
    except Exception:
        return "—"


# ============================================================
# Результаты методов кластеризации
# ============================================================

METHOD_RESULTS = {
    "KMeans": get_result_from_session(["clustering_result", "kmeans_result"]),
    "Agglomerative": get_result_from_session(["agglomerative_result"]),
    "GMM": get_result_from_session(["gmm_result"]),
    "HDBSCAN": get_result_from_session(["hdbscan_result"]),
    "DEC": get_result_from_session(["dec_result", "deep_embedding_result"]),
    "Autoencoder": get_result_from_session(
        ["autoencoder_result", "autoencoder_clustering_result"]
    ),
}

METHOD_KEY_MAP = {
    "KMeans": "kmeans",
    "Agglomerative": "agglomerative",
    "GMM": "gmm",
    "HDBSCAN": "hdbscan",
    "DEC": "dec",
    "Autoencoder": "autoencoder",
}

available_methods = [
    method_name
    for method_name, result in METHOD_RESULTS.items()
    if is_valid_method_result(result)
]

if not available_methods:
    st.warning(
        "Нет результатов кластеризации. Сначала запустите хотя бы один метод: "
        "KMeans, Agglomerative, GMM, HDBSCAN, DEC или Autoencoder."
    )
    st.stop()


# ============================================================
# Названия кластеров
# ============================================================

def get_cluster_names_for_method(method_name: str, result: dict) -> pd.DataFrame:
    if not is_valid_method_result(result):
        return pd.DataFrame()

    result_df = result.get("result_df").copy()
    result_df["student_id"] = result_df["student_id"].astype(str)

    cluster_profiles = result.get("cluster_profiles")

    if cluster_profiles is not None and isinstance(cluster_profiles, pd.DataFrame):
        cluster_profiles = cluster_profiles.copy()
    else:
        cluster_profiles = pd.DataFrame()

    names_df = pd.DataFrame()

    if build_cluster_names is not None and not cluster_profiles.empty:
        try:
            non_noise_result_df = result_df[result_df["cluster"] != -1].copy()

            if not non_noise_result_df.empty:
                names_df = build_cluster_names(
                    result_df=non_noise_result_df,
                    cluster_profiles=cluster_profiles,
                )
        except Exception:
            names_df = pd.DataFrame()

    if names_df.empty:
        names_df = (
            result_df.groupby("cluster")
            .agg(cluster_size=("student_id", "nunique"))
            .reset_index()
        )

        names_df["suggested_name"] = names_df["cluster"].apply(
            lambda x: (
                "Шумовые / нетипичные студенты"
                if x == -1
                else f"Кластер {x}"
            )
        )

        names_df["description"] = names_df["cluster"].apply(
            lambda x: (
                "Студенты, не вошедшие в устойчивую группу."
                if x == -1
                else "Группа студентов со схожими признаками активности."
            )
        )
    else:
        if "cluster_size" not in names_df.columns:
            size_df = (
                result_df.groupby("cluster")
                .agg(cluster_size=("student_id", "nunique"))
                .reset_index()
            )

            names_df = names_df.merge(size_df, on="cluster", how="left")

        if "suggested_name" not in names_df.columns:
            names_df["suggested_name"] = names_df["cluster"].apply(
                lambda x: f"Кластер {x}"
            )

        if "description" not in names_df.columns:
            names_df["description"] = "Группа студентов со схожими признаками активности."

        if -1 in result_df["cluster"].unique() and -1 not in names_df["cluster"].unique():
            noise_count = int((result_df["cluster"] == -1).sum())

            noise_row = pd.DataFrame(
                [
                    {
                        "cluster": -1,
                        "cluster_size": noise_count,
                        "suggested_name": "Шумовые / нетипичные студенты",
                        "description": "Студенты, не вошедшие в устойчивую группу.",
                    }
                ]
            )

            names_df = pd.concat([noise_row, names_df], ignore_index=True)

    if apply_custom_cluster_names is not None:
        try:
            method_key = METHOD_KEY_MAP.get(method_name, method_name.lower())

            names_df = apply_custom_cluster_names(
                method_key=method_key,
                cluster_names_df=names_df,
            )
        except Exception:
            pass

    names_df["suggested_name"] = names_df["suggested_name"].fillna(
        names_df["cluster"].apply(lambda x: f"Кластер {x}")
    )

    names_df["description"] = names_df["description"].fillna(
        "Группа студентов со схожими признаками активности."
    )

    return names_df


def build_method_student_table(method_name: str, result: dict) -> pd.DataFrame:
    if not is_valid_method_result(result):
        return pd.DataFrame()

    result_df = result.get("result_df").copy()
    result_df["student_id"] = result_df["student_id"].astype(str)

    result_df = result_df.drop_duplicates(
        subset=["student_id"],
        keep="first",
    )

    names_df = get_cluster_names_for_method(method_name, result)

    table = result_df[["student_id", "cluster"]].copy()

    if not names_df.empty:
        table = table.merge(
            names_df[["cluster", "suggested_name"]],
            on="cluster",
            how="left",
        )
    else:
        table["suggested_name"] = table["cluster"].apply(
            lambda x: f"Кластер {x}"
        )

    table["suggested_name"] = table["suggested_name"].fillna(
        table["cluster"].apply(lambda x: f"Кластер {x}")
    )

    table = table.rename(
        columns={
            "cluster": f"{method_name}_cluster",
            "suggested_name": f"{method_name}_resource_pattern",
        }
    )

    probability_candidates = [
        "cluster_probability",
        "probability",
        "membership_probability",
        "max_probability",
    ]

    for probability_col in probability_candidates:
        if probability_col in result_df.columns:
            probability_table = result_df[["student_id", probability_col]].copy()
            probability_table = probability_table.rename(
                columns={
                    probability_col: f"{method_name}_cluster_probability"
                }
            )

            table = table.merge(
                probability_table,
                on="student_id",
                how="left",
            )

            break

    return table


def build_ml_patterns_table(selected_methods: list[str]) -> pd.DataFrame:
    tables = []

    for method_name in selected_methods:
        result = METHOD_RESULTS.get(method_name)
        table = build_method_student_table(method_name, result)

        if not table.empty:
            tables.append(table)

    if not tables:
        return pd.DataFrame()

    ml_df = tables[0]

    for table in tables[1:]:
        ml_df = ml_df.merge(
            table,
            on="student_id",
            how="outer",
        )

    ml_df["student_id"] = ml_df["student_id"].astype(str)
    ml_df = ml_df.drop_duplicates(subset=["student_id"], keep="first")

    return ml_df


# ============================================================
# Process mining признаки
# ============================================================

def get_process_features_df() -> pd.DataFrame:
    candidates = [
        "process_behavior_features_df",
        "process_features_df",
        "process_mining_features_df",
        "final_behavior_df",
    ]

    for key in candidates:
        value = st.session_state.get(key)

        if value is None:
            continue

        if not isinstance(value, pd.DataFrame):
            continue

        if value.empty:
            continue

        df = value.copy()

        if "student_id" not in df.columns:
            continue

        df["student_id"] = df["student_id"].astype(str)
        df = df.drop_duplicates(subset=["student_id"], keep="first")

        return df

    return pd.DataFrame()


# ============================================================
# Event log для пути студента
# ============================================================

def get_event_log_from_session() -> pd.DataFrame:
    candidates = [
        "process_event_log",
        "df_human_events",
        "df_sessions",
        "df_clean",
    ]

    for key in candidates:
        value = st.session_state.get(key)

        if value is None:
            continue

        if not isinstance(value, pd.DataFrame):
            continue

        if value.empty:
            continue

        df = value.copy()

        if "student_id" not in df.columns:
            continue

        if "timestamp" not in df.columns:
            continue

        df["student_id"] = df["student_id"].astype(str)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df[df["timestamp"].notna()].copy()

        if df.empty:
            continue

        if "process_activity" not in df.columns:
            if "human_activity" in df.columns:
                df["process_activity"] = df["human_activity"]
            elif "activity" in df.columns:
                df["process_activity"] = df["activity"]
            else:
                df["process_activity"] = "Событие"

        if "activity" not in df.columns:
            df["activity"] = df["process_activity"]

        if "component" not in df.columns:
            df["component"] = "Нет данных"

        if "context" not in df.columns:
            df["context"] = "Нет данных"

        dedup_cols = [
            col for col in [
                "student_id",
                "timestamp",
                "process_activity",
                "component",
                "context",
                "activity",
            ]
            if col in df.columns
        ]

        df = df.drop_duplicates(subset=dedup_cols, keep="first")
        df = df.sort_values(["student_id", "timestamp"]).reset_index(drop=True)

        return df

    return pd.DataFrame()


# ============================================================
# Итоговая таблица студентов
# ============================================================

def build_ml_summary_for_row(row: pd.Series, selected_methods: list[str]) -> pd.Series:
    method_patterns = {}

    for method_name in selected_methods:
        pattern_col = f"{method_name}_resource_pattern"

        if pattern_col in row.index:
            pattern = safe_text(row.get(pattern_col), "")

            if pattern != "":
                method_patterns[method_name] = pattern

    available_patterns = [
        pattern
        for pattern in method_patterns.values()
        if pattern not in ["", "Нет данных"]
    ]

    if not available_patterns:
        return pd.Series(
            {
                "main_ml_pattern": "Нет данных",
                "ml_consensus_status": "Нет данных",
                "ml_signature": "Нет данных",
                "ml_methods_count": 0,
                "ml_unique_patterns_count": 0,
            }
        )

    pattern_counts = pd.Series(available_patterns).value_counts()
    top_count = int(pattern_counts.iloc[0])
    top_patterns = pattern_counts[pattern_counts == top_count].index.tolist()

    if len(top_patterns) == 1:
        main_ml_pattern = top_patterns[0]
    else:
        main_ml_pattern = "Смешанный ресурсный профиль"

    unique_patterns = sorted(set(available_patterns))

    if len(unique_patterns) == 1:
        ml_consensus_status = "Методы МО совпали"
    elif top_count > 1 and len(top_patterns) == 1:
        ml_consensus_status = "Есть преобладающий паттерн МО"
    else:
        ml_consensus_status = "Методы МО различаются"

    ml_signature = " | ".join(
        [
            f"{method}: {pattern}"
            for method, pattern in method_patterns.items()
            if pattern not in ["", "Нет данных"]
        ]
    )

    return pd.Series(
        {
            "main_ml_pattern": main_ml_pattern,
            "ml_consensus_status": ml_consensus_status,
            "ml_signature": ml_signature,
            "ml_methods_count": len(available_patterns),
            "ml_unique_patterns_count": len(unique_patterns),
        }
    )


def build_final_df(
    ml_df: pd.DataFrame,
    process_df: pd.DataFrame,
    selected_methods: list[str],
) -> pd.DataFrame:
    if ml_df is None or ml_df.empty:
        return pd.DataFrame()

    final_df = ml_df.copy()
    final_df["student_id"] = final_df["student_id"].astype(str)
    final_df = final_df.drop_duplicates(subset=["student_id"], keep="first")

    if process_df is not None and not process_df.empty:
        process_df = process_df.copy()
        process_df["student_id"] = process_df["student_id"].astype(str)
        process_df = process_df.drop_duplicates(subset=["student_id"], keep="first")

        process_columns = [
            "student_id",
            "process_pattern",
            "process_flags",
            "process_flags_count",
            "final_behavior_pattern",
            "final_behavior_description",
            "completed_assignments_count",
            "expected_assignments_count",
            "assignment_completion_ratio",
            "completed_assignments_list",
            "missing_assignments_list",
            "completed_tests_count",
            "expected_tests_count",
            "test_completion_ratio",
            "completed_tests_list",
            "missing_tests_list",
            "control_completion_ratio",
            "process_total_events",
            "process_active_days",
            "max_day_activity_ratio",
            "top_2_days_activity_ratio",
            "top_3_days_activity_ratio",
            "days_to_80_percent_events",
            "last_period_events_ratio",
            "fast_lecture_completion_count",
            "measured_lecture_completion_count",
            "fast_lecture_completion_ratio",
            "median_lecture_duration_min",
            "fast_test_completion_count",
            "measured_test_completion_count",
            "fast_test_completion_ratio",
            "median_test_duration_min",
            "fast_assignment_upload_count",
            "measured_assignment_upload_count",
            "fast_assignment_upload_ratio",
            "suspicious_first_assignment_upload_count",
            "measured_first_assignment_upload_count",
            "suspicious_first_assignment_upload_ratio",
            "trace_length",
            "linearity",
            "complexity",
            "returns_count",
            "variant_frequency",
            "required_assignments_count",
            "required_tests_count",
            "required_assignments_list",
            "required_tests_list",
        ]

        process_columns = [
            col for col in process_columns
            if col in process_df.columns
        ]

        final_df = final_df.merge(
            process_df[process_columns],
            on="student_id",
            how="left",
        )
    else:
        final_df["process_pattern"] = "Нет данных process mining"
        final_df["process_flags"] = "Нет данных process mining"
        final_df["process_flags_count"] = 0
        final_df["final_behavior_description"] = "Process mining признаки не рассчитаны."

    ml_summary_df = final_df.apply(
        lambda row: build_ml_summary_for_row(row, selected_methods),
        axis=1,
    )

    final_df = pd.concat([final_df, ml_summary_df], axis=1)

    if "process_pattern" not in final_df.columns:
        final_df["process_pattern"] = "Нет данных process mining"

    final_df["process_pattern"] = final_df["process_pattern"].fillna(
        "Нет данных process mining"
    )

    if "process_flags" not in final_df.columns:
        final_df["process_flags"] = "Нет дополнительных process-признаков"

    final_df["process_flags"] = final_df["process_flags"].fillna(
        "Нет дополнительных process-признаков"
    )

    if "process_flags_count" not in final_df.columns:
        final_df["process_flags_count"] = 0

    final_df["process_flags_count"] = pd.to_numeric(
        final_df["process_flags_count"],
        errors="coerce",
    ).fillna(0).astype(int)

    final_df["main_process_pattern"] = final_df["process_pattern"].astype(str)

    final_df["main_final_pattern"] = (
        final_df["main_ml_pattern"].astype(str)
        + " + "
        + final_df["main_process_pattern"].astype(str)
    )

    final_df["final_pattern_key"] = (
        final_df["main_ml_pattern"].astype(str)
        + " || "
        + final_df["main_process_pattern"].astype(str)
        + " || "
        + final_df["process_flags"].astype(str)
    )

    final_df["final_interpretation"] = final_df.apply(
        lambda row: (
            f"Основной паттерн по методам МО: «{safe_text(row.get('main_ml_pattern'))}». "
            f"Статус согласованности МО: «{safe_text(row.get('ml_consensus_status'))}». "
            f"Основной паттерн по process mining: «{safe_text(row.get('main_process_pattern'))}». "
            f"Дополнительные process-признаки: «{safe_text(row.get('process_flags'))}»."
        ),
        axis=1,
    )

    final_df = final_df.drop_duplicates(subset=["student_id"], keep="first")

    return final_df


def build_unique_patterns_df(final_df: pd.DataFrame) -> pd.DataFrame:
    if final_df is None or final_df.empty:
        return pd.DataFrame()

    temp = final_df.copy()
    temp["student_id"] = temp["student_id"].astype(str)
    temp = temp.drop_duplicates(subset=["student_id"], keep="first")

    group_cols = [
        "main_ml_pattern",
        "ml_consensus_status",
        "main_process_pattern",
        "process_flags",
        "main_final_pattern",
        "final_pattern_key",
    ]

    group_cols = [
        col for col in group_cols
        if col in temp.columns
    ]

    agg_dict = {
        "students_count": ("student_id", "nunique"),
        "students_preview": (
            "student_id",
            lambda s: ", ".join(sorted(set(s.astype(str)))[:20]),
        ),
    }

    optional_metrics = {
        "avg_control_completion": "control_completion_ratio",
        "avg_assignment_completion": "assignment_completion_ratio",
        "avg_test_completion": "test_completion_ratio",
        "avg_events": "process_total_events",
        "avg_active_days": "process_active_days",
        "avg_top_2_days_ratio": "top_2_days_activity_ratio",
        "avg_top_3_days_ratio": "top_3_days_activity_ratio",
        "avg_days_to_80": "days_to_80_percent_events",
        "avg_fast_lecture_ratio": "fast_lecture_completion_ratio",
        "avg_fast_test_ratio": "fast_test_completion_ratio",
        "avg_fast_assignment_upload_ratio": "suspicious_first_assignment_upload_ratio",
        "avg_trace_length": "trace_length",
        "avg_linearity": "linearity",
        "avg_complexity": "complexity",
        "avg_returns": "returns_count",
    }

    for new_col, source_col in optional_metrics.items():
        if source_col in temp.columns:
            agg_dict[new_col] = (source_col, "mean")

    grouped = (
        temp.groupby(group_cols, dropna=False)
        .agg(**agg_dict)
        .reset_index()
        .sort_values("students_count", ascending=False)
        .reset_index(drop=True)
    )

    grouped["pattern_id"] = range(1, len(grouped) + 1)

    grouped["pattern_label"] = grouped.apply(
        lambda row: (
            f"Паттерн {int(row['pattern_id'])}: "
            f"{short_text(row['main_ml_pattern'], 50)} + "
            f"{short_text(row['main_process_pattern'], 50)}"
        ),
        axis=1,
    )

    for col in grouped.columns:
        if col.startswith("avg_"):
            grouped[col] = pd.to_numeric(grouped[col], errors="coerce").round(3)

    return grouped


# ============================================================
# Визуализации
# ============================================================

def plot_fullscreen_pattern_tree(unique_patterns_df: pd.DataFrame, height: int):
    tree_df = unique_patterns_df.copy()

    tree_df["root"] = "Все итоговые паттерны"
    tree_df["ml_node"] = "МО: " + tree_df["main_ml_pattern"].astype(str)
    tree_df["pm_node"] = "Process mining: " + tree_df["main_process_pattern"].astype(str)
    tree_df["flags_node"] = "Признаки: " + tree_df["process_flags"].astype(str)

    fig = px.treemap(
        tree_df,
        path=[
            "root",
            "ml_node",
            "pm_node",
            "flags_node",
        ],
        values="students_count",
        custom_data=[
            "pattern_id",
            "pattern_label",
            "students_count",
            "main_ml_pattern",
            "main_process_pattern",
            "process_flags",
            "ml_consensus_status",
        ],
        title="Дерево итоговых паттернов поведения студентов",
        template="plotly_white",
    )

    fig.update_traces(
        textinfo="label+value+percent parent",
        hovertemplate=(
            "<b>%{label}</b><br><br>"
            "ID паттерна: %{customdata[0]}<br>"
            "Студентов: %{customdata[2]}<br>"
            "Паттерн: %{customdata[1]}<br><br>"
            "МО: %{customdata[3]}<br>"
            "Согласованность МО: %{customdata[6]}<br>"
            "Process mining: %{customdata[4]}<br>"
            "Process-признаки: %{customdata[5]}<br>"
            "<extra></extra>"
        ),
        marker=dict(
            line=dict(width=1.2, color="white")
        ),
    )

    fig.update_layout(
        height=height,
        margin=dict(t=80, l=5, r=5, b=5),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black", size=15),
        title=dict(
            font=dict(size=24),
            x=0.02,
        ),
    )

    return fig


def plot_main_patterns_bar(unique_patterns_df: pd.DataFrame, top_n: int):
    chart_df = unique_patterns_df.head(top_n).copy()
    chart_df = chart_df.sort_values("students_count", ascending=True)

    fig = px.bar(
        chart_df,
        x="students_count",
        y="pattern_label",
        orientation="h",
        color="main_process_pattern",
        hover_data=[
            col for col in [
                "main_ml_pattern",
                "ml_consensus_status",
                "main_process_pattern",
                "process_flags",
                "avg_control_completion",
                "avg_events",
                "avg_active_days",
                "avg_top_2_days_ratio",
                "avg_trace_length",
                "avg_returns",
            ]
            if col in chart_df.columns
        ],
        title="Рейтинг основных итоговых паттернов",
    )

    fig.update_layout(
        height=max(520, top_n * 44),
        xaxis_title="Количество студентов",
        yaxis_title="Итоговый паттерн",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        legend_title_text="Process mining паттерн",
    )

    return fig


def plot_ml_process_heatmap(final_df: pd.DataFrame):
    heat_df = (
        final_df.groupby(["main_ml_pattern", "main_process_pattern"], dropna=False)
        .agg(students_count=("student_id", "nunique"))
        .reset_index()
    )

    matrix = heat_df.pivot_table(
        index="main_ml_pattern",
        columns="main_process_pattern",
        values="students_count",
        fill_value=0,
    )

    fig = px.imshow(
        matrix,
        text_auto=True,
        aspect="auto",
        title="Матрица связи: паттерн МО × process mining паттерн",
    )

    fig.update_layout(
        height=700,
        xaxis_title="Process mining паттерн",
        yaxis_title="Паттерн методов МО",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
    )

    return fig, heat_df


def plot_patterns_sankey(unique_patterns_df: pd.DataFrame):
    if unique_patterns_df.empty:
        return None

    links_1 = unique_patterns_df[
        ["main_ml_pattern", "main_process_pattern", "students_count"]
    ].copy()

    links_1["source"] = "МО: " + links_1["main_ml_pattern"].astype(str)
    links_1["target"] = "PM: " + links_1["main_process_pattern"].astype(str)
    links_1["value"] = links_1["students_count"]

    links_2 = unique_patterns_df[
        ["main_process_pattern", "process_flags", "students_count"]
    ].copy()

    links_2["source"] = "PM: " + links_2["main_process_pattern"].astype(str)
    links_2["target"] = "Признаки: " + links_2["process_flags"].astype(str)
    links_2["value"] = links_2["students_count"]

    links = pd.concat(
        [
            links_1[["source", "target", "value"]],
            links_2[["source", "target", "value"]],
        ],
        ignore_index=True,
    )

    links = (
        links.groupby(["source", "target"], as_index=False)
        .agg(value=("value", "sum"))
    )

    labels = pd.Index(
        pd.concat([links["source"], links["target"]]).unique()
    ).tolist()

    label_to_id = {
        label: idx
        for idx, label in enumerate(labels)
    }

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=24,
                    thickness=19,
                    line=dict(color="black", width=0.3),
                    label=labels,
                ),
                link=dict(
                    source=links["source"].map(label_to_id),
                    target=links["target"].map(label_to_id),
                    value=links["value"],
                ),
            )
        ]
    )

    fig.update_layout(
        title_text="Поток формирования итоговых паттернов",
        height=760,
        font=dict(size=12, color="black"),
        paper_bgcolor="white",
    )

    return fig


def plot_pattern_quality_scatter(unique_patterns_df: pd.DataFrame):
    if unique_patterns_df.empty:
        return None

    x_col = "avg_control_completion" if "avg_control_completion" in unique_patterns_df.columns else None
    y_col = "avg_active_days" if "avg_active_days" in unique_patterns_df.columns else None

    if x_col is None or y_col is None:
        return None

    fig = px.scatter(
        unique_patterns_df,
        x=x_col,
        y=y_col,
        size="students_count",
        color="main_process_pattern",
        hover_name="pattern_label",
        hover_data=[
            col for col in [
                "main_ml_pattern",
                "ml_consensus_status",
                "process_flags",
                "avg_events",
                "avg_top_2_days_ratio",
                "avg_fast_lecture_ratio",
                "avg_fast_test_ratio",
                "avg_returns",
            ]
            if col in unique_patterns_df.columns
        ],
        title="Карта качества паттернов: полнота выполнения × активные дни",
    )

    fig.update_layout(
        height=650,
        xaxis_title="Средняя полнота выполнения контрольных активностей",
        yaxis_title="Среднее число активных дней",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        legend_title_text="Process mining паттерн",
    )

    return fig


def build_pattern_recommendations(pattern_row: pd.Series) -> list[str]:
    recommendations = []

    process_pattern = safe_text(pattern_row.get("main_process_pattern"))
    ml_status = safe_text(pattern_row.get("ml_consensus_status"))
    flags = safe_text(pattern_row.get("process_flags"))

    control_completion = pattern_row.get("avg_control_completion", np.nan)
    top_2_ratio = pattern_row.get("avg_top_2_days_ratio", np.nan)
    active_days = pattern_row.get("avg_active_days", np.nan)
    fast_lecture_ratio = pattern_row.get("avg_fast_lecture_ratio", np.nan)
    fast_test_ratio = pattern_row.get("avg_fast_test_ratio", np.nan)

    if "Неполное выполнение" in process_pattern or (
        pd.notna(control_completion) and control_completion < 0.5
    ):
        recommendations.append(
            "Паттерн связан с неполным выполнением контрольных активностей. "
            "Стоит проверить, какие задания или тесты чаще всего не выполняются, "
            "и добавить напоминания, промежуточные дедлайны или пояснения."
        )

    if "Авральное" in process_pattern or (
        pd.notna(top_2_ratio) and top_2_ratio >= 0.6
    ):
        recommendations.append(
            "Паттерн похож на Сжатое прохождение курса. "
            "Для таких студентов полезны дробные дедлайны и регулярные контрольные точки."
        )

    if "Формальное" in process_pattern or (
        pd.notna(fast_lecture_ratio) and fast_lecture_ratio >= 0.5
    ):
        recommendations.append(
            "Есть признаки формального прохождения лекционных материалов. "
            "Можно усилить связь лекций с заданиями или добавить короткие проверочные вопросы."
        )

    if "Быстрое прохождение тестов" in process_pattern or (
        pd.notna(fast_test_ratio) and fast_test_ratio >= 0.5
    ):
        recommendations.append(
            "Есть признаки слишком быстрого прохождения тестов. "
            "Стоит проверить сложность тестов, число попыток и возможное угадывание ответов."
        )

    if pd.notna(active_days) and active_days <= 2:
        recommendations.append(
            "Активность сконцентрирована в небольшом числе дней. "
            "Это может указывать на нерегулярное обучение или прохождение курса перед дедлайном."
        )

    if ml_status == "Методы МО различаются":
        recommendations.append(
            "Методы машинного обучения дали разные ресурсные паттерны. "
            "Такой профиль стоит рассматривать как смешанный и анализировать вместе с траекторией студента."
        )

    if flags == "Нет выраженных дополнительных признаков":
        recommendations.append(
            "Выраженных рискованных process-признаков нет. "
            "Такой паттерн можно использовать как относительно стабильный сценарий прохождения курса."
        )

    if not recommendations:
        recommendations.append(
            "Паттерн требует содержательной интерпретации по таблице студентов и индивидуальным траекториям."
        )

    return recommendations


def render_pattern_card(pattern_row: pd.Series):
    pattern_id = int(pattern_row["pattern_id"])
    students_count = int(pattern_row["students_count"])

    main_ml_pattern = safe_text(pattern_row.get("main_ml_pattern"))
    main_process_pattern = safe_text(pattern_row.get("main_process_pattern"))
    ml_status = safe_text(pattern_row.get("ml_consensus_status"))
    flags = safe_text(pattern_row.get("process_flags"))

    avg_control = pattern_row.get("avg_control_completion", np.nan)
    avg_events = pattern_row.get("avg_events", np.nan)
    avg_active_days = pattern_row.get("avg_active_days", np.nan)
    avg_top_2 = pattern_row.get("avg_top_2_days_ratio", np.nan)

    badge_class = "good-badge" if flags == "Нет выраженных дополнительных признаков" else "risk-badge"

    st.markdown(
        f"""
        <div class="pattern-card">
            <div class="pattern-title">Паттерн {pattern_id}: {main_ml_pattern} + {main_process_pattern}</div>
            <div class="pattern-subtitle">Количество студентов: <b>{students_count}</b></div>
            <span class="pattern-badge">МО: {ml_status}</span>
            <span class="pattern-badge">Process mining: {main_process_pattern}</span>
            <span class="{badge_class}">{flags}</span>
            <br><br>
            <b>Средние показатели паттерна:</b><br>
            Полнота контрольных активностей: <b>{percent_text(avg_control)}</b><br>
            Среднее число событий: <b>{number_text(avg_events)}</b><br>
            Среднее число активных дней: <b>{number_text(avg_active_days)}</b><br>
            Доля активности в 2 самых активных дня: <b>{percent_text(avg_top_2)}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Путь студента
# ============================================================

def build_student_path_df(event_log: pd.DataFrame, student_id: str) -> pd.DataFrame:
    if event_log is None or event_log.empty:
        return pd.DataFrame()

    df = event_log[
        event_log["student_id"].astype(str) == str(student_id)
    ].copy()

    if df.empty:
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df[df["timestamp"].notna()].copy()

    if df.empty:
        return pd.DataFrame()

    df = df.sort_values("timestamp").reset_index(drop=True)

    dedup_cols = [
        col for col in [
            "timestamp",
            "process_activity",
            "component",
            "context",
            "activity",
        ]
        if col in df.columns
    ]

    df = df.drop_duplicates(subset=dedup_cols, keep="first")
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["step"] = range(1, len(df) + 1)
    df["previous_timestamp"] = df["timestamp"].shift(1)
    df["minutes_from_previous"] = (
        (df["timestamp"] - df["previous_timestamp"])
        .dt.total_seconds()
        .div(60)
        .round(2)
    )

    df["date"] = df["timestamp"].dt.date.astype(str)
    df["time"] = df["timestamp"].dt.strftime("%H:%M:%S")

    columns = [
        "step",
        "timestamp",
        "date",
        "time",
        "minutes_from_previous",
        "process_activity",
        "component",
        "context",
        "activity",
    ]

    columns = [
        col for col in columns
        if col in df.columns
    ]

    return df[columns]


def plot_student_timeline(student_path_df: pd.DataFrame):
    if student_path_df is None or student_path_df.empty:
        return None

    y_col = "process_activity" if "process_activity" in student_path_df.columns else "activity"

    hover_cols = [
        col for col in [
            "step",
            "component",
            "context",
            "activity",
            "minutes_from_previous",
        ]
        if col in student_path_df.columns
    ]

    fig = px.scatter(
        student_path_df,
        x="timestamp",
        y=y_col,
        color=y_col,
        hover_data=hover_cols,
        title="Путь прохождения курса выбранного студента",
    )

    fig.update_traces(marker=dict(size=9))

    fig.update_layout(
        height=720,
        xaxis_title="Время",
        yaxis_title="Действие",
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
    )

    return fig


# ============================================================
# Настройки итогового анализа
# ============================================================

st.subheader("Настройки итогового анализа")

settings_col_1, settings_col_2, settings_col_3 = st.columns([2, 1, 1])

with settings_col_1:
    selected_methods = st.multiselect(
        "Методы машинного обучения, которые учитывать",
        options=available_methods,
        default=available_methods,
        key="final_selected_methods",
    )

with settings_col_2:
    top_n_patterns = st.slider(
        "Топ паттернов",
        min_value=3,
        max_value=50,
        value=15,
        step=1,
        key="final_top_n_patterns",
    )

with settings_col_3:
    tree_height = st.slider(
        "Высота дерева",
        min_value=800,
        max_value=1600,
        value=1150,
        step=50,
        key="final_tree_height",
    )

if not selected_methods:
    st.warning("Выберите хотя бы один метод машинного обучения.")
    st.stop()


# ============================================================
# Построение итоговых данных
# ============================================================

ml_df = build_ml_patterns_table(selected_methods)

if ml_df.empty:
    st.error("Не удалось собрать таблицу ML-паттернов.")
    st.stop()

process_df = get_process_features_df()

if process_df.empty:
    st.warning(
        "Process mining признаки не найдены. Итоговая страница будет построена "
        "только по методам машинного обучения."
    )

final_df = build_final_df(
    ml_df=ml_df,
    process_df=process_df,
    selected_methods=selected_methods,
)

if final_df.empty:
    st.error("Не удалось построить итоговую таблицу.")
    st.stop()

unique_patterns_df = build_unique_patterns_df(final_df)

if unique_patterns_df.empty:
    st.error("Не удалось построить таблицу основных паттернов.")
    st.stop()

st.session_state["final_results_df"] = final_df
st.session_state["unique_final_patterns_df"] = unique_patterns_df


# ============================================================
# Метрики
# ============================================================

students_count = final_df["student_id"].nunique()
patterns_count = len(unique_patterns_df)

ml_same_count = int(
    (final_df["ml_consensus_status"] == "Методы МО совпали").sum()
)

ml_diff_count = int(
    final_df["ml_consensus_status"].isin(
        [
            "Методы МО различаются",
            "Есть преобладающий паттерн МО",
        ]
    ).sum()
)

risk_count = int(
    final_df["process_flags_count"].fillna(0).astype(int).gt(0).sum()
)

largest_pattern_size = int(unique_patterns_df["students_count"].max())

m1, m2, m3, m4, m5 = st.columns(5)

m1.metric("Студентов", students_count)
m2.metric("Основных паттернов", patterns_count)
m3.metric("Самый крупный паттерн", largest_pattern_size)
m4.metric("МО совпали", ml_same_count)
m5.metric("Есть process-признаки", risk_count)


# ============================================================
# Вкладки
# ============================================================

tab_tree, tab_analysis, tab_pattern, tab_student, tab_tables, tab_text = st.tabs(
    [
        "Дерево паттернов",
        "Анализ",
        "Выбранный паттерн",
        "Студент",
        "Таблицы",
        " ",
    ]
)


# ============================================================
# TAB 1: Большое дерево
# ============================================================

with tab_tree:
    st.subheader("Большое дерево итоговых паттернов")

    st.write(
        """
        Дерево показывает вложенность паттернов:
        **все студенты → основной паттерн МО → основной process mining паттерн → дополнительные process-признаки**.
        Размер каждого прямоугольника соответствует количеству студентов.
        """
    )

    tree_fig = plot_fullscreen_pattern_tree(
        unique_patterns_df=unique_patterns_df,
        height=tree_height,
    )

    st.plotly_chart(
        tree_fig,
        use_container_width=True,
        config={
            "displayModeBar": True,
            "scrollZoom": True,
            "responsive": True,
        },
    )

    st.info(
        """
        Чтобы выбрать конкретный паттерн для анализа, используйте список ниже.
        Так страница работает стабильнее: дерево остаётся большим и красивым,
        а выбор паттерна не зависит от сторонних библиотек для кликов.
        """
    )

    pattern_options = unique_patterns_df["pattern_id"].astype(int).tolist()

    selected_pattern_id_from_state = st.session_state.get(
        "selected_final_pattern_id",
        pattern_options[0],
    )

    if selected_pattern_id_from_state not in pattern_options:
        selected_pattern_id_from_state = pattern_options[0]

    selected_pattern_index = pattern_options.index(selected_pattern_id_from_state)

    selected_pattern_id = st.selectbox(
        "Выберите паттерн для подробного анализа",
        options=pattern_options,
        index=selected_pattern_index,
        format_func=lambda pid: unique_patterns_df.loc[
            unique_patterns_df["pattern_id"] == pid,
            "pattern_label",
        ].iloc[0],
        key="selected_final_pattern_selectbox_tree",
    )

    st.session_state["selected_final_pattern_id"] = int(selected_pattern_id)

    selected_pattern_row = unique_patterns_df[
        unique_patterns_df["pattern_id"] == selected_pattern_id
    ].iloc[0]

    render_pattern_card(selected_pattern_row)


# ============================================================
# TAB 2: Красивый анализ
# ============================================================

with tab_analysis:
    st.subheader("Анализ основных паттернов")

    pattern_bar_fig = plot_main_patterns_bar(
        unique_patterns_df=unique_patterns_df,
        top_n=min(top_n_patterns, len(unique_patterns_df)),
    )

    st.plotly_chart(
        pattern_bar_fig,
        use_container_width=True,
        config={"displayModeBar": True},
    )

    col_a, col_b = st.columns(2)

    with col_a:
        ml_counts_df = (
            final_df.groupby("main_ml_pattern", dropna=False)
            .agg(students_count=("student_id", "nunique"))
            .reset_index()
            .sort_values("students_count", ascending=False)
        )

        fig_ml = px.bar(
            ml_counts_df.sort_values("students_count"),
            x="students_count",
            y="main_ml_pattern",
            orientation="h",
            title="Распределение основных паттернов по методам МО",
        )

        fig_ml.update_layout(
            height=max(450, len(ml_counts_df) * 35),
            xaxis_title="Количество студентов",
            yaxis_title="Паттерн МО",
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color="black"),
        )

        st.plotly_chart(fig_ml, use_container_width=True)

    with col_b:
        pm_counts_df = (
            final_df.groupby("main_process_pattern", dropna=False)
            .agg(students_count=("student_id", "nunique"))
            .reset_index()
            .sort_values("students_count", ascending=False)
        )

        fig_pm = px.bar(
            pm_counts_df.sort_values("students_count"),
            x="students_count",
            y="main_process_pattern",
            orientation="h",
            title="Распределение process mining паттернов",
        )

        fig_pm.update_layout(
            height=max(450, len(pm_counts_df) * 35),
            xaxis_title="Количество студентов",
            yaxis_title="Process mining паттерн",
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color="black"),
        )

        st.plotly_chart(fig_pm, use_container_width=True)

    st.subheader("Связь методов МО и process mining")

    heatmap_fig, heat_df = plot_ml_process_heatmap(final_df)

    st.plotly_chart(
        heatmap_fig,
        use_container_width=True,
        config={"displayModeBar": True},
    )

    st.subheader("Поток формирования паттернов")

    sankey_fig = plot_patterns_sankey(unique_patterns_df)

    if sankey_fig is not None:
        st.plotly_chart(
            sankey_fig,
            use_container_width=True,
            config={"displayModeBar": True},
        )

    st.subheader("Карта качества паттернов")

    quality_fig = plot_pattern_quality_scatter(unique_patterns_df)

    if quality_fig is not None:
        st.plotly_chart(
            quality_fig,
            use_container_width=True,
            config={"displayModeBar": True},
        )
    else:
        st.info(
            "Карта качества недоступна, потому что не найдены process mining метрики "
            "control_completion_ratio и process_active_days."
        )

    st.subheader("Интерпретация распределения")

    top_pattern = unique_patterns_df.iloc[0]
    top_share = top_pattern["students_count"] / max(students_count, 1)

    st.markdown(
        f"""
        - Самый крупный паттерн: **{top_pattern["main_final_pattern"]}**.
        - В нём находится **{int(top_pattern["students_count"])} студентов**, то есть **{top_share * 100:.1f}%** от всех студентов.
        - Всего выделено **{patterns_count} основных итоговых паттернов**.
        - Студентов с дополнительными process mining признаками: **{risk_count}**.
        """
    )


# ============================================================
# TAB 3: Выбранный паттерн
# ============================================================

with tab_pattern:
    st.subheader("Подробный анализ выбранного паттерна")

    pattern_options = unique_patterns_df["pattern_id"].astype(int).tolist()

    selected_pattern_id = st.session_state.get(
        "selected_final_pattern_id",
        pattern_options[0],
    )

    if selected_pattern_id not in pattern_options:
        selected_pattern_id = pattern_options[0]

    selected_pattern_id = st.selectbox(
        "Паттерн",
        options=pattern_options,
        index=pattern_options.index(selected_pattern_id),
        format_func=lambda pid: unique_patterns_df.loc[
            unique_patterns_df["pattern_id"] == pid,
            "pattern_label",
        ].iloc[0],
        key="selected_final_pattern_selectbox_analysis",
    )

    st.session_state["selected_final_pattern_id"] = int(selected_pattern_id)

    selected_pattern_row = unique_patterns_df[
        unique_patterns_df["pattern_id"] == selected_pattern_id
    ].iloc[0]

    render_pattern_card(selected_pattern_row)

    st.subheader("Что означает этот паттерн")

    recommendations = build_pattern_recommendations(selected_pattern_row)

    for idx, recommendation in enumerate(recommendations, start=1):
        st.markdown(f"**{idx}.** {recommendation}")

    pattern_students_df = final_df[
        final_df["final_pattern_key"].astype(str)
        == str(selected_pattern_row["final_pattern_key"])
    ].copy()

    pattern_students_df = pattern_students_df.sort_values("student_id")

    st.subheader("Студенты в выбранном паттерне")

    compact_columns = [
        "student_id",
        "main_ml_pattern",
        "ml_consensus_status",
        "main_process_pattern",
        "process_flags",
        "control_completion_ratio",
        "process_total_events",
        "process_active_days",
        "top_2_days_activity_ratio",
        "fast_lecture_completion_ratio",
        "fast_test_completion_ratio",
        "trace_length",
        "returns_count",
    ]

    compact_columns = [
        col for col in compact_columns
        if col in pattern_students_df.columns
    ]

    st.dataframe(
        pattern_students_df[compact_columns],
        use_container_width=True,
        height=450,
    )

    csv_pattern_students = (
        pattern_students_df[compact_columns]
        .to_csv(index=False)
        .encode("utf-8-sig")
    )

    st.download_button(
        "Скачать студентов выбранного паттерна CSV",
        data=csv_pattern_students,
        file_name=f"pattern_{selected_pattern_id}_students.csv",
        mime="text/csv",
        key="download_selected_pattern_students_csv",
    )


# ============================================================
# TAB 4: Студент
# ============================================================

with tab_student:
    st.subheader("Просмотр студента внутри выбранного паттерна")

    selected_pattern_id = st.session_state.get(
        "selected_final_pattern_id",
        int(unique_patterns_df["pattern_id"].iloc[0]),
    )

    selected_pattern_row = unique_patterns_df[
        unique_patterns_df["pattern_id"] == selected_pattern_id
    ]

    if selected_pattern_row.empty:
        selected_pattern_row = unique_patterns_df.iloc[[0]]
        selected_pattern_id = int(selected_pattern_row["pattern_id"].iloc[0])

    selected_pattern_row = selected_pattern_row.iloc[0]

    pattern_students_df = final_df[
        final_df["final_pattern_key"].astype(str)
        == str(selected_pattern_row["final_pattern_key"])
    ].copy()

    pattern_students_df = pattern_students_df.sort_values("student_id")

    if pattern_students_df.empty:
        st.warning("В выбранном паттерне нет студентов.")
        st.stop()

    st.write(
        f"Текущий паттерн: **{selected_pattern_row['pattern_label']}**. "
        f"Студентов: **{len(pattern_students_df)}**."
    )

    student_options = pattern_students_df["student_id"].astype(str).tolist()

    previous_selected_student = st.session_state.get(
        "selected_final_student_id",
        student_options[0],
    )

    if previous_selected_student not in student_options:
        previous_selected_student = student_options[0]

    selected_student_id = st.selectbox(
        "Выберите студента",
        options=student_options,
        index=student_options.index(previous_selected_student),
        format_func=make_student_label,
        key="selected_final_student_selectbox",
    )

    st.session_state["selected_final_student_id"] = selected_student_id

    selected_student_row = final_df[
        final_df["student_id"].astype(str) == str(selected_student_id)
    ].copy()

    if selected_student_row.empty:
        st.warning("Студент не найден в итоговой таблице.")
        st.stop()

    row = selected_student_row.iloc[0]

    st.success(f"Выбран студент: {selected_student_id}")

    st.info(row.get("final_interpretation", ""))

    student_col_1, student_col_2 = st.columns(2)

    with student_col_1:
        st.subheader("Паттерны методов МО")

        ml_rows = []

        for method_name in selected_methods:
            cluster_col = f"{method_name}_cluster"
            pattern_col = f"{method_name}_resource_pattern"
            probability_col = f"{method_name}_cluster_probability"

            if cluster_col not in selected_student_row.columns:
                continue

            ml_rows.append(
                {
                    "method": method_name,
                    "cluster": row.get(cluster_col),
                    "resource_pattern": row.get(pattern_col),
                    "cluster_probability": row.get(probability_col, np.nan),
                }
            )

        if ml_rows:
            st.dataframe(pd.DataFrame(ml_rows), use_container_width=True)
        else:
            st.info("Нет данных по методам МО.")

    with student_col_2:
        st.subheader("Process mining признаки")

        process_columns = [
            "main_process_pattern",
            "process_flags",
            "process_flags_count",
            "completed_assignments_count",
            "expected_assignments_count",
            "assignment_completion_ratio",
            "completed_tests_count",
            "expected_tests_count",
            "test_completion_ratio",
            "control_completion_ratio",
            "process_total_events",
            "process_active_days",
            "top_2_days_activity_ratio",
            "top_3_days_activity_ratio",
            "days_to_80_percent_events",
            "fast_lecture_completion_ratio",
            "fast_test_completion_ratio",
            "suspicious_first_assignment_upload_ratio",
            "trace_length",
            "linearity",
            "complexity",
            "returns_count",
        ]

        process_columns = [
            col for col in process_columns
            if col in selected_student_row.columns
        ]

        if process_columns:
            st.dataframe(
                selected_student_row[process_columns],
                use_container_width=True,
            )
        else:
            st.info("Нет process mining признаков.")

    st.subheader("Путь прохождения курса")

    event_log = get_event_log_from_session()

    if event_log.empty:
        st.warning(
            "Лог событий для построения пути не найден. "
            "Сначала постройте event log на странице Process Mining."
        )
    else:
        student_path_df = build_student_path_df(
            event_log=event_log,
            student_id=selected_student_id,
        )

        if student_path_df.empty:
            st.warning("Для выбранного студента не найдены события.")
        else:
            timeline_fig = plot_student_timeline(student_path_df)

            if timeline_fig is not None:
                st.plotly_chart(
                    timeline_fig,
                    use_container_width=True,
                    config={"displayModeBar": True},
                )

            st.dataframe(
                student_path_df,
                use_container_width=True,
                height=450,
            )

            csv_path = student_path_df.to_csv(index=False).encode("utf-8-sig")

            st.download_button(
                "Скачать путь выбранного студента CSV",
                data=csv_path,
                file_name=f"student_{selected_student_id}_path.csv",
                mime="text/csv",
                key="download_selected_student_path_csv",
            )


# ============================================================
# TAB 5: Таблицы
# ============================================================

with tab_tables:
    st.subheader("Таблица основных итоговых паттернов")

    unique_display_columns = [
        "pattern_id",
        "students_count",
        "main_ml_pattern",
        "ml_consensus_status",
        "main_process_pattern",
        "process_flags",
        "main_final_pattern",
        "avg_control_completion",
        "avg_assignment_completion",
        "avg_test_completion",
        "avg_events",
        "avg_active_days",
        "avg_top_2_days_ratio",
        "avg_top_3_days_ratio",
        "avg_days_to_80",
        "avg_fast_lecture_ratio",
        "avg_fast_test_ratio",
        "avg_fast_assignment_upload_ratio",
        "avg_trace_length",
        "avg_linearity",
        "avg_complexity",
        "avg_returns",
        "students_preview",
    ]

    unique_display_columns = [
        col for col in unique_display_columns
        if col in unique_patterns_df.columns
    ]

    st.dataframe(
        unique_patterns_df[unique_display_columns],
        use_container_width=True,
        height=520,
    )

    unique_csv = (
        unique_patterns_df[unique_display_columns]
        .to_csv(index=False)
        .encode("utf-8-sig")
    )

    st.download_button(
        "Скачать основные паттерны CSV",
        data=unique_csv,
        file_name="main_final_patterns.csv",
        mime="text/csv",
        key="download_main_final_patterns_csv",
    )

    st.subheader("Итоговая таблица студентов")

    base_columns = [
        "student_id",
        "main_ml_pattern",
        "ml_consensus_status",
        "ml_signature",
        "main_process_pattern",
        "process_flags",
        "process_flags_count",
        "main_final_pattern",
        "final_interpretation",
    ]

    method_columns = []

    for method_name in selected_methods:
        method_columns.extend(
            [
                f"{method_name}_cluster",
                f"{method_name}_resource_pattern",
                f"{method_name}_cluster_probability",
            ]
        )

    process_metric_columns = [
        "completed_assignments_count",
        "expected_assignments_count",
        "assignment_completion_ratio",
        "missing_assignments_list",
        "completed_tests_count",
        "expected_tests_count",
        "test_completion_ratio",
        "missing_tests_list",
        "control_completion_ratio",
        "process_total_events",
        "process_active_days",
        "max_day_activity_ratio",
        "top_2_days_activity_ratio",
        "top_3_days_activity_ratio",
        "days_to_80_percent_events",
        "last_period_events_ratio",
        "fast_lecture_completion_ratio",
        "fast_test_completion_ratio",
        "suspicious_first_assignment_upload_ratio",
        "trace_length",
        "linearity",
        "complexity",
        "returns_count",
        "required_assignments_count",
        "required_tests_count",
    ]

    final_display_columns = [
        col for col in (
            base_columns
            + method_columns
            + process_metric_columns
        )
        if col in final_df.columns
    ]

    st.dataframe(
        final_df[final_display_columns],
        use_container_width=True,
        height=600,
    )

    final_csv = (
        final_df[final_display_columns]
        .to_csv(index=False)
        .encode("utf-8-sig")
    )

    st.download_button(
        "Скачать итоговую таблицу студентов CSV",
        data=final_csv,
        file_name="final_results_by_students.csv",
        mime="text/csv",
        key="download_final_results_by_students_csv",
    )

