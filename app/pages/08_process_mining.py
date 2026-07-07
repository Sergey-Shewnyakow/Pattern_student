import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.state import init_session_state
from src.ui_styles import apply_global_styles

from src.process_mining_preprocessing import build_process_event_log

from src.process_mining_analysis import (
    calculate_process_metrics,
    calculate_directly_follows,
    calculate_activity_frequencies,
    calculate_variants,
    calculate_transition_matrix,
    compare_process_metrics_by_cluster,
)

from src.process_mining_visualization import (
    plot_process_map,
    plot_transition_heatmap,
    plot_top_transitions,
    plot_dotted_chart,
)

from src.pm4py_process_visualization import (
    build_pm4py_heuristic_svg,
    build_pm4py_process_tree_svg,
    save_svg_to_download_bytes,
)

from src.cluster_naming import build_cluster_names

try:
    from src.cluster_name_editor import apply_custom_cluster_names
except Exception:
    apply_custom_cluster_names = None

from src.process_behavior_features import (
    infer_required_course_elements,
    calculate_student_process_behavior_features,
    merge_resource_and_process_patterns,
)


# ============================================================
# Настройка страницы
# ============================================================

st.set_page_config(
    page_title="Process Mining",
    layout="wide",
)

init_session_state()
apply_global_styles()

st.title("Process Mining")




# ============================================================
# Получение исходного лога
# ============================================================

df_sessions = st.session_state.get("df_sessions")

if df_sessions is None:
    df_sessions = st.session_state.get("df_clean")

if df_sessions is None:
    st.warning(
        "Сначала выполните Data Preparation: загрузите лог и постройте сессии."
    )
    st.stop()

if df_sessions.empty:
    st.error("Лог пустой.")
    st.stop()

df_sessions = df_sessions.copy()

if "student_id" not in df_sessions.columns:
    st.error("В логе отсутствует колонка student_id.")
    st.stop()

df_sessions["student_id"] = df_sessions["student_id"].astype(str)


# ============================================================
# Учитываем исключённых пользователей
# ============================================================

anomaly_df = st.session_state.get("anomaly_df")
excluded_student_ids = []

if anomaly_df is not None and not anomaly_df.empty:
    anomaly_df = anomaly_df.copy()

    if "student_id" in anomaly_df.columns:
        anomaly_df["student_id"] = anomaly_df["student_id"].astype(str)

        if "exclude_final" in anomaly_df.columns:
            excluded_student_ids = (
                anomaly_df.loc[anomaly_df["exclude_final"], "student_id"]
                .astype(str)
                .tolist()
            )
        elif "exclude_manual" in anomaly_df.columns:
            excluded_student_ids = (
                anomaly_df.loc[anomaly_df["exclude_manual"], "student_id"]
                .astype(str)
                .tolist()
            )
        elif "final_anomaly" in anomaly_df.columns:
            excluded_student_ids = (
                anomaly_df.loc[anomaly_df["final_anomaly"], "student_id"]
                .astype(str)
                .tolist()
            )

if excluded_student_ids:
    before_users_count = df_sessions["student_id"].nunique()
    before_events_count = len(df_sessions)

    df_sessions = df_sessions[
        ~df_sessions["student_id"].isin(excluded_student_ids)
    ].copy()

    after_users_count = df_sessions["student_id"].nunique()
    after_events_count = len(df_sessions)

    st.info(
        "Из process mining исключены пользователи, отмеченные как не-студенты. "
        f"Исключено пользователей: {len(excluded_student_ids)}. "
        f"Пользователей до фильтрации: {before_users_count}, после: {after_users_count}. "
        f"Событий до фильтрации: {before_events_count}, после: {after_events_count}."
    )
else:
    st.warning(
        "Список исключённых пользователей не найден или пуст. "
        "Process mining будет построен по всем пользователям из подготовленного лога."
    )

if df_sessions.empty:
    st.error(
        "После исключения не-студентов в логе не осталось событий для process mining."
    )
    st.stop()


# ============================================================
# Вспомогательные функции для кластеров
# ============================================================

def get_result_from_session(possible_keys):
    for key in possible_keys:
        value = st.session_state.get(key)

        if value is not None:
            return value

    return None


METHOD_RESULTS = {
    "KMeans": get_result_from_session(["clustering_result", "kmeans_result"]),
    "Agglomerative": get_result_from_session(["agglomerative_result"]),
    "GMM": get_result_from_session(["gmm_result"]),
    "HDBSCAN": get_result_from_session(["hdbscan_result"]),
    "DEC": get_result_from_session(["dec_result", "deep_embedding_result"]),
    "Autoencoder": get_result_from_session(["autoencoder_result", "autoencoder_clustering_result"]),
}

METHOD_KEY_MAP = {
    "KMeans": "kmeans",
    "Agglomerative": "agglomerative",
    "GMM": "gmm",
    "HDBSCAN": "hdbscan",
    "DEC": "dec",
    "Autoencoder": "autoencoder",
}


def get_cluster_names_for_method(method_name, result):
    if result is None:
        return pd.DataFrame()

    if not isinstance(result, dict):
        return pd.DataFrame()

    result_df = result.get("result_df")
    cluster_profiles = result.get("cluster_profiles")

    if result_df is None or result_df.empty:
        return pd.DataFrame()

    if "cluster" not in result_df.columns:
        return pd.DataFrame()

    if cluster_profiles is None or not isinstance(cluster_profiles, pd.DataFrame):
        cluster_profiles = pd.DataFrame()

    if cluster_profiles.empty:
        names_df = (
            result_df.groupby("cluster")
            .agg(cluster_size=("student_id", "nunique"))
            .reset_index()
        )

        names_df["suggested_name"] = names_df["cluster"].apply(
            lambda x: "Шумовые / нетипичные студенты" if x == -1 else f"Кластер {x}"
        )

        names_df["description"] = names_df["cluster"].apply(
            lambda x: (
                "Студенты, не вошедшие в устойчивую группу."
                if x == -1
                else "Группа студентов со схожими признаками активности."
            )
        )
    else:
        try:
            non_noise_result_df = result_df[result_df["cluster"] != -1].copy()

            if non_noise_result_df.empty:
                names_df = pd.DataFrame(
                    columns=[
                        "cluster",
                        "cluster_size",
                        "suggested_name",
                        "description",
                    ]
                )
            else:
                names_df = build_cluster_names(
                    result_df=non_noise_result_df,
                    cluster_profiles=cluster_profiles,
                )
        except Exception:
            names_df = (
                result_df.groupby("cluster")
                .agg(cluster_size=("student_id", "nunique"))
                .reset_index()
            )

            names_df["suggested_name"] = names_df["cluster"].apply(
                lambda x: f"Кластер {x}"
            )

            names_df["description"] = "Группа студентов со схожими признаками активности."

        noise_count = int((result_df["cluster"] == -1).sum())

        if noise_count > 0 and -1 not in names_df.get("cluster", pd.Series()).tolist():
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

    return names_df


def build_clustered_students(method_name, result):
    if result is None:
        return pd.DataFrame()

    if not isinstance(result, dict):
        return pd.DataFrame()

    result_df = result.get("result_df")

    if result_df is None or result_df.empty:
        return pd.DataFrame()

    if "student_id" not in result_df.columns or "cluster" not in result_df.columns:
        return pd.DataFrame()

    names_df = get_cluster_names_for_method(method_name, result)

    students_df = result_df[["student_id", "cluster"]].copy()
    students_df["student_id"] = students_df["student_id"].astype(str)

    if not names_df.empty and "cluster" in names_df.columns:
        if "suggested_name" not in names_df.columns:
            names_df["suggested_name"] = names_df["cluster"].apply(
                lambda x: f"Кластер {x}"
            )

        students_df = students_df.merge(
            names_df[["cluster", "suggested_name"]],
            on="cluster",
            how="left",
        )
    else:
        students_df["suggested_name"] = students_df["cluster"].apply(
            lambda x: f"Кластер {x}"
        )

    students_df["suggested_name"] = students_df["suggested_name"].fillna(
        students_df["cluster"].apply(lambda x: f"Кластер {x}")
    )

    return students_df


def get_resource_patterns_for_method(method_name):
    method_result = METHOD_RESULTS.get(method_name)

    if method_result is None:
        return pd.DataFrame()

    return build_clustered_students(
        method_name=method_name,
        result=method_result,
    )


def prepare_required_editor_df(stats_df, element_type):
    if stats_df is None or stats_df.empty:
        return pd.DataFrame(
            columns=[
                "use_as_required",
                "element_type",
                "context",
                "students_completed",
                "completion_share",
            ]
        )

    editor_df = stats_df.copy()

    if "context" not in editor_df.columns:
        editor_df["context"] = ""

    if "students_completed" not in editor_df.columns:
        editor_df["students_completed"] = 0

    if "completion_share" not in editor_df.columns:
        editor_df["completion_share"] = 0.0

    if "is_required" not in editor_df.columns:
        editor_df["is_required"] = False

    editor_df["use_as_required"] = editor_df["is_required"].astype(bool)
    editor_df["element_type"] = element_type

    editor_df["completion_share"] = pd.to_numeric(
        editor_df["completion_share"],
        errors="coerce",
    ).fillna(0.0)

    editor_df["students_completed"] = pd.to_numeric(
        editor_df["students_completed"],
        errors="coerce",
    ).fillna(0).astype(int)

    editor_df = editor_df[
        [
            "use_as_required",
            "element_type",
            "context",
            "students_completed",
            "completion_share",
        ]
    ].copy()

    editor_df = editor_df.sort_values(
        ["use_as_required", "students_completed", "completion_share"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return editor_df


# ============================================================
# Настройки подготовки event log
# ============================================================

st.subheader("Настройки подготовки event log")

col_1, col_2, col_3 = st.columns(3)

with col_1:
    detail_level = st.selectbox(
        "Уровень детализации действий",
        options=["coarse", "medium", "detailed"],
        index=1,
        format_func=lambda x: {
            "coarse": "Крупный",
            "medium": "Средний — рекомендуется",
            "detailed": "Детальный",
        }.get(x, x),
        key="process_detail_level",
    )

with col_2:
    collapse_duplicates = st.checkbox(
        "Сжимать подряд идущие одинаковые действия",
        value=True,
        key="process_collapse_duplicates",
    )

with col_3:
    build_button = st.button(
        "Построить event log",
        key="build_process_event_log_button",
    )


# ============================================================
# Построение event log
# ============================================================

process_settings = {
    "detail_level": detail_level,
    "collapse_duplicates": collapse_duplicates,
    "excluded_student_ids": tuple(sorted(excluded_student_ids)),
}

previous_process_settings = st.session_state.get("process_settings")

need_rebuild_event_log = (
    build_button
    or st.session_state.get("process_event_log") is None
    or previous_process_settings != process_settings
)

if need_rebuild_event_log:
    try:
        process_event_log = build_process_event_log(
            df=df_sessions,
            detail_level=detail_level,
            collapse_duplicates=collapse_duplicates,
        )

        st.session_state["process_event_log"] = process_event_log
        st.session_state["process_settings"] = process_settings

        for key in [
            "pm4py_heuristic_svg",
            "pm4py_process_tree_svg",
            "pm4py_svg_scope_name",
            "required_elements_result",
            "manual_required_assignments_df",
            "manual_required_tests_df",
            "selected_required_assignments",
            "selected_required_tests",
        ]:
            if key in st.session_state:
                del st.session_state[key]

    except Exception as e:
        st.exception(e)
        st.stop()

process_event_log = st.session_state.get("process_event_log")

if process_event_log is None or process_event_log.empty:
    st.error("После фильтрации не осталось событий для process mining.")
    st.stop()

process_event_log = process_event_log.copy()
process_event_log["student_id"] = process_event_log["student_id"].astype(str)


# ============================================================
# Краткая информация по подготовке
# ============================================================

st.subheader("Итог подготовки данных для process mining")

prep_1, prep_2, prep_3, prep_4 = st.columns(4)

prep_1.metric("Пользователей в event log", process_event_log["student_id"].nunique())
prep_2.metric("Событий в event log", len(process_event_log))

if "process_activity" in process_event_log.columns:
    prep_3.metric("Типов действий", process_event_log["process_activity"].nunique())
else:
    prep_3.metric("Типов действий", "—")

prep_4.metric("Исключено не-студентов", len(excluded_student_ids))

with st.expander("Показать подготовленный event log"):
    display_columns = [
        "student_id",
        "timestamp",
        "component",
        "context",
        "activity",
        "process_activity",
    ]

    available_display_columns = [
        col for col in display_columns
        if col in process_event_log.columns
    ]

    st.dataframe(
        process_event_log[available_display_columns].head(300),
        use_container_width=True,
    )

    full_event_log_csv = (
        process_event_log[available_display_columns]
        .to_csv(index=False)
        .encode("utf-8-sig")
    )

    st.download_button(
        "Скачать полный подготовленный event log CSV",
        data=full_event_log_csv,
        file_name="full_process_event_log.csv",
        mime="text/csv",
        key="download_full_process_event_log_csv",
    )


# ============================================================
# Уникальные контексты событий
# ============================================================

st.subheader("Уникальные контексты событий")

context_columns = [
    col for col in ["component", "context", "activity", "process_activity"]
    if col in process_event_log.columns
]

if context_columns:
    context_counts_df = (
        process_event_log
        .groupby(context_columns)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    st.dataframe(
        context_counts_df,
        use_container_width=True,
    )

    csv_data = context_counts_df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        "Скачать таблицу уникальных контекстов CSV",
        data=csv_data,
        file_name="unique_contexts.csv",
        mime="text/csv",
        key="download_unique_contexts_csv",
    )
else:
    st.info("В event log нет колонок component, context, activity или process_activity.")


# ============================================================
# Область анализа
# ============================================================

st.subheader("Область анализа")

available_methods = [
    method_name
    for method_name, result in METHOD_RESULTS.items()
    if result is not None
]

analysis_mode = st.radio(
    "Для чего построить диаграммы?",
    options=["Весь курс", "Выбранный кластер", "Выбранный студент"],
    horizontal=True,
    key="process_analysis_mode",
)

filtered_event_log = process_event_log.copy()
selected_scope_name = "Весь курс"

if analysis_mode == "Выбранный кластер":
    if not available_methods:
        st.warning(
            "Сначала запустите хотя бы один метод кластеризации, чтобы анализировать "
            "process mining по кластерам."
        )
    else:
        selected_method = st.selectbox(
            "Метод кластеризации",
            options=available_methods,
            key="process_selected_method",
        )

        clustered_students_df = build_clustered_students(
            selected_method,
            METHOD_RESULTS[selected_method],
        )

        if clustered_students_df.empty:
            st.warning("Для выбранного метода нет данных о кластерах.")
        else:
            students_in_process_log = (
                process_event_log["student_id"].astype(str).unique().tolist()
            )

            clustered_students_df = clustered_students_df[
                clustered_students_df["student_id"].astype(str).isin(
                    students_in_process_log
                )
            ].copy()

            if clustered_students_df.empty:
                st.warning(
                    "После исключения не-студентов не осталось студентов выбранного метода "
                    "для process mining."
                )
                st.stop()

            cluster_options_df = (
                clustered_students_df[["cluster", "suggested_name"]]
                .drop_duplicates()
                .sort_values("cluster")
                .reset_index(drop=True)
            )

            cluster_options_df["label"] = (
                "Кластер "
                + cluster_options_df["cluster"].astype(str)
                + " — "
                + cluster_options_df["suggested_name"].astype(str)
            )

            selected_cluster_label = st.selectbox(
                "Кластер",
                options=cluster_options_df["label"].tolist(),
                key="process_selected_cluster_label",
            )

            selected_cluster_row = cluster_options_df[
                cluster_options_df["label"] == selected_cluster_label
            ].iloc[0]

            selected_cluster = selected_cluster_row["cluster"]
            selected_cluster_name = selected_cluster_row["suggested_name"]

            selected_student_ids = (
                clustered_students_df[
                    clustered_students_df["cluster"] == selected_cluster
                ]["student_id"]
                .astype(str)
                .tolist()
            )

            filtered_event_log = process_event_log[
                process_event_log["student_id"].astype(str).isin(
                    selected_student_ids
                )
            ].copy()

            selected_scope_name = (
                f"{selected_method}: кластер {selected_cluster} — "
                f"{selected_cluster_name}"
            )

            st.info(
                f"Анализируется: {selected_scope_name}. "
                f"Студентов в кластере: {len(selected_student_ids)}. "
                f"Событий: {len(filtered_event_log)}."
            )

elif analysis_mode == "Выбранный студент":
    student_ids = sorted(process_event_log["student_id"].astype(str).unique().tolist())

    selected_student_id = st.selectbox(
        "Студент",
        options=student_ids,
        key="process_selected_single_student",
    )

    filtered_event_log = process_event_log[
        process_event_log["student_id"].astype(str) == str(selected_student_id)
    ].copy()

    selected_scope_name = f"Студент {selected_student_id}"

    st.info(
        f"Анализируется студент {selected_student_id}. "
        f"Событий: {len(filtered_event_log)}."
    )

if filtered_event_log.empty:
    st.warning("Для выбранного фильтра нет событий.")
    st.stop()


# ============================================================
# Сброс старых SVG при смене области
# ============================================================

previous_svg_scope_name = st.session_state.get("pm4py_svg_scope_name")

if previous_svg_scope_name != selected_scope_name:
    for svg_key in [
        "pm4py_heuristic_svg",
        "pm4py_process_tree_svg",
    ]:
        if svg_key in st.session_state:
            del st.session_state[svg_key]

    st.session_state["pm4py_svg_scope_name"] = selected_scope_name


# ============================================================
# Метрики процесса
# ============================================================

st.subheader("Метрики процесса")

metrics = calculate_process_metrics(filtered_event_log)

m1, m2, m3, m4 = st.columns(4)

m1.metric("Пользователей", metrics["case_count"])
m2.metric("Событий", metrics["events_count"])
m3.metric("Действий", metrics["unique_activities_count"])
m4.metric("Вариантов", metrics["variants_count"])

m5, m6, m7, m8 = st.columns(4)

m5.metric("Средняя длина траектории", f"{metrics['avg_trace_length']:.2f}")
m6.metric("Средняя линейность", f"{metrics['avg_linearity']:.3f}")
m7.metric("Средние возвраты", f"{metrics['avg_rework_count']:.2f}")
m8.metric(
    "Доля самого частого варианта",
    f"{metrics['most_common_variant_share']:.3f}",
)

with st.expander("Показать все метрики process mining"):
    st.dataframe(
        pd.DataFrame([metrics]),
        use_container_width=True,
    )


# ============================================================
# Частоты действий и переходов
# ============================================================

activity_freq_df = calculate_activity_frequencies(filtered_event_log)
transitions_df = calculate_directly_follows(filtered_event_log)

st.subheader("Частоты действий и переходов")

col_freq_1, col_freq_2 = st.columns(2)

with col_freq_1:
    st.write("Частоты действий")
    st.dataframe(
        activity_freq_df,
        use_container_width=True,
    )

with col_freq_2:
    st.write("Directly-follows переходы")
    st.dataframe(
        transitions_df,
        use_container_width=True,
    )


# ============================================================
# Настройки диаграмм
# ============================================================

st.subheader("Настройки диаграмм")

v1, v2, v3 = st.columns(3)

with v1:
    top_transitions = st.slider(
        "Top переходов для обычной process map",
        min_value=10,
        max_value=100,
        value=30,
        step=5,
        key="process_top_transitions",
    )

with v2:
    min_transition_count = st.slider(
        "Мин. частота перехода",
        min_value=1,
        max_value=100,
        value=10,
        step=1,
        key="process_min_transition_count",
    )

with v3:
    top_transitions_bar = st.slider(
        "Top переходов для столбчатого графика",
        min_value=10,
        max_value=50,
        value=25,
        step=5,
        key="process_top_transitions_bar",
    )


# ============================================================
# Обычная process map
# ============================================================

st.subheader("Обычная process map / Directly-Follows Graph")

fig_process_map = plot_process_map(
    transitions_df=transitions_df,
    activity_freq_df=activity_freq_df,
    top_transitions=top_transitions,
    min_transition_count=min_transition_count,
)

st.plotly_chart(
    fig_process_map,
    use_container_width=True,
)

st.caption(
    "Обычная process map показывает самые частые переходы между действиями."
)


# ============================================================
# PM4Py SVG-диаграммы
# ============================================================

st.subheader("PM4Py-диаграммы процесса в SVG")

st.write(
    """
    Эти диаграммы строятся через PM4Py по нажатию на кнопку.
    Они строятся для текущей выбранной области анализа:
    весь курс, выбранный кластер или выбранный студент.
    """
)

pm4py_tab_1, pm4py_tab_2 = st.tabs(
    [
        "Эвристический метод",
        "Иерархический метод",
    ]
)

with pm4py_tab_1:
    st.write("### Эвристический метод / Heuristics Miner")

    h1, h2, h3 = st.columns(3)

    with h1:
        pm4py_dependency_threshold = st.slider(
            "Dependency threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            key="pm4py_dependency_threshold",
        )

    with h2:
        pm4py_min_act_count = st.slider(
            "Минимальная частота действия",
            min_value=1,
            max_value=100,
            value=5,
            step=1,
            key="pm4py_min_act_count",
        )

    with h3:
        pm4py_min_dfg_occurrences = st.slider(
            "Минимальная частота перехода",
            min_value=1,
            max_value=100,
            value=5,
            step=1,
            key="pm4py_min_dfg_occurrences",
        )

    st.caption(
        "Если диаграмма слишком перегружена, увеличьте dependency threshold, "
        "минимальную частоту действия или минимальную частоту перехода."
    )

    if st.button(
        "Построить эвристическую SVG-диаграмму",
        key="build_pm4py_heuristic_svg_button",
    ):
        try:
            with st.spinner("PM4Py строит эвристическую диаграмму..."):
                heuristic_svg = build_pm4py_heuristic_svg(
                    event_log=filtered_event_log,
                    dependency_threshold=pm4py_dependency_threshold,
                    min_act_count=pm4py_min_act_count,
                    min_dfg_occurrences=pm4py_min_dfg_occurrences,
                )

            st.session_state["pm4py_heuristic_svg"] = heuristic_svg
            st.session_state["pm4py_svg_scope_name"] = selected_scope_name

        except Exception as e:
            st.exception(e)

    if st.session_state.get("pm4py_heuristic_svg") is not None:
        heuristic_svg = st.session_state["pm4py_heuristic_svg"]

        components.html(
            heuristic_svg,
            height=850,
            scrolling=True,
        )

        st.download_button(
            "Скачать эвристическую диаграмму SVG",
            data=save_svg_to_download_bytes(heuristic_svg),
            file_name="heuristic_miner.svg",
            mime="image/svg+xml",
            key="download_heuristic_svg",
        )

with pm4py_tab_2:
    st.write("### Иерархический метод / Inductive Miner Process Tree")

    pm4py_noise_threshold = st.slider(
        "Noise threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
        key="pm4py_noise_threshold",
    )

    st.caption(
        "Иерархическая диаграмма строится как process tree. "
        "Если модель слишком сложная, увеличьте noise threshold."
    )

    if st.button(
        "Построить иерархическую SVG-диаграмму",
        key="build_pm4py_process_tree_svg_button",
    ):
        try:
            with st.spinner("PM4Py строит иерархическую диаграмму..."):
                process_tree_svg = build_pm4py_process_tree_svg(
                    event_log=filtered_event_log,
                    noise_threshold=pm4py_noise_threshold,
                )

            st.session_state["pm4py_process_tree_svg"] = process_tree_svg
            st.session_state["pm4py_svg_scope_name"] = selected_scope_name

        except Exception as e:
            st.exception(e)

    if st.session_state.get("pm4py_process_tree_svg") is not None:
        process_tree_svg = st.session_state["pm4py_process_tree_svg"]

        components.html(
            process_tree_svg,
            height=850,
            scrolling=True,
        )

        st.download_button(
            "Скачать иерархическую диаграмму SVG",
            data=save_svg_to_download_bytes(process_tree_svg),
            file_name="process_tree.svg",
            mime="image/svg+xml",
            key="download_process_tree_svg",
        )


# ============================================================
# Top transitions
# ============================================================

st.subheader("Top переходов")

fig_top_transitions = plot_top_transitions(
    transitions_df=transitions_df,
    top_n=top_transitions_bar,
)

st.plotly_chart(
    fig_top_transitions,
    use_container_width=True,
)


# ============================================================
# Heatmap переходов
# ============================================================

st.subheader("Heatmap переходов")

transition_matrix_df = calculate_transition_matrix(transitions_df)

fig_heatmap = plot_transition_heatmap(
    transition_matrix_df=transition_matrix_df,
    max_activities=20,
)

st.plotly_chart(
    fig_heatmap,
    use_container_width=True,
)


# ============================================================
# Dotted chart
# ============================================================

st.subheader("Dotted chart активности во времени")

d1, d2 = st.columns(2)

with d1:
    dotted_mode = st.radio(
        "Кого показать на Dotted chart?",
        options=["Всех студентов из выбранной области", "Выбранных студентов"],
        horizontal=False,
        key="dotted_mode",
    )

with d2:
    max_students_timeline = st.slider(
        "Максимум студентов, если выбраны все",
        min_value=10,
        max_value= 400,
        value=100,
        step=10,
        key="process_max_students_dotted",
    )

selected_students_for_dotted = None

if dotted_mode == "Выбранных студентов":
    available_students_for_dotted = sorted(
        filtered_event_log["student_id"].astype(str).unique().tolist()
    )

    default_students = available_students_for_dotted[
        : min(5, len(available_students_for_dotted))
    ]

    selected_students_for_dotted = st.multiselect(
        "Выберите студентов для Dotted chart",
        options=available_students_for_dotted,
        default=default_students,
        key="selected_students_for_dotted",
    )

fig_dotted = plot_dotted_chart(
    event_log=filtered_event_log,
    selected_students=selected_students_for_dotted,
    max_students=max_students_timeline,
)

st.plotly_chart(
    fig_dotted,
    use_container_width=True,
)

st.caption(
    "Dotted chart показывает распределение действий во времени. "
    "Он полезен для анализа регулярности, рывков активности и длинных пауз."
)


# ============================================================
# Варианты траекторий
# ============================================================

st.subheader("Наиболее частые варианты траекторий")

variants_df = calculate_variants(
    event_log=filtered_event_log,
    top_n=20,
)

st.dataframe(
    variants_df,
    use_container_width=True,
)

st.caption(
    "Если вариантов много, а самый частый вариант встречается редко, это означает "
    "высокую вариативность поведения студентов."
)


# ============================================================
# Сравнение process mining метрик по кластерам
# ============================================================

st.subheader("Сравнение process mining метрик по кластерам")

if not available_methods:
    st.info(
        "Для сравнения по кластерам сначала запустите один из методов кластеризации."
    )
else:
    selected_compare_method = st.selectbox(
        "Метод для сравнения кластеров",
        options=available_methods,
        key="process_compare_method",
    )

    clustered_students_df = build_clustered_students(
        selected_compare_method,
        METHOD_RESULTS[selected_compare_method],
    )

    if clustered_students_df.empty:
        st.info("Нет данных о кластерах для выбранного метода.")
    else:
        students_in_process_log = (
            process_event_log["student_id"].astype(str).unique().tolist()
        )

        clustered_students_df = clustered_students_df[
            clustered_students_df["student_id"].astype(str).isin(
                students_in_process_log
            )
        ].copy()

        if clustered_students_df.empty:
            st.info(
                "После исключения не-студентов нет пересечения между кластерами "
                "и event log process mining."
            )
        else:
            cluster_metrics_df = compare_process_metrics_by_cluster(
                event_log=process_event_log,
                clustered_students_df=clustered_students_df,
                cluster_col="cluster",
                cluster_name_col="suggested_name",
            )

            st.dataframe(
                cluster_metrics_df,
                use_container_width=True,
            )

            st.caption(
                "Эта таблица показывает, как различаются процессные характеристики "
                "между кластерами: длина траектории, линейность, возвраты, число вариантов."
            )


# ============================================================
# Процессные паттерны студентов
# ============================================================

st.subheader("Выявление процессных паттернов студентов")

st.write(
    """
    Этот блок уточняет результаты кластеризации.
    Кластеризация показывает, какие ресурсы студент использовал,
    а process mining показывает, как именно он проходил курс:
    регулярно, аврально, формально, с быстрым прохождением лекций
    или с неполным выполнением контрольных активностей.
    """
)

pb1, pb2, pb3, pb4, pb5 = st.columns(5)

with pb1:
    last_period_days = st.slider(
        "Последний период курса, дней",
        min_value=1,
        max_value=30,
        value=7,
        step=1,
        key="process_behavior_last_period_days",
    )

with pb2:
    fast_assignment_minutes = st.slider(
        "Быстрая загрузка ответа, минут",
        min_value=5,
        max_value=120,
        value=15,
        step=5,
        key="process_behavior_fast_assignment_minutes",
    )

with pb3:
    fast_test_minutes = st.slider(
        "Быстрое прохождение теста, минут",
        min_value=1,
        max_value=30,
        value=3,
        step=1,
        key="process_behavior_fast_test_minutes",
    )

with pb4:
    fast_lecture_minutes = st.slider(
        "Быстрое прохождение лекции, минут",
        min_value=1,
        max_value=20,
        value=2,
        step=1,
        key="process_behavior_fast_lecture_minutes",
    )

with pb5:
    min_required_completion_share = st.slider(
        "Порог автоматического определения обязательного элемента",
        min_value=0.10,
        max_value=0.95,
        value=0.50,
        step=0.05,
        key="process_behavior_min_required_completion_share",
    )

if available_methods:
    selected_resource_method = st.selectbox(
        "Метод кластеризации для ресурсного паттерна",
        options=available_methods,
        key="process_behavior_resource_method",
    )
else:
    selected_resource_method = None
    st.info(
        "Методы кластеризации ещё не запускались. "
        "Будут рассчитаны только процессные паттерны."
    )


# ============================================================
# 1. Автоматическое определение обязательных элементов
# ============================================================

st.subheader("1. Автоматическое определение обязательных элементов курса")

st.write(
    """
    Сначала система автоматически находит элементы, которые похожи на обязательные.
    После этого список можно вручную исправить: включить нужные элементы
    и исключить необязательные, тренировочные или технические элементы.
    """
)

required_detection_settings = {
    "min_required_completion_share": float(min_required_completion_share),
    "process_event_log_rows": int(len(process_event_log)),
    "process_event_log_students": int(process_event_log["student_id"].nunique()),
}

previous_required_detection_settings = st.session_state.get(
    "required_detection_settings"
)

need_recalculate_required_elements = (
    st.session_state.get("required_elements_result") is None
    or previous_required_detection_settings != required_detection_settings
)

detect_required_button = st.button(
    "Определить обязательные элементы курса",
    key="detect_required_course_elements_button",
)

if detect_required_button or need_recalculate_required_elements:
    try:
        required_elements_result = infer_required_course_elements(
            event_log=process_event_log,
            min_required_completion_share=min_required_completion_share,
        )

        st.session_state["required_elements_result"] = required_elements_result
        st.session_state["required_detection_settings"] = required_detection_settings

        for key in [
            "manual_required_assignments_df",
            "manual_required_tests_df",
            "selected_required_assignments",
            "selected_required_tests",
        ]:
            if key in st.session_state:
                del st.session_state[key]

    except Exception as e:
        st.exception(e)
        st.stop()

required_elements_result = st.session_state.get("required_elements_result")

if required_elements_result is None:
    st.warning("Сначала определите обязательные элементы курса.")
    st.stop()

assignment_stats_df = required_elements_result.get(
    "assignment_completion_stats",
    pd.DataFrame(),
)

test_stats_df = required_elements_result.get(
    "test_completion_stats",
    pd.DataFrame(),
)

default_assignment_editor_df = prepare_required_editor_df(
    assignment_stats_df,
    element_type="Задание",
)

default_test_editor_df = prepare_required_editor_df(
    test_stats_df,
    element_type="Тест",
)

if "manual_required_assignments_df" not in st.session_state:
    st.session_state["manual_required_assignments_df"] = default_assignment_editor_df

if "manual_required_tests_df" not in st.session_state:
    st.session_state["manual_required_tests_df"] = default_test_editor_df


# ============================================================
# 2. Ручная проверка обязательных элементов
# ============================================================

st.subheader("2. Ручная проверка обязательных элементов")

st.info(
    """
    Отметьте галочками только те элементы, которые действительно должны учитываться
    как обязательные. Например, если тест был тренировочным или техническим,
    его можно исключить из расчёта.
    """
)

required_tab_1, required_tab_2 = st.tabs(
    [
        "Обязательные задания",
        "Обязательные тесты",
    ]
)

with required_tab_1:
    st.write("### Выбор обязательных заданий")

    if st.session_state["manual_required_assignments_df"].empty:
        st.warning("Кандидаты в обязательные задания не найдены.")
        edited_assignments_df = pd.DataFrame()
    else:
        edited_assignments_df = st.data_editor(
            st.session_state["manual_required_assignments_df"],
            use_container_width=True,
            hide_index=True,
            key="manual_required_assignments_editor",
            disabled=[
                "element_type",
                "context",
                "students_completed",
                "completion_share",
            ],
            column_config={
                "use_as_required": st.column_config.CheckboxColumn(
                    "Учитывать как обязательное",
                    help="Если включено, элемент будет считаться обязательным при расчёте процессных паттернов.",
                    default=False,
                ),
                "element_type": st.column_config.TextColumn(
                    "Тип",
                ),
                "context": st.column_config.TextColumn(
                    "Элемент курса",
                ),
                "students_completed": st.column_config.NumberColumn(
                    "Студентов выполнили",
                ),
                "completion_share": st.column_config.ProgressColumn(
                    "Доля выполнения",
                    min_value=0.0,
                    max_value=1.0,
                    format="%.2f",
                ),
            },
        )

with required_tab_2:
    st.write("### Выбор обязательных тестов")

    if st.session_state["manual_required_tests_df"].empty:
        st.warning("Кандидаты в обязательные тесты не найдены.")
        edited_tests_df = pd.DataFrame()
    else:
        edited_tests_df = st.data_editor(
            st.session_state["manual_required_tests_df"],
            use_container_width=True,
            hide_index=True,
            key="manual_required_tests_editor",
            disabled=[
                "element_type",
                "context",
                "students_completed",
                "completion_share",
            ],
            column_config={
                "use_as_required": st.column_config.CheckboxColumn(
                    "Учитывать как обязательное",
                    help="Если включено, элемент будет считаться обязательным при расчёте процессных паттернов.",
                    default=False,
                ),
                "element_type": st.column_config.TextColumn(
                    "Тип",
                ),
                "context": st.column_config.TextColumn(
                    "Элемент курса",
                ),
                "students_completed": st.column_config.NumberColumn(
                    "Студентов начали / выполнили",
                ),
                "completion_share": st.column_config.ProgressColumn(
                    "Доля выполнения",
                    min_value=0.0,
                    max_value=1.0,
                    format="%.2f",
                ),
            },
        )

if not edited_assignments_df.empty and "use_as_required" in edited_assignments_df.columns:
    selected_required_assignments = (
        edited_assignments_df.loc[
            edited_assignments_df["use_as_required"] == True,
            "context",
        ]
        .dropna()
        .astype(str)
        .tolist()
    )
else:
    selected_required_assignments = []

if not edited_tests_df.empty and "use_as_required" in edited_tests_df.columns:
    selected_required_tests = (
        edited_tests_df.loc[
            edited_tests_df["use_as_required"] == True,
            "context",
        ]
        .dropna()
        .astype(str)
        .tolist()
    )
else:
    selected_required_tests = []

st.session_state["selected_required_assignments"] = selected_required_assignments
st.session_state["selected_required_tests"] = selected_required_tests

summary_col_1, summary_col_2, summary_col_3 = st.columns(3)

summary_col_1.metric(
    "Выбрано обязательных заданий",
    len(selected_required_assignments),
)

summary_col_2.metric(
    "Выбрано обязательных тестов",
    len(selected_required_tests),
)

summary_col_3.metric(
    "Всего обязательных элементов",
    len(selected_required_assignments) + len(selected_required_tests),
)

with st.expander("Показать выбранные обязательные элементы"):
    st.write("**Выбранные обязательные задания:**")
    st.write(selected_required_assignments if selected_required_assignments else "Не выбрано")

    st.write("**Выбранные обязательные тесты:**")
    st.write(selected_required_tests if selected_required_tests else "Не выбрано")


# ============================================================
# 3. Расчёт процессных паттернов
# ============================================================

st.subheader("3. Расчёт процессных паттернов по выбранным обязательным элементам")

st.warning(
    """
    Если не выбрать ни одного обязательного задания или теста,
    показатели полноты выполнения контрольных активностей будут рассчитаны
    относительно 0 обязательных элементов. Перед расчётом проверьте список выше.
    """
)

calculate_button = st.button(
    "Рассчитать процессные паттерны студентов",
    key="calculate_process_behavior_patterns_button",
)

if calculate_button:
    try:
        process_behavior_features_df = calculate_student_process_behavior_features(
            event_log=process_event_log,
            last_period_days=last_period_days,
            fast_assignment_minutes=fast_assignment_minutes,
            fast_test_minutes=fast_test_minutes,
            fast_lecture_minutes=fast_lecture_minutes,
            min_required_completion_share=min_required_completion_share,
            required_assignments=selected_required_assignments,
            required_tests=selected_required_tests,
        )

        if selected_resource_method is not None:
            resource_patterns_df = get_resource_patterns_for_method(
                selected_resource_method
            )
        else:
            resource_patterns_df = pd.DataFrame()

        final_behavior_df = merge_resource_and_process_patterns(
            resource_patterns_df=resource_patterns_df,
            process_features_df=process_behavior_features_df,
        )

        st.session_state["process_behavior_features_df"] = process_behavior_features_df
        st.session_state["final_behavior_df"] = final_behavior_df
        st.session_state["manual_required_assignments_used"] = selected_required_assignments
        st.session_state["manual_required_tests_used"] = selected_required_tests

        st.success("Процессные паттерны рассчитаны по выбранным обязательным элементам.")

    except Exception as e:
        st.exception(e)


# ============================================================
# 4. Отображение результатов
# ============================================================

if st.session_state.get("final_behavior_df") is not None:
    final_behavior_df = st.session_state["final_behavior_df"].copy()

    st.subheader("Итоговые ресурсно-процессные паттерны")

    used_assignments = st.session_state.get("manual_required_assignments_used", [])
    used_tests = st.session_state.get("manual_required_tests_used", [])

    with st.expander("Обязательные элементы, использованные в последнем расчёте"):
        st.write("**Задания:**")
        st.write(used_assignments if used_assignments else "Не выбраны")

        st.write("**Тесты:**")
        st.write(used_tests if used_tests else "Не выбраны")

    important_columns = [
        "student_id",
        "cluster",
        "resource_pattern",
        "process_pattern",
        "final_behavior_pattern",
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
        "median_assignment_upload_delay_min",
        "suspicious_first_assignment_upload_count",
        "measured_first_assignment_upload_count",
        "suspicious_first_assignment_upload_ratio",
        "median_first_assignment_upload_delay_min",
        "trace_length",
        "linearity",
        "complexity",
        "returns_count",
        "variant_frequency",
        "final_behavior_description",
        "process_flags",
        "process_flags_count",
        "required_assignments_count",
        "required_tests_count",
        "required_assignments_list",
        "required_tests_list",
    ]

    available_columns = [
        col for col in important_columns
        if col in final_behavior_df.columns
    ]

    st.dataframe(
        final_behavior_df[available_columns],
        use_container_width=True,
    )

    csv_data = (
        final_behavior_df[available_columns]
        .to_csv(index=False)
        .encode("utf-8-sig")
    )

    st.download_button(
        "Скачать итоговые процессные паттерны CSV",
        data=csv_data,
        file_name="final_process_behavior_patterns.csv",
        mime="text/csv",
        key="download_final_process_behavior_patterns_csv",
    )

    # ------------------------------------------------------------
    # Распределение процессных паттернов
    # ------------------------------------------------------------

    st.subheader("Распределение процессных паттернов")

    if "process_pattern" in final_behavior_df.columns:
        process_pattern_counts = (
            final_behavior_df["process_pattern"]
            .value_counts()
            .reset_index()
        )

        process_pattern_counts.columns = [
            "process_pattern",
            "students_count",
        ]

        st.bar_chart(
            process_pattern_counts.set_index("process_pattern")
        )

        st.dataframe(
            process_pattern_counts,
            use_container_width=True,
        )

    # ------------------------------------------------------------
    # Студенты с рискованными процессными признаками
    # ------------------------------------------------------------

    st.subheader("Студенты, у которых ресурсный паттерн требует уточнения")

    suspicious_patterns = [
        "Неполное выполнение контрольных активностей",
        "Сжатое прохождение курса",
        "Формально комплексное прохождение",
        "Формальное прохождение лекционных элементов",
        "Быстрое прохождение тестов",
        "Подозрительно быстрая первая загрузка ответов",
    ]

    if "process_pattern" in final_behavior_df.columns:
        suspicious_df = final_behavior_df[
            final_behavior_df["process_pattern"].isin(suspicious_patterns)
        ].copy()
    else:
        suspicious_df = pd.DataFrame()

    if suspicious_df.empty:
        st.success(
            "Студентов с авральным, формальным, неполным или быстрым прохождением не найдено."
        )
    else:
        st.warning(
            f"Найдено студентов, у которых ресурсный паттерн требует уточнения: "
            f"{len(suspicious_df)}."
        )

        st.dataframe(
            suspicious_df[available_columns],
            use_container_width=True,
        )

    # ------------------------------------------------------------
    # Анализ одного студента
    # ------------------------------------------------------------

    st.subheader("Подробная интерпретация выбранного студента")

    behavior_student_ids = sorted(
        final_behavior_df["student_id"].astype(str).unique().tolist()
    )

    selected_behavior_student = st.selectbox(
        "Выберите студента для процессной интерпретации",
        options=behavior_student_ids,
        key="selected_behavior_student",
    )

    selected_behavior_row = final_behavior_df[
        final_behavior_df["student_id"].astype(str) == str(selected_behavior_student)
    ].copy()

    if not selected_behavior_row.empty:
        row = selected_behavior_row.iloc[0]

        st.info(
            f"Студент **{selected_behavior_student}**: "
            f"**{row.get('final_behavior_pattern', row.get('process_pattern', 'Нет данных'))}**"
        )

        st.write(row.get("final_behavior_description", ""))

        st.dataframe(
            selected_behavior_row[available_columns],
            use_container_width=True,
        )


# ============================================================
# К каким кластерам относится выбранный студент
# ============================================================

st.subheader("Кластеры выбранного студента")

all_student_ids_for_cluster_info = sorted(
    process_event_log["student_id"].astype(str).unique().tolist()
)

selected_student_for_cluster_info = st.selectbox(
    "Выберите студента для просмотра его кластеров",
    options=all_student_ids_for_cluster_info,
    key="process_student_cluster_info_selectbox",
)

student_cluster_rows = []

for method_name, method_result in METHOD_RESULTS.items():
    if method_result is None:
        continue

    clustered_students_df = build_clustered_students(
        method_name,
        method_result,
    )

    if clustered_students_df.empty:
        continue

    student_row = clustered_students_df[
        clustered_students_df["student_id"].astype(str)
        == str(selected_student_for_cluster_info)
    ].copy()

    if student_row.empty:
        continue

    cluster_value = student_row["cluster"].iloc[0]
    pattern_value = student_row["suggested_name"].iloc[0]

    probability_value = None

    if isinstance(method_result, dict):
        result_df = method_result.get("result_df")
    else:
        result_df = None

    if (
        result_df is not None
        and "cluster_probability" in result_df.columns
        and "student_id" in result_df.columns
    ):
        probability_rows = result_df[
            result_df["student_id"].astype(str)
            == str(selected_student_for_cluster_info)
        ]

        if not probability_rows.empty:
            probability_value = probability_rows["cluster_probability"].iloc[0]

    student_cluster_rows.append(
        {
            "method": method_name,
            "cluster": cluster_value,
            "pattern": pattern_value,
            "cluster_probability": probability_value,
        }
    )

if student_cluster_rows:
    student_clusters_df = pd.DataFrame(student_cluster_rows)

    st.dataframe(
        student_clusters_df,
        use_container_width=True,
    )

    unique_patterns = (
        student_clusters_df["pattern"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if len(unique_patterns) <= 1:
        st.success(
            f"У студента {selected_student_for_cluster_info} во всех доступных "
            f"методах совпадает паттерн: {unique_patterns[0]}."
        )
    else:
        st.warning(
            f"У студента {selected_student_for_cluster_info} разные методы дали "
            f"разные паттерны: {', '.join(unique_patterns)}."
        )
else:
    st.info("Для выбранного студента нет данных по кластерам.")
    st.info("Для выбранного студента нет данных по кластерам.")