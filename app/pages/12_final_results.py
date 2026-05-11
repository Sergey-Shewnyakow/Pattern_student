import pandas as pd
import plotly.express as px
import streamlit as st

from src.state import init_session_state
from src.ui_styles import apply_global_styles
from src.cluster_naming import build_cluster_names

try:
    from src.cluster_name_editor import apply_custom_cluster_names
except Exception:
    apply_custom_cluster_names = None

try:
    from streamlit_plotly_events import plotly_events
except Exception:
    plotly_events = None

st.set_page_config(
    page_title="Final Behavior Patterns",
    layout="wide",
)

init_session_state()
apply_global_styles()


st.title("Итоговые паттерны поведения студентов")

st.write(
    "На этой странице объединяются результаты машинного обучения и process mining. "
    "Кластеризация показывает ресурсный паттерн студента, то есть какие элементы курса "
    "он использовал. Process mining показывает процессный паттерн, то есть как именно "
    "студент проходил курс: регулярно, аврально, формально, с быстрым прохождением "
    "лекций или тестов. Совпадающие комбинации считаются едиными итоговыми "
    "паттернами поведения."
)


# ------------------------------------------------------------
# Получение результатов методов
# ------------------------------------------------------------
def get_result_from_session(possible_keys: list[str]):
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
}


METHOD_KEY_MAP = {
    "KMeans": "kmeans",
    "Agglomerative": "agglomerative",
    "GMM": "gmm",
    "HDBSCAN": "hdbscan",
    "DEC": "dec",
}


def get_cluster_names_for_method(method_name: str, result: dict) -> pd.DataFrame:
    """
    Получает названия кластеров с учётом ручных правок.
    """
    if result is None:
        return pd.DataFrame()

    result_df = result.get("result_df")
    cluster_profiles = result.get("cluster_profiles")

    if result_df is None or cluster_profiles is None:
        return pd.DataFrame()

    if result_df.empty:
        return pd.DataFrame()

    if cluster_profiles.empty:
        names_df = pd.DataFrame(
            columns=["cluster", "cluster_size", "suggested_name", "description"]
        )
    else:
        if "cluster" in result_df.columns:
            non_noise_result_df = result_df[result_df["cluster"] != -1].copy()
        else:
            non_noise_result_df = result_df.copy()

        if non_noise_result_df.empty:
            names_df = pd.DataFrame(
                columns=["cluster", "cluster_size", "suggested_name", "description"]
            )
        else:
            names_df = build_cluster_names(
                result_df=non_noise_result_df,
                cluster_profiles=cluster_profiles,
            )

    if "cluster" in result_df.columns:
        noise_count = int((result_df["cluster"] == -1).sum())
    else:
        noise_count = 0

    if noise_count > 0:
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
        method_key = METHOD_KEY_MAP.get(method_name, method_name.lower())

        names_df = apply_custom_cluster_names(
            method_key=method_key,
            cluster_names_df=names_df,
        )

    return names_df


def build_method_student_table(method_name: str, result: dict) -> pd.DataFrame:
    """
    Возвращает student_id + кластер + название паттерна для одного метода.
    """
    if result is None:
        return pd.DataFrame()

    result_df = result.get("result_df")

    if result_df is None or result_df.empty:
        return pd.DataFrame()

    if "student_id" not in result_df.columns or "cluster" not in result_df.columns:
        return pd.DataFrame()

    names_df = get_cluster_names_for_method(method_name, result)

    if names_df.empty:
        return pd.DataFrame()

    table = result_df[["student_id", "cluster"]].copy()
    table["student_id"] = table["student_id"].astype(str)

    table = table.merge(
        names_df[["cluster", "suggested_name"]],
        on="cluster",
        how="left",
    )

    table = table.rename(
        columns={
            "cluster": f"{method_name}_cluster",
            "suggested_name": f"{method_name}_resource_pattern",
        }
    )

    if "cluster_probability" in result_df.columns:
        table[f"{method_name}_cluster_probability"] = result_df[
            "cluster_probability"
        ].values

    return table


def build_ml_patterns_table(selected_methods: list[str]) -> pd.DataFrame:
    """
    Собирает единую таблицу ML-паттернов по выбранным методам.
    """
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

    return ml_df


def normalize_pattern_value(value) -> str:
    if value is None or pd.isna(value):
        return "Нет данных"

    return str(value).strip()


def build_resource_summary(row: pd.Series, selected_methods: list[str]) -> pd.Series:
    """
    Формирует итоговую ресурсную интерпретацию по выбранным ML-методам.

    Если все методы дали одинаковое название паттерна — считаем, что ресурсный
    паттерн устойчивый. Если названия разные — фиксируем расхождение.
    """
    method_patterns = {}

    for method_name in selected_methods:
        col = f"{method_name}_resource_pattern"

        if col in row.index:
            method_patterns[method_name] = normalize_pattern_value(row[col])

    available_patterns = [
        pattern for pattern in method_patterns.values()
        if pattern != "Нет данных"
    ]

    unique_patterns = sorted(set(available_patterns))

    if not available_patterns:
        resource_consensus_status = "Нет данных"
        resource_consensus_pattern = "Нет данных"
        resource_signature = "Нет данных"
    elif len(unique_patterns) == 1:
        resource_consensus_status = "ML-методы совпали"
        resource_consensus_pattern = unique_patterns[0]
        resource_signature = unique_patterns[0]
    else:
        resource_consensus_status = "ML-методы различаются"
        resource_consensus_pattern = "Смешанный ресурсный профиль"
        resource_signature = " | ".join(
            [
                f"{method}: {pattern}"
                for method, pattern in method_patterns.items()
                if pattern != "Нет данных"
            ]
        )

    return pd.Series(
        {
            "resource_consensus_status": resource_consensus_status,
            "resource_consensus_pattern": resource_consensus_pattern,
            "resource_signature": resource_signature,
            "resource_unique_patterns_count": len(unique_patterns),
        }
    )


def build_final_behavior_table(
    ml_df: pd.DataFrame,
    process_df: pd.DataFrame,
    selected_methods: list[str],
) -> pd.DataFrame:
    """
    Объединяет ML-паттерны и process mining признаки.
    """
    if ml_df is None or ml_df.empty:
        return pd.DataFrame()

    result_df = ml_df.copy()
    result_df["student_id"] = result_df["student_id"].astype(str)

    if process_df is not None and not process_df.empty:
        process_df = process_df.copy()
        process_df["student_id"] = process_df["student_id"].astype(str)

        process_columns = [
            col for col in [
                "student_id",
                "process_pattern",
                "process_flags",
                "process_flags_count",
                "final_behavior_pattern",
                "final_behavior_description",

                "completed_assignments_count",
                "expected_assignments_count",
                "assignment_completion_ratio",

                "completed_tests_count",
                "expected_tests_count",
                "test_completion_ratio",
                "control_completion_ratio",

                "process_total_events",
                "process_active_days",
                "max_day_activity_ratio",
                "top_2_days_activity_ratio",
                "top_3_days_activity_ratio",
                "days_to_80_percent_events",

                "fast_lecture_completion_count",
                "measured_lecture_completion_count",
                "fast_lecture_completion_ratio",

                "fast_test_completion_count",
                "measured_test_completion_count",
                "fast_test_completion_ratio",

                "suspicious_first_assignment_upload_count",
                "measured_first_assignment_upload_count",
                "suspicious_first_assignment_upload_ratio",
            ]
            if col in process_df.columns
        ]

        result_df = result_df.merge(
            process_df[process_columns],
            on="student_id",
            how="left",
        )
    else:
        result_df["process_pattern"] = "Нет данных process mining"
        result_df["process_flags"] = "Нет данных process mining"
        result_df["process_flags_count"] = None
        result_df["final_behavior_description"] = "Process mining признаки не рассчитаны."

    resource_summary_df = result_df.apply(
        lambda row: build_resource_summary(row, selected_methods),
        axis=1,
    )

    result_df = pd.concat(
        [result_df, resource_summary_df],
        axis=1,
    )

    result_df["process_pattern"] = result_df["process_pattern"].fillna(
        "Нет данных process mining"
    )

    result_df["process_flags"] = result_df["process_flags"].fillna(
        "Нет дополнительных process-признаков"
    )

    result_df["final_pattern_key"] = (
        result_df["resource_signature"].astype(str)
        + " + "
        + result_df["process_pattern"].astype(str)
        + " + "
        + result_df["process_flags"].astype(str)
    )

    result_df["final_pattern_short"] = (
        result_df["resource_consensus_pattern"].astype(str)
        + " + "
        + result_df["process_pattern"].astype(str)
    )

    result_df["final_behavior_interpretation"] = result_df.apply(
        build_final_interpretation,
        axis=1,
    )

    return result_df


def build_final_interpretation(row: pd.Series) -> str:
    resource_status = row.get("resource_consensus_status", "Нет данных")
    resource_pattern = row.get("resource_consensus_pattern", "Нет данных")
    resource_signature = row.get("resource_signature", "Нет данных")
    process_pattern = row.get("process_pattern", "Нет данных")
    process_flags = row.get("process_flags", "Нет данных")

    if resource_status == "ML-методы совпали":
        resource_text = (
            f"Методы машинного обучения согласованно определили ресурсный паттерн: "
            f"«{resource_pattern}»."
        )
    elif resource_status == "ML-методы различаются":
        resource_text = (
            "Методы машинного обучения дали разные ресурсные паттерны, поэтому "
            f"профиль считается смешанным. Детализация: {resource_signature}."
        )
    else:
        resource_text = "Ресурсный паттерн не определён."

    process_text = (
        f"Process mining определил процессный паттерн: «{process_pattern}». "
        f"Дополнительные признаки: {process_flags}."
    )

    return resource_text + " " + process_text


def build_unique_final_patterns(final_df: pd.DataFrame) -> pd.DataFrame:
    """
    Группирует студентов по уникальному итоговому паттерну.
    """
    if final_df is None or final_df.empty:
        return pd.DataFrame()

    group_cols = [
        "resource_consensus_status",
        "resource_consensus_pattern",
        "resource_signature",
        "process_pattern",
        "process_flags",
        "final_pattern_short",
        "final_pattern_key",
    ]

    existing_group_cols = [
        col for col in group_cols if col in final_df.columns
    ]

    grouped = (
        final_df.groupby(existing_group_cols)
        .agg(
            students_count=("student_id", "nunique"),
            avg_events=("process_total_events", "mean"),
            avg_active_days=("process_active_days", "mean"),
            avg_control_completion=("control_completion_ratio", "mean"),
            avg_days_to_80=("days_to_80_percent_events", "mean"),
            students=("student_id", lambda s: ", ".join(s.astype(str).head(20))),
        )
        .reset_index()
        .sort_values("students_count", ascending=False)
        .reset_index(drop=True)
    )

    grouped["avg_events"] = grouped["avg_events"].round(2)
    grouped["avg_active_days"] = grouped["avg_active_days"].round(2)
    grouped["avg_control_completion"] = grouped["avg_control_completion"].round(3)
    grouped["avg_days_to_80"] = grouped["avg_days_to_80"].round(2)

    grouped["pattern_id"] = range(1, len(grouped) + 1)

    return grouped

def build_interactive_student_tree(final_df: pd.DataFrame):
    """
    Строит интерактивное дерево:
    ресурсный паттерн -> process pattern -> process flags -> студент.
    """
    tree_df = final_df.copy()

    tree_df["student_id"] = tree_df["student_id"].astype(str)

    tree_df["tree_root"] = "Все итоговые паттерны"

    tree_df["tree_resource"] = tree_df["resource_consensus_pattern"].fillna(
        "Нет ресурсного паттерна"
    )

    tree_df["tree_process"] = tree_df["process_pattern"].fillna(
        "Нет process pattern"
    )

    tree_df["tree_flags"] = tree_df["process_flags"].fillna(
        "Нет process flags"
    )

    # Чтобы дерево не превращалось в длинную строку из ID
    tree_df["tree_student"] = tree_df["student_id"].apply(
        lambda x: f"Студент {str(x)[:8]}...{str(x)[-6:]}"
    )

    tree_df["count"] = 1

    fig = px.treemap(
        tree_df,
        path=[
            "tree_root",
            "tree_resource",
            "tree_process",
            "tree_flags",
            "tree_student",
        ],
        values="count",
        custom_data=[
            "student_id",
            "final_pattern_short",
            "resource_consensus_pattern",
            "process_pattern",
            "process_flags",
        ],
        title="Интерактивное дерево: итоговый паттерн → process-признаки → студент",
        template="plotly_white",
    )

    fig.update_traces(
        textinfo="label+value",
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Студентов в узле: %{value}<br>"
            "ID студента: %{customdata[0]}<br>"
            "Итоговый паттерн: %{customdata[1]}<br>"
            "Ресурсный паттерн: %{customdata[2]}<br>"
            "Process pattern: %{customdata[3]}<br>"
            "Process flags: %{customdata[4]}<br>"
            "<extra></extra>"
        ),
        marker=dict(
            line=dict(width=1, color="white")
        ),
    )

    fig.update_layout(
        height=850,
        margin=dict(t=70, l=20, r=20, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black", size=12),
    )

    return fig

def extract_student_id_from_tree_click(clicked_points, final_df: pd.DataFrame):
    """
    Достаёт student_id из клика по treemap.

    Работает даже если кликнули по листу студента,
    где label отображается сокращённо, а полный student_id лежит в customdata.
    """
    if not clicked_points:
        return None

    point = clicked_points[0]

    all_student_ids = final_df["student_id"].astype(str).values

    customdata = point.get("customdata")

    if customdata is not None:
        try:
            possible_student_id = str(customdata[0])

            if possible_student_id in all_student_ids:
                return possible_student_id
        except Exception:
            pass

    label = str(point.get("label", "")).strip()

    if label.startswith("Студент "):
        possible_student_id = label.replace("Студент ", "").strip()

        if possible_student_id in all_student_ids:
            return possible_student_id

    if label in all_student_ids:
        return label

    return None


def get_process_event_log_from_session():
    """
    Берёт подготовленный process event log из session_state.
    Он появляется после страницы Process Mining.
    """
    event_log = st.session_state.get("process_event_log")

    if event_log is None or event_log.empty:
        return pd.DataFrame()

    event_log = event_log.copy()
    event_log["student_id"] = event_log["student_id"].astype(str)
    event_log["timestamp"] = pd.to_datetime(event_log["timestamp"], errors="coerce")
    event_log = event_log[event_log["timestamp"].notna()].copy()

    return event_log


def build_student_path_table(
    event_log: pd.DataFrame,
    student_id: str,
) -> pd.DataFrame:
    """
    Строит полный путь прохождения курса конкретного студента.
    """
    if event_log is None or event_log.empty:
        return pd.DataFrame()

    student_log = event_log[
        event_log["student_id"].astype(str) == str(student_id)
    ].copy()

    if student_log.empty:
        return pd.DataFrame()

    student_log = student_log.sort_values("timestamp").reset_index(drop=True)

    student_log["step"] = range(1, len(student_log) + 1)

    student_log["previous_time"] = student_log["timestamp"].shift(1)

    student_log["minutes_from_previous"] = (
        (student_log["timestamp"] - student_log["previous_time"])
        .dt.total_seconds()
        .div(60)
        .round(2)
    )

    student_log["date"] = student_log["timestamp"].dt.date.astype(str)
    student_log["time"] = student_log["timestamp"].dt.time.astype(str)

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

    available_columns = [
        col for col in columns if col in student_log.columns
    ]

    return student_log[available_columns]


def build_student_timeline_chart(student_path_df: pd.DataFrame):
    """
    Рисует timeline действий студента.
    """
    if student_path_df is None or student_path_df.empty:
        return None

    fig = px.scatter(
        student_path_df,
        x="timestamp",
        y="process_activity",
        color="process_activity",
        hover_data=[
            col for col in [
                "step",
                "component",
                "context",
                "activity",
                "minutes_from_previous",
            ]
            if col in student_path_df.columns
        ],
        title="Путь прохождения курса выбранного студента",
    )

    fig.update_layout(
        xaxis_title="Время",
        yaxis_title="Действие",
        height=700,
        showlegend=False,
    )

    return fig


def _normalize_context(value) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return " ".join(text.split())


def build_student_component_timings(
    event_log: pd.DataFrame,
    student_id: str,
) -> dict:
    """
    Считает основные тайминги студента по заданиям, тестам и лекциям.
    """
    if event_log is None or event_log.empty:
        return {
            "assignments": pd.DataFrame(),
            "tests": pd.DataFrame(),
            "lectures": pd.DataFrame(),
        }

    student_log = event_log[
        event_log["student_id"].astype(str) == str(student_id)
    ].copy()

    if student_log.empty:
        return {
            "assignments": pd.DataFrame(),
            "tests": pd.DataFrame(),
            "lectures": pd.DataFrame(),
        }

    student_log = student_log.sort_values("timestamp").copy()
    student_log["context_clean"] = student_log["context"].apply(_normalize_context)

    # ------------------------------------------------------------
    # Задания
    # ------------------------------------------------------------
    assignment_rows = []

    assignment_log = student_log[
        student_log["component"].astype(str).str.contains(
            "Задание|Ответ в виде файла",
            case=False,
            na=False,
            regex=True,
        )
    ].copy()

    assignment_open_actions = [
        "Открытие практического задания",
        "Открытие формы ответа",
        "Просмотр статуса задания",
    ]

    assignment_submit_actions = [
        "Загрузка файла",
        "Отправка практического задания",
        "Обновление ответа",
    ]

    for context, group in assignment_log.groupby("context_clean"):
        first_open = group[
            group["process_activity"].isin(assignment_open_actions)
        ]["timestamp"].min()

        first_submit = group[
            group["process_activity"].isin(assignment_submit_actions)
        ]["timestamp"].min()

        if pd.notna(first_open) and pd.notna(first_submit):
            delay_min = round((first_submit - first_open).total_seconds() / 60, 2)
        else:
            delay_min = None

        assignment_rows.append(
            {
                "type": "Задание",
                "context": context,
                "first_open_time": first_open,
                "first_submit_time": first_submit,
                "minutes_from_first_open_to_submit": delay_min,
                "events_count": len(group),
            }
        )

    assignments_df = pd.DataFrame(assignment_rows)

    # ------------------------------------------------------------
    # Тесты
    # ------------------------------------------------------------
    test_rows = []

    test_log = student_log[
        student_log["component"].astype(str).str.contains(
            "Тест",
            case=False,
            na=False,
            regex=False,
        )
    ].copy()

    for context, group in test_log.groupby("context_clean"):
        first_open = group[
            group["process_activity"].isin(["Открытие теста"])
        ]["timestamp"].min()

        first_start = group[
            group["process_activity"].isin(["Начало теста"])
        ]["timestamp"].min()

        first_finish = group[
            group["process_activity"].isin(["Завершение теста"])
        ]["timestamp"].min()

        if pd.notna(first_start) and pd.notna(first_finish):
            duration_min = round((first_finish - first_start).total_seconds() / 60, 2)
        else:
            duration_min = None

        test_rows.append(
            {
                "type": "Тест",
                "context": context,
                "first_open_time": first_open,
                "first_start_time": first_start,
                "first_finish_time": first_finish,
                "test_duration_min": duration_min,
                "was_started": pd.notna(first_start),
                "events_count": len(group),
            }
        )

    tests_df = pd.DataFrame(test_rows)

    # ------------------------------------------------------------
    # Лекции
    # ------------------------------------------------------------
    lecture_rows = []

    lecture_log = student_log[
        (
            student_log["component"].astype(str).str.contains(
                "Лекция",
                case=False,
                na=False,
                regex=False,
            )
        )
        | (
            student_log["context_clean"].astype(str).str.contains(
                "Лекция",
                case=False,
                na=False,
                regex=False,
            )
        )
    ].copy()

    for context, group in lecture_log.groupby("context_clean"):
        first_open = group[
            group["process_activity"].isin(["Открытие лекции"])
        ]["timestamp"].min()

        first_start = group[
            group["process_activity"].isin(["Начало лекции", "Повтор лекции"])
        ]["timestamp"].min()

        first_finish = group[
            group["process_activity"].isin(["Завершение лекции"])
        ]["timestamp"].min()

        if pd.notna(first_start) and pd.notna(first_finish):
            duration_min = round((first_finish - first_start).total_seconds() / 60, 2)
        else:
            duration_min = None

        lecture_rows.append(
            {
                "type": "Лекция",
                "context": context,
                "first_open_time": first_open,
                "first_start_time": first_start,
                "first_finish_time": first_finish,
                "lecture_duration_min": duration_min,
                "events_count": len(group),
            }
        )

    lectures_df = pd.DataFrame(lecture_rows)

    return {
        "assignments": assignments_df,
        "tests": tests_df,
        "lectures": lectures_df,
    }


# ------------------------------------------------------------
# Проверка доступных данных
# ------------------------------------------------------------
ready_methods = [
    method_name for method_name, result in METHOD_RESULTS.items()
    if result is not None
]

if not ready_methods:
    st.warning(
        "Нет результатов кластеризации. Сначала запустите хотя бы один метод "
        "машинного обучения: KMeans, GMM, Agglomerative, HDBSCAN или DEC."
    )
    st.stop()

process_df = st.session_state.get("final_behavior_df")

if process_df is None or process_df.empty:
    process_df = st.session_state.get("process_behavior_features_df")

if process_df is None or process_df.empty:
    st.warning(
        "Process mining признаки пока не рассчитаны. "
        "Сначала на странице Process Mining нажмите кнопку "
        "«Рассчитать процессные паттерны студентов». "
        "Страница всё равно может показать ML-паттерны, но итоговый анализ будет неполным."
    )



# ------------------------------------------------------------
# Выбор методов
# ------------------------------------------------------------
st.subheader("Настройки итоговой интерпретации")

selected_methods = st.multiselect(
    "Выберите методы машинного обучения, которые нужно учитывать",
    options=ready_methods,
    default=ready_methods,
    key="final_results_selected_methods",
)

if not selected_methods:
    st.warning("Выберите хотя бы один метод машинного обучения.")
    st.stop()

st.caption(
    "Если выбранные методы дают одинаковое название ресурсного паттерна, считается, "
    "что ресурсный профиль устойчивый. Если названия различаются, студент получает "
    "смешанный ресурсный профиль."
)


# ------------------------------------------------------------
# Построение итоговой таблицы
# ------------------------------------------------------------
ml_df = build_ml_patterns_table(selected_methods)

if ml_df.empty:
    st.error("Не удалось собрать таблицу ML-паттернов.")
    st.stop()

final_df = build_final_behavior_table(
    ml_df=ml_df,
    process_df=process_df,
    selected_methods=selected_methods,
)

if final_df.empty:
    st.error("Не удалось построить итоговую таблицу паттернов.")
    st.stop()

unique_patterns_df = build_unique_final_patterns(final_df)

st.session_state["final_results_df"] = final_df
st.session_state["unique_final_patterns_df"] = unique_patterns_df


# ------------------------------------------------------------
# Основные метрики
# ------------------------------------------------------------
st.subheader("Итоговые показатели")

metric_1, metric_2, metric_3, metric_4 = st.columns(4)

metric_1.metric("Студентов", final_df["student_id"].nunique())
metric_2.metric("Уникальных итоговых паттернов", len(unique_patterns_df))

if "resource_consensus_status" in final_df.columns:
    agreed_count = int(
        (final_df["resource_consensus_status"] == "ML-методы совпали").sum()
    )
else:
    agreed_count = 0

metric_3.metric("ML-методы совпали", agreed_count)

if "process_flags_count" in final_df.columns:
    complex_process_count = int(
        (final_df["process_flags_count"].fillna(0) >= 2).sum()
    )
else:
    complex_process_count = 0

metric_4.metric("2+ process-признака", complex_process_count)


# ------------------------------------------------------------
# Таблица итоговых паттернов по студентам
# ------------------------------------------------------------
st.subheader("Итоговая таблица студентов")

base_columns = [
    "student_id",
    "resource_consensus_status",
    "resource_consensus_pattern",
    "resource_signature",
    "process_pattern",
    "process_flags",
    "process_flags_count",
    "final_pattern_short",
    "final_behavior_interpretation",
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
    "completed_tests_count",
    "expected_tests_count",
    "control_completion_ratio",
    "process_total_events",
    "process_active_days",
    "days_to_80_percent_events",
    "top_2_days_activity_ratio",
    "fast_lecture_completion_ratio",
    "fast_test_completion_ratio",
    "suspicious_first_assignment_upload_ratio",
]

display_columns = [
    col for col in base_columns + method_columns + process_metric_columns
    if col in final_df.columns
]

st.dataframe(
    final_df[display_columns],
    use_container_width=True,
)

csv_data = final_df[display_columns].to_csv(index=False).encode("utf-8-sig")

st.download_button(
    "Скачать итоговую таблицу студентов CSV",
    data=csv_data,
    file_name="final_behavior_results_by_students.csv",
    mime="text/csv",
    key="download_final_behavior_results_by_students",
)


# ------------------------------------------------------------
# Уникальные итоговые паттерны
# ------------------------------------------------------------
st.subheader("Уникальные итоговые паттерны поведения")

unique_display_columns = [
    "pattern_id",
    "students_count",
    "resource_consensus_status",
    "resource_consensus_pattern",
    "resource_signature",
    "process_pattern",
    "process_flags",
    "final_pattern_short",
    "avg_events",
    "avg_active_days",
    "avg_control_completion",
    "avg_days_to_80",
    "students",
]

unique_display_columns = [
    col for col in unique_display_columns
    if col in unique_patterns_df.columns
]

st.dataframe(
    unique_patterns_df[unique_display_columns],
    use_container_width=True,
)

unique_csv_data = (
    unique_patterns_df[unique_display_columns]
    .to_csv(index=False)
    .encode("utf-8-sig")
)

st.download_button(
    "Скачать уникальные итоговые паттерны CSV",
    data=unique_csv_data,
    file_name="unique_final_behavior_patterns.csv",
    mime="text/csv",
    key="download_unique_final_behavior_patterns",
)


# ------------------------------------------------------------
# Визуализации
# ------------------------------------------------------------
st.subheader("Визуализация итоговых результатов")

viz_tab_1, viz_tab_2, viz_tab_3, viz_tab_4 = st.tabs(
    [
        "Итоговые паттерны",
        "Process mining паттерны",
        "Согласованность ML-методов",
        "Ресурсный × процессный паттерн",
    ]
)

with viz_tab_1:
    top_n_patterns = st.slider(
        "Сколько итоговых паттернов показать",
        min_value=5,
        max_value=min(50, max(5, len(unique_patterns_df))),
        value=min(15, max(5, len(unique_patterns_df))),
        step=1,
        key="final_patterns_top_n",
    )

    top_patterns_df = unique_patterns_df.head(top_n_patterns).copy()

    fig_final_patterns = px.bar(
        top_patterns_df.sort_values("students_count"),
        x="students_count",
        y="final_pattern_short",
        orientation="h",
        color="process_pattern",
        title="Самые распространённые итоговые паттерны поведения",
        hover_data=[
            col for col in [
                "resource_signature",
                "process_flags",
                "avg_active_days",
                "avg_days_to_80",
            ]
            if col in top_patterns_df.columns
        ],
    )

    fig_final_patterns.update_layout(
        xaxis_title="Число студентов",
        yaxis_title="Итоговый паттерн",
        height=max(600, top_n_patterns * 35),
    )

    st.plotly_chart(
        fig_final_patterns,
        use_container_width=True,
    )

    fig_treemap = px.treemap(
        unique_patterns_df,
        path=[
            "resource_consensus_pattern",
            "process_pattern",
            "process_flags",
        ],
        values="students_count",
        title="Дерево итоговых паттернов: ресурсный паттерн → process pattern → process flags",
    )

    st.plotly_chart(
        fig_treemap,
        use_container_width=True,
    )

with viz_tab_2:
    if "process_pattern" in final_df.columns:
        process_counts_df = (
            final_df["process_pattern"]
            .fillna("Нет данных")
            .value_counts()
            .reset_index()
        )

        process_counts_df.columns = ["process_pattern", "students_count"]

        fig_process = px.bar(
            process_counts_df.sort_values("students_count"),
            x="students_count",
            y="process_pattern",
            orientation="h",
            title="Распределение process mining паттернов",
        )

        fig_process.update_layout(
            xaxis_title="Число студентов",
            yaxis_title="Process pattern",
            height=max(500, len(process_counts_df) * 35),
        )

        st.plotly_chart(
            fig_process,
            use_container_width=True,
        )

    if "process_flags" in final_df.columns:
        flags_rows = []

        for _, row in final_df.iterrows():
            flags_text = row.get("process_flags", "")

            if pd.isna(flags_text):
                continue

            flags = [
                flag.strip()
                for flag in str(flags_text).split(";")
                if flag.strip()
                and flag.strip() != "Нет выраженных дополнительных признаков"
                and flag.strip() != "Нет дополнительных process-признаков"
            ]

            for flag in flags:
                flags_rows.append(
                    {
                        "student_id": row["student_id"],
                        "process_flag": flag,
                    }
                )

        flags_df = pd.DataFrame(flags_rows)

        if not flags_df.empty:
            flags_counts_df = (
                flags_df["process_flag"]
                .value_counts()
                .reset_index()
            )

            flags_counts_df.columns = ["process_flag", "students_count"]

            fig_flags = px.bar(
                flags_counts_df.sort_values("students_count"),
                x="students_count",
                y="process_flag",
                orientation="h",
                title="Распределение дополнительных process-признаков",
            )

            fig_flags.update_layout(
                xaxis_title="Число студентов",
                yaxis_title="Process flag",
                height=max(500, len(flags_counts_df) * 35),
            )

            st.plotly_chart(
                fig_flags,
                use_container_width=True,
            )

with viz_tab_3:
    if "resource_consensus_status" in final_df.columns:
        consensus_counts_df = (
            final_df["resource_consensus_status"]
            .fillna("Нет данных")
            .value_counts()
            .reset_index()
        )

        consensus_counts_df.columns = [
            "resource_consensus_status",
            "students_count",
        ]

        fig_consensus = px.pie(
            consensus_counts_df,
            names="resource_consensus_status",
            values="students_count",
            title="Согласованность ресурсных паттернов между ML-методами",
        )

        st.plotly_chart(
            fig_consensus,
            use_container_width=True,
        )

        st.dataframe(
            consensus_counts_df,
            use_container_width=True,
        )

with viz_tab_4:
    if (
        "resource_consensus_pattern" in final_df.columns
        and "process_pattern" in final_df.columns
    ):
        cross_df = (
            final_df.groupby(
                ["resource_consensus_pattern", "process_pattern"]
            )
            .size()
            .reset_index(name="students_count")
        )

        heatmap_df = cross_df.pivot_table(
            index="resource_consensus_pattern",
            columns="process_pattern",
            values="students_count",
            fill_value=0,
        )

        fig_heatmap = px.imshow(
            heatmap_df,
            text_auto=True,
            aspect="auto",
            title="Heatmap: ресурсный паттерн × process mining паттерн",
        )

        fig_heatmap.update_layout(
            xaxis_title="Process mining паттерн",
            yaxis_title="Ресурсный паттерн",
            height=700,
        )

        st.plotly_chart(
            fig_heatmap,
            use_container_width=True,
        )

        st.dataframe(
            cross_df.sort_values("students_count", ascending=False),
            use_container_width=True,
        )


# ------------------------------------------------------------
# Анализ выбранного студента
# ------------------------------------------------------------
st.subheader("Анализ выбранного студента")

# ------------------------------------------------------------
# ------------------------------------------------------------
# Интерактивное дерево со студентами
# ------------------------------------------------------------
st.subheader("Интерактивное дерево итоговых паттернов со студентами")

st.write(
    "В этом дереве можно раскрывать итоговые паттерны до конкретных студентов. "
    "Если клик по студенту не срабатывает, выберите студента вручную ниже."
)

interactive_tree_fig = build_interactive_student_tree(final_df)

selected_student_from_tree = None

if plotly_events is not None:
    clicked_points = plotly_events(
        interactive_tree_fig,
        click_event=True,
        hover_event=False,
        select_event=False,
        override_height=850,
        key="final_results_interactive_student_tree",
    )

    selected_student_from_tree = extract_student_id_from_tree_click(
        clicked_points=clicked_points,
        final_df=final_df,
    )

    if selected_student_from_tree is not None:
        st.session_state["final_results_tree_selected_student"] = selected_student_from_tree
        st.session_state["final_results_manual_student_for_path_fixed"] = selected_student_from_tree

else:
    st.plotly_chart(
        interactive_tree_fig,
        use_container_width=True,
    )

    st.warning(
        "Клик по дереву недоступен. Для клика установите пакет: "
        "`pip install streamlit-plotly-events`."
    )


# ------------------------------------------------------------
# Выбор студента
# ------------------------------------------------------------
st.subheader("Выбор студента для просмотра пути")

all_students_for_path = sorted(
    final_df["student_id"].astype(str).unique().tolist()
)

clicked_student = st.session_state.get("final_results_tree_selected_student")

if clicked_student in all_students_for_path:
    default_student = clicked_student
else:
    default_student = all_students_for_path[0]

if "final_results_manual_student_for_path_fixed" not in st.session_state:
    st.session_state["final_results_manual_student_for_path_fixed"] = default_student

if clicked_student in all_students_for_path:
    st.session_state["final_results_manual_student_for_path_fixed"] = clicked_student

selected_student_manual = st.selectbox(
    "Выберите студента",
    options=all_students_for_path,
    key="final_results_manual_student_for_path_fixed",
)

tree_selected_student = selected_student_manual

if clicked_student is not None:
    st.success(
        f"Студент выбран из дерева и подставлен в список: {clicked_student}"
    )

if selected_student_from_tree is not None:
    tree_selected_student = selected_student_from_tree


# ------------------------------------------------------------
# Вывод информации по студенту
# ------------------------------------------------------------
if tree_selected_student is not None:
    st.success(f"Выбран студент: {tree_selected_student}")

    tree_student_row = final_df[
        final_df["student_id"].astype(str) == str(tree_selected_student)
    ].copy()

    if not tree_student_row.empty:
        tree_row = tree_student_row.iloc[0]

        st.info(
            f"Итоговый паттерн студента **{tree_selected_student}**: "
            f"**{tree_row.get('final_pattern_short', 'Нет данных')}**"
        )

        st.write(tree_row.get("final_behavior_interpretation", ""))

    process_event_log_for_path = get_process_event_log_from_session()

    if process_event_log_for_path.empty:
        st.warning(
            "Подготовленный process event log не найден. "
            "Сначала откройте страницу Process Mining и нажмите «Построить event log». "
            "После этого вернитесь на страницу итоговых результатов."
        )
    else:
        st.subheader("Путь прохождения курса выбранного студента")

        student_path_df = build_student_path_table(
            event_log=process_event_log_for_path,
            student_id=tree_selected_student,
        )

        if student_path_df.empty:
            st.warning(
                "Для выбранного студента нет событий в process event log. "
                "Возможно, студент был исключён как не-студент или отсутствует "
                "в подготовленном process mining логе."
            )
        else:
            path_fig = build_student_timeline_chart(student_path_df)

            if path_fig is not None:
                st.plotly_chart(
                    path_fig,
                    use_container_width=True,
                )

            st.dataframe(
                student_path_df,
                use_container_width=True,
            )

            path_csv = student_path_df.to_csv(index=False).encode("utf-8-sig")

            st.download_button(
                "Скачать путь выбранного студента CSV",
                data=path_csv,
                file_name=f"student_{tree_selected_student}_path.csv",
                mime="text/csv",
                key="download_tree_student_path_csv_fixed",
            )

        st.subheader("Основные тайминги выбранного студента")

        timings = build_student_component_timings(
            event_log=process_event_log_for_path,
            student_id=tree_selected_student,
        )

        timing_tab_1, timing_tab_2, timing_tab_3 = st.tabs(
            [
                "Задания",
                "Тесты",
                "Лекции",
            ]
        )

        with timing_tab_1:
            assignments_df = timings["assignments"]

            if assignments_df.empty:
                st.info("Нет данных по заданиям.")
            else:
                st.dataframe(
                    assignments_df,
                    use_container_width=True,
                )

        with timing_tab_2:
            tests_df = timings["tests"]

            if tests_df.empty:
                st.info("Нет данных по тестам.")
            else:
                st.dataframe(
                    tests_df,
                    use_container_width=True,
                )

        with timing_tab_3:
            lectures_df = timings["lectures"]

            if lectures_df.empty:
                st.info("Нет данных по лекциям.")
            else:
                st.dataframe(
                    lectures_df,
                    use_container_width=True,
                )

student_ids = sorted(final_df["student_id"].astype(str).unique().tolist())

selected_student_id = st.selectbox(
    "Выберите студента",
    options=student_ids,
    key="final_results_selected_student",
)

student_row = final_df[
    final_df["student_id"].astype(str) == str(selected_student_id)
].copy()

if student_row.empty:
    st.warning("Студент не найден в итоговой таблице.")
else:
    row = student_row.iloc[0]

    st.info(
        f"Итоговый паттерн студента **{selected_student_id}**: "
        f"**{row.get('final_pattern_short', 'Нет данных')}**"
    )

    st.write(row.get("final_behavior_interpretation", ""))

    st.subheader("ML-кластеры выбранного студента")

    student_ml_rows = []

    for method_name in selected_methods:
        cluster_col = f"{method_name}_cluster"
        pattern_col = f"{method_name}_resource_pattern"
        probability_col = f"{method_name}_cluster_probability"

        if cluster_col not in student_row.columns:
            continue

        student_ml_rows.append(
            {
                "method": method_name,
                "cluster": row.get(cluster_col),
                "resource_pattern": row.get(pattern_col),
                "cluster_probability": row.get(probability_col, None),
            }
        )

    if student_ml_rows:
        st.dataframe(
            pd.DataFrame(student_ml_rows),
            use_container_width=True,
        )

    st.subheader("Process mining признаки выбранного студента")

    student_process_columns = [
        "process_pattern",
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
        "days_to_80_percent_events",
        "top_2_days_activity_ratio",
        "top_3_days_activity_ratio",

        "fast_lecture_completion_count",
        "measured_lecture_completion_count",
        "fast_lecture_completion_ratio",

        "fast_test_completion_count",
        "measured_test_completion_count",
        "fast_test_completion_ratio",

        "suspicious_first_assignment_upload_count",
        "measured_first_assignment_upload_count",
        "suspicious_first_assignment_upload_ratio",
    ]

    student_process_columns = [
        col for col in student_process_columns
        if col in student_row.columns
    ]

    st.dataframe(
        student_row[student_process_columns],
        use_container_width=True,
    )

    with st.expander("Показать полную строку студента"):
        st.dataframe(
            student_row,
            use_container_width=True,
        )


# ------------------------------------------------------------
# Текст для ВКР
# ------------------------------------------------------------
st.subheader("Формулировка для ВКР")

st.write(
    """
Итоговые паттерны поведения формировались путём объединения двух уровней анализа.
Первый уровень представлен методами машинного обучения, которые выделяют ресурсные
паттерны студентов по агрегированным признакам активности. Второй уровень представлен
process mining признаками, которые описывают характер прохождения курса во времени:
авральность, регулярность, формальное прохождение лекций, быстрое прохождение тестов,
подозрительно быструю первую загрузку ответов и полноту выполнения контрольных
активностей.

Если несколько студентов имели одинаковую комбинацию ресурсного паттерна и
process mining признаков, такая комбинация рассматривалась как единый итоговый
паттерн поведения. Такой подход позволяет не только определить, какие ресурсы курса
использовал студент, но и уточнить, каким образом он проходил курс.
"""
)