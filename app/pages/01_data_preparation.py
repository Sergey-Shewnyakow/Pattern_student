import streamlit as st

from src.state import init_session_state
from src.data_loader import load_file
from src.normalizer import normalize_columns
from src.validator import validate_columns
from src.preprocessing import preprocess_log
from src.sessionization import add_sessions
from src.features import build_student_features
from src.anomaly_detection import detect_role_action_anomalies
from src.event_classification import split_event_types
from src.ui_styles import apply_global_styles


apply_global_styles()

st.set_page_config(page_title="Data Preparation", layout="wide")
init_session_state()

st.title("Data Preparation")
st.write(
    "Загрузка логов, построение признаков и исключение пользователей, "
    "похожих на преподавателей или администраторов."
)

uploaded_file = st.file_uploader(
    "Загрузите файл лога",
    type=["csv", "xlsx", "xls"],
    key="data_preparation_uploader",
)


# -----------------------------
# Загрузка нового файла
# -----------------------------
if uploaded_file is not None:
    try:
        if st.session_state["last_filename"] != uploaded_file.name:
            st.session_state["last_filename"] = uploaded_file.name

            st.session_state["last_gap_minutes"] = None

            st.session_state["df_raw"] = None
            st.session_state["df_norm"] = None
            st.session_state["df_clean"] = None
            st.session_state["df_sessions"] = None

            st.session_state["df_human_events"] = None
            st.session_state["df_system_events"] = None
            st.session_state["df_other_events"] = None

            st.session_state["features_df"] = None
            st.session_state["anomaly_df"] = None
            st.session_state["features_df_for_clustering"] = None

            st.session_state["pending_filtered_df"] = None
            st.session_state["applied_filtered_df"] = None

            st.session_state["k_scores_df"] = None
            st.session_state["clustering_result"] = None

            df_raw = load_file(uploaded_file)
            df_norm = normalize_columns(df_raw)

            is_valid, missing = validate_columns(df_norm)

            if not is_valid:
                st.error("Не хватает обязательных колонок: " + ", ".join(missing))
                st.stop()

            df_clean = preprocess_log(df_norm)

            st.session_state["df_raw"] = df_raw
            st.session_state["df_norm"] = df_norm
            st.session_state["df_clean"] = df_clean

    except Exception as e:
        st.exception(e)


# -----------------------------
# Если файл ещё не загружен
# -----------------------------
df_raw = st.session_state.get("df_raw")
df_norm = st.session_state.get("df_norm")
df_clean = st.session_state.get("df_clean")

if df_raw is None or df_norm is None or df_clean is None:
    st.info("Загрузите файл, чтобы начать подготовку данных.")
    st.stop()


# -----------------------------
# Отображение исходных данных
# -----------------------------
st.subheader("Исходные данные")
st.dataframe(df_raw.head(), use_container_width=True)

st.subheader("После нормализации колонок")
st.write(list(df_norm.columns))


# -----------------------------
# Параметры сессий
# -----------------------------
default_gap = (
    st.session_state["last_gap_minutes"]
    if st.session_state["last_gap_minutes"] is not None
    else 30
)

gap_minutes = st.slider(
    "Порог новой сессии (минуты)",
    min_value=5,
    max_value=120,
    value=default_gap,
    step=5,
    key="gap_minutes_slider",
)


# Если изменился порог — пересчитываем признаки и сбрасываем downstream
if (
    st.session_state["last_gap_minutes"] is None
    or st.session_state["last_gap_minutes"] != gap_minutes
):
    df_sessions = add_sessions(df_clean, gap_minutes=gap_minutes)

    df_sessions, df_human_events, df_system_events, df_other_events = split_event_types(
        df_sessions
    )

    features_df = build_student_features(df_sessions)

    st.session_state["last_gap_minutes"] = gap_minutes

    st.session_state["df_sessions"] = df_sessions
    st.session_state["df_human_events"] = df_human_events
    st.session_state["df_system_events"] = df_system_events
    st.session_state["df_other_events"] = df_other_events

    st.session_state["features_df"] = features_df
    st.session_state["anomaly_df"] = None
    st.session_state["features_df_for_clustering"] = None

    st.session_state["pending_filtered_df"] = None
    st.session_state["applied_filtered_df"] = None

    st.session_state["k_scores_df"] = None
    st.session_state["clustering_result"] = None


df_sessions = st.session_state["df_sessions"]
df_human_events = st.session_state.get("df_human_events")
df_system_events = st.session_state.get("df_system_events")
df_other_events = st.session_state.get("df_other_events")
features_df = st.session_state["features_df"]


# -----------------------------
# Очищенные данные
# -----------------------------
st.subheader("Очищенные данные с session_id")
st.dataframe(df_sessions.head(), use_container_width=True)

c1, c2, c3 = st.columns(3)
c1.metric("Число строк", len(df_sessions))
c2.metric("Число пользователей", df_sessions["student_id"].nunique())
c3.metric("Число типов событий", df_sessions["activity"].nunique())


# -----------------------------
# Разделение событий
# -----------------------------
if df_human_events is not None and df_system_events is not None:
    staff_like_events_count = (
        int(df_sessions["is_staff_like_event"].sum())
        if "is_staff_like_event" in df_sessions.columns
        else 0
    )

    h1, h2, h3, h4 = st.columns(4)

    h1.metric("Человеческие события", len(df_human_events))
    h2.metric("Системные события", len(df_system_events))
    h3.metric(
        "Прочие события",
        0 if df_other_events is None else len(df_other_events),
    )
    h4.metric("Административные события", staff_like_events_count)

st.subheader("Разделение событий")

event_columns = [
    "student_id",
    "timestamp",
    "component",
    "activity",
    "event_type",
    "human_activity",
    "role_event_type",
    "is_staff_like_event",
    "role_event_reason",
]

available_event_columns = [
    col for col in event_columns if col in df_sessions.columns
]

st.dataframe(
    df_sessions[available_event_columns].head(30),
    use_container_width=True,
)

with st.expander("Показать административные события"):
    if "is_staff_like_event" in df_sessions.columns:
        staff_events_df = df_sessions[df_sessions["is_staff_like_event"]].copy()

        if staff_events_df.empty:
            st.success("Жёстких административных событий не найдено.")
        else:
            st.warning(
                f"Найдено административных событий: "
                f"{len(staff_events_df)}"
            )

            st.dataframe(
                staff_events_df[available_event_columns],
                use_container_width=True,
            )
    else:
        st.info("Колонка is_staff_like_event пока не создана.")


# -----------------------------
# Признаки студентов
# -----------------------------
st.subheader("Признаки пользователей")
st.dataframe(features_df.head(), use_container_width=True)


# -----------------------------
# Аномалии / исключение не-студентов
# -----------------------------
st.subheader("Исключение преподавателей и администраторов")

st.write(
    "Isolation Forest отключён. Исключение выполняется по двум автоматическим правилам: "
    "жёсткие административные действия и сверхбольшое количество событий."
)

st.caption(
    "Сверхбольшое количество событий определяется автоматически методом IQR: "
    "порог = Q3 + 1.5 × IQR."
)

st.caption(
    "Не исключаются: просмотр подтверждения удаления ответа, удаление ответа, "
    "удаление комментария, создание темы, выставление оценки, отчёты и календарные события."
)


if st.button("Найти преподавателей/администраторов", key="find_anomalies_button"):
    anomaly_df = detect_role_action_anomalies(
        features_df=features_df,
        log_df=df_sessions,
    )

    anomaly_df["exclude_manual"] = anomaly_df["final_anomaly"]
    anomaly_df["exclude_final"] = anomaly_df["exclude_manual"]

    st.session_state["anomaly_df"] = anomaly_df.copy()
    st.session_state["pending_filtered_df"] = anomaly_df.copy()
    st.session_state["applied_filtered_df"] = anomaly_df.copy()

    st.session_state["features_df_for_clustering"] = anomaly_df.loc[
        ~anomaly_df["exclude_final"]
    ].copy()

    st.session_state["k_scores_df"] = None
    st.session_state["clustering_result"] = None


# -----------------------------
# Отображение сохранённых исключений
# -----------------------------
if st.session_state["anomaly_df"] is not None:
    anomaly_df = st.session_state["anomaly_df"].copy()

    total_users = len(anomaly_df)
    total_anomalies = int(anomaly_df["final_anomaly"].sum())

    staff_action_anomalies = (
        int(anomaly_df["staff_action_anomaly"].sum())
        if "staff_action_anomaly" in anomaly_df.columns
        else 0
    )

    event_count_anomalies = (
        int(anomaly_df["event_count_anomaly"].sum())
        if "event_count_anomaly" in anomaly_df.columns
        else 0
    )

    event_count_threshold = (
        int(anomaly_df["event_count_threshold"].iloc[0])
        if "event_count_threshold" in anomaly_df.columns and len(anomaly_df) > 0
        else 0
    )

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Всего пользователей", total_users)
    a2.metric("Исключаются как не-студенты", total_anomalies)
    a3.metric("По админ-действиям", staff_action_anomalies)
    a4.metric("По числу событий", event_count_anomalies)

    st.info(f"Автоматический порог сверхбольшого количества событий: {event_count_threshold}")

    st.subheader("Таблица найденных преподавателей/администраторов")

    anomaly_columns = [
        "student_id",
        "final_anomaly",
        "anomaly_reason",
        "staff_action_anomaly",
        "staff_action_count",
        "admin_action_count",
        "event_count_anomaly",
        "total_events",
        "event_count_threshold",
        "event_count_q1",
        "event_count_q3",
        "event_count_iqr",
        "staff_action_examples",
        "staff_action_reasons",
    ]

    available_anomaly_columns = [
        col for col in anomaly_columns if col in anomaly_df.columns
    ]

    st.dataframe(
        anomaly_df[available_anomaly_columns].sort_values(
            by=["final_anomaly", "total_events"],
            ascending=[False, False],
        ),
        use_container_width=True,
    )

    st.subheader("Ручное исключение пользователей")

    editable_df = anomaly_df.copy()

    if "exclude_manual" not in editable_df.columns:
        editable_df["exclude_manual"] = editable_df["final_anomaly"]

    editable_columns = [
        "exclude_manual",
        "student_id",
        "final_anomaly",
        "staff_action_anomaly",
        "event_count_anomaly",
        "total_events",
        "event_count_threshold",
        "staff_action_count",
        "admin_action_count",
        "staff_action_examples",
    ]

    available_editable_columns = [
        col for col in editable_columns if col in editable_df.columns
    ]

    with st.form("apply_exclusion_form"):
        edited_df = st.data_editor(
            editable_df[available_editable_columns].sort_values(
                by=["exclude_manual", "final_anomaly", "total_events"],
                ascending=[False, False, False],
            ),
            use_container_width=True,
            hide_index=True,
        )

        apply_exclusion = st.form_submit_button("Применить исключение")

    if apply_exclusion:
        selected_ids = (
            edited_df.loc[edited_df["exclude_manual"], "student_id"]
            .astype(str)
            .tolist()
        )

        updated_df = anomaly_df.copy()
        updated_df["exclude_manual"] = (
            updated_df["student_id"].astype(str).isin(selected_ids)
        )
        updated_df["exclude_final"] = updated_df["exclude_manual"]

        st.session_state["anomaly_df"] = updated_df.copy()
        st.session_state["pending_filtered_df"] = updated_df.copy()
        st.session_state["applied_filtered_df"] = updated_df.copy()

        features_df_for_clustering = updated_df.loc[
            ~updated_df["exclude_final"]
        ].copy()

        st.session_state["features_df_for_clustering"] = features_df_for_clustering

        st.session_state["k_scores_df"] = None
        st.session_state["clustering_result"] = None

    current_df = st.session_state["applied_filtered_df"]

    if current_df is not None:
        st.subheader("Итоговая таблица исключения")

        exclude_columns = [
            "student_id",
            "exclude_manual",
            "exclude_final",
            "final_anomaly",
            "staff_action_anomaly",
            "event_count_anomaly",
            "total_events",
            "event_count_threshold",
            "staff_action_count",
            "admin_action_count",
            "staff_action_examples",
        ]

        available_exclude_columns = [
            col for col in exclude_columns if col in current_df.columns
        ]

        st.dataframe(
            current_df[available_exclude_columns].sort_values(
                by=["exclude_final", "final_anomaly", "total_events"],
                ascending=[False, False, False],
            ),
            use_container_width=True,
        )

        excluded_count = int(current_df["exclude_final"].sum())
        used_count = len(current_df) - excluded_count

        e1, e2 = st.columns(2)
        e1.metric("Исключено пользователей", excluded_count)
        e2.metric("Останется для кластеризации", used_count)

        features_df_for_clustering = current_df.loc[
            ~current_df["exclude_final"]
        ].copy()

        st.subheader("Данные для кластеризации")
        st.write(f"Будет использовано пользователей: {len(features_df_for_clustering)}")
        st.dataframe(features_df_for_clustering.head(), use_container_width=True)

        st.success("Исключение применено. Можно переходить на страницу KMeans Clustering.")

else:
    st.info(
        "Сначала нажмите «Найти преподавателей/администраторов». "
        "После этого можно будет вручную отметить пользователей для исключения."
    )