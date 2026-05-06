import streamlit as st

from src.state import init_session_state
from src.data_loader import load_file
from src.normalizer import normalize_columns
from src.validator import validate_columns
from src.preprocessing import preprocess_log
from src.sessionization import add_sessions
from src.features import build_student_features
from src.anomaly_detection import (
    detect_rule_based_anomalies,
    detect_isolation_forest_anomalies,
    combine_anomaly_flags,
)
from src.event_classification import split_event_types
from src.ui_styles import apply_global_styles

apply_global_styles()

st.set_page_config(page_title="Data Preparation", layout="wide")
init_session_state()

st.title("Data Preparation")
st.write("Загрузка логов, построение признаков и исключение аномальных пользователей.")

uploaded_file = st.file_uploader(
    "Загрузите файл лога",
    type=["csv", "xlsx", "xls"],
    key="data_preparation_uploader"
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
# Если файл уже был обработан ранее
# -----------------------------
df_raw = st.session_state.get("df_raw")
df_norm = st.session_state.get("df_norm")
df_clean = st.session_state.get("df_clean")

if df_raw is None or df_norm is None or df_clean is None:
    st.info("Загрузите файл, чтобы начать подготовку данных.")
    st.stop()

# -----------------------------
# Отображение уже сохранённых данных
# -----------------------------
st.subheader("Исходные данные")
st.dataframe(df_raw.head(), use_container_width=True)

st.subheader("После нормализации колонок")
st.write(list(df_norm.columns))

# -----------------------------
# Параметры сессий
# -----------------------------
default_gap = st.session_state["last_gap_minutes"] if st.session_state["last_gap_minutes"] is not None else 30

gap_minutes = st.slider(
    "Порог новой сессии (минуты)",
    min_value=5,
    max_value=120,
    value=default_gap,
    step=5,
    key="gap_minutes_slider"
)

# Если изменился порог — пересчитываем признаки и сбрасываем downstream
if st.session_state["last_gap_minutes"] is None or st.session_state["last_gap_minutes"] != gap_minutes:
    df_sessions = add_sessions(df_clean, gap_minutes=gap_minutes)
    df_sessions, df_human_events, df_system_events, df_other_events = split_event_types(df_sessions)

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

st.subheader("Очищенные данные с session_id")
st.dataframe(df_sessions.head(), use_container_width=True)

c1, c2, c3 = st.columns(3)
c1.metric("Число строк", len(df_sessions))
c2.metric("Число студентов", df_sessions["student_id"].nunique())
c3.metric("Число типов событий", df_sessions["activity"].nunique())

if df_human_events is not None and df_system_events is not None:
    h1, h2, h3 = st.columns(3)
    h1.metric("Человеческие события", len(df_human_events))
    h2.metric("Системные события", len(df_system_events))
    h3.metric("Прочие события", 0 if df_other_events is None else len(df_other_events))

    st.subheader("Разделение событий")
    st.dataframe(
        df_sessions[["student_id", "timestamp", "component", "activity", "event_type", "human_activity"]].head(20),
        use_container_width=True
    )

st.subheader("Признаки студентов")
st.dataframe(features_df.head(), use_container_width=True)

# -----------------------------
# Аномалии
# -----------------------------
st.subheader("Обнаружение аномальных пользователей")

anomaly_method = st.radio(
    "Метод обнаружения аномалий",
    options=[
        "Только правила",
        "Только Isolation Forest",
        "Оба метода"
    ],
    horizontal=True,
    key="anomaly_method_radio"
)

quantile_threshold = st.slider(
    "Порог для правил (перцентиль)",
    min_value=0.90,
    max_value=0.999,
    value=0.99,
    step=0.001,
    format="%.3f",
    key="quantile_threshold_slider"
)

contamination = st.slider(
    "Доля аномалий для Isolation Forest",
    min_value=0.01,
    max_value=0.30,
    value=0.05,
    step=0.01,
    key="contamination_slider"
)

if st.button("Найти аномалии", key="find_anomalies_button"):
    anomaly_df = features_df.copy()

    use_rule_based = anomaly_method in ["Только правила", "Оба метода"]
    use_iforest = anomaly_method in ["Только Isolation Forest", "Оба метода"]

    if use_rule_based:
        anomaly_df, _ = detect_rule_based_anomalies(
            anomaly_df,
            quantile_threshold=quantile_threshold
        )

    if use_iforest:
        anomaly_df = detect_isolation_forest_anomalies(
            anomaly_df,
            contamination=contamination
        )

    anomaly_df = combine_anomaly_flags(
        anomaly_df,
        use_rule_based=use_rule_based,
        use_iforest=use_iforest
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
# Отображение сохранённых аномалий
# -----------------------------
if st.session_state["anomaly_df"] is not None:
    anomaly_df = st.session_state["anomaly_df"].copy()

    total_users = len(anomaly_df)
    total_anomalies = int(anomaly_df["final_anomaly"].sum())

    a1, a2 = st.columns(2)
    a1.metric("Всего пользователей", total_users)
    a2.metric("Найдено аномальных", total_anomalies)

    st.subheader("Таблица аномалий")
    anomaly_columns = ["student_id", "final_anomaly"]

    for col in [
        "rule_based_anomaly",
        "triggered_rules",
        "iforest_anomaly",
        "iforest_score"
    ]:
        if col in anomaly_df.columns:
            anomaly_columns.append(col)

    st.dataframe(
        anomaly_df[anomaly_columns].sort_values(
            by="final_anomaly",
            ascending=False
        ),
        use_container_width=True
    )

    st.subheader("Ручное исключение пользователей")

    editable_df = anomaly_df.copy()
    if "exclude_manual" not in editable_df.columns:
        editable_df["exclude_manual"] = editable_df["final_anomaly"]

    editable_columns = [
        "exclude_manual",
        "student_id",
        "final_anomaly"
    ]

    for col in [
        "rule_based_anomaly",
        "triggered_rules",
        "iforest_anomaly",
        "iforest_score"
    ]:
        if col in editable_df.columns:
            editable_columns.append(col)

    with st.form("apply_exclusion_form"):
        edited_df = st.data_editor(
            editable_df[editable_columns].sort_values(
                by=["exclude_manual", "final_anomaly"],
                ascending=[False, False]
            ),
            use_container_width=True,
            hide_index=True
        )

        apply_exclusion = st.form_submit_button("Применить исключение")

    if apply_exclusion:
        selected_ids = edited_df.loc[
            edited_df["exclude_manual"], "student_id"
        ].astype(str).tolist()

        updated_df = anomaly_df.copy()
        updated_df["exclude_manual"] = updated_df["student_id"].astype(str).isin(selected_ids)
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
            "final_anomaly"
        ]

        for col in [
            "rule_based_anomaly",
            "triggered_rules",
            "iforest_anomaly",
            "iforest_score"
        ]:
            if col in current_df.columns:
                exclude_columns.append(col)

        st.dataframe(
            current_df[exclude_columns].sort_values(
                by=["exclude_final", "final_anomaly"],
                ascending=[False, False]
            ),
            use_container_width=True
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
        "Сначала нажмите «Найти аномалии». "
        "После этого можно будет вручную отметить пользователей для исключения."
    )