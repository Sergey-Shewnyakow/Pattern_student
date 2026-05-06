import streamlit as st
import streamlit.components.v1 as components

from src.state import init_session_state
from src.process_mining_features import (
    build_trace_preview,
    build_process_mining_features,
)
from src.process_mining_visualization import (
    build_heuristics_miner_svg,
    build_inductive_miner_svg,
    can_build_process_model,
)

st.set_page_config(page_title="Process Mining Features", layout="wide")
init_session_state()


def render_svg_zoomable(svg_path: str, title: str, height: int = 700):
    if not svg_path:
        st.warning(f"Нет SVG для отображения: {title}")
        return

    with open(svg_path, "r", encoding="utf-8") as f:
        svg_content = f.read()

    col1, col2, col3 = st.columns([4, 1, 1])

    with col1:
        st.markdown(f"**{title}**")

    with col2:
        st.download_button(
            label="Скачать SVG",
            data=svg_content,
            file_name=f"{title}.svg".replace(" ", "_"),
            mime="image/svg+xml",
            key=f"download_{title}"
        )

    with col3:
        fullscreen = st.checkbox(
            "На весь экран",
            value=False,
            key=f"fullscreen_{title}"
        )

    current_height = 1200 if fullscreen else height

    html = f"""
    <div style="
        border:1px solid #ddd;
        border-radius:8px;
        padding:12px;
        background:white;
        overflow:auto;
        height:{current_height}px;
    ">
        {svg_content}
    </div>
    """

    components.html(html, height=current_height + 20, scrolling=True)


st.title("Process Mining Features")
st.write(
    "Построение процессных признаков на основе человеческих траекторий студентов. "
    "Системные события исключаются из process mining-моделей."
)

df_sessions = st.session_state.get("df_sessions")
df_human_events = st.session_state.get("df_human_events")
features_df_for_clustering = st.session_state.get("features_df_for_clustering")

if df_sessions is None or df_human_events is None:
    st.warning(
        "Сначала подготовьте данные на странице Data Preparation: "
        "загрузите лог и выполните предобработку."
    )
    st.stop()

st.success(
    f"Данные готовы. Всего событий: {len(df_sessions)}, "
    f"человеческих событий: {len(df_human_events)}, "
    f"пользователей: {df_sessions['student_id'].nunique()}."
)

st.subheader("Предпросмотр human event log")
preview_cols = [col for col in ["student_id", "timestamp", "component", "activity", "human_activity"] if col in df_human_events.columns]
st.dataframe(
    df_human_events[preview_cols].head(),
    use_container_width=True
)

if "human_activity" in df_human_events.columns:
    st.subheader("Какие события попали в 'Прочее действие'")

    other_actions_df = (
        df_human_events[df_human_events["human_activity"] == "Прочее действие"]
        .groupby(["component", "activity"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    if other_actions_df.empty:
        st.success("Событий типа 'Прочее действие' не найдено.")
    else:
        st.dataframe(other_actions_df, use_container_width=True)

st.subheader("Настройка process mining-признаков")

use_filtered_users = st.radio(
    "Для каких пользователей строить process mining-признаки",
    options=[
        "Использовать всех пользователей из human event log",
        "Использовать только пользователей после фильтрации аномалий"
    ],
    horizontal=True
)

if st.button("Построить process mining-признаки"):
    pm_source_df = df_human_events.copy()

    if use_filtered_users == "Использовать только пользователей после фильтрации аномалий":
        if features_df_for_clustering is None:
            st.error(
                "Нет отфильтрованного набора пользователей. "
                "Сначала примените исключение аномалий на странице Data Preparation."
            )
            st.stop()

        allowed_ids = set(features_df_for_clustering["student_id"].astype(str))
        pm_source_df = pm_source_df[
            pm_source_df["student_id"].astype(str).isin(allowed_ids)
        ].copy()

    pm_source_df = pm_source_df.copy()
    if "human_activity" in pm_source_df.columns:
        pm_source_df["activity"] = pm_source_df["human_activity"]

    trace_preview_df = build_trace_preview(pm_source_df, max_students=50)
    pm_features_df, pm_variants_df = build_process_mining_features(pm_source_df)

    ok, message = can_build_process_model(pm_source_df)
    heuristics_svg_path = None
    inductive_svg_path = None

    if ok:
        heuristics_svg_path = build_heuristics_miner_svg(pm_source_df)
        inductive_svg_path = build_inductive_miner_svg(pm_source_df)
    else:
        st.warning(message)

    st.session_state["pm_trace_preview_df"] = trace_preview_df
    st.session_state["pm_features_df"] = pm_features_df
    st.session_state["pm_variants_df"] = pm_variants_df
    st.session_state["pm_heuristics_svg_path"] = heuristics_svg_path
    st.session_state["pm_inductive_svg_path"] = inductive_svg_path

    st.session_state["pm_clustering_result"] = None
    st.session_state["pm_k_scores_df"] = None

if st.session_state["pm_trace_preview_df"] is not None:
    st.subheader("Предпросмотр траекторий студентов")
    st.dataframe(
        st.session_state["pm_trace_preview_df"],
        use_container_width=True
    )

if st.session_state["pm_variants_df"] is not None:
    variants_df = st.session_state["pm_variants_df"]

    st.subheader("Варианты траекторий")
    st.dataframe(
        variants_df.sort_values("variant_frequency", ascending=False),
        use_container_width=True
    )

if st.session_state["pm_features_df"] is not None:
    pm_features_df = st.session_state["pm_features_df"]

    st.subheader("Process mining-признаки")
    st.dataframe(pm_features_df, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Число пользователей", len(pm_features_df))
    c2.metric("Число process-признаков", pm_features_df.shape[1] - 1)
    c3.metric("Число вариантов", pm_features_df["variant_id"].nunique())

    st.subheader("Краткие выводы по process mining")
    top_variant_freq = pm_features_df["variant_frequency"].max()
    avg_trace_length = pm_features_df["trace_length"].mean()
    avg_linearity = pm_features_df["linearity_score"].mean()
    avg_complexity = pm_features_df["path_complexity"].mean()
    avg_backtracks = pm_features_df["backtrack_count"].mean()

    st.write(
        f"""
- Средняя длина траектории: **{avg_trace_length:.2f}**
- Средняя линейность траектории: **{avg_linearity:.2f}**
- Средняя сложность траектории: **{avg_complexity:.2f}**
- Среднее число возвратов: **{avg_backtracks:.2f}**
- Частота самого распространённого варианта: **{top_variant_freq}**
"""
    )

    if avg_linearity > 0.8:
        st.info("В среднем траектории выглядят достаточно линейными.")
    else:
        st.info("В среднем траектории содержат заметную долю повторов и нелинейных переходов.")

    if avg_backtracks > 1:
        st.info("У студентов заметны возвраты к ранее выполненным действиям.")
    else:
        st.info("Возвраты к уже пройденным действиям выражены слабо.")

    st.subheader("Модель Heuristics Miner")
    if st.session_state.get("pm_heuristics_svg_path"):
        render_svg_zoomable(
            st.session_state["pm_heuristics_svg_path"],
            "Heuristics Miner (SVG)",
            height=750
        )

    st.subheader("Модель Inductive Miner")
    if st.session_state.get("pm_inductive_svg_path"):
        render_svg_zoomable(
            st.session_state["pm_inductive_svg_path"],
            "Inductive Miner (BPMN, SVG)",
            height=750
        )

    st.subheader("Process mining по отдельному студенту")

    student_ids = sorted(df_human_events["student_id"].astype(str).unique().tolist())
    if student_ids:
        default_student = st.session_state.get("pm_selected_student_id", student_ids[0])
        if default_student not in student_ids:
            default_student = student_ids[0]

        selected_student_id = st.selectbox(
            "Выберите студента для индивидуального process mining",
            student_ids,
            index=student_ids.index(default_student),
            key="pm_selected_student_id"
        )

        if st.button("Построить process mining для выбранного студента"):
            student_log_df = df_human_events[
                df_human_events["student_id"].astype(str) == str(selected_student_id)
            ].copy()

            if "human_activity" in student_log_df.columns:
                student_log_df["activity"] = student_log_df["human_activity"]

            st.session_state["pm_selected_student_log_df"] = student_log_df

            student_pm_features_df, _ = build_process_mining_features(student_log_df)
            st.session_state["pm_student_features_df"] = student_pm_features_df

            ok_student, message_student = can_build_process_model(student_log_df)

            if ok_student:
                try:
                    st.session_state["pm_student_heuristics_svg_path"] = build_heuristics_miner_svg(student_log_df)
                except Exception as e:
                    st.session_state["pm_student_heuristics_svg_path"] = None
                    st.warning(f"Heuristics Miner для выбранного студента не построен: {e}")

                try:
                    st.session_state["pm_student_inductive_svg_path"] = build_inductive_miner_svg(student_log_df)
                except Exception as e:
                    st.session_state["pm_student_inductive_svg_path"] = None
                    st.warning(f"Inductive Miner для выбранного студента не построен: {e}")
            else:
                st.warning(message_student)
                st.session_state["pm_student_heuristics_svg_path"] = None
                st.session_state["pm_student_inductive_svg_path"] = None

    if st.session_state.get("pm_selected_student_log_df") is not None:
        st.subheader("Лог выбранного студента")
        st.dataframe(
            st.session_state["pm_selected_student_log_df"],
            use_container_width=True
        )

    if st.session_state.get("pm_student_features_df") is not None:
        st.subheader("Process mining-признаки выбранного студента")
        st.dataframe(
            st.session_state["pm_student_features_df"],
            use_container_width=True
        )

    if st.session_state.get("pm_student_heuristics_svg_path"):
        st.subheader("Heuristics Miner для выбранного студента")
        render_svg_zoomable(
            st.session_state["pm_student_heuristics_svg_path"],
            "Heuristics Miner для выбранного студента (SVG)",
            height=650
        )

    if st.session_state.get("pm_student_inductive_svg_path"):
        st.subheader("Inductive Miner для выбранного студента")
        render_svg_zoomable(
            st.session_state["pm_student_inductive_svg_path"],
            "Inductive Miner для выбранного студента (BPMN, SVG)",
            height=650
        )

    st.success(
        "Process mining-признаки и модели построены по человеческим событиям. "
        "Теперь можно выполнять кластеризацию по process mining-признакам."
    )
else:
    st.info("Нажмите «Построить process mining-признаки».")