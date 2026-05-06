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
from src.cluster_naming import build_cluster_names

st.set_page_config(page_title="Cluster Process Mining Views", layout="wide")
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

    current_height = 1900 if fullscreen else height

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


def get_available_standard_results():
    results = []

    possible_results = [
        ("KMeans", st.session_state.get("clustering_result")),
        ("Agglomerative", st.session_state.get("agg_clustering_result")),
        ("GMM", st.session_state.get("gmm_clustering_result")),
        ("HDBSCAN", st.session_state.get("hdbscan_clustering_result")),
        ("Autoencoder + KMeans", st.session_state.get("ae_clustering_result")),
    ]

    for method_name, result_obj in possible_results:
        if result_obj is not None:
            results.append((method_name, result_obj))

    return results


st.title("Cluster Process Mining Views")
st.write(
    "Анализ process mining по кластерам, полученным при обычной кластеризации. "
    "Здесь можно посмотреть, как выглядят траектории и process-модели внутри каждого кластера."
)

df_sessions = st.session_state.get("df_sessions")

if df_sessions is None:
    st.warning(
        "Сначала подготовьте данные на странице Data Preparation."
    )
    st.stop()

available_results = get_available_standard_results()

if not available_results:
    st.warning(
        "Нет результатов обычной кластеризации. "
        "Сначала запустите хотя бы один метод на обычных признаках "
        "(KMeans, Agglomerative, GMM, HDBSCAN или Autoencoder + KMeans)."
    )
    st.stop()

method_labels = [name for name, _ in available_results]

selected_method = st.selectbox(
    "Выберите метод обычной кластеризации",
    method_labels,
    key="cluster_pm_method_selector"
)

selected_result_obj = None
for method_name, result_obj in available_results:
    if method_name == selected_method:
        selected_result_obj = result_obj
        break

result_df = selected_result_obj["result_df"]
cluster_profiles = selected_result_obj["cluster_profiles"]

non_noise_df = result_df[result_df["cluster"] != -1].copy() if "cluster" in result_df.columns else result_df

if len(cluster_profiles) > 0 and len(non_noise_df) > 0:
    cluster_names_df = build_cluster_names(non_noise_df, cluster_profiles)
else:
    cluster_names_df = None

cluster_values = sorted(result_df["cluster"].unique().tolist())

selected_cluster = st.selectbox(
    "Выберите кластер",
    cluster_values,
    key="cluster_pm_cluster_selector"
)

cluster_students_df = result_df[result_df["cluster"] == selected_cluster].copy()
cluster_student_ids = set(cluster_students_df["student_id"].astype(str))

cluster_log_df = df_sessions[
    df_sessions["student_id"].astype(str).isin(cluster_student_ids)
].copy()

st.subheader("Информация о выбранном кластере")

c1, c2, c3 = st.columns(3)
c1.metric("Метод", selected_method)
c2.metric("Кластер", selected_cluster)
c3.metric("Число студентов", len(cluster_students_df))

if cluster_names_df is not None and selected_cluster != -1:
    matched = cluster_names_df[cluster_names_df["cluster"] == selected_cluster]
    if not matched.empty:
        cluster_name = matched["suggested_name"].iloc[0]
        cluster_description = matched["description"].iloc[0]

        st.info(
            f"Кластер **{selected_cluster}** — **{cluster_name}**. "
            f"{cluster_description}"
        )

st.subheader("Студенты выбранного кластера")
st.dataframe(cluster_students_df, use_container_width=True)

st.subheader("События выбранного кластера")
st.dataframe(cluster_log_df.head(50), use_container_width=True)

if st.button("Построить process mining для выбранного кластера"):
    if cluster_log_df.empty:
        st.error("Для выбранного кластера нет событий.")
        st.stop()

    trace_preview_df = build_trace_preview(cluster_log_df, max_students=100)
    cluster_pm_features_df, cluster_pm_variants_df = build_process_mining_features(cluster_log_df)

    ok, message = can_build_process_model(cluster_log_df)
    heuristics_svg_path = None
    inductive_svg_path = None

    if ok:
        try:
            heuristics_svg_path = build_heuristics_miner_svg(cluster_log_df)
        except Exception as e:
            st.warning(f"Heuristics Miner не построен: {e}")

        try:
            inductive_svg_path = build_inductive_miner_svg(cluster_log_df)
        except Exception as e:
            st.warning(f"Inductive Miner не построен: {e}")
    else:
        st.warning(message)

    st.session_state["cluster_pm_trace_preview_df"] = trace_preview_df
    st.session_state["cluster_pm_features_df"] = cluster_pm_features_df
    st.session_state["cluster_pm_variants_df"] = cluster_pm_variants_df
    st.session_state["cluster_pm_heuristics_svg_path"] = heuristics_svg_path
    st.session_state["cluster_pm_inductive_svg_path"] = inductive_svg_path

if st.session_state.get("cluster_pm_trace_preview_df") is not None:
    st.subheader("Предпросмотр траекторий внутри кластера")
    st.dataframe(
        st.session_state["cluster_pm_trace_preview_df"],
        use_container_width=True
    )

if st.session_state.get("cluster_pm_variants_df") is not None:
    st.subheader("Варианты траекторий внутри кластера")
    st.dataframe(
        st.session_state["cluster_pm_variants_df"].sort_values(
            "variant_frequency",
            ascending=False
        ),
        use_container_width=True
    )

if st.session_state.get("cluster_pm_features_df") is not None:
    cluster_pm_features_df = st.session_state["cluster_pm_features_df"]

    st.subheader("Process mining-признаки выбранного кластера")
    st.dataframe(cluster_pm_features_df, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Число студентов", len(cluster_pm_features_df))
    c2.metric("Средняя длина траектории", f"{cluster_pm_features_df['trace_length'].mean():.2f}")
    c3.metric("Средняя линейность", f"{cluster_pm_features_df['linearity_score'].mean():.2f}")
    c4.metric("Средняя сложность", f"{cluster_pm_features_df['path_complexity'].mean():.2f}")

    st.subheader("Краткие выводы по кластеру")

    avg_trace_length = cluster_pm_features_df["trace_length"].mean()
    avg_linearity = cluster_pm_features_df["linearity_score"].mean()
    avg_complexity = cluster_pm_features_df["path_complexity"].mean()
    avg_backtracks = cluster_pm_features_df["backtrack_count"].mean()
    top_variant_freq = cluster_pm_features_df["variant_frequency"].max()

    st.write(
        f"""
- Средняя длина траектории в кластере: **{avg_trace_length:.2f}**
- Средняя линейность: **{avg_linearity:.2f}**
- Средняя сложность траектории: **{avg_complexity:.2f}**
- Среднее число возвратов: **{avg_backtracks:.2f}**
- Частота самого распространённого варианта: **{top_variant_freq}**
"""
    )

    if avg_linearity > 0.8:
        st.info("Для этого кластера характерны более линейные траектории.")
    else:
        st.info("Для этого кластера характерны нелинейные траектории с повторными переходами.")

    if avg_backtracks > 1:
        st.info("Внутри кластера заметны возвраты к уже пройденным действиям.")
    else:
        st.info("Внутри кластера возвраты выражены слабо.")

    st.subheader("Heuristics Miner для кластера")
    if st.session_state.get("cluster_pm_heuristics_svg_path"):
        render_svg_zoomable(
            st.session_state["cluster_pm_heuristics_svg_path"],
            f"Heuristics Miner для кластера {selected_cluster}",
            height=750
        )

    st.subheader("Inductive Miner для кластера")
    if st.session_state.get("cluster_pm_inductive_svg_path"):
        render_svg_zoomable(
            st.session_state["cluster_pm_inductive_svg_path"],
            f"Inductive Miner для кластера {selected_cluster}",
            height=750
        )

else:
    st.info("Нажмите «Построить process mining для выбранного кластера».")