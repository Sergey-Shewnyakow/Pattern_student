import pandas as pd
import streamlit as st

from src.state import init_session_state
from src.hybrid_features import build_hybrid_features
from src.visualization import (
    plot_cluster_counts,
    plot_pca_clusters,
    plot_cluster_profile_bar,
)
from src.cluster_naming import build_cluster_names

from src.clustering import run_kmeans, evaluate_kmeans_range
from src.agglomerative_clustering import run_agglomerative, evaluate_agglomerative_range
from src.gmm_clustering import run_gmm, evaluate_gmm_range
from src.hdbscan_clustering import run_hdbscan, evaluate_hdbscan_range

st.set_page_config(page_title="Hybrid Clustering", layout="wide")
init_session_state()


def build_student_cluster_comparison(result_df: pd.DataFrame, student_id: str) -> tuple[pd.DataFrame, int]:
    student_row = result_df[result_df["student_id"].astype(str) == str(student_id)].copy()

    if student_row.empty:
        raise ValueError("Выбранный студент не найден в result_df.")

    cluster_id = int(student_row["cluster"].iloc[0])

    if cluster_id == -1:
        return pd.DataFrame(), cluster_id

    cluster_df = result_df[result_df["cluster"] == cluster_id].copy()

    numeric_cols = [
        col for col in result_df.select_dtypes(include="number").columns
        if col != "cluster"
    ]

    rows = []

    for col in numeric_cols:
        student_value = float(student_row[col].iloc[0])
        cluster_mean = float(cluster_df[col].mean())
        cluster_std = float(cluster_df[col].std(ddof=0))

        diff = student_value - cluster_mean
        relative_diff_pct = 0.0
        if abs(cluster_mean) > 1e-9:
            relative_diff_pct = (diff / cluster_mean) * 100

        if cluster_std > 1e-9:
            z_score = diff / cluster_std
        else:
            z_score = 0.0

        is_strong_deviation = (abs(z_score) >= 1.5) or (abs(relative_diff_pct) >= 30)

        rows.append({
            "feature": col,
            "student_value": student_value,
            "cluster_mean": cluster_mean,
            "difference": diff,
            "relative_diff_pct": relative_diff_pct,
            "z_score": z_score,
            "strong_deviation": is_strong_deviation,
        })

    comparison_df = pd.DataFrame(rows).sort_values(
        by="z_score",
        key=lambda s: s.abs(),
        ascending=False
    )

    return comparison_df, cluster_id


def highlight_large_deviation(row: pd.Series):
    if bool(row["strong_deviation"]):
        return ["background-color: orange"] * len(row)
    return [""] * len(row)


st.title("Hybrid Clustering")
st.write(
    "Совместное использование обычных логовых признаков и process mining-признаков "
    "в едином признаковом пространстве."
)

log_features_df = st.session_state.get("features_df_for_clustering")
pm_features_df = st.session_state.get("pm_features_df")

if log_features_df is None:
    st.warning("Сначала подготовьте обычные признаки на странице Data Preparation.")
    st.stop()

if pm_features_df is None:
    st.warning("Сначала постройте process mining-признаки на странице Process Mining Features.")
    st.stop()

hybrid_df = build_hybrid_features(log_features_df, pm_features_df)
st.session_state["hybrid_features_df"] = hybrid_df

if len(hybrid_df) < 2:
    st.error("После объединения получилось слишком мало пользователей.")
    st.stop()

st.success(f"Гибридный набор готов. Пользователей: {len(hybrid_df)}")

st.subheader("Гибридные признаки")
st.dataframe(hybrid_df.head(), use_container_width=True)

st.subheader("Выбор метода")
method = st.selectbox(
    "Метод кластеризации",
    options=["KMeans", "Agglomerative", "GMM", "HDBSCAN"],
    index=0,
    key="hybrid_method"
)

if method in ["KMeans", "Agglomerative", "GMM"]:
    st.subheader("Оценка разных значений k")

    col1, col2 = st.columns(2)
    with col1:
        k_min = st.number_input("Минимальное k", min_value=2, max_value=20, value=2, step=1, key="hybrid_k_min")
    with col2:
        k_max = st.number_input("Максимальное k", min_value=2, max_value=20, value=6, step=1, key="hybrid_k_max")

    if k_min < k_max:
        if st.button("Оценить k", key="hybrid_evaluate_button"):
            max_allowed_k = min(int(k_max), len(hybrid_df) - 1)
            min_allowed_k = int(k_min)

            if method == "KMeans":
                st.session_state["hybrid_k_scores_df"] = evaluate_kmeans_range(
                    hybrid_df, k_min=min_allowed_k, k_max=max_allowed_k
                )
            elif method == "Agglomerative":
                st.session_state["hybrid_k_scores_df"] = evaluate_agglomerative_range(
                    hybrid_df, k_min=min_allowed_k, k_max=max_allowed_k
                )
            elif method == "GMM":
                st.session_state["hybrid_k_scores_df"] = evaluate_gmm_range(
                    hybrid_df, k_min=min_allowed_k, k_max=max_allowed_k
                )

    if st.session_state["hybrid_k_scores_df"] is not None:
        st.dataframe(st.session_state["hybrid_k_scores_df"], use_container_width=True)

    n_clusters = st.slider(
        "Число кластеров",
        min_value=2,
        max_value=min(8, len(hybrid_df)),
        value=min(4, len(hybrid_df)),
        step=1,
        key="hybrid_n_clusters"
    )

    if st.button("Запустить кластеризацию", key="hybrid_run_button"):
        if method == "KMeans":
            st.session_state["hybrid_clustering_result"] = run_kmeans(hybrid_df, n_clusters=n_clusters)
        elif method == "Agglomerative":
            st.session_state["hybrid_clustering_result"] = run_agglomerative(hybrid_df, n_clusters=n_clusters)
        elif method == "GMM":
            st.session_state["hybrid_clustering_result"] = run_gmm(hybrid_df, n_components=n_clusters)

else:
    st.subheader("Параметры HDBSCAN")

    min_cluster_size = st.number_input(
        "min_cluster_size",
        min_value=2,
        max_value=max(2, len(hybrid_df)),
        value=min(5, len(hybrid_df)),
        step=1,
        key="hybrid_hdbscan_min_cluster_size"
    )

    range_text = st.text_input(
        "Введите значения min_cluster_size через запятую",
        value="2,3,4,5,6,8,10",
        key="hybrid_hdbscan_range_text"
    )

    if st.button("Оценить HDBSCAN", key="hybrid_hdbscan_eval_button"):
        values = [int(x.strip()) for x in range_text.split(",") if x.strip()]
        values = sorted(set(v for v in values if v >= 2))

        st.session_state["hybrid_k_scores_df"] = evaluate_hdbscan_range(
            hybrid_df,
            min_cluster_size_values=values
        )

    if st.session_state["hybrid_k_scores_df"] is not None:
        st.dataframe(st.session_state["hybrid_k_scores_df"], use_container_width=True)

    if st.button("Запустить HDBSCAN", key="hybrid_hdbscan_run_button"):
        st.session_state["hybrid_clustering_result"] = run_hdbscan(
            hybrid_df,
            min_cluster_size=int(min_cluster_size)
        )

# -----------------------------
# Результаты
# -----------------------------
if st.session_state["hybrid_clustering_result"] is not None:
    result_obj = st.session_state["hybrid_clustering_result"]
    result_df = result_obj["result_df"]
    metrics = result_obj["metrics"]
    cluster_profiles = result_obj["cluster_profiles"]

    non_noise_df = result_df[result_df["cluster"] != -1].copy() if "cluster" in result_df.columns else result_df
    names_base_df = non_noise_df if len(non_noise_df) > 0 else result_df

    cluster_names_df = build_cluster_names(names_base_df, cluster_profiles) if len(cluster_profiles) > 0 else None

    st.subheader("Метрики")
    metrics_display = pd.DataFrame([metrics])
    st.dataframe(metrics_display, use_container_width=True)

    if cluster_names_df is not None:
        st.subheader("Автоматические названия кластеров")
        st.dataframe(cluster_names_df, use_container_width=True)

    st.subheader("Студенты и их кластеры")
    if cluster_names_df is not None:
        result_with_names = result_df.merge(
            cluster_names_df[["cluster", "suggested_name"]],
            on="cluster",
            how="left"
        )
        if "cluster" in result_with_names.columns:
            result_with_names["suggested_name"] = result_with_names["suggested_name"].fillna("Шум / выброс")
        st.dataframe(result_with_names, use_container_width=True)
    else:
        st.dataframe(result_df, use_container_width=True)

    if len(cluster_profiles) > 0:
        st.subheader("Средние профили кластеров")
        st.dataframe(cluster_profiles, use_container_width=True)

        st.subheader("Распределение студентов по кластерам")
        fig_counts = plot_cluster_counts(result_df)
        st.plotly_chart(fig_counts, use_container_width=True)

        if len(non_noise_df) >= 2 and non_noise_df["cluster"].nunique() >= 2:
            st.subheader("PCA-визуализация кластеров")
            fig_pca, pca_df = plot_pca_clusters(non_noise_df)
            st.plotly_chart(fig_pca, use_container_width=True)

        available_features = [col for col in cluster_profiles.columns if col != "cluster"]
        selected_feature = st.selectbox(
            "Выберите признак для сравнения кластеров",
            available_features,
            key="hybrid_selected_feature"
        )

        fig_profile = plot_cluster_profile_bar(cluster_profiles, selected_feature)
        st.plotly_chart(fig_profile, use_container_width=True)

    st.subheader("Анализ выбранного студента")
    student_ids = sorted(result_df["student_id"].astype(str).tolist())
    selected_student_id = st.selectbox(
        "Выберите студента",
        student_ids,
        key="hybrid_selected_student"
    )

    comparison_df, cluster_id = build_student_cluster_comparison(result_df, selected_student_id)

    if cluster_id == -1:
        st.info(f"Студент **{selected_student_id}** помечен как шумовая точка.")
    else:
        if cluster_names_df is not None:
            matched = cluster_names_df[cluster_names_df["cluster"] == cluster_id]
            if not matched.empty:
                cluster_name = matched["suggested_name"].iloc[0]
                st.info(
                    f"Студент **{selected_student_id}** относится к кластеру "
                    f"**{cluster_id}** — **{cluster_name}**."
                )
            else:
                st.info(f"Студент **{selected_student_id}** относится к кластеру **{cluster_id}**.")
        else:
            st.info(f"Студент **{selected_student_id}** относится к кластеру **{cluster_id}**.")

    student_row = result_df[result_df["student_id"].astype(str) == str(selected_student_id)].copy()
    st.dataframe(student_row, use_container_width=True)

    if cluster_id != -1 and not comparison_df.empty:
        show_only_strong = st.checkbox(
            "Показать только сильно отличающиеся признаки",
            value=False,
            key="hybrid_show_only_strong"
        )

        display_df = comparison_df.copy()
        if show_only_strong:
            display_df = display_df[display_df["strong_deviation"]].copy()

        if display_df.empty:
            st.success("Сильно отличающихся признаков не найдено.")
        else:
            styled_df = display_df.style.apply(highlight_large_deviation, axis=1).format({
                "student_value": "{:.4f}",
                "cluster_mean": "{:.4f}",
                "difference": "{:.4f}",
                "relative_diff_pct": "{:.2f}",
                "z_score": "{:.2f}",
            })
            st.write(styled_df)
