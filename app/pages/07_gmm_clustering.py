import pandas as pd
import streamlit as st

from src.state import init_session_state
from src.visualization import (
    plot_cluster_counts,
    plot_pca_clusters,
    plot_cluster_profile_bar,
)
from src.cluster_naming import build_cluster_names
from src.gmm_clustering import (
    run_gmm,
    evaluate_gmm_range,
)

st.set_page_config(page_title="Gaussian Mixture Models", layout="wide")
init_session_state()


def build_student_cluster_comparison(result_df: pd.DataFrame, student_id: str) -> tuple[pd.DataFrame, int]:
    student_row = result_df[result_df["student_id"].astype(str) == str(student_id)].copy()

    if student_row.empty:
        raise ValueError("Выбранный студент не найден в result_df.")

    cluster_id = int(student_row["cluster"].iloc[0])
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


def build_student_interpretation(comparison_df: pd.DataFrame, student_id: str, cluster_id: int) -> str:
    strong_df = comparison_df[comparison_df["strong_deviation"]].copy()

    if strong_df.empty:
        return (
            f"Студент {student_id} относится к кластеру {cluster_id} и в целом "
            f"ведёт себя близко к среднему профилю этого кластера."
        )

    top_df = strong_df.head(5)

    phrases = []
    for _, row in top_df.iterrows():
        direction = "выше" if row["difference"] > 0 else "ниже"
        phrases.append(
            f"{row['feature']} {direction} среднего "
            f"(значение {row['student_value']:.2f}, среднее кластера {row['cluster_mean']:.2f})"
        )

    joined = "; ".join(phrases)

    return (
        f"Студент {student_id} относится к кластеру {cluster_id}, "
        f"но по ряду признаков заметно отличается от типичного представителя этого кластера: "
        f"{joined}."
    )


st.title("Gaussian Mixture Models")
st.write("Кластеризация по обычным логовым признакам методом Gaussian Mixture Model.")

features_df_for_clustering = st.session_state.get("features_df_for_clustering")

if features_df_for_clustering is None:
    st.warning(
        "Сначала подготовьте данные на странице Data Preparation: "
        "загрузите лог, постройте признаки, найдите аномалии и примените исключение."
    )
    st.stop()

if len(features_df_for_clustering) < 2:
    st.error("Для кластеризации осталось слишком мало пользователей.")
    st.stop()

st.success(
    f"Данные готовы. Для кластеризации доступно пользователей: "
    f"{len(features_df_for_clustering)}"
)

st.subheader("Таблица признаков для кластеризации")
st.dataframe(features_df_for_clustering.head(), use_container_width=True)

# -----------------------------
# Параметры алгоритма
# -----------------------------
st.subheader("Параметры Gaussian Mixture Model")

covariance_type = st.selectbox(
    "Covariance type",
    options=["full", "tied", "diag", "spherical"],
    index=0,
    key="gmm_covariance_type"
)

# -----------------------------
# Оценка разных k
# -----------------------------
st.subheader("Оценка разных значений k")

col_k1, col_k2 = st.columns(2)

with col_k1:
    default_k_min = st.session_state.get("gmm_k_min", 2)
    k_min = st.number_input(
        "Минимальное k",
        min_value=2,
        max_value=20,
        value=default_k_min,
        step=1,
        key="gmm_k_min"
    )

with col_k2:
    default_k_max = st.session_state.get("gmm_k_max", 6)
    k_max = st.number_input(
        "Максимальное k",
        min_value=2,
        max_value=20,
        value=default_k_max,
        step=1,
        key="gmm_k_max"
    )

if k_min >= k_max:
    st.warning("Минимальное k должно быть меньше максимального k.")
else:
    if st.button("Оценить k", key="evaluate_gmm_button"):
        current_features = st.session_state["features_df_for_clustering"]

        if len(current_features) < 3:
            st.error("Для оценки диапазона k нужно хотя бы 3 пользователя.")
        else:
            max_allowed_k = min(int(k_max), len(current_features) - 1)
            min_allowed_k = int(k_min)

            if min_allowed_k > max_allowed_k:
                st.error("После исключения пользователей осталось слишком мало данных для выбранного диапазона k.")
            else:
                st.session_state["gmm_k_scores_df"] = evaluate_gmm_range(
                    current_features,
                    k_min=min_allowed_k,
                    k_max=max_allowed_k,
                    covariance_type=covariance_type
                )

if st.session_state["gmm_k_scores_df"] is not None:
    st.subheader("Метрики для разных k")
    st.dataframe(st.session_state["gmm_k_scores_df"], use_container_width=True)

# -----------------------------
# Запуск алгоритма
# -----------------------------
st.subheader("Запуск модели")

max_possible_k = min(8, len(features_df_for_clustering))
min_possible_k = 2

default_clusters = st.session_state.get("gmm_n_clusters", 4)
if default_clusters > max_possible_k:
    default_clusters = max_possible_k
if default_clusters < min_possible_k:
    default_clusters = min_possible_k

n_clusters = st.slider(
    "Число компонент",
    min_value=min_possible_k,
    max_value=max_possible_k,
    value=default_clusters,
    step=1,
    key="gmm_n_clusters"
)

if st.button("Запустить GMM", key="run_gmm_button"):
    current_features = st.session_state["features_df_for_clustering"]

    if len(current_features) < n_clusters:
        st.error("Число пользователей меньше числа компонент. Уменьшите k.")
    else:
        st.session_state["gmm_clustering_result"] = run_gmm(
            current_features,
            n_components=n_clusters,
            covariance_type=covariance_type
        )

# -----------------------------
# Результаты
# -----------------------------
if st.session_state["gmm_clustering_result"] is not None:
    clustering_result = st.session_state["gmm_clustering_result"]

    result_df = clustering_result["result_df"]
    metrics = clustering_result["metrics"]
    cluster_profiles = clustering_result["cluster_profiles"]

    cluster_names_df = build_cluster_names(result_df, cluster_profiles)

    st.subheader("Метрики кластеризации")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Silhouette", f"{metrics['silhouette_score']:.4f}")
    m2.metric("Calinski-Harabasz", f"{metrics['calinski_harabasz_score']:.4f}")
    m3.metric("Davies-Bouldin", f"{metrics['davies_bouldin_score']:.4f}")
    m4.metric("BIC", f"{metrics['bic']:.2f}")
    m5.metric("AIC", f"{metrics['aic']:.2f}")

    st.subheader("Автоматические названия кластеров")
    st.dataframe(cluster_names_df, use_container_width=True)

    st.subheader("Студенты и их кластеры")
    result_with_names = result_df.merge(
        cluster_names_df[["cluster", "suggested_name"]],
        on="cluster",
        how="left"
    )
    st.dataframe(result_with_names, use_container_width=True)

    st.subheader("Средние профили кластеров")
    profiles_with_names = cluster_profiles.merge(
        cluster_names_df[["cluster", "suggested_name"]],
        on="cluster",
        how="left"
    )
    st.dataframe(profiles_with_names, use_container_width=True)

    st.subheader("Распределение студентов по кластерам")
    fig_counts = plot_cluster_counts(result_df)
    st.plotly_chart(fig_counts, use_container_width=True)

    st.subheader("PCA-визуализация кластеров")
    fig_pca, pca_df = plot_pca_clusters(result_df)
    st.plotly_chart(fig_pca, use_container_width=True)

    with st.expander("Показать PCA-таблицу"):
        st.dataframe(pca_df, use_container_width=True)

    st.subheader("График среднего признака по кластерам")
    available_features = [
        col for col in cluster_profiles.columns
        if col != "cluster"
    ]

    selected_feature = st.selectbox(
        "Выберите признак для сравнения кластеров",
        available_features,
        key="selected_gmm_cluster_feature"
    )

    fig_profile = plot_cluster_profile_bar(cluster_profiles, selected_feature)
    st.plotly_chart(fig_profile, use_container_width=True)

    # Анализ студента
    st.subheader("Анализ выбранного студента")

    student_ids = sorted(result_df["student_id"].astype(str).tolist())

    default_student = st.session_state.get("selected_gmm_student_id")
    if default_student not in student_ids:
        default_student = student_ids[0]

    selected_student_id = st.selectbox(
        "Выберите студента",
        student_ids,
        index=student_ids.index(default_student),
        key="selected_gmm_student_id"
    )

    comparison_df, cluster_id = build_student_cluster_comparison(
        result_df,
        selected_student_id
    )

    cluster_name = cluster_names_df.loc[
        cluster_names_df["cluster"] == cluster_id, "suggested_name"
    ].iloc[0]

    student_confidence = result_df.loc[
        result_df["student_id"].astype(str) == str(selected_student_id), "gmm_confidence"
    ].iloc[0]

    st.info(
        f"Студент **{selected_student_id}** относится к кластеру "
        f"**{cluster_id}** — **{cluster_name}**. "
        f"Уверенность GMM: **{student_confidence:.4f}**."
    )

    student_row = result_df[result_df["student_id"].astype(str) == str(selected_student_id)].copy()
    st.subheader("Строка выбранного студента")
    st.dataframe(student_row, use_container_width=True)

    interpretation_text = build_student_interpretation(
        comparison_df,
        selected_student_id,
        cluster_id
    )
    st.subheader("Интерпретация")
    st.write(interpretation_text)

    st.subheader("Сравнение со средним по кластеру")
    show_only_strong = st.checkbox(
        "Показать только сильно отличающиеся признаки",
        value=False,
        key="show_only_strong_gmm_deviation"
    )

    display_df = comparison_df.copy()
    if show_only_strong:
        display_df = display_df[display_df["strong_deviation"]].copy()

    if display_df.empty:
        st.success("Сильно отличающихся признаков для выбранного студента не найдено.")
    else:
        styled_display_df = display_df.style.apply(highlight_large_deviation, axis=1).format({
            "student_value": "{:.4f}",
            "cluster_mean": "{:.4f}",
            "difference": "{:.4f}",
            "relative_diff_pct": "{:.2f}",
            "z_score": "{:.2f}",
        })

        st.write(styled_display_df)

else:
    st.info("Выберите параметры и нажмите «Запустить GMM».")