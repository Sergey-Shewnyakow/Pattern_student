import pandas as pd
import streamlit as st

from src.state import init_session_state
from src.visualization import (
    plot_cluster_counts,
    plot_pca_clusters,
    plot_cluster_profile_bar,
)
from src.cluster_naming import build_cluster_names
from src.hdbscan_clustering import (
    run_hdbscan,
    evaluate_hdbscan_range,
)

st.set_page_config(page_title="HDBSCAN Clustering", layout="wide")
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


def build_student_interpretation(comparison_df: pd.DataFrame, student_id: str, cluster_id: int) -> str:
    if cluster_id == -1:
        return (
            f"Студент {student_id} не был отнесён ни к одному плотному кластеру "
            f"и помечен как шумовая точка (cluster = -1)."
        )

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


st.title("HDBSCAN Clustering")
st.write("Кластеризация по обычным логовым признакам методом HDBSCAN.")

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
st.subheader("Параметры HDBSCAN")

col1, col2, col3 = st.columns(3)

with col1:
    min_cluster_size = st.number_input(
        "min_cluster_size",
        min_value=2,
        max_value=max(2, len(features_df_for_clustering)),
        value=min(5, len(features_df_for_clustering)),
        step=1,
        key="hdbscan_min_cluster_size"
    )

with col2:
    min_samples_choice = st.checkbox(
        "Задать min_samples вручную",
        value=False,
        key="hdbscan_manual_min_samples"
    )

with col3:
    cluster_selection_method = st.selectbox(
        "cluster_selection_method",
        options=["eom", "leaf"],
        index=0,
        key="hdbscan_cluster_selection_method"
    )

if min_samples_choice:
    min_samples = st.number_input(
        "min_samples",
        min_value=1,
        max_value=max(1, len(features_df_for_clustering)),
        value=min(5, len(features_df_for_clustering)),
        step=1,
        key="hdbscan_min_samples"
    )
else:
    min_samples = None
    st.caption("min_samples не задан вручную.")

metric = st.selectbox(
    "Metric",
    options=["euclidean", "manhattan", "l1", "l2", "cosine"],
    index=0,
    key="hdbscan_metric"
)

# -----------------------------
# Оценка диапазона параметров
# -----------------------------
st.subheader("Оценка разных значений min_cluster_size")

range_text = st.text_input(
    "Введите значения min_cluster_size через запятую",
    value="2,3,4,5,6,8,10",
    key="hdbscan_range_text"
)

if st.button("Оценить HDBSCAN", key="evaluate_hdbscan_button"):
    try:
        values = [
            int(x.strip())
            for x in range_text.split(",")
            if x.strip()
        ]
        values = sorted(set(v for v in values if v >= 2))

        if not values:
            st.error("Нужно указать хотя бы одно значение min_cluster_size >= 2.")
        else:
            st.session_state["hdbscan_scores_df"] = evaluate_hdbscan_range(
                st.session_state["features_df_for_clustering"],
                min_cluster_size_values=values,
                min_samples=min_samples,
                cluster_selection_method=cluster_selection_method,
                metric=metric
            )
    except Exception as e:
        st.error(f"Ошибка разбора списка значений: {e}")

if st.session_state["hdbscan_scores_df"] is not None:
    st.subheader("Результаты оценки HDBSCAN")
    st.dataframe(st.session_state["hdbscan_scores_df"], use_container_width=True)
    st.caption(
        "Метрики silhouette / CH / DB считаются только по точкам без шума. "
        "Если HDBSCAN не нашёл хотя бы 2 кластера, метрики могут быть пустыми."
    )

# -----------------------------
# Запуск алгоритма
# -----------------------------
st.subheader("Запуск модели")

if st.button("Запустить HDBSCAN", key="run_hdbscan_button"):
    st.session_state["hdbscan_clustering_result"] = run_hdbscan(
        st.session_state["features_df_for_clustering"],
        min_cluster_size=int(min_cluster_size),
        min_samples=(int(min_samples) if min_samples is not None else None),
        cluster_selection_method=cluster_selection_method,
        metric=metric
    )

# -----------------------------
# Результаты
# -----------------------------
if st.session_state["hdbscan_clustering_result"] is not None:
    clustering_result = st.session_state["hdbscan_clustering_result"]

    result_df = clustering_result["result_df"]
    metrics = clustering_result["metrics"]
    cluster_profiles = clustering_result["cluster_profiles"]

    non_noise_df = result_df[result_df["cluster"] != -1].copy()

    st.subheader("Метрики кластеризации")
    m1, m2, m3, m4, m5 = st.columns(5)

    m1.metric(
        "Silhouette",
        "-" if metrics["silhouette_score"] is None else f"{metrics['silhouette_score']:.4f}"
    )
    m2.metric(
        "Calinski-Harabasz",
        "-" if metrics["calinski_harabasz_score"] is None else f"{metrics['calinski_harabasz_score']:.4f}"
    )
    m3.metric(
        "Davies-Bouldin",
        "-" if metrics["davies_bouldin_score"] is None else f"{metrics['davies_bouldin_score']:.4f}"
    )
    m4.metric("Число кластеров", metrics["cluster_count"])
    m5.metric("Шумовых точек", metrics["noise_count"])

    st.subheader("Студенты и их кластеры")
    st.dataframe(result_df, use_container_width=True)

    if len(non_noise_df) > 0 and len(cluster_profiles) > 0:
        cluster_names_df = build_cluster_names(non_noise_df, cluster_profiles)

        st.subheader("Автоматические названия кластеров")
        st.dataframe(cluster_names_df, use_container_width=True)

        result_with_names = result_df.merge(
            cluster_names_df[["cluster", "suggested_name"]],
            on="cluster",
            how="left"
        )
        result_with_names["suggested_name"] = result_with_names["suggested_name"].fillna("Шум / выброс")

        st.subheader("Студенты и их кластеры с названиями")
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

        if metrics["cluster_count"] >= 2 and len(non_noise_df) >= 2:
            st.subheader("PCA-визуализация кластеров")
            fig_pca, pca_df = plot_pca_clusters(non_noise_df)
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
            key="selected_hdbscan_cluster_feature"
        )

        fig_profile = plot_cluster_profile_bar(cluster_profiles, selected_feature)
        st.plotly_chart(fig_profile, use_container_width=True)

    else:
        st.warning(
            "HDBSCAN не выделил достаточного числа непустых кластеров для построения профилей и названий. "
            "Попробуйте изменить min_cluster_size или min_samples."
        )

    # Анализ студента
    st.subheader("Анализ выбранного студента")

    student_ids = sorted(result_df["student_id"].astype(str).tolist())

    default_student = st.session_state.get("selected_hdbscan_student_id")
    if default_student not in student_ids:
        default_student = student_ids[0]

    selected_student_id = st.selectbox(
        "Выберите студента",
        student_ids,
        index=student_ids.index(default_student),
        key="selected_hdbscan_student_id"
    )

    comparison_df, cluster_id = build_student_cluster_comparison(
        result_df,
        selected_student_id
    )

    if cluster_id == -1:
        st.info(
            f"Студент **{selected_student_id}** помечен как **шумовая точка** "
            f"и не вошёл ни в один плотный кластер."
        )
    else:
        if len(non_noise_df) > 0 and len(cluster_profiles) > 0:
            cluster_names_df = build_cluster_names(non_noise_df, cluster_profiles)
            cluster_name = cluster_names_df.loc[
                cluster_names_df["cluster"] == cluster_id, "suggested_name"
            ].iloc[0]

            st.info(
                f"Студент **{selected_student_id}** относится к кластеру "
                f"**{cluster_id}** — **{cluster_name}**."
            )
        else:
            st.info(f"Студент **{selected_student_id}** относится к кластеру **{cluster_id}**.")

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

    if cluster_id != -1 and not comparison_df.empty:
        st.subheader("Сравнение со средним по кластеру")
        show_only_strong = st.checkbox(
            "Показать только сильно отличающиеся признаки",
            value=False,
            key="show_only_strong_hdbscan_deviation"
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
    st.info("Выберите параметры и нажмите «Запустить HDBSCAN».")