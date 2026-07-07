import pandas as pd
import streamlit as st

from src.state import init_session_state
from src.clustering import run_kmeans, evaluate_kmeans_range
from src.visualization import (
    plot_cluster_counts,
    plot_pca_clusters,
    plot_cluster_profile_bar,
)
from src.cluster_naming import build_cluster_names

st.set_page_config(page_title="KMeans Clustering", layout="wide")
init_session_state()

from src.ui_styles import apply_global_styles
from src.cluster_name_editor import render_editable_cluster_names


apply_global_styles()
RU_COLUMNS = {
    "cluster": "Кластер",
    "suggested_name": "Название кластера",
    "total_events": "Всего событий",
    "active_days": "Активные дни",
    "active_weeks": "Активные недели",
    "unique_activities": "Уникальные активности",
    "unique_contexts": "Уникальные контексты",
    "unique_components": "Уникальные компоненты",
    "sessions_count": "Число сессий",
    "avg_session_length_min": "Средняя длительность сессии, мин",
    "median_session_length_min": "Медианная длительность сессии, мин",
    "max_session_length_min": "Макс. длительность сессии, мин",
    "avg_events_per_session": "Среднее событий за сессию",
    "max_events_per_session": "Макс. событий за сессию",
    "avg_gap_between_events_min": "Средний интервал между событиями, мин",
    "median_gap_between_events_min": "Медианный интервал между событиями, мин",
    "max_gap_between_events_min": "Макс. интервал между событиями, мин",
    "avg_events_per_week": "Среднее событий в неделю",
    "std_events_per_week": "Ст. откл. событий в неделю",
    "weekly_regularity_cv": "Коэф. вариации по неделям",
    "night_activity_ratio": "Доля ночной активности",
    "weekend_activity_ratio": "Доля активности в выходные",
    "repeated_activities_count": "Число повторяющихся активностей",
    "avg_activity_repeats": "Среднее число повторов",
    "max_activity_repeats": "Макс. число повторов",
    "long_pauses_over_1d": "Паузы больше 1 дня",
    "long_pauses_over_3d": "Паузы больше 3 дней",
    "rule_based_anomaly": "Аномалия по правилам",
    "iforest_label": "Метка Isolation Forest",
    "iforest_anomaly": "Аномалия Isolation Forest",
    "iforest_score": "Оценка Isolation Forest",
    "final_anomaly": "Итоговая аномалия",
    "exclude_manual": "Исключено вручную",
    "exclude_final": "Итоговое исключение",
    "video_events": "События видеолекций",
    "notes_events": "События конспектов",
    "presentation_events": "События презентаций",
    "practice_events": "События практических",
    "other_material_events": "Прочие учебные события",
    "study_material_events": "События учебных материалов",
    "learning_related_events": "Учебные события всего",
    "video_share": "Доля видеолекций",
    "notes_share": "Доля конспектов",
    "presentation_share": "Доля презентаций",
    "practice_share": "Доля практических",
    "used_video": "Использовал видео",
    "used_notes": "Использовал конспекты",
    "used_presentation": "Использовал презентации",
    "used_practice": "Использовал практические",
    "material_diversity_count": "Разнообразие материалов",
    "all_materials_used": "Использовал все материалы",
    "practice_without_materials": "Практические без материалов",
}
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


st.title("KMeans Clustering")


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

st.subheader("Оценка разных значений k")

col_k1, col_k2 = st.columns(2)

with col_k1:
    default_k_min = st.session_state.get("kmeans_k_min", 2)
    k_min = st.number_input(
        "Минимальное k",
        min_value=2,
        max_value=20,
        value=default_k_min,
        step=1,
        key="kmeans_k_min"
    )

with col_k2:
    default_k_max = st.session_state.get("kmeans_k_max", 6)
    k_max = st.number_input(
        "Максимальное k",
        min_value=2,
        max_value=20,
        value=default_k_max,
        step=1,
        key="kmeans_k_max"
    )

if k_min >= k_max:
    st.warning("Минимальное k должно быть меньше максимального k.")
else:
    if st.button("Оценить k", key="evaluate_kmeans_button"):
        current_features = st.session_state["features_df_for_clustering"]

        if len(current_features) < 3:
            st.error("Для оценки диапазона k нужно хотя бы 3 пользователя.")
        else:
            max_allowed_k = min(int(k_max), len(current_features) - 1)
            min_allowed_k = int(k_min)

            if min_allowed_k > max_allowed_k:
                st.error(
                    "После исключения пользователей осталось слишком мало данных "
                    "для выбранного диапазона k."
                )
            else:
                st.session_state["k_scores_df"] = evaluate_kmeans_range(
                    current_features,
                    k_min=min_allowed_k,
                    k_max=max_allowed_k
                )

if st.session_state["k_scores_df"] is not None:
    st.subheader("Метрики для разных k")
    st.dataframe(st.session_state["k_scores_df"], use_container_width=True)

st.subheader("Параметры KMeans")

max_possible_k = min(8, len(features_df_for_clustering))
min_possible_k = 2

if max_possible_k < 2:
    st.error("Недостаточно данных для KMeans.")
    st.stop()

default_clusters = st.session_state.get("kmeans_n_clusters", 4)
if default_clusters > max_possible_k:
    default_clusters = max_possible_k
if default_clusters < min_possible_k:
    default_clusters = min_possible_k

n_clusters = st.slider(
    "Число кластеров",
    min_value=min_possible_k,
    max_value=max_possible_k,
    value=default_clusters,
    step=1,
    key="kmeans_n_clusters"
)

if st.button("Запустить KMeans", key="run_kmeans_button"):
    current_features = st.session_state["features_df_for_clustering"]

    if len(current_features) < n_clusters:
        st.error(
            "Число пользователей меньше числа кластеров. "
            "Уменьшите k или вернитесь на страницу Data Preparation."
        )
    else:
        st.session_state["clustering_result"] = run_kmeans(
            current_features,
            n_clusters=n_clusters
        )

if st.session_state["clustering_result"] is not None:
    clustering_result = st.session_state["clustering_result"]

    result_df = clustering_result["result_df"]
    metrics = clustering_result["metrics"]
    cluster_profiles = clustering_result["cluster_profiles"]

    cluster_names_df = build_cluster_names(
        result_df=result_df,
        cluster_profiles=cluster_profiles,
    )

    cluster_names_df = render_editable_cluster_names(
        method_key="kmeans",
        cluster_names_df=cluster_names_df,
        title="Названия кластеров KMeans",
    )

    st.subheader("Метрики кластеризации")
    m1, m2, m3 = st.columns(3)
    m1.metric("Silhouette", f"{metrics['silhouette_score']:.4f}")
    m2.metric("Calinski-Harabasz", f"{metrics['calinski_harabasz_score']:.4f}")
    m3.metric("Davies-Bouldin", f"{metrics['davies_bouldin_score']:.4f}")

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


    # Таблица с русскими названиями признаков
    profiles_with_names_ru = profiles_with_names.copy()

    # если вдруг есть техническая колонка
    profiles_with_names_ru = profiles_with_names_ru.drop(columns=["Unnamed: 0"], errors="ignore")

    # переименование колонок
    profiles_with_names_ru = profiles_with_names_ru.rename(columns=RU_COLUMNS)

    st.subheader("Средние профили кластеров ")
    st.dataframe(profiles_with_names_ru, use_container_width=True)



    st.subheader("Распределение студентов по кластерам")
    fig_counts = plot_cluster_counts(result_df)
    st.plotly_chart(fig_counts, use_container_width=True)

    st.subheader("PCA-визуализация кластеров")
    fig_pca, pca_df = plot_pca_clusters(result_df)
    st.plotly_chart(fig_pca, use_container_width=True)

    with st.expander("Показать PCA-таблицу"):
        st.dataframe(pca_df, use_container_width=True)

    st.subheader("PCA-визуализация для отчёта")

    if st.checkbox(
            "Показать крупный PCA-график для вставки в отчёт",
            value=False,
            key="show_report_pca_kmeans",
    ):
        fig_report_pca, report_pca_df = plot_pca_clusters_for_report(
            features_scaled=features_scaled,
            result_df=result_df,
            cluster_names_df=cluster_names_df,
            title="PCA-визуализация кластеров K-Means",
        )

        st.plotly_chart(
            fig_report_pca,
            use_container_width=True,
        )

        st.download_button(
            "Скачать PCA-график HTML",
            data=fig_report_pca.to_html().encode("utf-8"),
            file_name="pca_clusters_kmeans_report.html",
            mime="text/html",
            key="download_pca_kmeans_html",
        )

        st.dataframe(
            report_pca_df,
            use_container_width=True,
        )


    st.subheader("График среднего признака по кластерам")
    available_features = [
        col for col in cluster_profiles.columns
        if col != "cluster"
    ]

    default_feature = st.session_state.get("selected_cluster_feature", available_features[0])
    if default_feature not in available_features:
        default_feature = available_features[0]

    selected_feature = st.selectbox(
        "Выберите признак для сравнения кластеров",
        available_features,
        index=available_features.index(default_feature),
        key="selected_cluster_feature"
    )

    fig_profile = plot_cluster_profile_bar(cluster_profiles, selected_feature)
    st.plotly_chart(fig_profile, use_container_width=True)

    st.subheader("Анализ выбранного студента")

    student_ids = sorted(result_df["student_id"].astype(str).tolist())

    default_student = st.session_state.get("selected_student_id")
    if default_student not in student_ids:
        default_student = student_ids[0]

    selected_student_id = st.selectbox(
        "Выберите студента",
        student_ids,
        index=student_ids.index(default_student),
        key="selected_student_id"
    )

    comparison_df, cluster_id = build_student_cluster_comparison(
        result_df,
        selected_student_id
    )

    cluster_name = cluster_names_df.loc[
        cluster_names_df["cluster"] == cluster_id, "suggested_name"
    ].iloc[0]

    st.info(
        f"Студент **{selected_student_id}** относится к кластеру "
        f"**{cluster_id}** — **{cluster_name}**."
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
    st.caption(
        "Оранжевым подсвечены признаки, которые заметно отличаются от среднего "
        "по своему кластеру: |z-score| ≥ 1.5 или относительное отклонение ≥ 30%."
    )

    show_only_strong = st.checkbox(
        "Показать только сильно отличающиеся признаки",
        value=False,
        key="show_only_strong_deviation"
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
    st.info("Выберите параметры и нажмите «Запустить KMeans».")