import pandas as pd
import streamlit as st

from src.state import init_session_state
from src.gmm_clustering import (
    run_gmm,
    evaluate_gmm_range,
    compare_gmm_covariance_types,
)
from src.visualization import (
    plot_cluster_counts,
    plot_pca_clusters,
    plot_cluster_profile_bar,
)
from src.cluster_naming import build_cluster_names
from src.ui_styles import apply_global_styles

from src.cluster_name_editor import render_editable_cluster_names


st.set_page_config(
    page_title="GMM Clustering",
    layout="wide",
)

init_session_state()
apply_global_styles()


RU_COLUMNS = {
    "cluster": "Кластер",
    "suggested_name": "Название кластера",
    "description": "Описание",

    "cluster_probability": "Вероятность принадлежности к кластеру",

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

    "video_events": "События видеолекций",
    "lecture_events": "События лекций Moodle",
    "practice_events": "События практических",
    "test_events": "События тестов",
    "page_events": "События страниц",

    "study_material_events": "События учебных материалов",
    "control_activity_events": "Контрольные активности",
    "learning_related_events": "Учебные события всего",

    "video_share": "Доля видеолекций",
    "lecture_share": "Доля лекций Moodle",
    "practice_share": "Доля практических",
    "test_share": "Доля тестов",
    "page_share": "Доля страниц",
    "study_material_share": "Доля учебных материалов",
    "control_activity_share": "Доля контрольных активностей",

    "used_video": "Использовал видео",
    "used_lecture": "Использовал лекции Moodle",
    "used_practice": "Использовал практические",
    "used_test": "Использовал тесты",
    "used_page": "Использовал страницы",
    "used_study_materials": "Использовал учебные материалы",
    "used_control_activities": "Использовал контрольные активности",

    "material_diversity_count": "Разнообразие материалов",
    "full_course_activity": "Комплексное использование курса",
    "practice_test_without_materials": "Практика/тесты без материалов",

    "staff_action_anomaly": "Аномалия по админ-действиям",
    "event_count_anomaly": "Аномалия по числу событий",
    "final_anomaly": "Итоговая аномалия",
    "exclude_manual": "Исключено вручную",
    "exclude_final": "Итоговое исключение",
}


def build_student_cluster_comparison(
    result_df: pd.DataFrame,
    student_id: str,
) -> tuple[pd.DataFrame, int]:
    """
    Сравнивает выбранного студента со средним профилем его кластера.
    """
    student_row = result_df[
        result_df["student_id"].astype(str) == str(student_id)
    ].copy()

    if student_row.empty:
        raise ValueError("Выбранный студент не найден в result_df.")

    cluster_id = int(student_row["cluster"].iloc[0])

    cluster_df = result_df[result_df["cluster"] == cluster_id].copy()

    numeric_cols = [
        col
        for col in result_df.select_dtypes(include="number").columns
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

        is_strong_deviation = (
            abs(z_score) >= 1.5
            or abs(relative_diff_pct) >= 30
        )

        rows.append(
            {
                "feature": col,
                "student_value": student_value,
                "cluster_mean": cluster_mean,
                "difference": diff,
                "relative_diff_pct": relative_diff_pct,
                "z_score": z_score,
                "strong_deviation": is_strong_deviation,
            }
        )

    comparison_df = pd.DataFrame(rows).sort_values(
        by="z_score",
        key=lambda s: s.abs(),
        ascending=False,
    )

    return comparison_df, cluster_id


def highlight_large_deviation(row: pd.Series):
    """
    Подсветка сильных отклонений от среднего по кластеру.
    """
    if bool(row["strong_deviation"]):
        return ["background-color: orange"] * len(row)

    return [""] * len(row)


def build_student_interpretation(
    comparison_df: pd.DataFrame,
    student_id: str,
    cluster_id: int,
    cluster_probability: float | None = None,
) -> str:
    """
    Формирует текстовое описание отличий выбранного студента от своего кластера.
    """
    strong_df = comparison_df[comparison_df["strong_deviation"]].copy()

    probability_text = ""

    if cluster_probability is not None:
        probability_text = (
            f" Вероятность принадлежности к данному кластеру по GMM: "
            f"{cluster_probability:.3f}."
        )

        if cluster_probability < 0.6:
            probability_text += (
                " Это указывает на смешанный профиль поведения: студент может быть "
                "похож сразу на несколько кластеров."
            )

    if strong_df.empty:
        return (
            f"Студент {student_id} относится к кластеру {cluster_id} и в целом "
            f"ведёт себя близко к среднему профилю этого кластера."
            f"{probability_text}"
        )

    top_df = strong_df.head(5)

    phrases = []

    for _, row in top_df.iterrows():
        direction = "выше" if row["difference"] > 0 else "ниже"

        phrases.append(
            f"{row['feature']} {direction} среднего "
            f"(значение {row['student_value']:.2f}, "
            f"среднее кластера {row['cluster_mean']:.2f})"
        )

    joined = "; ".join(phrases)

    return (
        f"Студент {student_id} относится к кластеру {cluster_id}, "
        f"но по ряду признаков заметно отличается от типичного представителя "
        f"этого кластера: {joined}."
        f"{probability_text}"
    )


st.title("GMM Clustering")

st.write(
    "Кластеризация студентов методом Gaussian Mixture Model. "
    "GMM использует тот же набор признаков, что KMeans и Agglomerative, "
    "но дополнительно оценивает вероятность принадлежности студента к кластеру."
)


# ------------------------------------------------------------
# Данные для кластеризации
# ------------------------------------------------------------
features_df_for_clustering = st.session_state.get("features_df_for_clustering")

if features_df_for_clustering is None:
    st.warning(
        "Сначала подготовьте данные на странице Data Preparation: "
        "загрузите лог, постройте признаки, найдите не-студентов "
        "и примените исключение."
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
st.dataframe(
    features_df_for_clustering.head(),
    use_container_width=True,
)


# ------------------------------------------------------------
# Оценка разных k
# ------------------------------------------------------------
st.subheader("Оценка разных значений k")

col_k1, col_k2, col_k3 = st.columns(3)

with col_k1:
    default_k_min = st.session_state.get("gmm_k_min", 2)

    k_min = st.number_input(
        "Минимальное k",
        min_value=2,
        max_value=20,
        value=default_k_min,
        step=1,
        key="gmm_k_min",
    )

with col_k2:
    default_k_max = st.session_state.get("gmm_k_max", 6)

    k_max = st.number_input(
        "Максимальное k",
        min_value=2,
        max_value=20,
        value=default_k_max,
        step=1,
        key="gmm_k_max",
    )

with col_k3:
    covariance_for_range = st.selectbox(
        "Covariance type для оценки k",
        options=["full", "diag", "tied", "spherical"],
        index=0,
        key="gmm_covariance_for_range",
    )

if k_min >= k_max:
    st.warning("Минимальное k должно быть меньше максимального k.")
else:
    if st.button(
        "Оценить k",
        key="evaluate_gmm_button",
    ):
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
                st.session_state["gmm_scores_df"] = evaluate_gmm_range(
                    features_df=current_features,
                    k_min=min_allowed_k,
                    k_max=max_allowed_k,
                    covariance_type=covariance_for_range,
                )

if st.session_state.get("gmm_scores_df") is not None:
    st.subheader("Метрики для разных k")

    st.dataframe(
        st.session_state["gmm_scores_df"],
        use_container_width=True,
    )

    st.caption(
        "Для GMM особенно важны AIC и BIC: чем ниже значение, тем лучше модель "
        "с учётом её сложности. Однако для интерпретируемости в ВКР можно оставить 4 кластера."
    )


# ------------------------------------------------------------
# Сравнение covariance_type
# ------------------------------------------------------------
st.subheader("Сравнение covariance_type")

st.caption(
    "full — самый гибкий вариант; diag — более устойчивый и простой; "
    "tied и spherical можно использовать как проверочные варианты."
)

if st.button(
    "Сравнить covariance_type",
    key="compare_gmm_covariance_button",
):
    default_components_for_covariance = st.session_state.get(
        "gmm_n_components",
        4,
    )

    st.session_state["gmm_covariance_comparison_df"] = (
        compare_gmm_covariance_types(
            features_df=features_df_for_clustering,
            n_components=default_components_for_covariance,
        )
    )

if st.session_state.get("gmm_covariance_comparison_df") is not None:
    st.dataframe(
        st.session_state["gmm_covariance_comparison_df"],
        use_container_width=True,
    )


# ------------------------------------------------------------
# Параметры GMM
# ------------------------------------------------------------
st.subheader("Параметры GMM")

max_possible_k = min(8, len(features_df_for_clustering))
min_possible_k = 2

if max_possible_k < 2:
    st.error("Недостаточно данных для GMM.")
    st.stop()

default_components = st.session_state.get("gmm_n_components", 4)

if default_components > max_possible_k:
    default_components = max_possible_k

if default_components < min_possible_k:
    default_components = min_possible_k

col_p1, col_p2 = st.columns(2)

with col_p1:
    n_components = st.slider(
        "Число компонент",
        min_value=min_possible_k,
        max_value=max_possible_k,
        value=default_components,
        step=1,
        key="gmm_n_components",
    )

with col_p2:
    covariance_type = st.selectbox(
        "Covariance type",
        options=["full", "diag", "tied", "spherical"],
        index=0,
        key="gmm_covariance_type",
    )

st.caption(
    "Для основного сравнения с KMeans можно использовать 4 компоненты и covariance_type='full'."
)


# ------------------------------------------------------------
# Запуск GMM
# ------------------------------------------------------------
if st.button(
    "Запустить GMM",
    key="run_gmm_button",
):
    current_features = st.session_state["features_df_for_clustering"]

    if len(current_features) < n_components:
        st.error(
            "Число пользователей меньше числа компонент. "
            "Уменьшите k или вернитесь на страницу Data Preparation."
        )
    else:
        st.session_state["gmm_result"] = run_gmm(
            features_df=current_features,
            n_components=n_components,
            covariance_type=covariance_type,
        )


# ------------------------------------------------------------
# Результаты
# ------------------------------------------------------------
if st.session_state.get("gmm_result") is not None:
    gmm_result = st.session_state["gmm_result"]

    result_df = gmm_result["result_df"]
    metrics = gmm_result["metrics"]
    cluster_profiles = gmm_result["cluster_profiles"]
    used_features = gmm_result["used_features"]

    cluster_names_df = build_cluster_names(
        result_df=result_df,
        cluster_profiles=cluster_profiles,
    )

    cluster_names_df = render_editable_cluster_names(
        method_key="gmm",
        cluster_names_df=cluster_names_df,
        title="Названия кластеров GMM",
    )

    st.subheader("Метрики кластеризации")

    m1, m2, m3, m4, m5 = st.columns(5)

    silhouette = metrics.get("silhouette_score")
    calinski = metrics.get("calinski_harabasz_score")
    davies = metrics.get("davies_bouldin_score")
    aic = metrics.get("aic")
    bic = metrics.get("bic")

    m1.metric(
        "Silhouette",
        "—" if silhouette is None else f"{silhouette:.4f}",
    )

    m2.metric(
        "Calinski-Harabasz",
        "—" if calinski is None else f"{calinski:.4f}",
    )

    m3.metric(
        "Davies-Bouldin",
        "—" if davies is None else f"{davies:.4f}",
    )

    m4.metric(
        "AIC",
        "—" if aic is None else f"{aic:.2f}",
    )

    m5.metric(
        "BIC",
        "—" if bic is None else f"{bic:.2f}",
    )

    st.subheader("Использованные признаки")
    st.write(used_features)

    st.subheader("Автоматические названия кластеров")
    st.dataframe(
        cluster_names_df,
        use_container_width=True,
    )

    st.subheader("Студенты и их кластеры")

    result_with_names = result_df.merge(
        cluster_names_df[["cluster", "suggested_name"]],
        on="cluster",
        how="left",
    )

    important_student_columns = [
        "student_id",
        "cluster",
        "suggested_name",
        "cluster_probability",
        "total_events",
        "active_days",
        "active_weeks",
        "video_share",
        "lecture_share",
        "practice_share",
        "test_share",
        "study_material_share",
        "control_activity_share",
    ]

    available_student_columns = [
        col for col in important_student_columns if col in result_with_names.columns
    ]

    st.dataframe(
        result_with_names[available_student_columns],
        use_container_width=True,
    )

    with st.expander("Показать полную таблицу студентов"):
        st.dataframe(
            result_with_names,
            use_container_width=True,
        )

    st.subheader("Средние профили кластеров")

    profiles_with_names = cluster_profiles.merge(
        cluster_names_df[["cluster", "suggested_name"]],
        on="cluster",
        how="left",
    )

    st.dataframe(
        profiles_with_names,
        use_container_width=True,
    )

    profiles_with_names_ru = profiles_with_names.copy()
    profiles_with_names_ru = profiles_with_names_ru.drop(
        columns=["Unnamed: 0"],
        errors="ignore",
    )
    profiles_with_names_ru = profiles_with_names_ru.rename(columns=RU_COLUMNS)

    st.subheader("Средние профили кластеров с русскими названиями признаков")
    st.dataframe(
        profiles_with_names_ru,
        use_container_width=True,
    )

    # ------------------------------------------------------------
    # Вероятности принадлежности к кластерам
    # ------------------------------------------------------------
    st.subheader("Уверенность GMM в назначении кластеров")

    low_confidence_threshold = st.slider(
        "Порог низкой уверенности",
        min_value=0.40,
        max_value=0.90,
        value=0.60,
        step=0.05,
        key="gmm_low_confidence_threshold",
    )

    low_confidence_df = result_with_names[
        result_with_names["cluster_probability"] < low_confidence_threshold
    ].copy()

    c_low_1, c_low_2 = st.columns(2)

    c_low_1.metric(
        "Студентов со смешанным профилем",
        len(low_confidence_df),
    )

    c_low_2.metric(
        "Средняя уверенность GMM",
        f"{result_with_names['cluster_probability'].mean():.3f}",
    )

    st.caption(
        "Низкая вероятность принадлежности к кластеру означает, что студент может быть "
        "похож сразу на несколько поведенческих групп."
    )

    if not low_confidence_df.empty:
        st.dataframe(
            low_confidence_df[available_student_columns],
            use_container_width=True,
        )
    else:
        st.success("Студентов с низкой уверенностью принадлежности к кластеру не найдено.")

    # ------------------------------------------------------------
    # Графики
    # ------------------------------------------------------------
    st.subheader("Распределение студентов по кластерам")

    fig_counts = plot_cluster_counts(result_df)
    st.plotly_chart(
        fig_counts,
        use_container_width=True,
    )

    st.subheader("PCA-визуализация кластеров")

    fig_pca, pca_df = plot_pca_clusters(result_df)
    st.plotly_chart(
        fig_pca,
        use_container_width=True,
    )

    with st.expander("Показать PCA-таблицу"):
        st.dataframe(
            pca_df,
            use_container_width=True,
        )

    st.subheader("График среднего признака по кластерам")

    available_features = [
        col
        for col in cluster_profiles.columns
        if col != "cluster"
    ]

    if available_features:
        default_feature = st.session_state.get(
            "selected_gmm_cluster_feature",
            available_features[0],
        )

        if default_feature not in available_features:
            default_feature = available_features[0]

        selected_feature = st.selectbox(
            "Выберите признак для сравнения кластеров",
            available_features,
            index=available_features.index(default_feature),
            key="selected_gmm_cluster_feature",
        )

        fig_profile = plot_cluster_profile_bar(
            cluster_profiles,
            selected_feature,
        )

        st.plotly_chart(
            fig_profile,
            use_container_width=True,
        )
    else:
        st.info("Нет числовых признаков для построения графика профиля.")

    # ------------------------------------------------------------
    # Анализ выбранного студента
    # ------------------------------------------------------------
    st.subheader("Анализ выбранного студента")

    student_ids = sorted(result_df["student_id"].astype(str).tolist())

    if student_ids:
        default_student = st.session_state.get("selected_gmm_student_id")

        if default_student not in student_ids:
            default_student = student_ids[0]

        selected_student_id = st.selectbox(
            "Выберите студента",
            student_ids,
            index=student_ids.index(default_student),
            key="selected_gmm_student_id",
        )

        comparison_df, cluster_id = build_student_cluster_comparison(
            result_df=result_df,
            student_id=selected_student_id,
        )

        cluster_name_rows = cluster_names_df.loc[
            cluster_names_df["cluster"] == cluster_id,
            "suggested_name",
        ]

        if cluster_name_rows.empty:
            cluster_name = "Без названия"
        else:
            cluster_name = cluster_name_rows.iloc[0]

        student_row = result_df[
            result_df["student_id"].astype(str) == str(selected_student_id)
        ].copy()

        cluster_probability = None

        if "cluster_probability" in student_row.columns:
            cluster_probability = float(student_row["cluster_probability"].iloc[0])

        if cluster_probability is None:
            st.info(
                f"Студент **{selected_student_id}** относится к кластеру "
                f"**{cluster_id}** — **{cluster_name}**."
            )
        else:
            st.info(
                f"Студент **{selected_student_id}** относится к кластеру "
                f"**{cluster_id}** — **{cluster_name}**. "
                f"Вероятность принадлежности: **{cluster_probability:.3f}**."
            )

        st.subheader("Строка выбранного студента")
        st.dataframe(
            student_row,
            use_container_width=True,
        )

        probability_columns = [
            col for col in student_row.columns
            if col.startswith("gmm_probability_cluster_")
        ]

        if probability_columns:
            st.subheader("Вероятности принадлежности к каждому кластеру")
            probability_view = student_row[
                ["student_id", "cluster", "cluster_probability"] + probability_columns
            ]
            st.dataframe(
                probability_view,
                use_container_width=True,
            )

        interpretation_text = build_student_interpretation(
            comparison_df=comparison_df,
            student_id=selected_student_id,
            cluster_id=cluster_id,
            cluster_probability=cluster_probability,
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
            key="show_only_strong_deviation_gmm",
        )

        display_df = comparison_df.copy()

        if show_only_strong:
            display_df = display_df[display_df["strong_deviation"]].copy()

        if display_df.empty:
            st.success(
                "Сильно отличающихся признаков для выбранного студента не найдено."
            )
        else:
            styled_display_df = display_df.style.apply(
                highlight_large_deviation,
                axis=1,
            ).format(
                {
                    "student_value": "{:.4f}",
                    "cluster_mean": "{:.4f}",
                    "difference": "{:.4f}",
                    "relative_diff_pct": "{:.2f}",
                    "z_score": "{:.2f}",
                }
            )

            st.write(styled_display_df)
    else:
        st.info("Нет студентов для анализа.")

else:
    st.info("Выберите параметры и нажмите «Запустить GMM».")