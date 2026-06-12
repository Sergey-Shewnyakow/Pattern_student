import pandas as pd
import streamlit as st

from src.state import init_session_state
from src.hdbscan_clustering import (
    run_hdbscan,
    evaluate_hdbscan_params,
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
    page_title="HDBSCAN Clustering",
    layout="wide",
)

init_session_state()
apply_global_styles()


RU_COLUMNS = {
    "cluster": "Кластер",
    "suggested_name": "Название кластера",
    "description": "Описание",

    "cluster_probability": "Уверенность HDBSCAN",
    "is_noise": "Шумовой объект",

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
    Для шумовых объектов cluster = -1 сравнение выполняется со всеми шумовыми объектами.
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
    Формирует текстовое описание выбранного студента.
    """
    if cluster_id == -1:
        return (
            f"Студент {student_id} отнесён HDBSCAN к шумовым объектам "
            f"(cluster = -1). Это означает, что его поведение не вошло "
            f"ни в одну устойчивую плотностную группу. Такой студент может иметь "
            f"нетипичный или смешанный цифровой след."
        )

    strong_df = comparison_df[comparison_df["strong_deviation"]].copy()

    probability_text = ""

    if cluster_probability is not None:
        probability_text = (
            f" Уверенность HDBSCAN в принадлежности к кластеру: "
            f"{cluster_probability:.3f}."
        )

        if cluster_probability < 0.6:
            probability_text += (
                " Низкое значение может указывать на пограничный или смешанный профиль."
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


st.title("HDBSCAN Clustering")

st.write(
    "Плотностная кластеризация студентов методом HDBSCAN. "
    "Метод использует тот же набор признаков, что KMeans, Agglomerative и GMM, "
    "но не требует заранее задавать число кластеров и может выделять шумовые объекты."
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
# Оценка параметров
# ------------------------------------------------------------
st.subheader("Оценка параметров HDBSCAN")

st.write(
    "HDBSCAN не использует заранее заданное число кластеров. "
    "Вместо этого на результат влияют параметры `min_cluster_size` и `min_samples`."
)

col_eval_1, col_eval_2 = st.columns(2)

with col_eval_1:
    min_cluster_size_values_text = st.text_input(
        "Значения min_cluster_size через запятую",
        value="5,10,15,20",
        key="hdbscan_min_cluster_size_values",
    )

with col_eval_2:
    min_samples_values_text = st.text_input(
        "Значения min_samples через запятую или None",
        value="None,5,10",
        key="hdbscan_min_samples_values",
    )


def parse_int_list(text: str) -> list[int]:
    values = []

    for part in text.split(","):
        part = part.strip()

        if not part:
            continue

        values.append(int(part))

    return values


def parse_min_samples_list(text: str) -> list[int | None]:
    values = []

    for part in text.split(","):
        part = part.strip()

        if not part:
            continue

        if part.lower() == "none":
            values.append(None)
        else:
            values.append(int(part))

    return values


if st.button(
    "Оценить параметры HDBSCAN",
    key="evaluate_hdbscan_params_button",
):
    try:
        min_cluster_size_values = parse_int_list(min_cluster_size_values_text)
        min_samples_values = parse_min_samples_list(min_samples_values_text)

        st.session_state["hdbscan_params_scores_df"] = evaluate_hdbscan_params(
            features_df=features_df_for_clustering,
            min_cluster_size_values=min_cluster_size_values,
            min_samples_values=min_samples_values,
        )
    except Exception as e:
        st.exception(e)

if st.session_state.get("hdbscan_params_scores_df") is not None:
    st.subheader("Метрики для разных параметров")
    st.dataframe(
        st.session_state["hdbscan_params_scores_df"],
        use_container_width=True,
    )

    st.caption(
        "Слишком большая доля шума означает, что параметры слишком строгие. "
        "Если clusters_count = 0 или 1, метод не нашёл устойчивую структуру кластеров."
    )


# ------------------------------------------------------------
# Параметры HDBSCAN
# ------------------------------------------------------------
st.subheader("Параметры HDBSCAN")

max_cluster_size = max(2, min(100, len(features_df_for_clustering)))

col_p1, col_p2 = st.columns(2)

with col_p1:
    min_cluster_size = st.slider(
        "min_cluster_size",
        min_value=2,
        max_value=max_cluster_size,
        value=min(10, max_cluster_size),
        step=1,
        key="hdbscan_min_cluster_size",
    )

with col_p2:
    min_samples_mode = st.selectbox(
        "min_samples",
        options=["None", "Задать вручную"],
        index=0,
        key="hdbscan_min_samples_mode",
    )

    if min_samples_mode == "None":
        min_samples = None
    else:
        min_samples = st.slider(
            "Значение min_samples",
            min_value=1,
            max_value=max_cluster_size,
            value=min(5, max_cluster_size),
            step=1,
            key="hdbscan_min_samples_manual",
        )

st.caption(
    "Рекомендуемый старт: min_cluster_size = 10, min_samples = None. "
    "Если слишком много шума, уменьшите min_cluster_size или min_samples."
)


# ------------------------------------------------------------
# Запуск HDBSCAN
# ------------------------------------------------------------
if st.button(
    "Запустить HDBSCAN",
    key="run_hdbscan_button",
):
    try:
        st.session_state["hdbscan_result"] = run_hdbscan(
            features_df=features_df_for_clustering,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
        )
    except Exception as e:
        st.exception(e)


# ------------------------------------------------------------
# Результаты
# ------------------------------------------------------------
if st.session_state.get("hdbscan_result") is not None:
    hdbscan_result = st.session_state["hdbscan_result"]

    result_df = hdbscan_result["result_df"]
    metrics = hdbscan_result["metrics"]
    cluster_profiles = hdbscan_result["cluster_profiles"]
    used_features = hdbscan_result["used_features"]

    if cluster_profiles.empty:
        cluster_names_df = pd.DataFrame(
            columns=["cluster", "cluster_size", "suggested_name", "description"]
        )
    else:
        cluster_names_df = build_cluster_names(
            result_df=result_df[result_df["cluster"] != -1].copy(),
            cluster_profiles=cluster_profiles,
        )



    # Добавляем строку для шума
    noise_count = int((result_df["cluster"] == -1).sum())

    if noise_count > 0:
        noise_row = pd.DataFrame(
            [
                {
                    "cluster": -1,
                    "cluster_size": noise_count,
                    "suggested_name": "Шумовые / нетипичные студенты",
                    "description": (
                        "Пользователи, поведение которых не вошло ни в одну "
                        "устойчивую плотностную группу."
                    ),
                }
            ]
        )

        cluster_names_df = pd.concat(
            [noise_row, cluster_names_df],
            ignore_index=True,
        )

    cluster_names_df = render_editable_cluster_names(
        method_key="hdbscan",
        cluster_names_df=cluster_names_df,
        title="Названия кластеров HDBSCAN",
    )

    st.subheader("Метрики кластеризации")

    m1, m2, m3, m4, m5 = st.columns(5)

    silhouette = metrics.get("silhouette_score")
    calinski = metrics.get("calinski_harabasz_score")
    davies = metrics.get("davies_bouldin_score")
    clusters_count = metrics.get("clusters_count")
    noise_share = metrics.get("noise_share")

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
        "Число кластеров",
        "—" if clusters_count is None else str(clusters_count),
    )

    m5.metric(
        "Доля шума",
        "—" if noise_share is None else f"{noise_share:.2%}",
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
        "is_noise",
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

    if cluster_profiles.empty:
        st.info("HDBSCAN не выделил устойчивые кластеры, кроме шума.")
    else:
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
    # Шумовые объекты
    # ------------------------------------------------------------
    st.subheader("Шумовые / нетипичные студенты")

    noise_df = result_with_names[result_with_names["cluster"] == -1].copy()

    if noise_df.empty:
        st.success("Шумовых объектов не найдено.")
    else:
        st.warning(
            f"HDBSCAN пометил как шум {len(noise_df)} пользователей."
        )

        st.dataframe(
            noise_df[available_student_columns],
            use_container_width=True,
        )

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

    if cluster_profiles.empty:
        st.info("Нет устойчивых кластеров для построения профиля.")
    else:
        available_features = [
            col
            for col in cluster_profiles.columns
            if col != "cluster"
        ]

        if available_features:
            default_feature = st.session_state.get(
                "selected_hdbscan_cluster_feature",
                available_features[0],
            )

            if default_feature not in available_features:
                default_feature = available_features[0]

            selected_feature = st.selectbox(
                "Выберите признак для сравнения кластеров",
                available_features,
                index=available_features.index(default_feature),
                key="selected_hdbscan_cluster_feature",
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
        default_student = st.session_state.get("selected_hdbscan_student_id")

        if default_student not in student_ids:
            default_student = student_ids[0]

        selected_student_id = st.selectbox(
            "Выберите студента",
            student_ids,
            index=student_ids.index(default_student),
            key="selected_hdbscan_student_id",
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

        if cluster_id == -1:
            st.warning(
                f"Студент **{selected_student_id}** отнесён к шуму "
                f"(**cluster = -1**) — **{cluster_name}**."
            )
        else:
            st.info(
                f"Студент **{selected_student_id}** относится к кластеру "
                f"**{cluster_id}** — **{cluster_name}**. "
                f"Уверенность HDBSCAN: **{cluster_probability:.3f}**."
            )

        st.subheader("Строка выбранного студента")
        st.dataframe(
            student_row,
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
            key="show_only_strong_deviation_hdbscan",
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
    st.info("Выберите параметры и нажмите «Запустить HDBSCAN».")