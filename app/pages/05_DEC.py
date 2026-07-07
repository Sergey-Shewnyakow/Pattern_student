import pandas as pd
import plotly.express as px
import streamlit as st

from src.state import init_session_state
from src.deep_embedding_clustering import (
    run_deep_embedding_clustering,
    evaluate_dec_range,
)
from src.cluster_naming import build_cluster_names
from src.visualization import (
    plot_cluster_counts,
    plot_pca_clusters,
    plot_cluster_profile_bar,
)
from src.ui_styles import apply_global_styles

from src.cluster_name_editor import render_editable_cluster_names

st.set_page_config(
    page_title="DEC",
    layout="wide",
)

init_session_state()
apply_global_styles()


st.title("Deep Embedding Clustering")



def build_student_cluster_comparison(
    result_df: pd.DataFrame,
    student_id: str,
):
    student_row = result_df[
        result_df["student_id"].astype(str) == str(student_id)
    ].copy()

    if student_row.empty:
        raise ValueError("Выбранный студент не найден.")

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

        if abs(cluster_mean) > 1e-9:
            relative_diff_pct = (diff / cluster_mean) * 100
        else:
            relative_diff_pct = 0.0

        if cluster_std > 1e-9:
            z_score = diff / cluster_std
        else:
            z_score = 0.0

        strong_deviation = (
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
                "strong_deviation": strong_deviation,
            }
        )

    comparison_df = pd.DataFrame(rows).sort_values(
        by="z_score",
        key=lambda s: s.abs(),
        ascending=False,
    )

    return comparison_df, cluster_id


def highlight_large_deviation(row: pd.Series):
    if bool(row["strong_deviation"]):
        return ["background-color: orange"] * len(row)

    return [""] * len(row)


# ------------------------------------------------------------
# Данные
# ------------------------------------------------------------
features_df_for_clustering = st.session_state.get("features_df_for_clustering")

if features_df_for_clustering is None:
    st.warning(
        "Сначала выполните Data Preparation: загрузите лог, постройте признаки "
        "и примените исключение не-студентов."
    )
    st.stop()

if len(features_df_for_clustering) < 2:
    st.error("Для кластеризации осталось слишком мало пользователей.")
    st.stop()

st.success(
    f"Данные готовы. Пользователей для кластеризации: "
    f"{len(features_df_for_clustering)}"
)

st.subheader("Таблица признаков для кластеризации")
st.dataframe(
    features_df_for_clustering.head(),
    use_container_width=True,
)


# ------------------------------------------------------------
# Оценка k
# ------------------------------------------------------------
st.subheader("Быстрая оценка разных k")

col_k1, col_k2, col_k3 = st.columns(3)

with col_k1:
    k_min = st.number_input(
        "Минимальное k",
        min_value=2,
        max_value=20,
        value=2,
        step=1,
        key="dec_k_min",
    )

with col_k2:
    k_max = st.number_input(
        "Максимальное k",
        min_value=2,
        max_value=20,
        value=6,
        step=1,
        key="dec_k_max",
    )

with col_k3:
    quick_embedding_dim = st.number_input(
        "Embedding dim для оценки",
        min_value=2,
        max_value=16,
        value=2,
        step=1,
        key="dec_quick_embedding_dim",
    )

if st.button("Оценить DEC для разных k", key="evaluate_dec_button"):
    if k_min >= k_max:
        st.warning("Минимальное k должно быть меньше максимального k.")
    else:
        with st.spinner("Выполняется быстрая оценка DEC..."):
            st.session_state["dec_scores_df"] = evaluate_dec_range(
                features_df=features_df_for_clustering,
                k_min=int(k_min),
                k_max=min(int(k_max), len(features_df_for_clustering) - 1),
                embedding_dim=int(quick_embedding_dim),
                hidden_dim=32,
                pretrain_epochs=15,
            )

if st.session_state.get("dec_scores_df") is not None:
    st.dataframe(
        st.session_state["dec_scores_df"],
        use_container_width=True,
    )


# ------------------------------------------------------------
# Параметры DEC
# ------------------------------------------------------------
st.subheader("Параметры DEC")

max_possible_k = min(8, len(features_df_for_clustering))
min_possible_k = 2

numeric_features_count = len(
    features_df_for_clustering.select_dtypes(include="number").columns
)

max_embedding_dim = max(2, min(16, numeric_features_count - 1))

col_p1, col_p2, col_p3 = st.columns(3)

with col_p1:
    n_clusters = st.slider(
        "Число кластеров",
        min_value=min_possible_k,
        max_value=max_possible_k,
        value=min(4, max_possible_k),
        step=1,
        key="dec_n_clusters",
    )

with col_p2:
    embedding_dim = st.slider(
        "Размер embedding",
        min_value=2,
        max_value=max_embedding_dim,
        value=2,
        step=1,
        key="dec_embedding_dim",
    )

with col_p3:
    hidden_dim = st.selectbox(
        "Размер скрытого слоя",
        options=[16, 32, 64, 128],
        index=1,
        key="dec_hidden_dim",
    )

col_t1, col_t2 = st.columns(2)

with col_t1:
    pretrain_epochs = st.slider(
        "Эпохи обучения автоэнкодера",
        min_value=10,
        max_value=200,
        value=30,
        step=10,
        key="dec_pretrain_epochs",
    )

with col_t2:
    learning_rate = st.selectbox(
        "Learning rate",
        options=[1e-2, 1e-3, 1e-4],
        index=1,
        format_func=lambda x: f"{x:g}",
        key="dec_learning_rate",
    )

st.caption(
    "Рекомендуемые параметры для первого запуска: 4 кластера, embedding_dim=2, "
    "hidden_dim=32, epochs=30, learning_rate=0.001."
)


# ------------------------------------------------------------
# Запуск DEC
# ------------------------------------------------------------
if st.button("Запустить Deep Embedding Clustering", key="run_dec_button"):
    with st.spinner("Обучается автоэнкодер и выполняется кластеризация..."):
        try:
            st.session_state["dec_result"] = run_deep_embedding_clustering(
                features_df=features_df_for_clustering,
                n_clusters=int(n_clusters),
                embedding_dim=int(embedding_dim),
                hidden_dim=int(hidden_dim),
                pretrain_epochs=int(pretrain_epochs),
                learning_rate=float(learning_rate),
            )
        except Exception as e:
            st.exception(e)


# ------------------------------------------------------------
# Результаты
# ------------------------------------------------------------
if st.session_state.get("dec_result") is not None:
    dec_result = st.session_state["dec_result"]

    result_df = dec_result["result_df"]
    metrics = dec_result["metrics"]
    cluster_profiles = dec_result["cluster_profiles"]
    used_features = dec_result["used_features"]
    history = dec_result["history"]

    cluster_names_df = build_cluster_names(
        result_df=result_df,
        cluster_profiles=cluster_profiles,
    )

    cluster_names_df = render_editable_cluster_names(
        method_key="dec",
        cluster_names_df=cluster_names_df,
        title="Названия кластеров DEC",
    )

    st.subheader("Метрики кластеризации")

    m1, m2, m3 = st.columns(3)

    silhouette = metrics.get("silhouette_score")
    calinski = metrics.get("calinski_harabasz_score")
    davies = metrics.get("davies_bouldin_score")

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

    st.subheader("Использованные признаки")
    st.write(used_features)

    # ------------------------------------------------------------
    # Loss автоэнкодера
    # ------------------------------------------------------------
    st.subheader("График обучения автоэнкодера")

    if history.pretrain_losses:
        loss_df = pd.DataFrame(
            {
                "epoch": list(range(1, len(history.pretrain_losses) + 1)),
                "loss": history.pretrain_losses,
            }
        )

        fig_loss = px.line(
            loss_df,
            x="epoch",
            y="loss",
            title="Autoencoder reconstruction loss",
        )

        st.plotly_chart(fig_loss, use_container_width=True)

    # ------------------------------------------------------------
    # Интерпретация кластеров
    # ------------------------------------------------------------
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

    important_columns = [
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

    available_columns = [
        col for col in important_columns
        if col in result_with_names.columns
    ]

    st.dataframe(
        result_with_names[available_columns],
        use_container_width=True,
    )

    with st.expander("Показать полную таблицу студентов"):
        st.dataframe(result_with_names, use_container_width=True)

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

    # ------------------------------------------------------------
    # Embedding-график
    # ------------------------------------------------------------
    st.subheader("Embedding-пространство DEC")

    embedding_cols = [
        col for col in result_with_names.columns
        if col.startswith("embedding_")
    ]

    if len(embedding_cols) >= 2:
        fig_embedding = px.scatter(
            result_with_names,
            x="embedding_1",
            y="embedding_2",
            color="cluster",
            hover_data=[
                "student_id",
                "suggested_name",
                "cluster_probability",
            ],
            title="DEC embedding-пространство",
        )

        st.plotly_chart(fig_embedding, use_container_width=True)
    else:
        st.info("Для 2D-графика нужно embedding_dim >= 2.")

    # ------------------------------------------------------------
    # Графики как у остальных методов
    # ------------------------------------------------------------
    st.subheader("Распределение студентов по кластерам")

    fig_counts = plot_cluster_counts(result_df)
    st.plotly_chart(fig_counts, use_container_width=True)

    st.subheader("PCA-визуализация по исходным признакам")

    pca_input_df = result_df.drop(
        columns=[
            col for col in result_df.columns
            if col.startswith("embedding_")
            or col == "cluster_probability"
        ],
        errors="ignore",
    )

    fig_pca, pca_df = plot_pca_clusters(pca_input_df)
    st.plotly_chart(fig_pca, use_container_width=True)

    with st.expander("Показать PCA-таблицу"):
        st.dataframe(pca_df, use_container_width=True)

    st.subheader("График среднего признака по кластерам")

    available_features = [
        col for col in cluster_profiles.columns
        if col != "cluster"
    ]

    if available_features:
        selected_feature = st.selectbox(
            "Выберите признак для сравнения кластеров",
            available_features,
            key="selected_dec_cluster_feature",
        )

        fig_profile = plot_cluster_profile_bar(
            cluster_profiles,
            selected_feature,
        )

        st.plotly_chart(fig_profile, use_container_width=True)

    # ------------------------------------------------------------
    # Анализ студента
    # ------------------------------------------------------------
    st.subheader("Анализ выбранного студента")

    student_ids = sorted(result_df["student_id"].astype(str).tolist())

    selected_student_id = st.selectbox(
        "Выберите студента",
        student_ids,
        key="selected_dec_student_id",
    )

    comparison_df, cluster_id = build_student_cluster_comparison(
        result_df=result_df,
        student_id=selected_student_id,
    )

    student_row = result_with_names[
        result_with_names["student_id"].astype(str) == str(selected_student_id)
    ].copy()

    cluster_name = student_row["suggested_name"].iloc[0]
    cluster_probability = student_row["cluster_probability"].iloc[0]

    st.info(
        f"Студент **{selected_student_id}** относится к кластеру "
        f"**{cluster_id}** — **{cluster_name}**. "
        f"Вероятность принадлежности: **{cluster_probability:.3f}**."
    )

    st.dataframe(student_row, use_container_width=True)

    st.subheader("Сравнение со средним по кластеру")

    show_only_strong = st.checkbox(
        "Показать только сильно отличающиеся признаки",
        value=False,
        key="show_only_strong_deviation_dec",
    )

    display_df = comparison_df.copy()

    if show_only_strong:
        display_df = display_df[display_df["strong_deviation"]].copy()

    if display_df.empty:
        st.success("Сильно отличающихся признаков не найдено.")
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
    st.info("Выберите параметры и нажмите «Запустить Deep Embedding Clustering».")