import streamlit as st

from src.state import init_session_state
from src.visualization import (
    plot_cluster_counts,
    plot_pca_clusters,
    plot_cluster_profile_bar,
)
from src.cluster_naming import build_cluster_names
from src.deep_embedding_clustering import (
    run_deep_embedding_clustering,
    evaluate_dec_range,
)

st.set_page_config(page_title="Deep Embedded Clustering", layout="wide")
init_session_state()

st.title("Deep Embedded Clustering (DEC)")
st.write(
    "Кластеризация обычных признаков с помощью Deep Embedded Clustering: "
    "автоэнкодер + уточнение кластеров в latent space."
)

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

st.subheader("Параметры DEC")

c1, c2, c3 = st.columns(3)
with c1:
    latent_dim = st.slider("Размер latent space", 2, 32, 10, 1, key="dec_latent_dim")
with c2:
    pretrain_epochs = st.slider("Эпохи предобучения", 10, 200, 50, 10, key="dec_pretrain_epochs")
with c3:
    batch_size = st.slider("Batch size", 8, 128, 32, 8, key="dec_batch_size")

st.subheader("Подбор числа кластеров")

k1, k2 = st.columns(2)
with k1:
    k_min = st.number_input("Минимальное k", min_value=2, max_value=20, value=2, step=1, key="dec_k_min")
with k2:
    k_max = st.number_input("Максимальное k", min_value=2, max_value=20, value=6, step=1, key="dec_k_max")

if st.button("Оценить k для DEC", key="dec_eval_button"):
    if k_min >= k_max:
        st.warning("Минимальное k должно быть меньше максимального.")
    else:
        st.session_state["dec_k_scores_df"] = evaluate_dec_range(
            features_df=features_df_for_clustering,
            k_min=int(k_min),
            k_max=int(k_max),
            latent_dim=int(latent_dim),
            pretrain_epochs=int(pretrain_epochs),
            dec_max_iter=800,
            batch_size=int(batch_size),
        )

if st.session_state.get("dec_k_scores_df") is not None:
    st.subheader("Метрики для разных k")
    st.dataframe(st.session_state["dec_k_scores_df"], use_container_width=True)

st.subheader("Запуск DEC")

max_possible_k = min(8, len(features_df_for_clustering))
n_clusters = st.slider(
    "Число кластеров",
    min_value=2,
    max_value=max_possible_k,
    value=4,
    step=1,
    key="dec_n_clusters"
)

dec_max_iter = st.slider(
    "Максимум итераций DEC",
    min_value=200,
    max_value=4000,
    value=2000,
    step=200,
    key="dec_max_iter"
)

if st.button("Запустить Deep Embedded Clustering", key="run_dec_button"):
    st.session_state["dec_clustering_result"] = run_deep_embedding_clustering(
        features_df=features_df_for_clustering,
        n_clusters=int(n_clusters),
        latent_dim=int(latent_dim),
        pretrain_epochs=int(pretrain_epochs),
        dec_max_iter=int(dec_max_iter),
        batch_size=int(batch_size),
    )

if st.session_state.get("dec_clustering_result") is not None:
    clustering_result = st.session_state["dec_clustering_result"]

    result_df = clustering_result["result_df"]
    metrics = clustering_result["metrics"]
    cluster_profiles = clustering_result["cluster_profiles"]

    cluster_names_df = build_cluster_names(result_df, cluster_profiles)

    st.subheader("Метрики кластеризации")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Silhouette", "-" if metrics["silhouette_score"] is None else f"{metrics['silhouette_score']:.4f}")
    m2.metric("Calinski-Harabasz", "-" if metrics["calinski_harabasz_score"] is None else f"{metrics['calinski_harabasz_score']:.4f}")
    m3.metric("Davies-Bouldin", "-" if metrics["davies_bouldin_score"] is None else f"{metrics['davies_bouldin_score']:.4f}")
    m4.metric("Final pretrain loss", f"{metrics['pretrain_loss_final']:.6f}")

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

    if result_df["cluster"].nunique() >= 2:
        st.subheader("PCA-визуализация кластеров")
        fig_pca, pca_df = plot_pca_clusters(result_df)
        st.plotly_chart(fig_pca, use_container_width=True)

        with st.expander("Показать PCA-таблицу"):
            st.dataframe(pca_df, use_container_width=True)

    st.subheader("График среднего признака по кластерам")
    available_features = [col for col in cluster_profiles.columns if col != "cluster"]
    selected_feature = st.selectbox(
        "Выберите признак для сравнения кластеров",
        available_features,
        key="selected_dec_cluster_feature"
    )

    fig_profile = plot_cluster_profile_bar(cluster_profiles, selected_feature)
    st.plotly_chart(fig_profile, use_container_width=True)

    st.success("DEC выполнен. Результаты можно использовать в сравнении методов.")
else:
    st.info("Выберите параметры и запустите Deep Embedded Clustering.")