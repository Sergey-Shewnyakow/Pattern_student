import pandas as pd
import streamlit as st

from src.state import init_session_state
from src.cluster_naming import build_cluster_names
from src.pm_cluster_naming import build_pm_cluster_names

st.set_page_config(page_title="Comparison", layout="wide")
init_session_state()

st.title("Comparison of Two Approaches")
st.write(
    "Сравнение результатов кластеризации на обычных логовых признаках "
    "и на process mining-признаках."
)

log_clustering_result = st.session_state.get("clustering_result")
pm_clustering_result = st.session_state.get("pm_clustering_result")

if log_clustering_result is None or pm_clustering_result is None:
    st.warning(
        "Для сравнения нужно сначала выполнить:\n"
        "- KMeans Clustering\n"
        "- Process Mining Clustering"
    )
    st.stop()

# -----------------------------
# Данные двух контуров
# -----------------------------
log_result_df = log_clustering_result["result_df"]
log_metrics = log_clustering_result["metrics"]
log_cluster_profiles = log_clustering_result["cluster_profiles"]
log_cluster_names_df = build_cluster_names(log_result_df, log_cluster_profiles)

pm_result_df = pm_clustering_result["result_df"]
pm_metrics = pm_clustering_result["metrics"]
pm_cluster_profiles = pm_clustering_result["cluster_profiles"]
pm_cluster_names_df = build_pm_cluster_names(pm_result_df, pm_cluster_profiles)

# -----------------------------
# Метрики
# -----------------------------
st.subheader("Сравнение метрик кластеризации")

metrics_df = pd.DataFrame([
    {
        "approach": "Обычные логовые признаки",
        "silhouette_score": log_metrics["silhouette_score"],
        "calinski_harabasz_score": log_metrics["calinski_harabasz_score"],
        "davies_bouldin_score": log_metrics["davies_bouldin_score"],
    },
    {
        "approach": "Process mining-признаки",
        "silhouette_score": pm_metrics["silhouette_score"],
        "calinski_harabasz_score": pm_metrics["calinski_harabasz_score"],
        "davies_bouldin_score": pm_metrics["davies_bouldin_score"],
    },
])

st.dataframe(metrics_df, use_container_width=True)

st.caption(
    "Silhouette и Calinski-Harabasz: выше — лучше. "
    "Davies-Bouldin: ниже — лучше."
)

# -----------------------------
# Названия кластеров
# -----------------------------
st.subheader("Автоматические названия кластеров")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Обычные логовые признаки**")
    st.dataframe(log_cluster_names_df, use_container_width=True)

with col2:
    st.markdown("**Process mining-признаки**")
    st.dataframe(pm_cluster_names_df, use_container_width=True)

# -----------------------------
# Размеры кластеров
# -----------------------------
st.subheader("Сравнение размеров кластеров")

log_sizes = (
    log_result_df.groupby("cluster")
    .size()
    .reset_index(name="log_cluster_size")
    .merge(log_cluster_names_df[["cluster", "suggested_name"]], on="cluster", how="left")
)

pm_sizes = (
    pm_result_df.groupby("cluster")
    .size()
    .reset_index(name="pm_cluster_size")
    .merge(pm_cluster_names_df[["cluster", "suggested_name"]], on="cluster", how="left")
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Размеры логовых кластеров**")
    st.dataframe(log_sizes, use_container_width=True)

with col2:
    st.markdown("**Размеры process-кластеров**")
    st.dataframe(pm_sizes, use_container_width=True)

# -----------------------------
# Сопоставление студентов
# -----------------------------
st.subheader("Сопоставление кластеров по студентам")

common_students = sorted(
    set(log_result_df["student_id"].astype(str)) &
    set(pm_result_df["student_id"].astype(str))
)

mapping_df = (
    log_result_df[["student_id", "cluster"]]
    .rename(columns={"cluster": "log_cluster"})
    .merge(
        log_cluster_names_df[["cluster", "suggested_name"]]
        .rename(columns={
            "cluster": "log_cluster",
            "suggested_name": "log_cluster_name"
        }),
        on="log_cluster",
        how="left"
    )
    .merge(
        pm_result_df[["student_id", "cluster"]]
        .rename(columns={"cluster": "pm_cluster"}),
        on="student_id",
        how="inner"
    )
    .merge(
        pm_cluster_names_df[["cluster", "suggested_name"]]
        .rename(columns={
            "cluster": "pm_cluster",
            "suggested_name": "pm_cluster_name"
        }),
        on="pm_cluster",
        how="left"
    )
)

st.dataframe(mapping_df, use_container_width=True)

# -----------------------------
# Анализ выбранного студента
# -----------------------------
st.subheader("Сравнение двух подходов для выбранного студента")

if not common_students:
    st.error("Не найдено общих студентов между двумя результатами кластеризации.")
    st.stop()

default_student = st.session_state.get("comparison_selected_student", common_students[0])
if default_student not in common_students:
    default_student = common_students[0]

selected_student = st.selectbox(
    "Выберите студента",
    common_students,
    index=common_students.index(default_student),
    key="comparison_selected_student"
)

student_log_row = log_result_df[log_result_df["student_id"].astype(str) == str(selected_student)].copy()
student_pm_row = pm_result_df[pm_result_df["student_id"].astype(str) == str(selected_student)].copy()

student_log_cluster = int(student_log_row["cluster"].iloc[0])
student_pm_cluster = int(student_pm_row["cluster"].iloc[0])

student_log_cluster_name = log_cluster_names_df.loc[
    log_cluster_names_df["cluster"] == student_log_cluster, "suggested_name"
].iloc[0]

student_pm_cluster_name = pm_cluster_names_df.loc[
    pm_cluster_names_df["cluster"] == student_pm_cluster, "suggested_name"
].iloc[0]

c1, c2 = st.columns(2)

with c1:
    st.markdown("### Обычные логовые признаки")
    st.info(
        f"Студент **{selected_student}** относится к кластеру "
        f"**{student_log_cluster}** — **{student_log_cluster_name}**."
    )
    st.dataframe(student_log_row, use_container_width=True)

with c2:
    st.markdown("### Process mining-признаки")
    st.info(
        f"Студент **{selected_student}** относится к process-кластеру "
        f"**{student_pm_cluster}** — **{student_pm_cluster_name}**."
    )
    st.dataframe(student_pm_row, use_container_width=True)

# -----------------------------
# Краткий автоматический вывод
# -----------------------------
st.subheader("Краткий вывод по сравнению")

def choose_better_text(metrics_table: pd.DataFrame) -> str:
    log_row = metrics_table.iloc[0]
    pm_row = metrics_table.iloc[1]

    wins_log = 0
    wins_pm = 0

    # больше лучше
    if log_row["silhouette_score"] > pm_row["silhouette_score"]:
        wins_log += 1
    else:
        wins_pm += 1

    if log_row["calinski_harabasz_score"] > pm_row["calinski_harabasz_score"]:
        wins_log += 1
    else:
        wins_pm += 1

    # меньше лучше
    if log_row["davies_bouldin_score"] < pm_row["davies_bouldin_score"]:
        wins_log += 1
    else:
        wins_pm += 1

    if wins_log > wins_pm:
        return (
            "По внутренним метрикам в текущем эксперименте лучше выглядит подход "
            "на обычных логовых признаках. Это может означать, что агрегированные "
            "поведенческие характеристики лучше разделяют студентов по группам активности."
        )

    if wins_pm > wins_log:
        return (
            "По внутренним метрикам в текущем эксперименте лучше выглядит подход "
            "на process mining-признаках. Это может означать, что траекторные и процессные "
            "характеристики лучше отражают различия в поведении студентов."
        )

    return (
        "Оба подхода показывают сопоставимый результат по внутренним метрикам, "
        "поэтому ключевым критерием сравнения становится интерпретируемость кластеров "
        "и содержательность выявляемых паттернов."
    )

summary_text = choose_better_text(metrics_df)
st.write(summary_text)

