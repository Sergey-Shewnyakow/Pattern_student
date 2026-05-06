import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.state import init_session_state
from src.cluster_naming import build_cluster_names
from src.pm_cluster_naming import build_pm_cluster_names

st.set_page_config(page_title="All Methods Comparison", layout="wide")
init_session_state()


def safe_metric(value):
    if value is None:
        return None
    return float(value)


def extract_standard_result(method_name: str, approach_name: str, result_obj):
    """
    Для методов без шума:
    KMeans / Agglomerative / GMM / Autoencoder+KMeans / Hybrid без шума
    """
    if result_obj is None:
        return None

    metrics = result_obj["metrics"]
    result_df = result_obj["result_df"]

    return {
        "method": method_name,
        "approach": approach_name,
        "silhouette_score": safe_metric(metrics.get("silhouette_score")),
        "calinski_harabasz_score": safe_metric(metrics.get("calinski_harabasz_score")),
        "davies_bouldin_score": safe_metric(metrics.get("davies_bouldin_score")),
        "bic": safe_metric(metrics.get("bic")),
        "aic": safe_metric(metrics.get("aic")),
        "cluster_count": int(result_df["cluster"].nunique()) if "cluster" in result_df.columns else None,
        "noise_count": 0,
        "n_users": len(result_df),
    }


def extract_hdbscan_result(method_name: str, approach_name: str, result_obj):
    """
    Для HDBSCAN с учетом шума.
    """
    if result_obj is None:
        return None

    metrics = result_obj["metrics"]
    result_df = result_obj["result_df"]

    return {
        "method": method_name,
        "approach": approach_name,
        "silhouette_score": safe_metric(metrics.get("silhouette_score")),
        "calinski_harabasz_score": safe_metric(metrics.get("calinski_harabasz_score")),
        "davies_bouldin_score": safe_metric(metrics.get("davies_bouldin_score")),
        "bic": None,
        "aic": None,
        "cluster_count": metrics.get("cluster_count"),
        "noise_count": metrics.get("noise_count"),
        "n_users": len(result_df),
    }


def build_metrics_table():
    rows = []

    # Обычные признаки
    rows.append(extract_standard_result("KMeans", "Обычные признаки", st.session_state.get("clustering_result")))
    rows.append(extract_standard_result("Agglomerative", "Обычные признаки", st.session_state.get("agg_clustering_result")))
    rows.append(extract_standard_result("GMM", "Обычные признаки", st.session_state.get("gmm_clustering_result")))
    rows.append(extract_hdbscan_result("HDBSCAN", "Обычные признаки", st.session_state.get("hdbscan_clustering_result")))
    rows.append(extract_standard_result("Autoencoder + KMeans", "Обычные признаки", st.session_state.get("ae_clustering_result")))
    rows.append(extract_standard_result("DEC", "Обычные признаки", st.session_state.get("dec_clustering_result")))
    # Process mining-признаки
    rows.append(extract_standard_result("KMeans", "Process mining-признаки", st.session_state.get("pm_clustering_result")))
    rows.append(extract_standard_result("Agglomerative", "Process mining-признаки", st.session_state.get("pm_agg_clustering_result")))
    rows.append(extract_standard_result("GMM", "Process mining-признаки", st.session_state.get("pm_gmm_clustering_result")))
    rows.append(extract_hdbscan_result("HDBSCAN", "Process mining-признаки", st.session_state.get("pm_hdbscan_clustering_result")))

    # Гибридный подход
    hybrid_result = st.session_state.get("hybrid_clustering_result")
    if hybrid_result is not None:
        hybrid_method = st.session_state.get("hybrid_method", "Hybrid Method")
        if hybrid_method == "HDBSCAN":
            rows.append(extract_hdbscan_result(hybrid_method, "Гибридные признаки", hybrid_result))
        else:
            rows.append(extract_standard_result(hybrid_method, "Гибридные признаки", hybrid_result))

    rows = [r for r in rows if r is not None]

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def get_cluster_names_for_result(result_obj, approach_name: str):
    if result_obj is None:
        return None

    result_df = result_obj["result_df"]
    cluster_profiles = result_obj["cluster_profiles"]

    if len(cluster_profiles) == 0:
        return None

    if approach_name == "Process mining-признаки":
        base_df = result_df[result_df["cluster"] != -1].copy() if "cluster" in result_df.columns else result_df
        if len(base_df) == 0:
            return None
        return build_pm_cluster_names(base_df, cluster_profiles)

    base_df = result_df[result_df["cluster"] != -1].copy() if "cluster" in result_df.columns else result_df
    if len(base_df) == 0:
        return None
    return build_cluster_names(base_df, cluster_profiles)


def choose_best_method(metrics_df: pd.DataFrame):
    """
    Простая сводная логика:
    - silhouette: больше лучше
    - calinski_harabasz: больше лучше
    - davies_bouldin: меньше лучше
    """
    if metrics_df.empty:
        return None

    score_df = metrics_df.copy()

    for col in ["silhouette_score", "calinski_harabasz_score", "davies_bouldin_score"]:
        if col not in score_df.columns:
            score_df[col] = None

    score_df["rank_silhouette"] = score_df["silhouette_score"].rank(ascending=False, method="min")
    score_df["rank_ch"] = score_df["calinski_harabasz_score"].rank(ascending=False, method="min")
    score_df["rank_db"] = score_df["davies_bouldin_score"].rank(ascending=True, method="min")

    score_df["total_rank"] = score_df[["rank_silhouette", "rank_ch", "rank_db"]].sum(axis=1, min_count=1)

    best_idx = score_df["total_rank"].idxmin()
    return score_df.loc[best_idx]


def build_method_results():
    results = [
        # Обычные признаки
        ("KMeans", "Обычные признаки", st.session_state.get("clustering_result")),
        ("Agglomerative", "Обычные признаки", st.session_state.get("agg_clustering_result")),
        ("GMM", "Обычные признаки", st.session_state.get("gmm_clustering_result")),
        ("HDBSCAN", "Обычные признаки", st.session_state.get("hdbscan_clustering_result")),
        ("Autoencoder + KMeans", "Обычные признаки", st.session_state.get("ae_clustering_result")),
        ("DEC", "Обычные признаки", st.session_state.get("dec_clustering_result")),

        # Process mining-признаки
        ("KMeans", "Process mining-признаки", st.session_state.get("pm_clustering_result")),
        ("Agglomerative", "Process mining-признаки", st.session_state.get("pm_agg_clustering_result")),
        ("GMM", "Process mining-признаки", st.session_state.get("pm_gmm_clustering_result")),
        ("HDBSCAN", "Process mining-признаки", st.session_state.get("pm_hdbscan_clustering_result")),
    ]

    hybrid_result = st.session_state.get("hybrid_clustering_result")
    hybrid_method = st.session_state.get("hybrid_method")
    if hybrid_result is not None:
        if hybrid_method is None:
            hybrid_method = "Hybrid Method"
        results.append((hybrid_method, "Гибридные признаки", hybrid_result))

    results = [
        (method_name, approach_name, result_obj)
        for method_name, approach_name, result_obj in results
        if result_obj is not None
    ]

    return results


def get_result_by_label(method_results, label: str):
    for method_name, approach_name, result_obj in method_results:
        full_label = f"{method_name} | {approach_name}"
        if full_label == label:
            return method_name, approach_name, result_obj
    return None, None, None


def get_cluster_name_map(result_obj, approach_name: str) -> dict:
    """
    cluster_id -> красивое название
    """
    if result_obj is None:
        return {}

    cluster_names_df = get_cluster_names_for_result(result_obj, approach_name)
    result_df = result_obj["result_df"]

    cluster_ids = sorted(result_df["cluster"].unique().tolist())
    name_map = {}

    for cluster_id in cluster_ids:
        if cluster_id == -1:
            name_map[cluster_id] = "Шум / выброс"
            continue

        if cluster_names_df is not None:
            matched = cluster_names_df[cluster_names_df["cluster"] == cluster_id]
            if not matched.empty:
                name_map[cluster_id] = matched["suggested_name"].iloc[0]
            else:
                name_map[cluster_id] = f"Кластер {cluster_id}"
        else:
            name_map[cluster_id] = f"Кластер {cluster_id}"

    return name_map


def format_cluster_label(method_label: str, cluster_id: int, cluster_name_map: dict) -> str:
    cluster_name = cluster_name_map.get(cluster_id, f"Кластер {cluster_id}")
    return f"{method_label} | {cluster_id} | {cluster_name}"


def build_cluster_overlap_df(
    left_result_df: pd.DataFrame,
    right_result_df: pd.DataFrame,
    left_method_label: str,
    right_method_label: str,
    left_cluster_name_map: dict,
    right_cluster_name_map: dict,
) -> pd.DataFrame:
    merged = (
        left_result_df[["student_id", "cluster"]]
        .rename(columns={"cluster": "left_cluster"})
        .merge(
            right_result_df[["student_id", "cluster"]].rename(columns={"cluster": "right_cluster"}),
            on="student_id",
            how="inner"
        )
    )

    overlap_df = pd.crosstab(
        merged["left_cluster"],
        merged["right_cluster"]
    )

    overlap_df.index = [
        format_cluster_label(left_method_label, idx, left_cluster_name_map)
        for idx in overlap_df.index
    ]
    overlap_df.columns = [
        format_cluster_label(right_method_label, col, right_cluster_name_map)
        for col in overlap_df.columns
    ]

    return overlap_df


def build_sankey_figure(
    left_result_df: pd.DataFrame,
    right_result_df: pd.DataFrame,
    left_label: str,
    right_label: str,
    left_cluster_name_map: dict,
    right_cluster_name_map: dict,
):
    merged = (
        left_result_df[["student_id", "cluster"]]
        .rename(columns={"cluster": "left_cluster"})
        .merge(
            right_result_df[["student_id", "cluster"]].rename(columns={"cluster": "right_cluster"}),
            on="student_id",
            how="inner"
        )
    )

    flow_df = (
        merged.groupby(["left_cluster", "right_cluster"])
        .size()
        .reset_index(name="count")
    )

    left_clusters = sorted(flow_df["left_cluster"].unique().tolist())
    right_clusters = sorted(flow_df["right_cluster"].unique().tolist())

    left_nodes = [
        format_cluster_label(left_label, c, left_cluster_name_map)
        for c in left_clusters
    ]
    right_nodes = [
        format_cluster_label(right_label, c, right_cluster_name_map)
        for c in right_clusters
    ]

    all_nodes = left_nodes + right_nodes

    left_index = {cluster: idx for idx, cluster in enumerate(left_clusters)}
    right_index = {cluster: idx + len(left_clusters) for idx, cluster in enumerate(right_clusters)}

    sources = [left_index[row["left_cluster"]] for _, row in flow_df.iterrows()]
    targets = [right_index[row["right_cluster"]] for _, row in flow_df.iterrows()]
    values = flow_df["count"].tolist()

    left_colors = [
        "rgba(79, 70, 229, 0.35)" if c != -1 else "rgba(107, 114, 128, 0.30)"
        for c in left_clusters
    ]
    right_colors = [
        "rgba(14, 165, 233, 0.35)" if c != -1 else "rgba(107, 114, 128, 0.30)"
        for c in right_clusters
    ]
    node_colors = left_colors + right_colors

    link_colors = []
    for _, row in flow_df.iterrows():
        if row["left_cluster"] == -1 or row["right_cluster"] == -1:
            link_colors.append("rgba(107, 114, 128, 0.45)")
        else:
            link_colors.append("rgba(59, 130, 246, 0.28)")

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=28,
                    thickness=28,
                    line=dict(color="rgba(120,120,120,0.35)", width=0.8),
                    label=all_nodes,
                    color=node_colors,
                    hovertemplate="%{customdata}<extra></extra>",
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color=link_colors,
                    hovertemplate="Число студентов: %{value}<extra></extra>",
                ),
            )
        ]
    )

    fig.update_layout(
        title="Sankey-диаграмма соответствия кластеров",
        font=dict(
            size=16,
            color="#000000",
            family="Arial"
        ),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        height=850,
        margin=dict(l=30, r=30, t=90, b=30),
        annotations=[
            dict(
                x=0.01,
                y=1.08,
                xref="paper",
                yref="paper",
                text=f"<b>{left_label}</b>",
                showarrow=False,
                font=dict(size=18, color="#000000"),
                xanchor="left"
            ),
            dict(
                x=0.99,
                y=1.08,
                xref="paper",
                yref="paper",
                text=f"<b>{right_label}</b>",
                showarrow=False,
                font=dict(size=18, color="#000000"),
                xanchor="right"
            ),
        ]
    )

    return fig, merged


def build_agreement_matrix(method_results, metric_name: str = "ARI") -> pd.DataFrame:
    available = [
        (method_name, approach_name, result_obj)
        for method_name, approach_name, result_obj in method_results
        if result_obj is not None
    ]

    labels = [f"{m} | {a}" for m, a, _ in available]
    matrix = pd.DataFrame(index=labels, columns=labels, dtype=float)

    for i, (m1, a1, r1) in enumerate(available):
        df1 = r1["result_df"][["student_id", "cluster"]].copy()
        df1["student_id"] = df1["student_id"].astype(str)

        for j, (m2, a2, r2) in enumerate(available):
            df2 = r2["result_df"][["student_id", "cluster"]].copy()
            df2["student_id"] = df2["student_id"].astype(str)

            merged = df1.merge(df2, on="student_id", how="inner", suffixes=("_1", "_2"))

            if merged.empty:
                score = None
            else:
                labels_1 = merged["cluster_1"].tolist()
                labels_2 = merged["cluster_2"].tolist()

                if metric_name == "ARI":
                    score = adjusted_rand_score(labels_1, labels_2)
                else:
                    score = normalized_mutual_info_score(labels_1, labels_2)

            matrix.iloc[i, j] = score

    return matrix


st.title("All Methods Comparison")
st.write("Сравнение всех методов кластеризации для обычных, process mining и гибридных признаков.")

metrics_df = build_metrics_table()

if metrics_df.empty:
    st.warning(
        "Пока нет результатов для сравнения. "
        "Сначала запустите страницы с кластеризацией."
    )
    st.stop()

st.subheader("Сводная таблица метрик")
st.dataframe(metrics_df, use_container_width=True)

st.caption(
    "Silhouette и Calinski-Harabasz: выше — лучше. "
    "Davies-Bouldin: ниже — лучше. "
    "Для HDBSCAN метрики считаются по точкам без шума."
)

# Сравнение внутри подходов
st.subheader("Сравнение внутри каждого подхода")

approaches = metrics_df["approach"].unique().tolist()
cols = st.columns(len(approaches))

for col, approach in zip(cols, approaches):
    with col:
        st.markdown(f"### {approach}")
        approach_df = metrics_df[metrics_df["approach"] == approach].copy()
        st.dataframe(approach_df, use_container_width=True)

# Лучшие методы
st.subheader("Автоматический выбор лучших методов")

best_results = {}
for approach in metrics_df["approach"].unique():
    best_results[approach] = choose_best_method(metrics_df[metrics_df["approach"] == approach].copy())

best_all = choose_best_method(metrics_df.copy())

summary_cols = st.columns(len(best_results) + 1)

for i, (approach, best_row) in enumerate(best_results.items()):
    with summary_cols[i]:
        if best_row is not None:
            st.success(
                f"Лучший для {approach}:\n\n"
                f"**{best_row['method']}**"
            )

with summary_cols[-1]:
    if best_all is not None:
        st.success(
            f"Лучший среди всех:\n\n"
            f"**{best_all['method']}**\n"
            f"({best_all['approach']})"
        )

# Названия кластеров
st.subheader("Названия кластеров по методам")

method_results_for_names = build_method_results()

available_methods = [
    f"{method} | {approach}"
    for method, approach, result_obj in method_results_for_names
]

if available_methods:
    selected_method = st.selectbox(
        "Выберите метод для просмотра названий кластеров",
        available_methods,
        key="all_methods_cluster_name_selector"
    )

    selected_tuple = None
    for item in method_results_for_names:
        label = f"{item[0]} | {item[1]}"
        if label == selected_method:
            selected_tuple = item
            break

    if selected_tuple is not None:
        method_name, approach_name, result_obj = selected_tuple
        cluster_names_df = get_cluster_names_for_result(result_obj, approach_name)

        if cluster_names_df is not None:
            st.dataframe(cluster_names_df, use_container_width=True)
        else:
            st.info("Для этого результата нет доступных названий кластеров.")
else:
    st.info("Нет доступных результатов для показа названий кластеров.")

# Сравнение по выбранному студенту
st.subheader("Сравнение методов для выбранного студента")

student_sets = []

for _, _, result_obj in method_results_for_names:
    if result_obj is not None:
        student_sets.append(set(result_obj["result_df"]["student_id"].astype(str)))

if not student_sets:
    st.info("Нет результатов для анализа студентов.")
    st.stop()

common_students = sorted(set.intersection(*student_sets)) if len(student_sets) > 1 else sorted(student_sets[0])

if not common_students:
    st.info("Нет общего набора студентов между всеми доступными результатами.")
else:
    selected_student = st.selectbox(
        "Выберите студента",
        common_students,
        key="all_methods_selected_student"
    )

    student_rows = []

    for method_name, approach_name, result_obj in method_results_for_names:
        if result_obj is None:
            continue

        result_df = result_obj["result_df"]
        row = result_df[result_df["student_id"].astype(str) == str(selected_student)].copy()

        if row.empty:
            continue

        cluster_id = int(row["cluster"].iloc[0])

        cluster_name = None
        cluster_names_df = get_cluster_names_for_result(result_obj, approach_name)
        if cluster_names_df is not None and cluster_id != -1:
            matched = cluster_names_df[cluster_names_df["cluster"] == cluster_id]
            if not matched.empty:
                cluster_name = matched["suggested_name"].iloc[0]

        if cluster_id == -1:
            cluster_name = "Шум / выброс"

        confidence = None
        if "gmm_confidence" in row.columns:
            confidence = float(row["gmm_confidence"].iloc[0])

        student_rows.append({
            "method": method_name,
            "approach": approach_name,
            "cluster": cluster_id,
            "cluster_name": cluster_name,
            "confidence": confidence,
        })

    student_compare_df = pd.DataFrame(student_rows)
    st.dataframe(student_compare_df, use_container_width=True)

# Визуальное сравнение двух методов
st.subheader("Визуальное сравнение двух методов")

method_results = build_method_results()

available_method_labels = [
    f"{method_name} | {approach_name}"
    for method_name, approach_name, result_obj in method_results
]

if len(available_method_labels) >= 2:
    col_v1, col_v2 = st.columns(2)

    with col_v1:
        left_method_label = st.selectbox(
            "Первый метод",
            available_method_labels,
            index=0,
            key="comparison_left_method"
        )

    with col_v2:
        default_right_index = 1 if len(available_method_labels) > 1 else 0
        right_method_label = st.selectbox(
            "Второй метод",
            available_method_labels,
            index=default_right_index,
            key="comparison_right_method"
        )

    if left_method_label == right_method_label:
        st.info("Выберите два разных метода для сравнения.")
    else:
        left_method_name, left_approach_name, left_result_obj = get_result_by_label(method_results, left_method_label)
        right_method_name, right_approach_name, right_result_obj = get_result_by_label(method_results, right_method_label)

        left_result_df = left_result_obj["result_df"].copy()
        right_result_df = right_result_obj["result_df"].copy()

        left_result_df["student_id"] = left_result_df["student_id"].astype(str)
        right_result_df["student_id"] = right_result_df["student_id"].astype(str)

        merged_compare = (
            left_result_df[["student_id", "cluster"]]
            .rename(columns={"cluster": "left_cluster"})
            .merge(
                right_result_df[["student_id", "cluster"]].rename(columns={"cluster": "right_cluster"}),
                on="student_id",
                how="inner"
            )
        )

        if merged_compare.empty:
            st.warning("У выбранных методов нет пересекающихся студентов для сравнения.")
        else:
            ari_value = adjusted_rand_score(
                merged_compare["left_cluster"],
                merged_compare["right_cluster"]
            )
            nmi_value = normalized_mutual_info_score(
                merged_compare["left_cluster"],
                merged_compare["right_cluster"]
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Общих студентов", len(merged_compare))
            c2.metric("ARI", f"{ari_value:.4f}")
            c3.metric("NMI", f"{nmi_value:.4f}")

            left_cluster_name_map = get_cluster_name_map(left_result_obj, left_approach_name)
            right_cluster_name_map = get_cluster_name_map(right_result_obj, right_approach_name)

            st.markdown("### Таблица пересечения кластеров")
            overlap_df = build_cluster_overlap_df(
                left_result_df,
                right_result_df,
                left_method_label,
                right_method_label,
                left_cluster_name_map,
                right_cluster_name_map,
            )
            st.dataframe(overlap_df, use_container_width=True)

            st.markdown("### Sankey-диаграмма")
            fig_sankey, sankey_merged = build_sankey_figure(
                left_result_df,
                right_result_df,
                left_method_label,
                right_method_label,
                left_cluster_name_map,
                right_cluster_name_map,
            )
            st.plotly_chart(fig_sankey, use_container_width=True)

            st.caption(
                "Можно сравнивать любые доступные результаты: обычные признаки, "
                "process mining-признаки, нейросетевой и гибридный подходы."
            )

            with st.expander("Показать таблицу соответствия студентов"):
                st.dataframe(sankey_merged, use_container_width=True)
else:
    st.info("Для визуального сравнения нужно минимум два результата кластеризации.")

# Матрица согласованности
st.subheader("Матрица согласованности методов")

agreement_metric = st.radio(
    "Выберите метрику согласованности",
    options=["ARI", "NMI"],
    horizontal=True,
    key="agreement_metric_selector"
)

agreement_matrix = build_agreement_matrix(method_results, metric_name=agreement_metric)
st.dataframe(agreement_matrix.round(4), use_container_width=True)


text_parts = []

for approach, best_row in best_results.items():
    if best_row is not None:
        text_parts.append(
            f"Для подхода «{approach}» лучшим по совокупности внутренних метрик "
            f"оказался метод {best_row['method']}."
        )

if best_all is not None:
    text_parts.append(
        f"Если рассматривать все эксперименты вместе, наилучший результат "
        f"в текущем запуске показал {best_all['method']} "
        f"для подхода «{best_all['approach']}»."
    )

if "Гибридные признаки" in metrics_df["approach"].unique():
    text_parts.append(
        "Гибридный подход позволяет проверить, даёт ли совместное использование "
        "обычных логовых и process mining-признаков дополнительный выигрыш "
        "по качеству кластеризации и интерпретируемости паттернов."
    )

text_parts.append(
    "Финальный выбор метода следует делать не только по внутренним метрикам, "
    "но и по интерпретируемости кластеров, устойчивости результатов и содержательности "
    "выявленных поведенческих паттернов."
)

st.write(" ".join(text_parts))

