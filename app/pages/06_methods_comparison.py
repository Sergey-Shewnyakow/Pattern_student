import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.state import init_session_state
from src.cluster_naming import build_cluster_names
from src.ui_styles import apply_global_styles

from src.cluster_name_editor import apply_custom_cluster_names

st.set_page_config(
    page_title="Methods Comparison",
    layout="wide",
)

init_session_state()
apply_global_styles()


st.title("Сравнение методов кластеризации")



# ------------------------------------------------------------
# Получение результатов из session_state
# ------------------------------------------------------------
def get_result_from_session(possible_keys: list[str]):
    for key in possible_keys:
        value = st.session_state.get(key)
        if value is not None:
            return value
    return None


method_results_all = {
    "KMeans": get_result_from_session(["clustering_result", "kmeans_result"]),
    "Agglomerative": get_result_from_session(["agglomerative_result"]),
    "GMM": get_result_from_session(["gmm_result"]),
    "HDBSCAN": get_result_from_session(["hdbscan_result"]),
    "DEC": get_result_from_session(["dec_result", "deep_embedding_result"]),
}


METHOD_INFO = {
    "KMeans": {
        "type": "Центроидный метод",
        "comment": "Основной интерпретируемый метод. Удобен, когда нужно заранее задать 4 кластера.",
    },
    "Agglomerative": {
        "type": "Иерархический метод",
        "comment": "Проверяет устойчивость паттернов при иерархическом объединении студентов.",
    },
    "GMM": {
        "type": "Вероятностный метод",
        "comment": "Позволяет анализировать смешанные профили через вероятность принадлежности к кластеру.",
    },
    "HDBSCAN": {
        "type": "Плотностный метод",
        "comment": "Не требует заранее задавать число кластеров и может выделять шумовые объекты.",
    },
    "DEC": {
        "type": "Нейросетевой метод",
        "comment": "Дополнительный эксперимент с embedding-представлением студентов.",
    },
}


ready_methods = [
    method_name
    for method_name, result in method_results_all.items()
    if result is not None
]

st.subheader("Выбор методов для сравнения")

status_df = pd.DataFrame(
    [
        {
            "method": method_name,
            "status": "Готово" if result is not None else "Не запускался",
        }
        for method_name, result in method_results_all.items()
    ]
)

st.dataframe(status_df, use_container_width=True)

if not ready_methods:
    st.warning(
        "Пока нет результатов для сравнения. Сначала запустите хотя бы один метод "
        "кластеризации на соответствующей странице."
    )
    st.stop()

selected_methods = st.multiselect(
    "Выберите методы, которые нужно включить в сравнение",
    options=list(method_results_all.keys()),
    default=ready_methods,
)

selected_methods = [
    method for method in selected_methods
    if method_results_all.get(method) is not None
]

if not selected_methods:
    st.warning("Выберите хотя бы один уже запущенный метод.")
    st.stop()

method_results = {
    method: method_results_all[method]
    for method in selected_methods
}


# ------------------------------------------------------------
# Вспомогательные функции
# ------------------------------------------------------------
def safe_metric(metrics: dict, key: str):
    if metrics is None:
        return None

    value = metrics.get(key)

    if value is None:
        return None

    try:
        return float(value)
    except Exception:
        return None


def get_clusters_count(result_df: pd.DataFrame) -> int:
    if result_df is None or result_df.empty or "cluster" not in result_df.columns:
        return 0

    labels = set(result_df["cluster"].dropna().astype(int).tolist())

    if -1 in labels:
        labels.remove(-1)

    return len(labels)


def get_noise_count(result_df: pd.DataFrame) -> int:
    if result_df is None or result_df.empty or "cluster" not in result_df.columns:
        return 0

    return int((result_df["cluster"] == -1).sum())


def get_cluster_sizes(result_df: pd.DataFrame) -> pd.DataFrame:
    if result_df is None or result_df.empty or "cluster" not in result_df.columns:
        return pd.DataFrame(columns=["cluster", "cluster_size"])

    return (
        result_df.groupby("cluster")
        .size()
        .reset_index(name="cluster_size")
        .sort_values("cluster")
        .reset_index(drop=True)
    )


def get_cluster_names(method_name: str, result: dict) -> pd.DataFrame:
    """
    Строит названия кластеров через общий cluster_naming.py
    и применяет ручные названия, если пользователь их сохранял.
    """
    if result is None:
        return pd.DataFrame(
            columns=["cluster", "cluster_size", "suggested_name", "description"]
        )

    result_df = result.get("result_df")
    cluster_profiles = result.get("cluster_profiles")

    if result_df is None or cluster_profiles is None:
        return pd.DataFrame(
            columns=["cluster", "cluster_size", "suggested_name", "description"]
        )

    if cluster_profiles.empty:
        names_df = pd.DataFrame(
            columns=["cluster", "cluster_size", "suggested_name", "description"]
        )
    else:
        non_noise_result_df = result_df[result_df["cluster"] != -1].copy()

        if non_noise_result_df.empty:
            names_df = pd.DataFrame(
                columns=["cluster", "cluster_size", "suggested_name", "description"]
            )
        else:
            names_df = build_cluster_names(
                result_df=non_noise_result_df,
                cluster_profiles=cluster_profiles,
            )

    noise_count = get_noise_count(result_df)

    if noise_count > 0:
        noise_row = pd.DataFrame(
            [
                {
                    "cluster": -1,
                    "cluster_size": noise_count,
                    "suggested_name": "Шумовые / нетипичные студенты",
                    "description": (
                        "Пользователи, поведение которых не вошло ни в одну "
                        "устойчивую группу. Такое поведение может быть нетипичным, "
                        "смешанным или редким."
                    ),
                }
            ]
        )

        names_df = pd.concat([noise_row, names_df], ignore_index=True)

    method_key_map = {
        "KMeans": "kmeans",
        "Agglomerative": "agglomerative",
        "GMM": "gmm",
        "HDBSCAN": "hdbscan",
        "DEC": "dec",
    }

    method_key = method_key_map.get(method_name, method_name.lower())

    names_df = apply_custom_cluster_names(
        method_key=method_key,
        cluster_names_df=names_df,
    )

    return names_df

def build_method_summary(
    method_name: str,
    result: dict,
) -> dict:
    result_df = result.get("result_df")
    metrics = result.get("metrics", {})

    return {
        "method": method_name,
        "method_type": METHOD_INFO[method_name]["type"],
        "clusters_count": get_clusters_count(result_df),
        "noise_count": get_noise_count(result_df),
        "students_count": 0 if result_df is None else len(result_df),
        "silhouette_score": safe_metric(metrics, "silhouette_score"),
        "calinski_harabasz_score": safe_metric(metrics, "calinski_harabasz_score"),
        "davies_bouldin_score": safe_metric(metrics, "davies_bouldin_score"),
        "comment": METHOD_INFO[method_name]["comment"],
    }


def attach_cluster_names_to_students(
    method_name: str,
    result: dict,
) -> pd.DataFrame:
    """
    Возвращает таблицу:
    student_id, <method>_cluster, <method>_pattern, <method>_probability.
    """
    result_df = result.get("result_df")

    if result_df is None or result_df.empty:
        return pd.DataFrame()

    if "student_id" not in result_df.columns or "cluster" not in result_df.columns:
        return pd.DataFrame()

    names_df = get_cluster_names(method_name, result)

    temp_df = result_df[["student_id", "cluster"]].copy()
    temp_df["student_id"] = temp_df["student_id"].astype(str)

    temp_df = temp_df.merge(
        names_df[["cluster", "suggested_name"]],
        on="cluster",
        how="left",
    )

    temp_df = temp_df.rename(
        columns={
            "cluster": f"{method_name}_cluster",
            "suggested_name": f"{method_name}_pattern",
        }
    )

    if "cluster_probability" in result_df.columns:
        temp_df[f"{method_name}_probability"] = result_df["cluster_probability"].values

    return temp_df


def build_student_patterns_table(method_results: dict) -> pd.DataFrame:
    """
    Собирает таблицу назначений студентов по выбранным методам.
    """
    tables = []

    for method_name, result in method_results.items():
        table = attach_cluster_names_to_students(method_name, result)

        if not table.empty:
            tables.append(table)

    if not tables:
        return pd.DataFrame()

    result_table = tables[0]

    for table in tables[1:]:
        result_table = result_table.merge(
            table,
            on="student_id",
            how="outer",
        )

    pattern_columns = [
        col for col in result_table.columns
        if col.endswith("_pattern")
    ]

    def count_unique_patterns(row):
        values = [
            str(row[col])
            for col in pattern_columns
            if pd.notna(row[col])
        ]

        return len(set(values))

    def agreement_status(row):
        unique_count = row["unique_patterns_count"]

        if unique_count <= 1:
            return "Паттерн совпадает"

        return "Паттерн различается"

    result_table["unique_patterns_count"] = result_table.apply(
        count_unique_patterns,
        axis=1,
    )

    result_table["agreement_status"] = result_table.apply(
        agreement_status,
        axis=1,
    )

    return result_table


def highlight_agreement_rows(row: pd.Series):
    if row.get("agreement_status") == "Паттерн совпадает":
        return ["background-color: #d9f7d9"] * len(row)

    if row.get("agreement_status") == "Паттерн различается":
        return ["background-color: #ffe1e1"] * len(row)

    return [""] * len(row)


def build_sankey_between_methods(
    student_patterns_df: pd.DataFrame,
    method_a: str,
    method_b: str,
):
    """
    Строит Sankey между названиями паттернов двух методов.
    """
    col_a = f"{method_a}_pattern"
    col_b = f"{method_b}_pattern"

    if col_a not in student_patterns_df.columns or col_b not in student_patterns_df.columns:
        return None

    flow_df = student_patterns_df[[col_a, col_b]].dropna().copy()

    if flow_df.empty:
        return None

    flow_df = (
        flow_df.groupby([col_a, col_b])
        .size()
        .reset_index(name="count")
    )

    left_labels = sorted(flow_df[col_a].unique().tolist())
    right_labels = sorted(flow_df[col_b].unique().tolist())

    labels = (
        [f"{method_a}: {label}" for label in left_labels]
        + [f"{method_b}: {label}" for label in right_labels]
    )

    label_to_index = {
        label: idx
        for idx, label in enumerate(labels)
    }

    sources = []
    targets = []
    values = []

    for _, row in flow_df.iterrows():
        source_label = f"{method_a}: {row[col_a]}"
        target_label = f"{method_b}: {row[col_b]}"

        sources.append(label_to_index[source_label])
        targets.append(label_to_index[target_label])
        values.append(int(row["count"]))

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=20,
                    thickness=18,
                    line=dict(width=0.5),
                    label=labels,
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                ),
            )
        ]
    )

    fig.update_layout(
        title_text=f"Sankey-сравнение паттернов: {method_a} → {method_b}",
        font_size=11,
    )

    return fig


def build_interpretation_text(summary_df: pd.DataFrame, student_patterns_df: pd.DataFrame) -> str:
    text_parts = []

    silhouette_df = summary_df.dropna(subset=["silhouette_score"]).copy()

    if not silhouette_df.empty:
        best_silhouette_row = silhouette_df.sort_values(
            "silhouette_score",
            ascending=False,
        ).iloc[0]

        text_parts.append(
            f"Наибольшее значение Silhouette среди выбранных методов показал "
            f"{best_silhouette_row['method']} "
            f"({best_silhouette_row['silhouette_score']:.4f}). "
            f"Это означает более выраженное разделение студентов в выбранном "
            f"признаковом пространстве."
        )

    davies_df = summary_df.dropna(subset=["davies_bouldin_score"]).copy()

    if not davies_df.empty:
        best_davies_row = davies_df.sort_values(
            "davies_bouldin_score",
            ascending=True,
        ).iloc[0]

        text_parts.append(
            f"Наименьшее значение Davies-Bouldin получил метод "
            f"{best_davies_row['method']} "
            f"({best_davies_row['davies_bouldin_score']:.4f}). "
            f"Для этой метрики меньшее значение считается более предпочтительным."
        )

    if not student_patterns_df.empty and "agreement_status" in student_patterns_df.columns:
        agreement_counts = student_patterns_df["agreement_status"].value_counts()

        same_count = int(agreement_counts.get("Паттерн совпадает", 0))
        diff_count = int(agreement_counts.get("Паттерн различается", 0))

        text_parts.append(
            f"По выбранным методам у {same_count} студентов паттерн совпадает, "
            f"а у {diff_count} студентов хотя бы один метод дал другой паттерн. "
            f"Студенты с различающимися паттернами требуют дополнительного анализа, "
            f"так как их поведение может находиться на границе нескольких групп."
        )

    if "HDBSCAN" in summary_df["method"].values:
        hdbscan_row = summary_df[summary_df["method"] == "HDBSCAN"].iloc[0]

        if hdbscan_row["noise_count"] and hdbscan_row["noise_count"] > 0:
            text_parts.append(
                f"HDBSCAN выделил {int(hdbscan_row['noise_count'])} шумовых объектов. "
                f"Их можно рассматривать как нетипичные или смешанные цифровые следы."
            )

    if "GMM" in method_results:
        gmm_df = method_results["GMM"].get("result_df")

        if gmm_df is not None and "cluster_probability" in gmm_df.columns:
            low_confidence_count = int((gmm_df["cluster_probability"] < 0.6).sum())

            text_parts.append(
                f"GMM выявил {low_confidence_count} студентов со сниженной уверенностью "
                f"принадлежности к кластеру при пороге 0.6. Такие студенты могут иметь "
                f"смешанный поведенческий профиль."
            )

    text_parts.append(
        "Для итоговой интерпретации в ВКР рекомендуется учитывать не только численные "
        "метрики, но и содержательную понятность паттернов. KMeans удобно оставить "
        "основным методом, а остальные методы использовать для проверки устойчивости "
        "и выявления спорных студентов."
    )

    return "\n\n".join(text_parts)


# ------------------------------------------------------------
# Итоговая таблица метрик
# ------------------------------------------------------------
st.subheader("Сравнение метрик качества")

summary_df = pd.DataFrame(
    [
        build_method_summary(method_name, result)
        for method_name, result in method_results.items()
    ]
)

st.dataframe(summary_df, use_container_width=True)

st.caption(
    "Silhouette и Calinski-Harabasz: больше — лучше. "
    "Davies-Bouldin: меньше — лучше. "
    "AIC/BIC здесь не используются, чтобы сравнение методов было единым."
)


# ------------------------------------------------------------
# Отдельные диаграммы для каждой метрики
# ------------------------------------------------------------
st.subheader("Диаграммы метрик")

metric_configs = [
    ("silhouette_score", "Silhouette Score", "Больше — лучше"),
    ("calinski_harabasz_score", "Calinski-Harabasz Score", "Больше — лучше"),
    ("davies_bouldin_score", "Davies-Bouldin Score", "Меньше — лучше"),
]

for metric_col, metric_title, metric_note in metric_configs:
    metric_df = summary_df.dropna(subset=[metric_col]).copy()

    if metric_df.empty:
        st.info(f"Нет данных для метрики {metric_title}.")
        continue

    fig = px.bar(
        metric_df,
        x="method",
        y=metric_col,
        text=metric_col,
        title=f"{metric_title} ({metric_note})",
    )

    fig.update_traces(
        texttemplate="%{text:.4f}",
        textposition="outside",
    )

    fig.update_layout(
        xaxis_title="Метод",
        yaxis_title=metric_title,
    )

    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------
# Сравнение одинаковых названий паттернов
# ------------------------------------------------------------
st.subheader("Сравнение паттернов с одинаковыми названиями")

cluster_name_frames = []

for method_name, result in method_results.items():
    names_df = get_cluster_names(method_name, result)

    if names_df.empty:
        continue

    names_df = names_df.copy()
    names_df["method"] = method_name

    cluster_name_frames.append(names_df)

if cluster_name_frames:
    all_cluster_names_df = pd.concat(
        cluster_name_frames,
        ignore_index=True,
    )

    st.dataframe(
        all_cluster_names_df[
            [
                "method",
                "cluster",
                "cluster_size",
                "suggested_name",
                "description",
            ]
        ],
        use_container_width=True,
    )

    same_pattern_df = (
        all_cluster_names_df.groupby(["suggested_name", "method"])["cluster_size"]
        .sum()
        .reset_index()
        .rename(columns={"cluster_size": "students_count"})
    )

    fig_patterns = px.bar(
        same_pattern_df,
        x="suggested_name",
        y="students_count",
        color="method",
        barmode="group",
        title="Сравнение одинаковых паттернов между методами",
    )

    fig_patterns.update_layout(
        xaxis_title="Название паттерна",
        yaxis_title="Число студентов",
    )

    st.plotly_chart(fig_patterns, use_container_width=True)

    st.caption(
        "Здесь сравниваются не номера кластеров, а именно одинаковые названия "
        "паттернов. Это корректнее, потому что cluster 0 у разных методов может "
        "означать разные группы."
    )
else:
    st.info("Нет данных для сравнения паттернов.")


# ------------------------------------------------------------
# Таблица и график согласованности студентов
# ------------------------------------------------------------
st.subheader("Согласованность паттернов по студентам")

student_patterns_df = build_student_patterns_table(method_results)

if student_patterns_df.empty:
    st.info("Нет данных для сравнения студентов по методам.")
else:
    agreement_counts = (
        student_patterns_df["agreement_status"]
        .value_counts()
        .reset_index()
    )
    agreement_counts.columns = ["agreement_status", "students_count"]

    c1, c2 = st.columns(2)

    c1.metric(
        "Паттерн совпадает",
        int(
            agreement_counts.loc[
                agreement_counts["agreement_status"] == "Паттерн совпадает",
                "students_count",
            ].sum()
        ),
    )

    c2.metric(
        "Паттерн различается",
        int(
            agreement_counts.loc[
                agreement_counts["agreement_status"] == "Паттерн различается",
                "students_count",
            ].sum()
        ),
    )

    fig_agreement_counts = px.bar(
        agreement_counts,
        x="agreement_status",
        y="students_count",
        color="agreement_status",
        title="Сколько студентов получили одинаковый или различающийся паттерн",
    )

    fig_agreement_counts.update_layout(
        xaxis_title="Согласованность паттерна",
        yaxis_title="Число студентов",
        showlegend=False,
    )

    st.plotly_chart(fig_agreement_counts, use_container_width=True)

    student_plot_df = student_patterns_df.copy()
    student_plot_df["student_order"] = range(1, len(student_plot_df) + 1)

    fig_students = px.scatter(
        student_plot_df,
        x="student_order",
        y="unique_patterns_count",
        color="agreement_status",
        hover_data=[
            col for col in student_plot_df.columns
            if col == "student_id" or col.endswith("_pattern")
        ],
        title="Согласованность паттернов по каждому студенту",
    )

    fig_students.update_layout(
        xaxis_title="Порядковый номер студента",
        yaxis_title="Число уникальных паттернов среди выбранных методов",
    )

    st.plotly_chart(fig_students, use_container_width=True)

    st.caption(
        "Если у студента один уникальный паттерн среди выбранных методов, значит методы "
        "согласованы. Если уникальных паттернов больше одного, хотя бы один метод "
        "отнёс студента к другому типу поведения."
    )

    with st.expander("Показать таблицу согласованности студентов"):
        styled_student_patterns_df = student_patterns_df.style.apply(
            highlight_agreement_rows,
            axis=1,
        )

        st.write(styled_student_patterns_df)


# ------------------------------------------------------------
# Sankey-сравнение двух методов
# ------------------------------------------------------------
st.subheader("Sankey-сравнение двух методов")

if student_patterns_df.empty:
    st.info("Для Sankey-диаграммы нужны результаты хотя бы двух методов.")
elif len(selected_methods) < 2:
    st.info("Выберите хотя бы два метода для построения Sankey-диаграммы.")
else:
    col_s1, col_s2 = st.columns(2)

    with col_s1:
        sankey_method_a = st.selectbox(
            "Метод слева",
            options=selected_methods,
            index=0,
            key="sankey_method_a",
        )

    with col_s2:
        available_method_b = [
            method for method in selected_methods
            if method != sankey_method_a
        ]

        sankey_method_b = st.selectbox(
            "Метод справа",
            options=available_method_b,
            index=0,
            key="sankey_method_b",
        )

    sankey_fig = build_sankey_between_methods(
        student_patterns_df=student_patterns_df,
        method_a=sankey_method_a,
        method_b=sankey_method_b,
    )

    if sankey_fig is None:
        st.info("Недостаточно данных для Sankey-диаграммы.")
    else:
        st.plotly_chart(sankey_fig, use_container_width=True)

    st.caption(
        "Sankey-диаграмма показывает, как студенты переходят между паттернами "
        "при сравнении двух методов. Чем толще поток, тем больше студентов имеют "
        "соответствующее сочетание паттернов."
    )


# ------------------------------------------------------------
# Анализ одного студента
# ------------------------------------------------------------
st.subheader("Анализ выбранного студента по всем выбранным методам")

if student_patterns_df.empty:
    st.info("Нет данных для анализа отдельного студента.")
else:
    student_ids = sorted(
        student_patterns_df["student_id"].dropna().astype(str).tolist()
    )

    selected_student_id = st.selectbox(
        "Выберите студента",
        student_ids,
        key="comparison_selected_student_id",
    )

    selected_student_row = student_patterns_df[
        student_patterns_df["student_id"].astype(str) == str(selected_student_id)
    ].copy()

    st.dataframe(selected_student_row, use_container_width=True)

    st.write("### Кластеры и паттерны выбранного студента")

    student_method_rows = []

    for method_name in selected_methods:
        cluster_col = f"{method_name}_cluster"
        pattern_col = f"{method_name}_pattern"
        probability_col = f"{method_name}_probability"

        if cluster_col not in selected_student_row.columns:
            continue

        cluster_value = selected_student_row[cluster_col].iloc[0]
        pattern_value = (
            selected_student_row[pattern_col].iloc[0]
            if pattern_col in selected_student_row.columns
            else "Нет названия"
        )

        probability_value = None

        if probability_col in selected_student_row.columns:
            probability_value = selected_student_row[probability_col].iloc[0]

        student_method_rows.append(
            {
                "method": method_name,
                "cluster": cluster_value,
                "pattern": pattern_value,
                "probability": probability_value,
            }
        )

    student_methods_df = pd.DataFrame(student_method_rows)

    st.dataframe(student_methods_df, use_container_width=True)

    unique_patterns = (
        student_methods_df["pattern"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if len(unique_patterns) <= 1:
        st.success(
            f"У студента {selected_student_id} паттерн совпадает во всех выбранных "
            f"методах: {unique_patterns[0] if unique_patterns else 'нет данных'}."
        )
    else:
        st.warning(
            f"У студента {selected_student_id} разные методы выделили разные паттерны: "
            f"{', '.join(unique_patterns)}. Это может означать пограничное или "
            f"смешанное поведение."
        )

