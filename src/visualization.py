import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def plot_cluster_counts(result_df: pd.DataFrame):
    """
    Строит диаграмму количества студентов в каждом кластере.
    """
    counts_df = (
        result_df["cluster"]
        .value_counts()
        .sort_index()
        .reset_index()
    )
    counts_df.columns = ["cluster", "students_count"]

    fig = px.bar(
        counts_df,
        x="cluster",
        y="students_count",
        text="students_count",
        title="Количество студентов по кластерам"
    )
    fig.update_traces(textposition="outside")
    return fig


def plot_pca_clusters(features_with_clusters: pd.DataFrame):
    """
    Строит 2D-визуализацию кластеров через PCA.
    """
    numeric_df = features_with_clusters.select_dtypes(include="number").copy()

    if "cluster" not in numeric_df.columns:
        raise ValueError("В таблице нет колонки 'cluster'.")

    X = numeric_df.drop(columns=["cluster"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame({
        "PC1": components[:, 0],
        "PC2": components[:, 1],
        "cluster": features_with_clusters["cluster"].astype(str),
        "student_id": features_with_clusters["student_id"].astype(str),
    })

    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="cluster",
        hover_data=["student_id"],
        title="PCA-проекция кластеров"
    )

    return fig, pca_df


def plot_cluster_profile_bar(cluster_profiles: pd.DataFrame, feature_name: str):
    """
    Строит bar chart по одному признаку для всех кластеров.
    """
    if feature_name not in cluster_profiles.columns:
        raise ValueError(f"Признак '{feature_name}' отсутствует в cluster_profiles.")

    fig = px.bar(
        cluster_profiles,
        x="cluster",
        y=feature_name,
        text=feature_name,
        title=f"Среднее значение признака '{feature_name}' по кластерам"
    )
    fig.update_traces(textposition="outside")
    return fig