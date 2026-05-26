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
    fig.update_traces(textposition="outside", textfont_size = 25)
    return fig


def plot_pca_clusters(features_with_clusters: pd.DataFrame):
    """
    Строит 2D-визуализацию кластеров через PCA.
    Версия оформлена так, чтобы график было хорошо видно в отчёте.
    """
    numeric_df = features_with_clusters.select_dtypes(include="number").copy()

    if "cluster" not in numeric_df.columns:
        raise ValueError("В таблице нет колонки 'cluster'.")

    X = numeric_df.drop(columns=["cluster"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(X_scaled)

    explained_1 = pca.explained_variance_ratio_[0] * 100
    explained_2 = pca.explained_variance_ratio_[1] * 100

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
        title="PCA-визуализация кластеров студентов",
        template="plotly_white",
    )

    fig.update_traces(
        marker=dict(
            size=11,
            opacity=0.85,
            line=dict(
                width=0.7,
                color="black",
            ),
        )
    )

    fig.update_layout(
        width=1400,
        height=900,
        title=dict(
            text="PCA-визуализация кластеров студентов",
            font=dict(size=28, color="black"),
            x=0.5,
        ),
        xaxis_title=f"PC1 ({explained_1:.1f}% дисперсии)",
        yaxis_title=f"PC2 ({explained_2:.1f}% дисперсии)",
        font=dict(
            size=18,
            color="black",
        ),
        legend=dict(
            title="Кластер",
            font=dict(size=16, color="black"),
            title_font=dict(size=18, color="black"),
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="black",
            borderwidth=1,
        ),
        margin=dict(l=80, r=40, t=100, b=80),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor="gray",
        tickfont=dict(size=16, color="black"),
        title_font=dict(size=20, color="black"),
    )

    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor="gray",
        tickfont=dict(size=16, color="black"),
        title_font=dict(size=20, color="black"),
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


