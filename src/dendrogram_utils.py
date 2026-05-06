import numpy as np
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram


def plot_agglomerative_dendrogram(model, truncate_mode=None, p=30):
    """
    Строит дендрограмму по обученной модели AgglomerativeClustering.
    """
    counts = np.zeros(model.children_.shape[0])

    n_samples = len(model.labels_)

    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(
        linkage_matrix,
        truncate_mode=truncate_mode,
        p=p,
        ax=ax
    )
    ax.set_title("Дендрограмма Agglomerative Clustering")
    ax.set_xlabel("Объекты / объединённые группы")
    ax.set_ylabel("Расстояние объединения")
    plt.xticks(rotation=90)

    return fig