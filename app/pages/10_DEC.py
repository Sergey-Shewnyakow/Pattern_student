import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

from src.cluster_features import prepare_cluster_matrix


@dataclass
class DECTrainingHistory:
    pretrain_losses: list[float]
    clustering_losses: list[float]


def _set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _calculate_metrics(x_embedding: np.ndarray, labels: np.ndarray):
    unique_labels = set(labels)

    if len(unique_labels) <= 1:
        return {
            "silhouette_score": None,
            "calinski_harabasz_score": None,
            "davies_bouldin_score": None,
        }

    return {
        "silhouette_score": silhouette_score(x_embedding, labels),
        "calinski_harabasz_score": calinski_harabasz_score(x_embedding, labels),
        "davies_bouldin_score": davies_bouldin_score(x_embedding, labels),
    }


def _target_distribution(q):
    """
    Target distribution из DEC.

    q — мягкие вероятности принадлежности к кластерам.
    """
    weight = (q**2) / q.sum(axis=0)
    return (weight.T / weight.sum(axis=1)).T


def run_deep_embedding_clustering(
    features_df: pd.DataFrame,
    n_clusters: int = 4,
    embedding_dim: int = 2,
    hidden_dim: int = 64,
    pretrain_epochs: int = 100,
    clustering_epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    random_state: int = 42,
):
    """
    Deep Embedding Clustering.

    Этапы:
    1. Берём единый набор признаков через cluster_features.py.
    2. Стандартизируем признаки.
    3. Обучаем автоэнкодер восстанавливать исходные признаки.
    4. Получаем embedding-представление студентов.
    5. Запускаем KMeans в embedding-пространстве.
    6. Выполняем несколько эпох DEC-уточнения.
    7. Интерпретируем кластеры по исходным признакам, а не по embedding.

    Важно:
    DEC используется как дополнительный экспериментальный метод.
    """
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise ImportError(
            "Для Deep Embedding Clustering нужен PyTorch. "
            "Установите его командой: pip install torch"
        ) from exc

    _set_seed(random_state)

    numeric_df, x_scaled, scaler = prepare_cluster_matrix(features_df)

    x_scaled = np.asarray(x_scaled, dtype=np.float32)

    input_dim = x_scaled.shape[1]

    if embedding_dim >= input_dim:
        embedding_dim = max(2, input_dim // 2)

    class Autoencoder(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int):
            super().__init__()

            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embedding_dim),
            )

            self.decoder = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
            )

        def forward(self, x):
            z = self.encoder(x)
            x_reconstructed = self.decoder(z)
            return x_reconstructed, z

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Autoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    dataset = TensorDataset(x_tensor)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    pretrain_losses = []

    # ------------------------------------------------------------
    # 1. Предобучение автоэнкодера
    # ------------------------------------------------------------
    model.train()

    for _ in range(pretrain_epochs):
        epoch_losses = []

        for (batch_x,) in dataloader:
            batch_x = batch_x.to(device)

            optimizer.zero_grad()

            x_reconstructed, _ = model(batch_x)

            loss = F.mse_loss(x_reconstructed, batch_x)

            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        pretrain_losses.append(float(np.mean(epoch_losses)))

    # ------------------------------------------------------------
    # 2. Получение embedding после предобучения
    # ------------------------------------------------------------
    model.eval()

    with torch.no_grad():
        _, z_tensor = model(x_tensor.to(device))

    z = z_tensor.cpu().numpy()

    # ------------------------------------------------------------
    # 3. Инициализация центров кластеров через KMeans
    # ------------------------------------------------------------
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )

    labels = kmeans.fit_predict(z)
    cluster_centers = kmeans.cluster_centers_

    cluster_centers_tensor = torch.tensor(
        cluster_centers,
        dtype=torch.float32,
        device=device,
    )

    cluster_centers_parameter = torch.nn.Parameter(cluster_centers_tensor)

    dec_optimizer = torch.optim.Adam(
        list(model.encoder.parameters()) + [cluster_centers_parameter],
        lr=learning_rate,
    )

    clustering_losses = []

    # ------------------------------------------------------------
    # 4. DEC-уточнение embedding-пространства
    # ------------------------------------------------------------
    alpha = 1.0

    for _ in range(clustering_epochs):
        model.train()

        x_all = x_tensor.to(device)

        z_all = model.encoder(x_all)

        # Student t-distribution
        distances = torch.sum(
            (z_all.unsqueeze(1) - cluster_centers_parameter.unsqueeze(0)) ** 2,
            dim=2,
        )

        q = 1.0 / (1.0 + distances / alpha)
        q = q ** ((alpha + 1.0) / 2.0)
        q = q / torch.sum(q, dim=1, keepdim=True)

        q_np = q.detach().cpu().numpy()
        p_np = _target_distribution(q_np)

        p = torch.tensor(
            p_np,
            dtype=torch.float32,
            device=device,
        )

        dec_optimizer.zero_grad()

        loss = F.kl_div(
            torch.log(q + 1e-10),
            p,
            reduction="batchmean",
        )

        loss.backward()
        dec_optimizer.step()

        clustering_losses.append(float(loss.item()))

    # ------------------------------------------------------------
    # 5. Финальные embedding и кластеры
    # ------------------------------------------------------------
    model.eval()

    with torch.no_grad():
        final_z_tensor = model.encoder(x_tensor.to(device))

        distances = torch.sum(
            (
                final_z_tensor.unsqueeze(1)
                - cluster_centers_parameter.unsqueeze(0)
            )
            ** 2,
            dim=2,
        )

        q = 1.0 / (1.0 + distances / alpha)
        q = q ** ((alpha + 1.0) / 2.0)
        q = q / torch.sum(q, dim=1, keepdim=True)

    final_embedding = final_z_tensor.cpu().numpy()
    probabilities = q.cpu().numpy()
    labels = probabilities.argmax(axis=1)

    result_df = features_df.copy()
    result_df["cluster"] = labels
    result_df["cluster_probability"] = probabilities.max(axis=1)

    for dim_idx in range(final_embedding.shape[1]):
        result_df[f"embedding_{dim_idx + 1}"] = final_embedding[:, dim_idx]

    for cluster_idx in range(n_clusters):
        result_df[f"dec_probability_cluster_{cluster_idx}"] = probabilities[
            :,
            cluster_idx,
        ]

    metrics = _calculate_metrics(
        x_embedding=final_embedding,
        labels=labels,
    )

    cluster_profiles = (
        result_df.groupby("cluster")
        .mean(numeric_only=True)
        .reset_index()
    )

    history = DECTrainingHistory(
        pretrain_losses=pretrain_losses,
        clustering_losses=clustering_losses,
    )

    return {
        "result_df": result_df,
        "metrics": metrics,
        "cluster_profiles": cluster_profiles,
        "model": model,
        "scaler": scaler,
        "used_features": list(numeric_df.columns),
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "n_clusters": n_clusters,
        "history": history,
    }


def evaluate_dec_range(
    features_df: pd.DataFrame,
    k_min: int = 2,
    k_max: int = 8,
    embedding_dim: int = 2,
    hidden_dim: int = 64,
    pretrain_epochs: int = 50,
    clustering_epochs: int = 20,
    random_state: int = 42,
):
    """
    Быстрая оценка DEC для разных k.

    Для ускорения используется меньше эпох, чем при финальном запуске.
    """
    rows = []

    for k in range(k_min, k_max + 1):
        if k >= len(features_df):
            continue

        result = run_deep_embedding_clustering(
            features_df=features_df,
            n_clusters=k,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            pretrain_epochs=pretrain_epochs,
            clustering_epochs=clustering_epochs,
            random_state=random_state,
        )

        metrics = result["metrics"]

        rows.append(
            {
                "k": k,
                "embedding_dim": embedding_dim,
                "hidden_dim": hidden_dim,
                "silhouette_score": metrics["silhouette_score"],
                "calinski_harabasz_score": metrics["calinski_harabasz_score"],
                "davies_bouldin_score": metrics["davies_bouldin_score"],
                "used_features": ", ".join(result["used_features"]),
            }
        )

    return pd.DataFrame(rows)