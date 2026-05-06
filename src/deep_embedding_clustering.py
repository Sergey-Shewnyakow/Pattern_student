import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)


class ClusteringLayer(tf.keras.layers.Layer):
    """
    DEC clustering layer.
    Возвращает soft assignment q_ij по распределению Student's t.
    """

    def __init__(self, n_clusters: int, alpha: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.clusters = None

    def build(self, input_shape):
        latent_dim = int(input_shape[-1])
        self.clusters = self.add_weight(
            shape=(self.n_clusters, latent_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="clusters",
        )
        super().build(input_shape)

    def call(self, inputs):
        # Student t-distribution, как в DEC
        expanded_inputs = tf.expand_dims(inputs, axis=1)              # (batch, 1, d)
        expanded_centers = tf.expand_dims(self.clusters, axis=0)      # (1, k, d)

        distances = tf.reduce_sum(tf.square(expanded_inputs - expanded_centers), axis=2)
        q = 1.0 / (1.0 + distances / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = q / tf.reduce_sum(q, axis=1, keepdims=True)
        return q


def target_distribution(q: np.ndarray) -> np.ndarray:
    """
    Target distribution p из статьи DEC.
    """
    weight = (q ** 2) / np.sum(q, axis=0, keepdims=True)
    return (weight.T / np.sum(weight, axis=1)).T


def build_autoencoder(input_dim: int, latent_dim: int = 10):
    """
    Небольшой автоэнкодер для табличных признаков.
    """
    inputs = tf.keras.Input(shape=(input_dim,), name="input")

    x = tf.keras.layers.Dense(128, activation="relu")(inputs)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    latent = tf.keras.layers.Dense(latent_dim, activation="linear", name="latent")(x)

    x = tf.keras.layers.Dense(64, activation="relu")(latent)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(input_dim, activation="linear", name="reconstruction")(x)

    autoencoder = tf.keras.Model(inputs=inputs, outputs=outputs, name="autoencoder")
    encoder = tf.keras.Model(inputs=inputs, outputs=latent, name="encoder")

    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder, encoder


def build_dec_model(encoder: tf.keras.Model, n_clusters: int):
    """
    DEC model = encoder + clustering layer
    """
    inputs = encoder.input
    latent = encoder.output
    clustering_output = ClusteringLayer(n_clusters, name="clustering")(latent)

    dec_model = tf.keras.Model(inputs=inputs, outputs=clustering_output, name="DEC")
    dec_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="kld",
    )
    return dec_model


def prepare_features_for_dec(features_df: pd.DataFrame):
    numeric_df = features_df.select_dtypes(include="number").copy()

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(numeric_df)

    return numeric_df, x_scaled, scaler


def _safe_cluster_metrics(x_scaled: np.ndarray, labels: np.ndarray):
    unique_labels = pd.Series(labels).nunique()

    if unique_labels < 2:
        return {
            "silhouette_score": None,
            "calinski_harabasz_score": None,
            "davies_bouldin_score": None,
        }

    return {
        "silhouette_score": silhouette_score(x_scaled, labels),
        "calinski_harabasz_score": calinski_harabasz_score(x_scaled, labels),
        "davies_bouldin_score": davies_bouldin_score(x_scaled, labels),
    }


def run_deep_embedding_clustering(
    features_df: pd.DataFrame,
    n_clusters: int = 4,
    latent_dim: int = 10,
    pretrain_epochs: int = 50,
    dec_max_iter: int = 2000,
    update_interval: int = 50,
    batch_size: int = 32,
    tol: float = 1e-3,
    random_state: int = 42,
):
    """
    DEC для обычных признаков.
    1) Масштабирование признаков
    2) Предобучение автоэнкодера
    3) KMeans в latent space
    4) DEC refinement
    """
    tf.keras.utils.set_random_seed(random_state)

    numeric_df, x_scaled, scaler = prepare_features_for_dec(features_df)
    input_dim = x_scaled.shape[1]

    autoencoder, encoder = build_autoencoder(input_dim=input_dim, latent_dim=latent_dim)

    # Предобучение автоэнкодера
    history = autoencoder.fit(
        x_scaled,
        x_scaled,
        epochs=pretrain_epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=0,
    )

    z = encoder.predict(x_scaled, verbose=0)

    # Инициализация центров через KMeans
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
    y_pred = kmeans.fit_predict(z)
    cluster_centers = kmeans.cluster_centers_

    # DEC model
    dec_model = build_dec_model(encoder, n_clusters=n_clusters)
    dec_model.get_layer(name="clustering").set_weights([cluster_centers])

    index_array = np.arange(x_scaled.shape[0])
    y_pred_last = np.copy(y_pred)

    for ite in range(dec_max_iter):
        if ite % update_interval == 0:
            q = dec_model.predict(x_scaled, verbose=0)
            p = target_distribution(q)

            y_pred = q.argmax(axis=1)
            delta_label = np.mean(y_pred != y_pred_last)
            y_pred_last = np.copy(y_pred)

            if ite > 0 and delta_label < tol:
                break

        # train on one batch
        batch_idx = index_array[(ite * batch_size) % x_scaled.shape[0]: ((ite + 1) * batch_size) % x_scaled.shape[0]]

        if len(batch_idx) == 0:
            batch_idx = index_array[:batch_size]

        x_batch = x_scaled[batch_idx]
        p_batch = p[batch_idx]

        dec_model.train_on_batch(x_batch, p_batch)

    # Финальные предсказания
    q_final = dec_model.predict(x_scaled, verbose=0)
    labels = q_final.argmax(axis=1)
    confidence = q_final.max(axis=1)

    result_df = features_df.copy()
    result_df["cluster"] = labels
    result_df["dec_confidence"] = confidence

    cluster_profiles = (
        result_df.groupby("cluster")
        .mean(numeric_only=True)
        .reset_index()
    )

    metrics = _safe_cluster_metrics(x_scaled, labels)
    metrics["n_clusters"] = int(n_clusters)
    metrics["cluster_count"] = int(pd.Series(labels).nunique())
    metrics["pretrain_loss_final"] = float(history.history["loss"][-1])

    return {
        "result_df": result_df,
        "metrics": metrics,
        "cluster_profiles": cluster_profiles,
        "autoencoder": autoencoder,
        "encoder": encoder,
        "dec_model": dec_model,
        "scaler": scaler,
        "latent_features": z,
    }


def evaluate_dec_range(
    features_df: pd.DataFrame,
    k_min: int = 2,
    k_max: int = 6,
    latent_dim: int = 10,
    pretrain_epochs: int = 30,
    dec_max_iter: int = 800,
    batch_size: int = 32,
    random_state: int = 42,
):
    rows = []

    for k in range(k_min, k_max + 1):
        result = run_deep_embedding_clustering(
            features_df=features_df,
            n_clusters=k,
            latent_dim=latent_dim,
            pretrain_epochs=pretrain_epochs,
            dec_max_iter=dec_max_iter,
            batch_size=batch_size,
            random_state=random_state,
        )

        rows.append({
            "k": k,
            "silhouette_score": result["metrics"]["silhouette_score"],
            "calinski_harabasz_score": result["metrics"]["calinski_harabasz_score"],
            "davies_bouldin_score": result["metrics"]["davies_bouldin_score"],
            "pretrain_loss_final": result["metrics"]["pretrain_loss_final"],
        })

    return pd.DataFrame(rows)