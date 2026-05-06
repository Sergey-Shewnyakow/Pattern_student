import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

from keras import Model, Input
from keras import layers
from keras.optimizers import Adam


def prepare_features_for_autoencoder(features_df: pd.DataFrame):
    """
    Подготовка данных:
    - только числовые признаки
    - масштабирование
    """
    numeric_df = features_df.select_dtypes(include="number").copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df)

    return numeric_df, X_scaled, scaler


def build_autoencoder(input_dim: int, latent_dim: int = 3):
    """
    Простой dense-autoencoder.
    """
    encoder_input = Input(shape=(input_dim,), name="encoder_input")
    x = layers.Dense(32, activation="relu")(encoder_input)
    x = layers.Dense(16, activation="relu")(x)
    latent = layers.Dense(latent_dim, activation="linear", name="latent")(x)

    x = layers.Dense(16, activation="relu")(latent)
    x = layers.Dense(32, activation="relu")(x)
    decoder_output = layers.Dense(input_dim, activation="linear", name="decoder_output")(x)

    autoencoder = Model(encoder_input, decoder_output, name="autoencoder")
    encoder = Model(encoder_input, latent, name="encoder")

    autoencoder.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mse"
    )

    return autoencoder, encoder


def run_autoencoder_kmeans(
    features_df: pd.DataFrame,
    latent_dim: int = 3,
    n_clusters: int = 4,
    epochs: int = 100,
    batch_size: int = 16,
    random_state: int = 42,
    validation_split: float = 0.1,
):
    """
    Обучение autoencoder и кластеризация KMeans на латентных признаках.
    """
    numeric_df, X_scaled, scaler = prepare_features_for_autoencoder(features_df)

    input_dim = X_scaled.shape[1]
    autoencoder, encoder = build_autoencoder(input_dim=input_dim, latent_dim=latent_dim)

    history = autoencoder.fit(
        X_scaled,
        X_scaled,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=0,
        shuffle=True,
    )

    latent_features = encoder.predict(X_scaled, verbose=0)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto"
    )
    labels = kmeans.fit_predict(latent_features)

    result_df = features_df.copy()
    result_df["cluster"] = labels

    latent_df = pd.DataFrame(
        latent_features,
        columns=[f"latent_{i+1}" for i in range(latent_features.shape[1])]
    )
    latent_df.insert(0, "student_id", features_df["student_id"].values)
    latent_df["cluster"] = labels

    metrics = {
        "silhouette_score": silhouette_score(latent_features, labels),
        "calinski_harabasz_score": calinski_harabasz_score(latent_features, labels),
        "davies_bouldin_score": davies_bouldin_score(latent_features, labels),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]) if "val_loss" in history.history else None,
    }

    cluster_profiles = (
        result_df.groupby("cluster")
        .mean(numeric_only=True)
        .reset_index()
    )

    return {
        "result_df": result_df,
        "latent_df": latent_df,
        "metrics": metrics,
        "cluster_profiles": cluster_profiles,
        "autoencoder": autoencoder,
        "encoder": encoder,
        "scaler": scaler,
        "history": history.history,
    }


def evaluate_autoencoder_kmeans_range(
    features_df: pd.DataFrame,
    latent_dim: int = 3,
    k_min: int = 2,
    k_max: int = 6,
    epochs: int = 100,
    batch_size: int = 16,
    validation_split: float = 0.1,
    random_state: int = 42,
):
    """
    Один раз обучает autoencoder, потом оценивает разные k в латентном пространстве.
    """
    numeric_df, X_scaled, scaler = prepare_features_for_autoencoder(features_df)

    input_dim = X_scaled.shape[1]
    autoencoder, encoder = build_autoencoder(input_dim=input_dim, latent_dim=latent_dim)

    history = autoencoder.fit(
        X_scaled,
        X_scaled,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=0,
        shuffle=True,
    )

    latent_features = encoder.predict(X_scaled, verbose=0)

    rows = []
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init="auto"
        )
        labels = kmeans.fit_predict(latent_features)

        rows.append({
            "k": k,
            "silhouette_score": silhouette_score(latent_features, labels),
            "calinski_harabasz_score": calinski_harabasz_score(latent_features, labels),
            "davies_bouldin_score": davies_bouldin_score(latent_features, labels),
            "latent_dim": latent_dim,
            "final_train_loss": float(history.history["loss"][-1]),
            "final_val_loss": float(history.history["val_loss"][-1]) if "val_loss" in history.history else None,
        })

    return pd.DataFrame(rows)