import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def _format_value(value):
    if pd.isna(value):
        return "NaN"
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        if abs(value) >= 100:
            return f"{value:.1f}"
        if abs(value) >= 1:
            return f"{value:.2f}"
        return f"{value:.4f}"
    return str(value)


def detect_rule_based_anomalies(
    features_df: pd.DataFrame,
    selected_columns: list[str] | None = None,
    quantile_threshold: float = 0.99
) -> tuple[pd.DataFrame, dict]:
    """
    Помечает аномальных пользователей по простому правилу:
    если хотя бы один выбранный признак выше заданного перцентиля,
    пользователь считается подозрительным.

    Возвращает:
    - DataFrame с колонками rule_based_anomaly и triggered_rules
    - словарь с порогами
    """
    df = features_df.copy()

    if selected_columns is None:
        selected_columns = [
            col for col in [
                "total_events",
                "active_days",
                "active_weeks",
                "sessions_count",
                "unique_activities",
                "unique_contexts",
                "unique_components",
                "avg_events_per_session",
                "max_events_per_session",
            ]
            if col in df.columns
        ]

    thresholds = {
        col: df[col].quantile(quantile_threshold)
        for col in selected_columns
    }

    anomaly_flags = []
    triggered_rules = []

    for _, row in df.iterrows():
        triggered = []
        for col in selected_columns:
            if row[col] > thresholds[col]:
                triggered.append(f"{col} ({_format_value(row[col])})")

        anomaly_flags.append(len(triggered) > 0)
        triggered_rules.append(", ".join(triggered) if triggered else "")

    df["rule_based_anomaly"] = anomaly_flags
    df["triggered_rules"] = triggered_rules

    return df, thresholds


def detect_isolation_forest_anomalies(
    features_df: pd.DataFrame,
    contamination: float = 0.05,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Помечает аномальных пользователей через IsolationForest.
    """
    df = features_df.copy()

    numeric_df = df.select_dtypes(include="number").copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df)

    model = IsolationForest(
        contamination=contamination,
        random_state=random_state
    )
    labels = model.fit_predict(X_scaled)
    scores = model.decision_function(X_scaled)

    df["iforest_label"] = labels
    df["iforest_anomaly"] = df["iforest_label"] == -1
    df["iforest_score"] = scores

    return df


def combine_anomaly_flags(
    features_df: pd.DataFrame,
    use_rule_based: bool = True,
    use_iforest: bool = True
) -> pd.DataFrame:
    """
    Объединяет пометки аномалий в одну колонку final_anomaly.
    """
    df = features_df.copy()

    final_flag = pd.Series(False, index=df.index)

    if use_rule_based and "rule_based_anomaly" in df.columns:
        final_flag = final_flag | df["rule_based_anomaly"]

    if use_iforest and "iforest_anomaly" in df.columns:
        final_flag = final_flag | df["iforest_anomaly"]

    df["final_anomaly"] = final_flag
    return df