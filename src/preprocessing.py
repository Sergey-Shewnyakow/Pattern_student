import pandas as pd


def preprocess_log(df: pd.DataFrame) -> pd.DataFrame:
    """
    Базовая предобработка логов:
    - перевод timestamp в datetime
    - удаление пустых строк по ключевым колонкам
    - сортировка по студенту и времени
    """
    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        dayfirst=True
    )

    df = df.dropna(subset=["student_id", "timestamp", "activity"])
    df["student_id"] = df["student_id"].astype(str)
    df["activity"] = df["activity"].astype(str).str.strip()

    if "context" in df.columns:
        df["context"] = df["context"].astype(str).str.strip()

    if "component" in df.columns:
        df["component"] = df["component"].astype(str).str.strip()

    df = df.sort_values(["student_id", "timestamp"]).reset_index(drop=True)

    return df