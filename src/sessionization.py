import pandas as pd


def add_sessions(df: pd.DataFrame, gap_minutes: int = 120) -> pd.DataFrame:
    """
    Добавляет session_id на основе разрыва во времени между событиями.
    Если между соседними событиями одного студента прошло больше gap_minutes,
    считается, что началась новая сессия.
    """
    df = df.copy()
    df = df.sort_values(["student_id", "timestamp"]).reset_index(drop=True)

    df["prev_timestamp"] = df.groupby("student_id")["timestamp"].shift(1)
    df["gap_minutes"] = (
        (df["timestamp"] - df["prev_timestamp"]).dt.total_seconds() / 60
    )

    df["new_session"] = (
        df["gap_minutes"].isna() | (df["gap_minutes"] > gap_minutes)
    ).astype(int)

    df["session_number"] = df.groupby("student_id")["new_session"].cumsum()
    df["session_id"] = (
        df["student_id"].astype(str) + "_s" + df["session_number"].astype(str)
    )

    return df