import pandas as pd
from src.config import COLUMN_ALIASES


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приводит названия колонок из исходного лога
    к единому внутреннему формату.
    """
    df = df.copy()

    rename_map = {}

    for target_col, possible_names in COLUMN_ALIASES.items():
        for source_col in possible_names:
            if source_col in df.columns:
                rename_map[source_col] = target_col
                break

    df = df.rename(columns=rename_map)
    return df