import pandas as pd
from src.config import REQUIRED_INTERNAL_COLUMNS


def validate_columns(df: pd.DataFrame) -> tuple[bool, list[str]]:
    missing = [col for col in REQUIRED_INTERNAL_COLUMNS if col not in df.columns]
    return len(missing) == 0, missing