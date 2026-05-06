import pandas as pd


def build_hybrid_features(
    log_features_df: pd.DataFrame,
    pm_features_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Объединяет обычные признаки и process mining-признаки по student_id.
    """
    hybrid_df = log_features_df.merge(
        pm_features_df,
        on="student_id",
        how="inner",
        suffixes=("_log", "_pm")
    )

    # Удаляем явно категориальные process-поля, которые мешают числовой кластеризации
    drop_cols = [
        "first_activity",
        "last_activity",
        "variant_id",
    ]

    existing_drop_cols = [col for col in drop_cols if col in hybrid_df.columns]
    hybrid_df = hybrid_df.drop(columns=existing_drop_cols)

    return hybrid_df