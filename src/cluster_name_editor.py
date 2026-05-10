import pandas as pd
import streamlit as st


def _get_storage_key(method_key: str) -> str:
    return f"{method_key}_custom_cluster_names"


def apply_custom_cluster_names(
    method_key: str,
    cluster_names_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Применяет сохранённые вручную названия кластеров.

    Если пользователь уже редактировал названия для метода,
    они подставляются вместо автоматических.
    """
    if cluster_names_df is None or cluster_names_df.empty:
        return cluster_names_df

    storage_key = _get_storage_key(method_key)

    custom_names = st.session_state.get(storage_key)

    if custom_names is None:
        return cluster_names_df

    result_df = cluster_names_df.copy()

    for cluster_id, values in custom_names.items():
        mask = result_df["cluster"].astype(str) == str(cluster_id)

        if not mask.any():
            continue

        if "suggested_name" in values:
            result_df.loc[mask, "suggested_name"] = values["suggested_name"]

        if "description" in values:
            result_df.loc[mask, "description"] = values["description"]

    return result_df


def render_editable_cluster_names(
    method_key: str,
    cluster_names_df: pd.DataFrame,
    title: str = "Названия кластеров",
) -> pd.DataFrame:
    """
    Показывает редактируемую таблицу названий кластеров.

    Пользователь может изменить:
    - suggested_name;
    - description.

    Колонки cluster и cluster_size не редактируются.
    """
    if cluster_names_df is None or cluster_names_df.empty:
        st.info("Нет кластеров для редактирования названий.")
        return cluster_names_df

    st.subheader(title)

    st.caption(
        "Программа автоматически предлагает название кластера, "
        "но его можно изменить вручную. Изменённые названия будут использоваться "
        "в таблицах, анализе студентов и сравнении методов."
    )

    storage_key = _get_storage_key(method_key)

    editable_df = apply_custom_cluster_names(
        method_key=method_key,
        cluster_names_df=cluster_names_df,
    ).copy()

    required_columns = [
        "cluster",
        "cluster_size",
        "suggested_name",
        "description",
    ]

    available_columns = [
        col for col in required_columns if col in editable_df.columns
    ]

    editable_view = editable_df[available_columns].copy()

    edited_df = st.data_editor(
        editable_view,
        use_container_width=True,
        hide_index=True,
        disabled=[
            col for col in ["cluster", "cluster_size"]
            if col in editable_view.columns
        ],
        key=f"{method_key}_cluster_names_editor",
    )

    col_save, col_reset = st.columns(2)

    with col_save:
        save_clicked = st.button(
            "Сохранить названия кластеров",
            key=f"{method_key}_save_cluster_names_button",
        )

    with col_reset:
        reset_clicked = st.button(
            "Сбросить ручные названия",
            key=f"{method_key}_reset_cluster_names_button",
        )

    if save_clicked:
        custom_names = {}

        for _, row in edited_df.iterrows():
            cluster_id = str(row["cluster"])

            custom_names[cluster_id] = {
                "suggested_name": row.get("suggested_name", ""),
                "description": row.get("description", ""),
            }

        st.session_state[storage_key] = custom_names

        st.success("Названия кластеров сохранены.")

    if reset_clicked:
        if storage_key in st.session_state:
            del st.session_state[storage_key]

        st.success("Ручные названия сброшены. Используются автоматические названия.")

    final_df = apply_custom_cluster_names(
        method_key=method_key,
        cluster_names_df=cluster_names_df,
    )

    return final_df