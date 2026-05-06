from pathlib import Path
import tempfile
import uuid

import pandas as pd
import pm4py


def prepare_pm4py_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приводит DataFrame к формату, удобному для PM4Py.
    """
    pm_df = df.copy()

    pm_df = pm_df.rename(columns={
        "student_id": "case:concept:name",
        "activity": "concept:name",
        "timestamp": "time:timestamp",
    })

    pm_df["case:concept:name"] = pm_df["case:concept:name"].astype(str)
    pm_df["concept:name"] = pm_df["concept:name"].astype(str)
    pm_df["time:timestamp"] = pd.to_datetime(pm_df["time:timestamp"], errors="coerce")

    pm_df = pm_df.dropna(subset=["case:concept:name", "concept:name", "time:timestamp"])
    pm_df = pm_df.sort_values(["case:concept:name", "time:timestamp"]).reset_index(drop=True)

    return pm_df


def _unique_temp_svg_path(prefix: str) -> str:
    filename = f"{prefix}_{uuid.uuid4().hex}.svg"
    return str(Path(tempfile.gettempdir()) / filename)


def build_heuristics_miner_svg(df: pd.DataFrame) -> str:
    """
    Строит SVG для Heuristics Miner и возвращает путь к файлу.
    """
    pm_df = prepare_pm4py_dataframe(df)

    heu_net = pm4py.discovery.discover_heuristics_net(pm_df)
    output_path = _unique_temp_svg_path("heuristics_miner")
    pm4py.vis.save_vis_heuristics_net(heu_net, output_path)
    return output_path


def build_inductive_miner_svg(df: pd.DataFrame) -> str:
    """
    Строит SVG для Inductive Miner (BPMN) и возвращает путь к файлу.
    """
    pm_df = prepare_pm4py_dataframe(df)

    bpmn_graph = pm4py.discovery.discover_bpmn_inductive(pm_df)
    output_path = _unique_temp_svg_path("inductive_miner_bpmn")
    pm4py.vis.save_vis_bpmn(bpmn_graph, output_path)
    return output_path


def can_build_process_model(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Простая проверка, достаточно ли данных для process mining-модели.
    """
    if df is None or df.empty:
        return False, "Нет событий для построения модели."

    if "student_id" not in df.columns or "activity" not in df.columns or "timestamp" not in df.columns:
        return False, "В данных нет обязательных колонок student_id, activity, timestamp."

    if len(df) < 2:
        return False, "Слишком мало событий для построения модели."

    if df["activity"].nunique() < 1:
        return False, "Нет активностей для построения модели."

    return True, ""