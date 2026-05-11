import os
import tempfile

import pandas as pd


def prepare_pm4py_dataframe(
    event_log: pd.DataFrame,
    case_id_col: str = "student_id",
    activity_col: str = "process_activity",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Подготавливает DataFrame для PM4Py.

    PM4Py ожидает колонки:
    - case:concept:name
    - concept:name
    - time:timestamp
    """
    if event_log is None or event_log.empty:
        raise ValueError("Event log пустой.")

    required_columns = [case_id_col, activity_col, timestamp_col]

    missing_columns = [
        col for col in required_columns
        if col not in event_log.columns
    ]

    if missing_columns:
        raise ValueError(
            "Для PM4Py не хватает колонок: " + ", ".join(missing_columns)
        )

    pm_df = event_log[[case_id_col, activity_col, timestamp_col]].copy()

    pm_df = pm_df.rename(
        columns={
            case_id_col: "case:concept:name",
            activity_col: "concept:name",
            timestamp_col: "time:timestamp",
        }
    )

    pm_df["case:concept:name"] = pm_df["case:concept:name"].astype(str)
    pm_df["concept:name"] = pm_df["concept:name"].astype(str)
    pm_df["time:timestamp"] = pd.to_datetime(pm_df["time:timestamp"])

    pm_df = pm_df.sort_values(
        ["case:concept:name", "time:timestamp"]
    ).reset_index(drop=True)

    try:
        import pm4py

        pm_df = pm4py.format_dataframe(
            pm_df,
            case_id="case:concept:name",
            activity_key="concept:name",
            timestamp_key="time:timestamp",
        )
    except Exception:
        pass

    return pm_df


def _read_svg(svg_path: str) -> str:
    with open(svg_path, "r", encoding="utf-8") as file:
        return file.read()


def build_pm4py_heuristic_svg(
    event_log: pd.DataFrame,
    dependency_threshold: float = 0.5,
    and_threshold: float = 0.65,
    loop_two_threshold: float = 0.5,
    min_act_count: int = 1,
    min_dfg_occurrences: int = 1,
) -> str:
    """
    Строит эвристическую диаграмму процесса через PM4Py и возвращает SVG.

    Эвристический майнер показывает устойчивые зависимости между действиями.
    """
    try:
        import pm4py
    except ImportError as exc:
        raise ImportError(
            "Библиотека pm4py не установлена. Установите её командой: pip install pm4py"
        ) from exc

    pm_df = prepare_pm4py_dataframe(event_log)

    with tempfile.TemporaryDirectory() as tmp_dir:
        svg_path = os.path.join(tmp_dir, "heuristic_miner.svg")

        try:
            heuristics_net = pm4py.discover_heuristics_net(
                pm_df,
                dependency_threshold=dependency_threshold,
                and_threshold=and_threshold,
                loop_two_threshold=loop_two_threshold,
                min_act_count=min_act_count,
                min_dfg_occurrences=min_dfg_occurrences,
            )

            pm4py.save_vis_heuristics_net(
                heuristics_net,
                svg_path,
            )

        except Exception:
            from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
            from pm4py.visualization.heuristics_net import visualizer as hn_visualizer

            parameters = {
                heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: dependency_threshold,
                heuristics_miner.Variants.CLASSIC.value.Parameters.AND_MEASURE_THRESH: and_threshold,
                heuristics_miner.Variants.CLASSIC.value.Parameters.MIN_ACT_COUNT: min_act_count,
                heuristics_miner.Variants.CLASSIC.value.Parameters.MIN_DFG_OCCURRENCES: min_dfg_occurrences,
            }

            heuristics_net = heuristics_miner.apply_heu(
                pm_df,
                parameters=parameters,
            )

            gviz = hn_visualizer.apply(heuristics_net)
            hn_visualizer.save(gviz, svg_path)

        return _read_svg(svg_path)


def build_pm4py_process_tree_svg(
    event_log: pd.DataFrame,
    noise_threshold: float = 0.2,
) -> str:
    """
    Строит иерархическую диаграмму процесса через PM4Py.

    Используется Inductive Miner, который строит process tree.
    """
    try:
        import pm4py
    except ImportError as exc:
        raise ImportError(
            "Библиотека pm4py не установлена. Установите её командой: pip install pm4py"
        ) from exc

    pm_df = prepare_pm4py_dataframe(event_log)

    with tempfile.TemporaryDirectory() as tmp_dir:
        svg_path = os.path.join(tmp_dir, "process_tree.svg")

        try:
            process_tree = pm4py.discover_process_tree_inductive(
                pm_df,
                noise_threshold=noise_threshold,
            )

            pm4py.save_vis_process_tree(
                process_tree,
                svg_path,
            )

        except Exception:
            from pm4py.algo.discovery.inductive import algorithm as inductive_miner
            from pm4py.visualization.process_tree import visualizer as pt_visualizer

            try:
                process_tree = inductive_miner.apply(
                    pm_df,
                    parameters={
                        inductive_miner.Variants.IMf.value.Parameters.NOISE_THRESHOLD: noise_threshold,
                    },
                )
            except Exception:
                process_tree = inductive_miner.apply(pm_df)

            gviz = pt_visualizer.apply(process_tree)
            pt_visualizer.save(gviz, svg_path)

        return _read_svg(svg_path)


def save_svg_to_download_bytes(svg_text: str) -> bytes:
    """
    Готовит SVG для st.download_button.
    """
    return svg_text.encode("utf-8")