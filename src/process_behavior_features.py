import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any


# ============================================================
# Нормализация и служебные функции
# ============================================================

def _normalize_context(value) -> str:
    """
    Приводит название элемента курса к единому виду.
    Это важно, чтобы ручной выбор обязательных элементов совпадал
    с названиями внутри event log.
    """
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    text = str(value).strip()
    text = " ".join(text.split())
    return text


def _safe_ratio(numerator, denominator) -> float:
    if denominator is None or denominator == 0:
        return 0.0
    try:
        return float(numerator) / float(denominator)
    except Exception:
        return 0.0


def _contains_any(text: str, keywords: List[str]) -> bool:
    text = str(text).lower()
    return any(keyword.lower() in text for keyword in keywords)


def _prepare_event_log(event_log: pd.DataFrame) -> pd.DataFrame:
    """
    Подготавливает event log к расчёту процессных признаков.
    """
    if event_log is None or event_log.empty:
        return pd.DataFrame()

    df = event_log.copy()

    if "student_id" not in df.columns:
        raise ValueError("В event log отсутствует колонка student_id.")

    if "timestamp" not in df.columns:
        raise ValueError("В event log отсутствует колонка timestamp.")

    df["student_id"] = df["student_id"].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df[df["timestamp"].notna()].copy()

    if "component" not in df.columns:
        df["component"] = ""

    if "context" not in df.columns:
        df["context"] = ""

    if "activity" not in df.columns:
        df["activity"] = ""

    if "process_activity" not in df.columns:
        df["process_activity"] = df["activity"]

    df["component"] = df["component"].fillna("").astype(str)
    df["context"] = df["context"].fillna("").astype(str)
    df["activity"] = df["activity"].fillna("").astype(str)
    df["process_activity"] = df["process_activity"].fillna("").astype(str)

    df["context_norm"] = df["context"].apply(_normalize_context)

    df = df.sort_values(["student_id", "timestamp"]).reset_index(drop=True)

    dedup_cols = [
        col for col in [
            "student_id",
            "timestamp",
            "component",
            "context",
            "activity",
            "process_activity",
        ]
        if col in df.columns
    ]

    df = df.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)

    return df


# ============================================================
# Определение типов событий
# ============================================================

ASSIGNMENT_KEYWORDS = [
    "задание",
    "ответ в виде файла",
    "assignment",
    "assign",
    "submit",
    "submission",
    "uploaded",
    "upload",
    "сдача",
    "отправка",
    "загрузка ответа",
]

TEST_KEYWORDS = [
    "тест",
    "quiz",
    "test",
    "attempt",
    "попытка",
]

LECTURE_KEYWORDS = [
    "лекция",
    "lecture",
    "lesson",
]

VIEW_KEYWORDS = [
    "просмотр",
    "открытие",
    "view",
    "viewed",
    "opened",
]

SUBMIT_KEYWORDS = [
    "загрузка",
    "отправка",
    "представлена",
    "submit",
    "submitted",
    "upload",
    "uploaded",
    "ответ",
]

TEST_START_KEYWORDS = [
    "начало",
    "начата",
    "started",
    "attempt started",
    "попытка начата",
]

TEST_FINISH_KEYWORDS = [
    "завершение",
    "завершена",
    "finished",
    "completed",
    "attempt submitted",
    "попытка завершена",
]

LECTURE_FINISH_KEYWORDS = [
    "завершение",
    "завершена",
    "finished",
    "completed",
]


def _build_search_text(df: pd.DataFrame) -> pd.Series:
    return (
        df["component"].fillna("").astype(str)
        + " "
        + df["context"].fillna("").astype(str)
        + " "
        + df["activity"].fillna("").astype(str)
        + " "
        + df["process_activity"].fillna("").astype(str)
    ).str.lower()


def _is_assignment_event(df: pd.DataFrame) -> pd.Series:
    text = _build_search_text(df)
    return text.apply(lambda x: _contains_any(x, ASSIGNMENT_KEYWORDS))


def _is_test_event(df: pd.DataFrame) -> pd.Series:
    text = _build_search_text(df)
    return text.apply(lambda x: _contains_any(x, TEST_KEYWORDS))


def _is_lecture_event(df: pd.DataFrame) -> pd.Series:
    text = _build_search_text(df)
    return text.apply(lambda x: _contains_any(x, LECTURE_KEYWORDS))


def _is_view_event(df: pd.DataFrame) -> pd.Series:
    text = _build_search_text(df)
    return text.apply(lambda x: _contains_any(x, VIEW_KEYWORDS))


def _is_submit_event(df: pd.DataFrame) -> pd.Series:
    text = _build_search_text(df)
    return text.apply(lambda x: _contains_any(x, SUBMIT_KEYWORDS))


def _is_test_start_event(df: pd.DataFrame) -> pd.Series:
    text = _build_search_text(df)
    return text.apply(lambda x: _contains_any(x, TEST_START_KEYWORDS))


def _is_test_finish_event(df: pd.DataFrame) -> pd.Series:
    text = _build_search_text(df)
    return text.apply(lambda x: _contains_any(x, TEST_FINISH_KEYWORDS))


def _is_lecture_finish_event(df: pd.DataFrame) -> pd.Series:
    text = _build_search_text(df)
    return text.apply(lambda x: _contains_any(x, LECTURE_FINISH_KEYWORDS))


# ============================================================
# Автоматическое определение обязательных элементов
# ============================================================

def infer_required_course_elements(
    event_log: pd.DataFrame,
    min_required_completion_share: float = 0.5,
) -> Dict[str, Any]:
    """
    Автоматически находит кандидаты в обязательные задания и тесты.

    Логика:
    - для заданий считаем студентов, у которых было событие отправки/загрузки;
    - для тестов считаем студентов, у которых было начало или завершение попытки;
    - элемент считается обязательным кандидатом, если доля студентов >= порога.
    """
    df = _prepare_event_log(event_log)

    if df.empty:
        return {
            "required_assignments": [],
            "required_tests": [],
            "assignment_completion_stats": pd.DataFrame(),
            "test_completion_stats": pd.DataFrame(),
        }

    total_students = df["student_id"].nunique()

    # ----------------------------
    # Задания
    # ----------------------------
    assignment_df = df[_is_assignment_event(df)].copy()

    if not assignment_df.empty:
        submit_mask = _is_submit_event(assignment_df)
        assignment_completed_df = assignment_df[submit_mask].copy()

        if assignment_completed_df.empty:
            assignment_completed_df = assignment_df.copy()

        assignment_stats = (
            assignment_completed_df
            .groupby("context_norm")
            .agg(
                students_completed=("student_id", "nunique"),
                events_count=("student_id", "count"),
                first_event=("timestamp", "min"),
                last_event=("timestamp", "max"),
            )
            .reset_index()
            .rename(columns={"context_norm": "context"})
        )

        assignment_stats = assignment_stats[assignment_stats["context"] != ""].copy()
        assignment_stats["completion_share"] = assignment_stats["students_completed"] / max(total_students, 1)
        assignment_stats["is_required"] = assignment_stats["completion_share"] >= min_required_completion_share

        assignment_stats = assignment_stats.sort_values(
            ["is_required", "students_completed", "completion_share"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    else:
        assignment_stats = pd.DataFrame(
            columns=[
                "context",
                "students_completed",
                "events_count",
                "first_event",
                "last_event",
                "completion_share",
                "is_required",
            ]
        )

    # ----------------------------
    # Тесты
    # ----------------------------
    test_df = df[_is_test_event(df)].copy()

    if not test_df.empty:
        test_active_mask = _is_test_start_event(test_df) | _is_test_finish_event(test_df)

        test_completed_df = test_df[test_active_mask].copy()

        if test_completed_df.empty:
            test_completed_df = test_df.copy()

        test_stats = (
            test_completed_df
            .groupby("context_norm")
            .agg(
                students_completed=("student_id", "nunique"),
                events_count=("student_id", "count"),
                first_event=("timestamp", "min"),
                last_event=("timestamp", "max"),
            )
            .reset_index()
            .rename(columns={"context_norm": "context"})
        )

        test_stats = test_stats[test_stats["context"] != ""].copy()
        test_stats["completion_share"] = test_stats["students_completed"] / max(total_students, 1)
        test_stats["is_required"] = test_stats["completion_share"] >= min_required_completion_share

        test_stats = test_stats.sort_values(
            ["is_required", "students_completed", "completion_share"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    else:
        test_stats = pd.DataFrame(
            columns=[
                "context",
                "students_completed",
                "events_count",
                "first_event",
                "last_event",
                "completion_share",
                "is_required",
            ]
        )

    required_assignments = (
        assignment_stats.loc[assignment_stats["is_required"], "context"]
        .dropna()
        .astype(str)
        .map(_normalize_context)
        .tolist()
    )

    required_tests = (
        test_stats.loc[test_stats["is_required"], "context"]
        .dropna()
        .astype(str)
        .map(_normalize_context)
        .tolist()
    )

    return {
        "required_assignments": required_assignments,
        "required_tests": required_tests,
        "assignment_completion_stats": assignment_stats,
        "test_completion_stats": test_stats,
    }


# ============================================================
# Расчёт таймингов по студенту
# ============================================================

def _calculate_assignment_stats_for_student(
    student_log: pd.DataFrame,
    required_assignments: List[str],
    fast_assignment_minutes: int,
) -> Dict[str, Any]:
    assignment_log = student_log[_is_assignment_event(student_log)].copy()

    completed_assignments = []
    missing_assignments = []
    upload_delays = []
    suspicious_first_upload_count = 0
    measured_first_upload_count = 0

    for assignment in required_assignments:
        assignment_norm = _normalize_context(assignment)

        item_log = assignment_log[
            assignment_log["context_norm"] == assignment_norm
        ].copy()

        if item_log.empty:
            missing_assignments.append(assignment_norm)
            continue

        view_log = item_log[_is_view_event(item_log)].copy()
        submit_log = item_log[_is_submit_event(item_log)].copy()

        if submit_log.empty:
            # Если явной отправки нет, но есть активность по заданию,
            # считаем, что студент хотя бы взаимодействовал с элементом.
            completed_assignments.append(assignment_norm)
            continue

        completed_assignments.append(assignment_norm)

        first_submit_time = submit_log["timestamp"].min()

        if not view_log.empty:
            first_view_time = view_log["timestamp"].min()
        else:
            first_view_time = item_log["timestamp"].min()

        if pd.notna(first_view_time) and pd.notna(first_submit_time):
            delay_min = (first_submit_time - first_view_time).total_seconds() / 60

            if delay_min >= 0:
                upload_delays.append(delay_min)
                measured_first_upload_count += 1

                if delay_min <= fast_assignment_minutes:
                    suspicious_first_upload_count += 1

    expected_count = len(required_assignments)
    completed_count = len(set(completed_assignments))
    missing_assignments = sorted(set(missing_assignments))

    return {
        "completed_assignments_count": completed_count,
        "expected_assignments_count": expected_count,
        "assignment_completion_ratio": _safe_ratio(completed_count, expected_count),
        "completed_assignments_list": "; ".join(sorted(set(completed_assignments))),
        "missing_assignments_list": "; ".join(missing_assignments),
        "fast_assignment_upload_count": suspicious_first_upload_count,
        "measured_assignment_upload_count": measured_first_upload_count,
        "fast_assignment_upload_ratio": _safe_ratio(
            suspicious_first_upload_count,
            measured_first_upload_count,
        ),
        "median_assignment_upload_delay_min": (
            float(np.median(upload_delays)) if upload_delays else np.nan
        ),
        "suspicious_first_assignment_upload_count": suspicious_first_upload_count,
        "measured_first_assignment_upload_count": measured_first_upload_count,
        "suspicious_first_assignment_upload_ratio": _safe_ratio(
            suspicious_first_upload_count,
            measured_first_upload_count,
        ),
        "median_first_assignment_upload_delay_min": (
            float(np.median(upload_delays)) if upload_delays else np.nan
        ),
    }


def _calculate_test_stats_for_student(
    student_log: pd.DataFrame,
    required_tests: List[str],
    fast_test_minutes: int,
) -> Dict[str, Any]:
    test_log = student_log[_is_test_event(student_log)].copy()

    completed_tests = []
    missing_tests = []
    test_durations = []
    fast_test_count = 0
    measured_test_count = 0

    for test in required_tests:
        test_norm = _normalize_context(test)

        item_log = test_log[
            test_log["context_norm"] == test_norm
        ].copy()

        if item_log.empty:
            missing_tests.append(test_norm)
            continue

        start_log = item_log[_is_test_start_event(item_log)].copy()
        finish_log = item_log[_is_test_finish_event(item_log)].copy()

        if not finish_log.empty:
            completed_tests.append(test_norm)
        elif not start_log.empty:
            completed_tests.append(test_norm)
        else:
            completed_tests.append(test_norm)

        if not start_log.empty and not finish_log.empty:
            first_start = start_log["timestamp"].min()
            first_finish = finish_log["timestamp"].min()

            duration_min = (first_finish - first_start).total_seconds() / 60

            if duration_min >= 0:
                test_durations.append(duration_min)
                measured_test_count += 1

                if duration_min <= fast_test_minutes:
                    fast_test_count += 1

    expected_count = len(required_tests)
    completed_count = len(set(completed_tests))
    missing_tests = sorted(set(missing_tests))

    return {
        "completed_tests_count": completed_count,
        "expected_tests_count": expected_count,
        "test_completion_ratio": _safe_ratio(completed_count, expected_count),
        "completed_tests_list": "; ".join(sorted(set(completed_tests))),
        "missing_tests_list": "; ".join(missing_tests),
        "fast_test_completion_count": fast_test_count,
        "measured_test_completion_count": measured_test_count,
        "fast_test_completion_ratio": _safe_ratio(fast_test_count, measured_test_count),
        "median_test_duration_min": (
            float(np.median(test_durations)) if test_durations else np.nan
        ),
    }


def _calculate_lecture_stats_for_student(
    student_log: pd.DataFrame,
    fast_lecture_minutes: int,
) -> Dict[str, Any]:
    lecture_log = student_log[_is_lecture_event(student_log)].copy()

    if lecture_log.empty:
        return {
            "fast_lecture_completion_count": 0,
            "measured_lecture_completion_count": 0,
            "fast_lecture_completion_ratio": 0.0,
            "median_lecture_duration_min": np.nan,
        }

    lecture_durations = []
    fast_lecture_count = 0
    measured_lecture_count = 0

    for context_norm, item_log in lecture_log.groupby("context_norm"):
        if context_norm == "":
            continue

        first_open = item_log["timestamp"].min()

        finish_log = item_log[_is_lecture_finish_event(item_log)].copy()

        if finish_log.empty:
            continue

        first_finish = finish_log["timestamp"].min()

        duration_min = (first_finish - first_open).total_seconds() / 60

        if duration_min >= 0:
            lecture_durations.append(duration_min)
            measured_lecture_count += 1

            if duration_min <= fast_lecture_minutes:
                fast_lecture_count += 1

    return {
        "fast_lecture_completion_count": fast_lecture_count,
        "measured_lecture_completion_count": measured_lecture_count,
        "fast_lecture_completion_ratio": _safe_ratio(
            fast_lecture_count,
            measured_lecture_count,
        ),
        "median_lecture_duration_min": (
            float(np.median(lecture_durations)) if lecture_durations else np.nan
        ),
    }


def _calculate_temporal_stats_for_student(
    student_log: pd.DataFrame,
    course_start: pd.Timestamp,
    course_end: pd.Timestamp,
    last_period_days: int,
) -> Dict[str, Any]:
    student_log = student_log.sort_values("timestamp").copy()

    total_events = len(student_log)

    if total_events == 0:
        return {
            "process_total_events": 0,
            "process_active_days": 0,
            "max_day_activity_ratio": 0.0,
            "top_2_days_activity_ratio": 0.0,
            "top_3_days_activity_ratio": 0.0,
            "days_to_80_percent_events": np.nan,
            "last_period_events_ratio": 0.0,
        }

    student_log["date"] = student_log["timestamp"].dt.date

    daily_counts = (
        student_log
        .groupby("date")
        .size()
        .sort_values(ascending=False)
    )

    active_days = len(daily_counts)

    max_day_activity_ratio = _safe_ratio(daily_counts.iloc[0], total_events)

    top_2_days_activity_ratio = _safe_ratio(
        daily_counts.head(2).sum(),
        total_events,
    )

    top_3_days_activity_ratio = _safe_ratio(
        daily_counts.head(3).sum(),
        total_events,
    )

    ordered_log = student_log.sort_values("timestamp").copy()
    ordered_log["event_number"] = np.arange(1, len(ordered_log) + 1)
    threshold_80 = total_events * 0.8

    event_80 = ordered_log[ordered_log["event_number"] >= threshold_80].head(1)

    if not event_80.empty:
        days_to_80 = (
            event_80["timestamp"].iloc[0] - ordered_log["timestamp"].min()
        ).total_seconds() / 86400
    else:
        days_to_80 = np.nan

    last_period_start = course_end - pd.Timedelta(days=last_period_days)

    last_period_events = student_log[
        student_log["timestamp"] >= last_period_start
    ]

    last_period_events_ratio = _safe_ratio(len(last_period_events), total_events)

    return {
        "process_total_events": total_events,
        "process_active_days": active_days,
        "max_day_activity_ratio": max_day_activity_ratio,
        "top_2_days_activity_ratio": top_2_days_activity_ratio,
        "top_3_days_activity_ratio": top_3_days_activity_ratio,
        "days_to_80_percent_events": days_to_80,
        "last_period_events_ratio": last_period_events_ratio,
    }


def _calculate_trace_stats_for_student(student_log: pd.DataFrame) -> Dict[str, Any]:
    student_log = student_log.sort_values("timestamp").copy()

    activities = student_log["process_activity"].fillna("").astype(str).tolist()
    trace_length = len(activities)

    if trace_length == 0:
        return {
            "trace_length": 0,
            "linearity": 0.0,
            "complexity": 0.0,
            "returns_count": 0,
            "variant": "",
            "variant_frequency": 1,
        }

    unique_activities = len(set(activities))
    linearity = _safe_ratio(unique_activities, trace_length)

    returns_count = 0
    seen = set()

    for activity in activities:
        if activity in seen:
            returns_count += 1
        seen.add(activity)

    complexity = float(trace_length + returns_count)

    variant = " → ".join(activities)

    return {
        "trace_length": trace_length,
        "linearity": linearity,
        "complexity": complexity,
        "returns_count": returns_count,
        "variant": variant,
        "variant_frequency": 1,
    }


# ============================================================
# Интерпретация процессного паттерна
# ============================================================

def _build_process_interpretation(row: Dict[str, Any]) -> Dict[str, Any]:
    flags = []

    control_completion_ratio = row.get("control_completion_ratio", 0.0)
    assignment_completion_ratio = row.get("assignment_completion_ratio", 0.0)
    test_completion_ratio = row.get("test_completion_ratio", 0.0)

    top_2_days_activity_ratio = row.get("top_2_days_activity_ratio", 0.0)
    last_period_events_ratio = row.get("last_period_events_ratio", 0.0)

    fast_lecture_ratio = row.get("fast_lecture_completion_ratio", 0.0)
    fast_test_ratio = row.get("fast_test_completion_ratio", 0.0)
    fast_assignment_upload_ratio = row.get("suspicious_first_assignment_upload_ratio", 0.0)

    active_days = row.get("process_active_days", 0)
    total_events = row.get("process_total_events", 0)

    if control_completion_ratio < 0.5:
        flags.append("Неполное выполнение контрольных активностей")

    if top_2_days_activity_ratio >= 0.6 or last_period_events_ratio >= 0.6:
        flags.append("Сжатое прохождение курса")

    if fast_lecture_ratio >= 0.5 and row.get("measured_lecture_completion_count", 0) > 0:
        flags.append("Формальное прохождение лекционных элементов")

    if fast_test_ratio >= 0.5 and row.get("measured_test_completion_count", 0) > 0:
        flags.append("Быстрое прохождение тестов")

    if fast_assignment_upload_ratio >= 0.5 and row.get("measured_first_assignment_upload_count", 0) > 0:
        flags.append("Подозрительно быстрая первая загрузка ответов")

    if active_days <= 2 and total_events >= 20:
        flags.append("Концентрированная активность за короткий период")

    if not flags:
        if control_completion_ratio >= 0.8 and active_days >= 5:
            process_pattern = "Регулярное прохождение"
        elif control_completion_ratio >= 0.8:
            process_pattern = "Выполнение основных контрольных активностей"
        else:
            process_pattern = "Умеренно выраженный процессный паттерн"
    else:
        if "Неполное выполнение контрольных активностей" in flags:
            process_pattern = "Неполное выполнение контрольных активностей"
        elif "Сжатое прохождение курса" in flags:
            process_pattern = "Сжатое прохождение курса"
        elif (
            "Формальное прохождение лекционных элементов" in flags
            and "Быстрое прохождение тестов" in flags
        ):
            process_pattern = "Формально комплексное прохождение"
        elif "Формальное прохождение лекционных элементов" in flags:
            process_pattern = "Формальное прохождение лекционных элементов"
        elif "Быстрое прохождение тестов" in flags:
            process_pattern = "Быстрое прохождение тестов"
        elif "Подозрительно быстрая первая загрузка ответов" in flags:
            process_pattern = "Подозрительно быстрая первая загрузка ответов"
        else:
            process_pattern = flags[0]

    process_flags = "; ".join(flags) if flags else "Нет выраженных дополнительных признаков"

    description_parts = [
        f"Процессный паттерн: {process_pattern}.",
        f"Выполнение заданий: {assignment_completion_ratio:.2f}.",
        f"Выполнение тестов: {test_completion_ratio:.2f}.",
        f"Общая полнота контрольных активностей: {control_completion_ratio:.2f}.",
        f"Активных дней: {active_days}.",
        f"Доля активности в два самых активных дня: {top_2_days_activity_ratio:.2f}.",
    ]

    if flags:
        description_parts.append(f"Дополнительные признаки: {process_flags}.")
    else:
        description_parts.append("Выраженных рискованных процессных признаков не обнаружено.")

    return {
        "process_pattern": process_pattern,
        "process_flags": process_flags,
        "process_flags_count": len(flags),
        "final_behavior_description": " ".join(description_parts),
    }


# ============================================================
# Главная функция расчёта process behavior features
# ============================================================

def calculate_student_process_behavior_features(
    event_log: pd.DataFrame,
    last_period_days: int = 7,
    fast_assignment_minutes: int = 15,
    fast_test_minutes: int = 3,
    fast_lecture_minutes: int = 2,
    min_required_completion_share: float = 0.5,
    required_assignments: Optional[List[str]] = None,
    required_tests: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Рассчитывает процессные признаки студентов.

    Если required_assignments и required_tests переданы вручную,
    расчёт идёт именно по ним.

    Если они не переданы, список обязательных элементов определяется автоматически.
    """
    df = _prepare_event_log(event_log)

    if df.empty:
        return pd.DataFrame()

    if required_assignments is None or required_tests is None:
        inferred_elements = infer_required_course_elements(
            event_log=df,
            min_required_completion_share=min_required_completion_share,
        )

        if required_assignments is None:
            required_assignments = inferred_elements["required_assignments"]

        if required_tests is None:
            required_tests = inferred_elements["required_tests"]

    required_assignments = [
        _normalize_context(item)
        for item in required_assignments
        if _normalize_context(item)
    ]

    required_tests = [
        _normalize_context(item)
        for item in required_tests
        if _normalize_context(item)
    ]

    required_assignments = sorted(set(required_assignments))
    required_tests = sorted(set(required_tests))

    course_start = df["timestamp"].min()
    course_end = df["timestamp"].max()

    rows = []

    for student_id, student_log in df.groupby("student_id"):
        student_log = student_log.sort_values("timestamp").copy()

        assignment_stats = _calculate_assignment_stats_for_student(
            student_log=student_log,
            required_assignments=required_assignments,
            fast_assignment_minutes=fast_assignment_minutes,
        )

        test_stats = _calculate_test_stats_for_student(
            student_log=student_log,
            required_tests=required_tests,
            fast_test_minutes=fast_test_minutes,
        )

        lecture_stats = _calculate_lecture_stats_for_student(
            student_log=student_log,
            fast_lecture_minutes=fast_lecture_minutes,
        )

        temporal_stats = _calculate_temporal_stats_for_student(
            student_log=student_log,
            course_start=course_start,
            course_end=course_end,
            last_period_days=last_period_days,
        )

        trace_stats = _calculate_trace_stats_for_student(student_log)

        expected_control_count = (
            assignment_stats["expected_assignments_count"]
            + test_stats["expected_tests_count"]
        )

        completed_control_count = (
            assignment_stats["completed_assignments_count"]
            + test_stats["completed_tests_count"]
        )

        row = {
            "student_id": str(student_id),
            **assignment_stats,
            **test_stats,
            **lecture_stats,
            **temporal_stats,
            **trace_stats,
            "completed_control_count": completed_control_count,
            "expected_control_count": expected_control_count,
            "control_completion_ratio": _safe_ratio(
                completed_control_count,
                expected_control_count,
            ),
            "required_assignments_count": len(required_assignments),
            "required_tests_count": len(required_tests),
            "required_assignments_list": "; ".join(required_assignments),
            "required_tests_list": "; ".join(required_tests),
        }

        interpretation = _build_process_interpretation(row)
        row.update(interpretation)

        rows.append(row)

    result_df = pd.DataFrame(rows)

    if result_df.empty:
        return result_df

    # Частота варианта траектории
    if "variant" in result_df.columns:
        variant_counts = result_df["variant"].value_counts().to_dict()
        result_df["variant_frequency"] = result_df["variant"].map(variant_counts)

    return result_df


# ============================================================
# Объединение ресурсных и процессных паттернов
# ============================================================

def merge_resource_and_process_patterns(
    resource_patterns_df: pd.DataFrame,
    process_features_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Объединяет:
    - ресурсный паттерн из ML-кластеризации;
    - процессный паттерн из process mining.
    """
    if process_features_df is None or process_features_df.empty:
        return pd.DataFrame()

    process_df = process_features_df.copy()
    process_df["student_id"] = process_df["student_id"].astype(str)

    if resource_patterns_df is None or resource_patterns_df.empty:
        result_df = process_df.copy()
        result_df["cluster"] = np.nan
        result_df["resource_pattern"] = "Нет ресурсного паттерна"
    else:
        resource_df = resource_patterns_df.copy()
        resource_df["student_id"] = resource_df["student_id"].astype(str)

        if "suggested_name" in resource_df.columns and "resource_pattern" not in resource_df.columns:
            resource_df = resource_df.rename(columns={"suggested_name": "resource_pattern"})

        if "resource_pattern" not in resource_df.columns:
            resource_df["resource_pattern"] = "Ресурсный паттерн не определён"

        merge_cols = [
            col for col in ["student_id", "cluster", "resource_pattern"]
            if col in resource_df.columns
        ]

        result_df = process_df.merge(
            resource_df[merge_cols].drop_duplicates(subset=["student_id"]),
            on="student_id",
            how="left",
        )

        result_df["resource_pattern"] = result_df["resource_pattern"].fillna(
            "Нет ресурсного паттерна"
        )

    result_df["final_behavior_pattern"] = (
        result_df["resource_pattern"].astype(str)
        + " + "
        + result_df["process_pattern"].astype(str)
    )

    return result_df