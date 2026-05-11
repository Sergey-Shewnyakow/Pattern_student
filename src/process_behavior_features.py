import pandas as pd
import numpy as np


MATERIAL_ACTIONS = [
    "Открытие лекции",
    "Начало лекции",
    "Продолжение лекции",
    "Завершение лекции",
    "Повтор лекции",
    "Изучение лекции",
    "Просмотр видеолекции",
    "Просмотр страницы курса",
    "Открытие страницы курса",
]


CONTROL_ACTIONS = [
    "Открытие теста",
    "Начало теста",
    "Работа с тестом",
    "Завершение теста",
    "Просмотр попытки теста",
    "Просмотр результата теста",
    "Просмотр сводки теста",
    "Открытие практического задания",
    "Открытие формы ответа",
    "Загрузка файла",
    "Отправка практического задания",
    "Просмотр статуса задания",
]


ASSIGNMENT_OPEN_ACTIONS = [
    "Открытие практического задания",
    "Открытие формы ответа",
    "Просмотр статуса задания",
]


ASSIGNMENT_SUBMIT_ACTIONS = [
    "Загрузка файла",
    "Отправка практического задания",
    "Обновление ответа",
]


LECTURE_START_ACTIONS = [
    "Начало лекции",
    "Повтор лекции",
]


LECTURE_END_ACTIONS = [
    "Завершение лекции",
]


TEST_START_ACTIONS = [
    "Начало теста",
]


TEST_COMPLETION_OR_PARTICIPATION_ACTIONS = [
    "Начало теста",
    "Завершение теста",
    "Просмотр результата теста",
    "Просмотр сводки теста",
    "Просмотр попытки теста",
]


TEST_FINISH_ACTIONS = [
    "Завершение теста",
]


def _safe_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _safe_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _normalize_context(value) -> str:
    text = _safe_text(value).strip()
    text = " ".join(text.split())
    return text


def _calculate_top_days_ratio(group: pd.DataFrame, top_n_days: int) -> float:
    if group.empty:
        return 0.0

    total_events = len(group)

    if total_events == 0:
        return 0.0

    events_by_day = (
        group.groupby(group["timestamp"].dt.date)
        .size()
        .sort_values(ascending=False)
    )

    top_events = events_by_day.head(top_n_days).sum()

    return float(top_events / total_events)


def _calculate_days_to_percent_events(
    group: pd.DataFrame,
    percent: float = 0.8,
) -> int:
    if group.empty:
        return 0

    total_events = len(group)

    if total_events == 0:
        return 0

    events_by_day = (
        group.groupby(group["timestamp"].dt.date)
        .size()
        .sort_values(ascending=False)
    )

    cumulative_events = 0

    for day_index, events_count in enumerate(events_by_day, start=1):
        cumulative_events += int(events_count)

        if cumulative_events / total_events >= percent:
            return day_index

    return len(events_by_day)


def infer_required_course_elements(
    event_log: pd.DataFrame,
    min_required_completion_share: float = 0.5,
) -> dict:
    """
    Универсально определяет обязательные задания и тесты курса.

    Элемент считается обязательным, если его выполнила/начала значительная часть студентов.
    """
    if event_log is None or event_log.empty:
        return {
            "required_assignments": [],
            "required_tests": [],
            "assignment_completion_stats": pd.DataFrame(),
            "test_completion_stats": pd.DataFrame(),
        }

    df = event_log.copy()

    if "component" not in df.columns:
        df["component"] = ""

    if "context" not in df.columns:
        df["context"] = ""

    df["student_id"] = df["student_id"].astype(str)
    df["context_clean"] = df["context"].apply(_normalize_context)

    total_students = df["student_id"].nunique()

    if total_students == 0:
        return {
            "required_assignments": [],
            "required_tests": [],
            "assignment_completion_stats": pd.DataFrame(),
            "test_completion_stats": pd.DataFrame(),
        }

    assignment_completed_df = df[
        df["process_activity"].isin(ASSIGNMENT_SUBMIT_ACTIONS)
        & df["component"].astype(str).str.contains(
            "Задание|Ответ в виде файла",
            case=False,
            na=False,
            regex=True,
        )
    ].copy()

    if assignment_completed_df.empty:
        assignment_stats_df = pd.DataFrame(
            columns=[
                "context",
                "students_completed",
                "completion_share",
                "is_required",
            ]
        )
    else:
        assignment_stats_df = (
            assignment_completed_df
            .groupby("context_clean")["student_id"]
            .nunique()
            .reset_index(name="students_completed")
            .rename(columns={"context_clean": "context"})
        )

        assignment_stats_df["completion_share"] = (
            assignment_stats_df["students_completed"] / total_students
        )

        assignment_stats_df["is_required"] = (
            assignment_stats_df["completion_share"] >= min_required_completion_share
        )

        assignment_stats_df = assignment_stats_df.sort_values(
            ["is_required", "students_completed"],
            ascending=[False, False],
        ).reset_index(drop=True)

    required_assignments = (
        assignment_stats_df.loc[
            assignment_stats_df["is_required"],
            "context",
        ]
        .astype(str)
        .tolist()
        if not assignment_stats_df.empty
        else []
    )

    test_completed_df = df[
        df["process_activity"].isin(TEST_COMPLETION_OR_PARTICIPATION_ACTIONS)
        & df["component"].astype(str).str.contains(
            "Тест",
            case=False,
            na=False,
            regex=False,
        )
    ].copy()

    if test_completed_df.empty:
        test_stats_df = pd.DataFrame(
            columns=[
                "context",
                "students_completed",
                "completion_share",
                "is_required",
            ]
        )
    else:
        test_stats_df = (
            test_completed_df
            .groupby("context_clean")["student_id"]
            .nunique()
            .reset_index(name="students_completed")
            .rename(columns={"context_clean": "context"})
        )

        test_stats_df["completion_share"] = (
            test_stats_df["students_completed"] / total_students
        )

        test_stats_df["is_required"] = (
            test_stats_df["completion_share"] >= min_required_completion_share
        )

        test_stats_df = test_stats_df.sort_values(
            ["is_required", "students_completed"],
            ascending=[False, False],
        ).reset_index(drop=True)

    required_tests = (
        test_stats_df.loc[
            test_stats_df["is_required"],
            "context",
        ]
        .astype(str)
        .tolist()
        if not test_stats_df.empty
        else []
    )

    return {
        "required_assignments": required_assignments,
        "required_tests": required_tests,
        "assignment_completion_stats": assignment_stats_df,
        "test_completion_stats": test_stats_df,
    }


def _calculate_assignment_completion(
    group: pd.DataFrame,
    required_assignments: list[str],
) -> dict:
    completed_assignments = set()

    group = group.copy()
    group["context_clean"] = group["context"].apply(_normalize_context)

    for _, row in group.iterrows():
        action = row.get("process_activity")
        context = row.get("context_clean")

        if context not in required_assignments:
            continue

        if action in ASSIGNMENT_SUBMIT_ACTIONS:
            completed_assignments.add(context)

    expected_count = len(required_assignments)
    completed_count = len(completed_assignments)

    missing_assignments = sorted(
        set(required_assignments) - completed_assignments
    )

    return {
        "completed_assignments_count": completed_count,
        "expected_assignments_count": expected_count,
        "assignment_completion_ratio": (
            completed_count / expected_count if expected_count > 0 else 0
        ),
        "completed_assignments_list": ", ".join(sorted(completed_assignments)),
        "missing_assignments_list": ", ".join(missing_assignments),
    }


def _calculate_test_completion(
    group: pd.DataFrame,
    required_tests: list[str],
) -> dict:
    """
    Тест считается выполненным/учтённым, если он был начат.

    Это сделано потому, что если тест начат, он фактически должен завершиться,
    даже если событие завершения не попало в лог.
    """
    completed_tests = set()

    group = group.copy()
    group["context_clean"] = group["context"].apply(_normalize_context)

    for _, row in group.iterrows():
        action = row.get("process_activity")
        context = row.get("context_clean")

        if context not in required_tests:
            continue

        if action in TEST_COMPLETION_OR_PARTICIPATION_ACTIONS:
            completed_tests.add(context)

    expected_count = len(required_tests)
    completed_count = len(completed_tests)

    missing_tests = sorted(
        set(required_tests) - completed_tests
    )

    return {
        "completed_tests_count": completed_count,
        "expected_tests_count": expected_count,
        "test_completion_ratio": (
            completed_count / expected_count if expected_count > 0 else 0
        ),
        "completed_tests_list": ", ".join(sorted(completed_tests)),
        "missing_tests_list": ", ".join(missing_tests),
    }


def _calculate_suspicious_first_assignment_uploads(
    group: pd.DataFrame,
    required_assignments: list[str],
    fast_assignment_minutes: int = 15,
) -> dict:
    """
    Считает подозрительную быструю загрузку только в случае:
    первое открытие задания -> быстрая загрузка ответа.

    Если студент открывал задание раньше, а потом загрузил ответ быстро,
    это не считается подозрительным.
    """
    suspicious_upload_count = 0
    measured_first_upload_count = 0
    durations = []

    group = group.copy()
    group["context_clean"] = group["context"].apply(_normalize_context)

    first_open_time_by_assignment = {}
    first_submit_time_by_assignment = {}

    for _, row in group.sort_values("timestamp").iterrows():
        action = row.get("process_activity")
        context = row.get("context_clean")
        current_time = row.get("timestamp")

        if context not in required_assignments:
            continue

        if action in ASSIGNMENT_OPEN_ACTIONS:
            if context not in first_open_time_by_assignment:
                first_open_time_by_assignment[context] = current_time

        if action in ASSIGNMENT_SUBMIT_ACTIONS:
            if context not in first_submit_time_by_assignment:
                first_submit_time_by_assignment[context] = current_time

    for assignment_context, submit_time in first_submit_time_by_assignment.items():
        if assignment_context not in first_open_time_by_assignment:
            continue

        open_time = first_open_time_by_assignment[assignment_context]

        diff_minutes = (submit_time - open_time).total_seconds() / 60

        if diff_minutes < 0:
            continue

        measured_first_upload_count += 1
        durations.append(diff_minutes)

        if diff_minutes <= fast_assignment_minutes:
            suspicious_upload_count += 1

    suspicious_upload_ratio = (
        suspicious_upload_count / measured_first_upload_count
        if measured_first_upload_count > 0
        else 0
    )

    return {
        "suspicious_first_assignment_upload_count": suspicious_upload_count,
        "measured_first_assignment_upload_count": measured_first_upload_count,
        "suspicious_first_assignment_upload_ratio": round(suspicious_upload_ratio, 4),
        "median_first_assignment_upload_delay_min": (
            round(float(np.median(durations)), 2) if durations else None
        ),
        "avg_first_assignment_upload_delay_min": (
            round(float(np.mean(durations)), 2) if durations else None
        ),

        # Старые названия оставлены для совместимости со страницей.
        "fast_assignment_upload_count": suspicious_upload_count,
        "measured_assignment_upload_count": measured_first_upload_count,
        "fast_assignment_upload_ratio": round(suspicious_upload_ratio, 4),
        "median_assignment_upload_delay_min": (
            round(float(np.median(durations)), 2) if durations else None
        ),
        "avg_assignment_upload_delay_min": (
            round(float(np.mean(durations)), 2) if durations else None
        ),
    }


def _calculate_fast_tests(
    group: pd.DataFrame,
    required_tests: list[str],
    fast_test_minutes: int = 3,
) -> dict:
    fast_test_count = 0
    measured_test_count = 0
    durations = []

    group = group.copy()
    group["context_clean"] = group["context"].apply(_normalize_context)

    test_start_times = {}

    for _, row in group.sort_values("timestamp").iterrows():
        action = row.get("process_activity")
        context = row.get("context_clean")
        current_time = row.get("timestamp")

        if context not in required_tests:
            continue

        if action in TEST_START_ACTIONS:
            test_start_times[context] = current_time

        if action in TEST_FINISH_ACTIONS:
            if context not in test_start_times:
                continue

            start_time = test_start_times[context]
            diff_minutes = (current_time - start_time).total_seconds() / 60

            if diff_minutes < 0:
                continue

            measured_test_count += 1
            durations.append(diff_minutes)

            if diff_minutes <= fast_test_minutes:
                fast_test_count += 1

            test_start_times.pop(context, None)

    fast_test_ratio = (
        fast_test_count / measured_test_count
        if measured_test_count > 0
        else 0
    )

    return {
        "fast_test_completion_count": fast_test_count,
        "measured_test_completion_count": measured_test_count,
        "fast_test_completion_ratio": round(fast_test_ratio, 4),
        "median_test_duration_min": (
            round(float(np.median(durations)), 2) if durations else None
        ),
        "avg_test_duration_min": (
            round(float(np.mean(durations)), 2) if durations else None
        ),
    }


def _calculate_fast_lectures(
    group: pd.DataFrame,
    fast_lecture_minutes: int = 2,
) -> dict:
    fast_lecture_count = 0
    measured_lecture_count = 0
    durations = []

    lecture_start_times = {}

    for _, row in group.sort_values("timestamp").iterrows():
        action = row.get("process_activity")
        context = _normalize_context(row.get("context"))
        current_time = row.get("timestamp")

        if "лекция" not in context.lower():
            continue

        if action in LECTURE_START_ACTIONS:
            lecture_start_times[context] = current_time

        if action in LECTURE_END_ACTIONS:
            if context not in lecture_start_times:
                continue

            start_time = lecture_start_times[context]
            diff_minutes = (current_time - start_time).total_seconds() / 60

            if diff_minutes < 0:
                continue

            measured_lecture_count += 1
            durations.append(diff_minutes)

            if diff_minutes <= fast_lecture_minutes:
                fast_lecture_count += 1

            lecture_start_times.pop(context, None)

    fast_lecture_ratio = (
        fast_lecture_count / measured_lecture_count
        if measured_lecture_count > 0
        else 0
    )

    return {
        "fast_lecture_completion_count": fast_lecture_count,
        "measured_lecture_completion_count": measured_lecture_count,
        "fast_lecture_completion_ratio": round(fast_lecture_ratio, 4),
        "median_lecture_duration_min": (
            round(float(np.median(durations)), 2) if durations else None
        ),
        "avg_lecture_duration_min": (
            round(float(np.mean(durations)), 2) if durations else None
        ),
    }


def _build_process_flags(row: pd.Series) -> list[str]:
    flags = []

    total_events = float(row.get("process_total_events", 0))
    active_days = float(row.get("process_active_days", 0))

    completed_assignments = float(row.get("completed_assignments_count", 0))
    expected_assignments = float(row.get("expected_assignments_count", 0))

    completed_tests = float(row.get("completed_tests_count", 0))
    expected_tests = float(row.get("expected_tests_count", 0))

    max_day_ratio = float(row.get("max_day_activity_ratio", 0))
    top_2_ratio = float(row.get("top_2_days_activity_ratio", 0))
    top_3_ratio = float(row.get("top_3_days_activity_ratio", 0))
    days_to_80 = float(row.get("days_to_80_percent_events", 999))

    material_share = float(row.get("process_material_share", 0))
    control_share = float(row.get("process_control_share", 0))

    fast_lecture_ratio = float(row.get("fast_lecture_completion_ratio", 0))
    measured_lecture_count = float(row.get("measured_lecture_completion_count", 0))

    fast_test_count = float(row.get("fast_test_completion_count", 0))

    suspicious_assignment_count = float(
        row.get("suspicious_first_assignment_upload_count", 0)
    )

    if expected_assignments > 0 and completed_assignments < expected_assignments:
        flags.append("Неполное выполнение заданий")

    if expected_tests > 0 and completed_tests < expected_tests:
        flags.append("Неполное выполнение тестов")

    if total_events <= 10 or active_days <= 1:
        flags.append("Эпизодическое прохождение")

    if days_to_80 <= 2 or top_2_ratio >= 0.85:
        flags.append("Авральное прохождение курса")
    elif days_to_80 <= 3 and top_3_ratio >= 0.90:
        flags.append("Авральное прохождение курса")

    if (
        material_share > 0
        and control_share > 0
        and (
            max_day_ratio >= 0.65
            or top_2_ratio >= 0.80
            or days_to_80 <= 3
        )
    ):
        flags.append("Формально комплексное прохождение")

    if measured_lecture_count >= 5 and fast_lecture_ratio >= 0.70:
        flags.append("Формальное прохождение лекционных элементов")

    if fast_test_count >= 2:
        flags.append("Быстрое прохождение тестов")

    if suspicious_assignment_count >= 2:
        flags.append("Подозрительно быстрая первая загрузка ответов")

    if active_days >= 5 and days_to_80 >= 4 and top_2_ratio < 0.75:
        flags.append("Регулярное прохождение")

    return flags


def _choose_main_process_pattern(flags: list[str]) -> str:
    """
    Основной паттерн выбирается по приоритету, но все остальные признаки
    сохраняются в process_flags.
    """
    priority = [
        "Неполное выполнение заданий",
        "Неполное выполнение тестов",
        "Эпизодическое прохождение",
        "Авральное прохождение курса",
        "Формально комплексное прохождение",
        "Формальное прохождение лекционных элементов",
        "Быстрое прохождение тестов",
        "Подозрительно быстрая первая загрузка ответов",
        "Регулярное прохождение",
    ]

    for pattern in priority:
        if pattern in flags:
            if pattern in [
                "Неполное выполнение заданий",
                "Неполное выполнение тестов",
            ]:
                return "Неполное выполнение контрольных активностей"

            if pattern == "Подозрительно быстрая первая загрузка ответов":
                return "Подозрительно быстрая первая загрузка ответов"

            return pattern

    return "Смешанный процессный профиль"


def calculate_student_process_behavior_features(
    event_log: pd.DataFrame,
    last_period_days: int = 7,
    fast_assignment_minutes: int = 15,
    fast_test_minutes: int = 3,
    fast_lecture_minutes: int = 2,
    min_required_completion_share: float = 0.5,
) -> pd.DataFrame:
    if event_log is None or event_log.empty:
        return pd.DataFrame()

    required_columns = ["student_id", "timestamp", "process_activity"]

    missing_columns = [
        col for col in required_columns
        if col not in event_log.columns
    ]

    if missing_columns:
        raise ValueError(
            "Для расчёта процессных паттернов не хватает колонок: "
            + ", ".join(missing_columns)
        )

    df = event_log.copy()

    if "component" not in df.columns:
        df["component"] = ""

    if "context" not in df.columns:
        df["context"] = ""

    df["student_id"] = df["student_id"].astype(str)
    df["timestamp"] = _safe_datetime(df["timestamp"])

    df = df[df["timestamp"].notna()].copy()

    if df.empty:
        return pd.DataFrame()

    inferred_elements = infer_required_course_elements(
        event_log=df,
        min_required_completion_share=min_required_completion_share,
    )

    required_assignments = inferred_elements["required_assignments"]
    required_tests = inferred_elements["required_tests"]

    course_start = df["timestamp"].min()
    course_end = df["timestamp"].max()
    last_period_start = course_end - pd.Timedelta(days=last_period_days)

    rows = []

    for student_id, group in df.sort_values("timestamp").groupby("student_id"):
        group = group.copy()

        total_events = len(group)

        if total_events == 0:
            continue

        first_event_time = group["timestamp"].min()
        last_event_time = group["timestamp"].max()
        active_days = group["timestamp"].dt.date.nunique()

        activity_span_days = (
            (last_event_time - first_event_time).total_seconds() / 86400
        )

        events_by_day = group.groupby(group["timestamp"].dt.date).size()
        max_day_events = int(events_by_day.max())

        max_day_activity_ratio = (
            max_day_events / total_events if total_events > 0 else 0
        )

        top_2_days_activity_ratio = _calculate_top_days_ratio(group, 2)
        top_3_days_activity_ratio = _calculate_top_days_ratio(group, 3)

        days_to_80_percent_events = _calculate_days_to_percent_events(
            group,
            percent=0.8,
        )

        last_period_events = group[group["timestamp"] >= last_period_start]

        last_period_activity_ratio = (
            len(last_period_events) / total_events if total_events > 0 else 0
        )

        material_events = group[
            group["process_activity"].isin(MATERIAL_ACTIONS)
        ]

        control_events = group[
            group["process_activity"].isin(CONTROL_ACTIONS)
        ]

        material_events_count = len(material_events)
        control_events_count = len(control_events)

        material_share = (
            material_events_count / total_events if total_events > 0 else 0
        )

        control_share = (
            control_events_count / total_events if total_events > 0 else 0
        )

        assignment_completion = _calculate_assignment_completion(
            group=group,
            required_assignments=required_assignments,
        )

        test_completion = _calculate_test_completion(
            group=group,
            required_tests=required_tests,
        )

        suspicious_assignments = _calculate_suspicious_first_assignment_uploads(
            group=group,
            required_assignments=required_assignments,
            fast_assignment_minutes=fast_assignment_minutes,
        )

        fast_tests = _calculate_fast_tests(
            group=group,
            required_tests=required_tests,
            fast_test_minutes=fast_test_minutes,
        )

        fast_lectures = _calculate_fast_lectures(
            group=group,
            fast_lecture_minutes=fast_lecture_minutes,
        )

        completed_assignments_count = assignment_completion[
            "completed_assignments_count"
        ]

        expected_assignments_count = assignment_completion[
            "expected_assignments_count"
        ]

        completed_tests_count = test_completion[
            "completed_tests_count"
        ]

        expected_tests_count = test_completion[
            "expected_tests_count"
        ]

        expected_control_count = (
            expected_assignments_count + expected_tests_count
        )

        completed_control_count = (
            completed_assignments_count + completed_tests_count
        )

        control_completion_ratio = (
            completed_control_count / expected_control_count
            if expected_control_count > 0
            else 0
        )

        row = {
            "student_id": student_id,

            "process_total_events": total_events,
            "process_active_days": active_days,
            "process_activity_span_days": round(activity_span_days, 2),

            "max_day_activity_ratio": round(max_day_activity_ratio, 4),
            "top_2_days_activity_ratio": round(top_2_days_activity_ratio, 4),
            "top_3_days_activity_ratio": round(top_3_days_activity_ratio, 4),
            "days_to_80_percent_events": days_to_80_percent_events,

            "last_period_activity_ratio": round(last_period_activity_ratio, 4),
            "last_period_days": last_period_days,

            "material_events_count": material_events_count,
            "control_events_count": control_events_count,
            "process_material_share": round(material_share, 4),
            "process_control_share": round(control_share, 4),

            "expected_control_count": expected_control_count,
            "completed_control_count": completed_control_count,
            "control_completion_ratio": round(control_completion_ratio, 4),

            "required_assignments_count": len(required_assignments),
            "required_tests_count": len(required_tests),

            "first_event_time": first_event_time,
            "last_event_time": last_event_time,
            "course_start": course_start,
            "course_end": course_end,
        }

        row.update(assignment_completion)
        row.update(test_completion)
        row.update(suspicious_assignments)
        row.update(fast_tests)
        row.update(fast_lectures)

        rows.append(row)

    features_df = pd.DataFrame(rows)

    if features_df.empty:
        return features_df

    flags_series = features_df.apply(_build_process_flags, axis=1)

    features_df["process_flags"] = flags_series.apply(
        lambda flags: "; ".join(flags) if flags else "Нет выраженных дополнительных признаков"
    )

    features_df["process_flags_count"] = flags_series.apply(len)

    features_df["process_pattern"] = flags_series.apply(
        _choose_main_process_pattern
    )

    features_df["process_pattern_description"] = features_df.apply(
        build_process_pattern_description,
        axis=1,
    )

    features_df["required_assignments_list"] = ", ".join(required_assignments)
    features_df["required_tests_list"] = ", ".join(required_tests)

    return features_df


def build_process_pattern_description(row: pd.Series) -> str:
    pattern = row.get("process_pattern", "Смешанный процессный профиль")
    flags = row.get("process_flags", "")

    total_events = int(row.get("process_total_events", 0))
    active_days = int(row.get("process_active_days", 0))

    completed_assignments = int(row.get("completed_assignments_count", 0))
    expected_assignments = int(row.get("expected_assignments_count", 0))

    completed_tests = int(row.get("completed_tests_count", 0))
    expected_tests = int(row.get("expected_tests_count", 0))

    control_completion_ratio = float(row.get("control_completion_ratio", 0))

    max_day_ratio = float(row.get("max_day_activity_ratio", 0))
    top_2_ratio = float(row.get("top_2_days_activity_ratio", 0))
    top_3_ratio = float(row.get("top_3_days_activity_ratio", 0))
    days_to_80 = int(row.get("days_to_80_percent_events", 0))

    fast_lecture_count = int(row.get("fast_lecture_completion_count", 0))
    measured_lecture_count = int(row.get("measured_lecture_completion_count", 0))
    fast_lecture_ratio = float(row.get("fast_lecture_completion_ratio", 0))

    fast_test_count = int(row.get("fast_test_completion_count", 0))
    measured_test_count = int(row.get("measured_test_completion_count", 0))

    suspicious_upload_count = int(
        row.get("suspicious_first_assignment_upload_count", 0)
    )

    measured_first_upload_count = int(
        row.get("measured_first_assignment_upload_count", 0)
    )

    base = ""

    if pattern == "Неполное выполнение контрольных активностей":
        base = (
            "Студент выполнил не все обязательные контрольные активности курса. "
            f"Выполнено заданий: {completed_assignments} из {expected_assignments}. "
            f"Выполнено тестов: {completed_tests} из {expected_tests}. "
            f"Общая полнота выполнения контрольных активностей: "
            f"{control_completion_ratio:.2f}."
        )

    elif pattern == "Авральное прохождение курса":
        base = (
            "Большая часть действий студента была выполнена в короткий промежуток времени. "
            f"80% всех действий сделаны за {days_to_80} активн. дн.; "
            f"доля действий в два самых активных дня: {top_2_ratio:.2f}; "
            f"доля действий в три самых активных дня: {top_3_ratio:.2f}."
        )

    elif pattern == "Формально комплексное прохождение":
        base = (
            "Студент использовал разные элементы курса, однако активность сильно "
            "сконцентрирована во времени. "
            f"Доля действий в самый активный день: {max_day_ratio:.2f}; "
            f"80% действий сделаны за {days_to_80} активн. дн."
        )

    elif pattern == "Формальное прохождение лекционных элементов":
        base = (
            "У студента обнаружены признаки формального прохождения лекций. "
            f"Быстро завершённых лекций: {fast_lecture_count} из "
            f"{measured_lecture_count}; доля быстрых лекций: {fast_lecture_ratio:.2f}."
        )

    elif pattern == "Быстрое прохождение тестов":
        base = (
            "У студента обнаружены тесты, завершённые за очень короткое время. "
            f"Быстрых завершений тестов: {fast_test_count} из "
            f"{measured_test_count}."
        )

    elif pattern == "Подозрительно быстрая первая загрузка ответов":
        base = (
            "У студента обнаружены случаи, когда задание впервые открывалось "
            "и ответ загружался почти сразу. "
            f"Таких случаев: {suspicious_upload_count} из "
            f"{measured_first_upload_count} измеренных первых загрузок."
        )

    elif pattern == "Регулярное прохождение":
        base = (
            "Активность студента распределена относительно равномерно. "
            f"Активных дней: {active_days}; всего действий: {total_events}; "
            f"80% действий сделаны за {days_to_80} активн. дн."
        )

    elif pattern == "Эпизодическое прохождение":
        base = (
            "Студент редко взаимодействовал с курсом. "
            f"Активных дней: {active_days}; всего действий: {total_events}."
        )

    else:
        base = (
            "Поведение студента не относится к одному явно выраженному процессному типу. "
            f"Активных дней: {active_days}; всего действий: {total_events}; "
            f"80% действий сделаны за {days_to_80} активн. дн."
        )

    if flags and flags != "Нет выраженных дополнительных признаков":
        base += f" Дополнительные выявленные признаки: {flags}."

    return base


def merge_resource_and_process_patterns(
    resource_patterns_df: pd.DataFrame,
    process_features_df: pd.DataFrame,
) -> pd.DataFrame:
    if process_features_df is None or process_features_df.empty:
        return pd.DataFrame()

    process_df = process_features_df.copy()
    process_df["student_id"] = process_df["student_id"].astype(str)

    if resource_patterns_df is None or resource_patterns_df.empty:
        result_df = process_df.copy()
        result_df["cluster"] = None
        result_df["resource_pattern"] = None
    else:
        resource_df = resource_patterns_df.copy()
        resource_df["student_id"] = resource_df["student_id"].astype(str)

        resource_df = resource_df.rename(
            columns={
                "suggested_name": "resource_pattern",
            }
        )

        keep_columns = [
            col for col in ["student_id", "cluster", "resource_pattern"]
            if col in resource_df.columns
        ]

        result_df = process_df.merge(
            resource_df[keep_columns],
            on="student_id",
            how="left",
        )

    result_df["final_behavior_pattern"] = result_df.apply(
        build_final_behavior_pattern,
        axis=1,
    )

    result_df["final_behavior_description"] = result_df.apply(
        build_final_behavior_description,
        axis=1,
    )

    return result_df


def build_final_behavior_pattern(row: pd.Series) -> str:
    resource_pattern = row.get("resource_pattern")
    process_pattern = row.get("process_pattern")

    if pd.isna(resource_pattern) or not resource_pattern:
        return str(process_pattern)

    return f"{resource_pattern} + {process_pattern}"


def build_final_behavior_description(row: pd.Series) -> str:
    resource_pattern = row.get("resource_pattern")
    process_pattern = row.get("process_pattern")
    process_description = row.get("process_pattern_description")

    if pd.isna(resource_pattern) or not resource_pattern:
        return str(process_description)

    return (
        f"По ресурсному профилю студент относится к группе: «{resource_pattern}». "
        f"Процессный анализ уточняет характер прохождения: «{process_pattern}». "
        f"{process_description}"
    )