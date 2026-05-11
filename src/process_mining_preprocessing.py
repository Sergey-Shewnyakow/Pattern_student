import pandas as pd


ADMIN_KEYWORDS = [
    "роль назначена",
    "назначение роли снято",
    "создан способ зачисления",
    "пользователь зачислен на курс",
    "пользователь отчислен из курса",
    "член группы добавлен",
    "член группы удален",
    "группа создана",
    "курс обновлен",
    "опубликована некоторая информация",
]


EXCLUDED_KEYWORDS = [
    "отчет",
    "отчёт",
    "профиль пользователя просмотрен",
    "список пользователей просмотрен",
    "пользователю поставлена оценка",
    "представленный ответ был оценен",
    "представленный ответ был оценён",
    "оценка удалена",
    "роль назначена",
    "назначение роли снято",
    "пользователь зачислен",
    "пользователь отчислен",
    "член группы добавлен",
    "член группы удален",
    "группа создана",
]


def _safe_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().lower()


def _safe_original(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def normalize_process_activity(row: pd.Series, detail_level: str = "medium") -> str:
    """
    Нормализует события Moodle для process mining.

    Важно:
    одно и то же activity может означать разные действия,
    поэтому учитываются component + context + activity.
    """
    component = _safe_text(row.get("component"))
    context = _safe_text(row.get("context"))
    activity = _safe_text(row.get("activity"))

    # ------------------------------------------------------------
    # Крупный уровень детализации
    # ------------------------------------------------------------
    if detail_level == "coarse":
        if "курс просмотрен" in activity:
            return "Просмотр курса"

        if component == "страница" and "видеолекция" in context:
            return "Просмотр видеолекции"

        if component == "страница":
            return "Просмотр страницы курса"

        if component == "лекция":
            return "Изучение лекции"

        if component == "тест":
            return "Работа с тестом"

        if component in ["задание", "ответ в виде файла"]:
            if (
                "работа представлена" in activity
                or "представлен ответ" in activity
                or "файл был загружен" in activity
            ):
                return "Отправка задания"
            return "Работа с практическим заданием"

        if component in ["форум", "комментарии к ответу"]:
            return "Форум / коммуникация"

        if "отзыв просмотрен" in activity:
            return "Просмотр отзыва"

        return "Прочее учебное действие"

    # ------------------------------------------------------------
    # Средний уровень детализации — основной для ВКР
    # ------------------------------------------------------------

    # Курс
    if "курс просмотрен" in activity:
        return "Просмотр курса"

    # Страницы и видео
    if component == "страница" and "видеолекция" in context:
        return "Просмотр видеолекции"

    if component == "страница":
        if "содержимое страницы просмотрено" in activity:
            return "Просмотр страницы курса"
        if "модуль курса просмотрен" in activity:
            return "Открытие страницы курса"
        return "Просмотр страницы курса"

    # Лекции Moodle
    if component == "лекция":
        if "лекция начата заново" in activity:
            return "Повтор лекции"
        if "лекция начата" in activity:
            return "Начало лекции"
        if "лекция продолжена" in activity:
            return "Продолжение лекции"
        if "лекция закончена" in activity:
            return "Завершение лекции"
        if "модуль курса просмотрен" in activity:
            return "Открытие лекции"
        return "Изучение лекции"

    # Тесты
    if component == "тест":
        if "начата попытка теста" in activity:
            return "Начало теста"
        if "попытка теста завершена" in activity:
            return "Завершение теста"
        if "попытка теста просмотрена" in activity:
            return "Просмотр попытки теста"
        if "завершенная попытка теста просмотрена" in activity:
            return "Просмотр результата теста"
        if "завершённая попытка теста просмотрена" in activity:
            return "Просмотр результата теста"
        if "сводка попытки теста просмотрена" in activity:
            return "Просмотр сводки теста"
        if "модуль курса просмотрен" in activity:
            return "Открытие теста"
        return "Работа с тестом"

    # Практические задания
    if component == "задание":
        if "модуль курса просмотрен" in activity:
            return "Открытие практического задания"
        if "форма представления ответов просмотрена" in activity:
            return "Открытие формы ответа"
        if "страница состояния представленных ответов просмотрена" in activity:
            return "Просмотр статуса задания"
        if "отзыв просмотрен" in activity:
            return "Просмотр отзыва"
        if "работа представлена" in activity:
            return "Отправка практического задания"
        return "Работа с практическим заданием"

    if component == "ответ в виде файла":
        if "файл был загружен" in activity:
            return "Загрузка файла"
        if "представлен ответ" in activity:
            return "Отправка практического задания"
        if "представленный ответ обновлен" in activity:
            return "Обновление ответа"
        if "пользователь продублировал свой ответ" in activity:
            return "Дублирование ответа"
        return "Работа с ответом"

    # Форум и коммуникации
    if component == "форум":
        if "тема создана" in activity:
            return "Создание темы форума"
        if "тема просмотрена" in activity:
            return "Просмотр форума"
        return "Форум"

    if component == "комментарии к ответу":
        if "комментарий создан" in activity:
            return "Создание комментария"
        if "комментарий просмотрен" in activity:
            return "Просмотр комментария"
        return "Комментарии"

    # Отзывы
    if "отзыв просмотрен" in activity:
        return "Просмотр отзыва"

    # Детальный режим: если ничего не распознано, оставляем activity
    if detail_level == "detailed":
        original_activity = _safe_original(row.get("activity"))
        original_component = _safe_original(row.get("component"))

        if original_component:
            return f"{original_component}: {original_activity}"

        return original_activity or "Прочее действие"

    return "Прочее учебное действие"


def is_process_event_allowed(row: pd.Series) -> bool:
    """
    Фильтрует события для process mining.

    Оставляем только действия, которые могут отражать маршрут студента.
    Убираем административные, отчётные и оценочные события.
    """
    activity = _safe_text(row.get("activity"))
    component = _safe_text(row.get("component"))

    role_event_type = _safe_text(row.get("role_event_type"))

    if role_event_type == "admin_action":
        return False

    if any(keyword in activity for keyword in ADMIN_KEYWORDS):
        return False

    if any(keyword in activity for keyword in EXCLUDED_KEYWORDS):
        return False

    # Полностью служебные компоненты, если это не просмотр курса
    if component in [
        "отчет по пользователю",
        "отчет по оценкам",
        "отчет по элементам",
        "обзорный отчет",
    ]:
        return False

    return True


def collapse_consecutive_duplicates(
    event_log: pd.DataFrame,
    case_id_col: str = "student_id",
    activity_col: str = "process_activity",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Сжимает подряд идущие одинаковые действия внутри траектории студента.

    Пример:
    Работа с тестом -> Работа с тестом -> Работа с тестом
    превращается в:
    Работа с тестом
    """
    if event_log.empty:
        return event_log

    result = event_log.sort_values([case_id_col, timestamp_col]).copy()

    result["prev_activity"] = result.groupby(case_id_col)[activity_col].shift(1)

    result = result[
        result[activity_col] != result["prev_activity"]
    ].copy()

    result = result.drop(columns=["prev_activity"], errors="ignore")

    return result.reset_index(drop=True)


def build_process_event_log(
    df: pd.DataFrame,
    detail_level: str = "medium",
    collapse_duplicates: bool = True,
) -> pd.DataFrame:
    """
    Готовит event log для process mining.

    На выходе:
    - student_id;
    - timestamp;
    - process_activity;
    - component;
    - context;
    - activity.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    required_columns = ["student_id", "timestamp", "activity"]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Для process mining не хватает колонок: "
            + ", ".join(missing_columns)
        )

    event_log = df.copy()

    event_log["student_id"] = event_log["student_id"].astype(str)

    event_log = event_log[event_log.apply(is_process_event_allowed, axis=1)].copy()

    event_log["process_activity"] = event_log.apply(
        lambda row: normalize_process_activity(
            row,
            detail_level=detail_level,
        ),
        axis=1,
    )

    event_log = event_log[
        event_log["process_activity"].notna()
        & (event_log["process_activity"].astype(str).str.strip() != "")
    ].copy()

    event_log = event_log.sort_values(
        ["student_id", "timestamp"],
    ).reset_index(drop=True)

    if collapse_duplicates:
        event_log = collapse_consecutive_duplicates(event_log)

    return event_log