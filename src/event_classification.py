import pandas as pd


SYSTEM_COMPONENTS = {
    "Система",
    "Отчет по пользователю",
    "Отчет по оценкам",
    "Отчет по элементам",
    "Обзорный отчет",
    "Туры для пользователей",
}


SYSTEM_EVENT_KEYWORDS = [
    "сохранена автоматически",
    "отчет",
]


HUMAN_EVENT_PATTERNS = [
    ("начата попытка теста", "Начало теста"),
    ("попытка теста завершена", "Завершение теста"),
    ("попытка теста просмотрена", "Просмотр попытки теста"),
    ("завершенная попытка теста просмотрена", "Просмотр завершенной попытки"),
    ("сводка попытки теста просмотрена", "Просмотр сводки теста"),

    ("лекция начата", "Начало лекции"),
    ("лекция продолжена", "Продолжение лекции"),
    ("лекция закончена", "Завершение лекции"),
    ("лекция начата заново", "Повторное начало лекции"),

    ("модуль курса просмотрен", "Просмотр модуля"),
    ("содержимое страницы просмотрено", "Просмотр страницы"),
    ("курс просмотрен", "Просмотр курса"),

    ("работа представлена", "Сдача задания"),
    ("представлен ответ", "Отправка ответа"),
    ("представленный ответ обновлен", "Обновление ответа"),
    ("файл был загружен", "Загрузка файла"),
    ("пользователь продублировал свой ответ", "Дублирование ответа"),

    ("отзыв просмотрен", "Просмотр отзыва"),
    ("страница состояния представленных ответов просмотрена", "Просмотр статуса задания"),
    ("форма представления ответов просмотрена", "Просмотр формы ответа"),

    ("комментарий создан", "Создание комментария"),
    ("тема создана", "Создание темы форума"),
    ("тема просмотрена", "Просмотр форума"),
]


HUMAN_COMPONENTS = {
    "Тест",
    "Лекция",
    "Задание",
    "Ответ в виде файла",
    "Страница",
    "Форум",
    "Комментарии к ответу",
}


# Только жёсткие административные действия.
# Эти события не должны появляться у обычного студента.
#
# ВАЖНО:
# - "Пользователю поставлена оценка" НЕ включаем, потому что в Moodle это массовое событие.
# - "Просмотрено подтверждение удаления ответа" НЕ включаем, потому что оно встречается у многих студентов.
# - "Ответ удален" НЕ включаем, потому что студент может удалить/изменить свой ответ или это может быть служебная запись.
# - "Комментарий удален" НЕ включаем по той же причине.
# - "Тема создана" НЕ включаем, потому что в форуме студент может создавать темы.
# - "Событие календаря создано" НЕ включаем, потому что оно может быть пользовательским событием.
STRICT_ADMIN_EVENT_KEYWORDS = [
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


def _safe_lower(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().lower()


def _safe_str(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def classify_event(component: str, activity: str) -> str:
    """
    Делит события на:
    - human: обычная учебная активность студента;
    - system: системные/служебные события платформы;
    - other: прочие события.

    Это разделение НЕ означает автоматическое исключение пользователя.
    Исключение выполняется только по STRICT_ADMIN_EVENT_KEYWORDS.
    """
    component_s = _safe_str(component)
    activity_l = _safe_lower(activity)

    if component_s in HUMAN_COMPONENTS:
        return "human"

    for pattern, _ in HUMAN_EVENT_PATTERNS:
        if pattern in activity_l:
            return "human"

    if component_s in SYSTEM_COMPONENTS:
        return "system"

    if any(keyword in activity_l for keyword in SYSTEM_EVENT_KEYWORDS):
        return "system"

    return "other"


def normalize_human_activity(activity: str) -> str:
    """
    Преобразует длинные события Moodle в короткие названия.
    """
    activity_l = _safe_lower(activity)

    for pattern, normalized in HUMAN_EVENT_PATTERNS:
        if pattern in activity_l:
            return normalized

    return "Прочее действие"


def classify_role_event(activity: str) -> str:
    """
    Классифицирует событие с точки зрения роли пользователя:
    - admin_action: жёсткое административное действие;
    - student_or_system_action: всё остальное, не основание для исключения.
    """
    activity_l = _safe_lower(activity)

    if any(keyword in activity_l for keyword in STRICT_ADMIN_EVENT_KEYWORDS):
        return "admin_action"

    return "student_or_system_action"


def is_staff_like_event(activity: str) -> bool:
    """
    Проверяет, является ли событие основанием считать пользователя
    преподавателем/администратором.
    """
    return classify_role_event(activity) == "admin_action"


def explain_role_event(activity: str) -> str:
    """
    Возвращает понятное объяснение для таблицы исключений.
    """
    activity_l = _safe_lower(activity)

    for keyword in STRICT_ADMIN_EVENT_KEYWORDS:
        if keyword in activity_l:
            return f"Административное действие: «{keyword}»"

    return ""


def add_event_type_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет к логу колонки:
    - event_type;
    - human_activity;
    - role_event_type;
    - is_staff_like_event;
    - role_event_reason.
    """
    result = df.copy()

    result["event_type"] = result.apply(
        lambda row: classify_event(
            row.get("component"),
            row.get("activity"),
        ),
        axis=1,
    )

    result["human_activity"] = result["activity"].apply(normalize_human_activity)

    result["role_event_type"] = result["activity"].apply(classify_role_event)

    result["is_staff_like_event"] = result["role_event_type"].eq("admin_action")

    result["role_event_reason"] = result["activity"].apply(explain_role_event)

    return result


def split_event_types(df: pd.DataFrame):
    """
    Разделяет лог на:
    - все события с классификацией;
    - человеческие события;
    - системные события;
    - прочие события.
    """
    df = add_event_type_columns(df)

    human_df = df[df["event_type"] == "human"].copy()
    system_df = df[df["event_type"] == "system"].copy()
    other_df = df[df["event_type"] == "other"].copy()

    return df, human_df, system_df, other_df