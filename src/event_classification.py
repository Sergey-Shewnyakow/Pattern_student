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
    "обновлен",
    "обновлена",
    "обновлено",
    "сохранена автоматически",
    "поставлена оценка",
    "оценен",
    "оценивания",
    "отчет",
    "зачислен",
    "роль назначена",
    "член группы",
    "создан способ зачисления",
    "подтверждение удаления",
    "редактирования",
    "список экземпляров модуля",
    "опубликована некоторая информация",
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

    ("модуль курса просмотрен", "Просмотр модуля"),
    ("содержимое страницы просмотрено", "Просмотр страницы"),

    ("работа представлена", "Сдача задания"),
    ("представлен ответ", "Отправка ответа"),
    ("файл был загружен", "Загрузка файла"),
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


def classify_event(component: str, activity: str) -> str:
    component = str(component).strip() if component is not None else ""
    activity_l = str(activity).strip().lower() if activity is not None else ""

    if component in SYSTEM_COMPONENTS:
        return "system"

    if any(keyword in activity_l for keyword in SYSTEM_EVENT_KEYWORDS):
        return "system"

    # специальные события, которые лучше исключить из human-модели
    if "ответ удален" in activity_l:
        return "system"
    if "комментарий удален" in activity_l:
        return "system"

    if component in HUMAN_COMPONENTS:
        return "human"

    for pattern, _ in HUMAN_EVENT_PATTERNS:
        if pattern in activity_l:
            return "human"

    return "other"


def normalize_human_activity(activity: str) -> str:
    activity_l = str(activity).strip().lower() if activity is not None else ""

    for pattern, normalized in HUMAN_EVENT_PATTERNS:
        if pattern in activity_l:
            return normalized

    return "Прочее действие"


def add_event_type_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result["event_type"] = result.apply(
        lambda row: classify_event(
            row.get("component"),
            row.get("activity"),
        ),
        axis=1,
    )

    result["human_activity"] = result["activity"].apply(normalize_human_activity)

    return result


def split_event_types(df: pd.DataFrame):
    df = add_event_type_columns(df)

    human_df = df[df["event_type"] == "human"].copy()
    system_df = df[df["event_type"] == "system"].copy()
    other_df = df[df["event_type"] == "other"].copy()

    return df, human_df, system_df, other_df