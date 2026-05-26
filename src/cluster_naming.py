import pandas as pd


TARGET_CLUSTER_NAMES = {
    "complex": "Изучение всех основных элементов курса",
    "lecture": "Изучение лекционных материалов без акцента на видео",
    "video": "Преимущественный просмотр видеолекций",
    "practice_test": "Выполнение заданий и тестов при редком обращении к материалам",
    "low_activity": "Минимальная активность в курсе",
    "episodic": "Нерегулярное взаимодействие с курсом",
    "mixed": "Смешанный ресурсный профиль",
}

def _safe_float(row: pd.Series, col: str, default: float = 0.0) -> float:
    value = row.get(col, default)

    if pd.isna(value):
        return default

    return float(value)


def _activity_level(
    total_events: float,
    active_days: float,
    global_means: dict,
) -> str:
    """
    Оценивает общий уровень активности кластера.
    """
    global_total_events = float(global_means.get("total_events", 0))
    global_active_days = float(global_means.get("active_days", 0))

    if global_total_events <= 0:
        return "средний"

    if (
        total_events >= global_total_events * 1.3
        and active_days >= max(1, global_active_days)
    ):
        return "высокий"

    if (
        total_events <= global_total_events * 0.4
        or active_days <= max(1, global_active_days * 0.4)
    ):
        return "низкий"

    return "средний"


def _regularity_description(
    weekly_regularity_cv: float,
    long_pauses_over_3d: float,
) -> str:
    """
    Даёт текстовое описание регулярности.

    weekly_regularity_cv:
    - низкое значение — более равномерная активность;
    - высокое значение — более рывковая активность.
    """
    if long_pauses_over_3d >= 2:
        return "наблюдаются длительные паузы более 3 дней"

    if weekly_regularity_cv <= 0.7:
        return "активность относительно регулярная"

    if weekly_regularity_cv >= 1.5:
        return "активность выражена рывками"

    return "активность умеренно неравномерная"


def suggest_cluster_name(cluster_row: pd.Series, global_means: dict) -> tuple[str, str]:

    # ------------------------------------------------------------
    # Типы учебной активности
    # ------------------------------------------------------------
    video_share = _safe_float(cluster_row, "video_share")
    lecture_share = _safe_float(cluster_row, "lecture_share")
    practice_share = _safe_float(cluster_row, "practice_share")
    test_share = _safe_float(cluster_row, "test_share")
    page_share = _safe_float(cluster_row, "page_share")
    study_material_share = _safe_float(cluster_row, "study_material_share")
    control_activity_share = _safe_float(cluster_row, "control_activity_share")

    used_video = _safe_float(cluster_row, "used_video")
    used_lecture = _safe_float(cluster_row, "used_lecture")
    used_practice = _safe_float(cluster_row, "used_practice")
    used_test = _safe_float(cluster_row, "used_test")
    used_page = _safe_float(cluster_row, "used_page")

    material_diversity_count = _safe_float(cluster_row, "material_diversity_count")
    full_course_activity = _safe_float(cluster_row, "full_course_activity")
    practice_test_without_materials = _safe_float(
        cluster_row,
        "practice_test_without_materials",
    )

    # ------------------------------------------------------------
    # Общая активность
    # ------------------------------------------------------------
    total_events = _safe_float(cluster_row, "total_events")
    active_days = _safe_float(cluster_row, "active_days")
    active_weeks = _safe_float(cluster_row, "active_weeks")
    sessions_count = _safe_float(cluster_row, "sessions_count")
    unique_activities = _safe_float(cluster_row, "unique_activities")

    # ------------------------------------------------------------
    # Регулярность
    # ------------------------------------------------------------
    weekly_regularity_cv = _safe_float(cluster_row, "weekly_regularity_cv")
    long_pauses_over_3d = _safe_float(cluster_row, "long_pauses_over_3d")

    level = _activity_level(
        total_events=total_events,
        active_days=active_days,
        global_means=global_means,
    )

    regularity_text = _regularity_description(
        weekly_regularity_cv=weekly_regularity_cv,
        long_pauses_over_3d=long_pauses_over_3d,
    )

    behavior_text = (
        f" Уровень активности: {level}. "
        f"В среднем событий: {total_events:.1f}, "
        f"активных дней: {active_days:.1f}, "
        f"активных недель: {active_weeks:.1f}, "
        f"сессий: {sessions_count:.1f}. "
        f"{regularity_text}."
    )

    global_total_events = float(global_means.get("total_events", 0))
    global_active_days = float(global_means.get("active_days", 0))

    # ------------------------------------------------------------
    # 1. Низкая учебная активность
    # ------------------------------------------------------------
    if (
        total_events <= max(5, global_total_events * 0.25)
        or active_days <= max(1, global_active_days * 0.25)
    ):
        return (
            TARGET_CLUSTER_NAMES["low_activity"],
            (
                "Студенты редко взаимодействуют с курсом, имеют малое число действий "
                "и ограниченное количество активных дней."
                + behavior_text
            ),
        )

    # ------------------------------------------------------------
    # 2. Контрольно-практические подтипы
    # ------------------------------------------------------------

    # 2.1. Практические и тесты почти без изучения материалов
    if (
            practice_test_without_materials >= 0.5
            or (
            control_activity_share >= 0.90
            and study_material_share <= 0.03
    )
    ):
        return (
            "Выполнение заданий и тестов без изучения материалов",
            (
                    "Основная активность почти полностью связана с практическими заданиями "
                    "и тестами. Обращение к лекциям, видеолекциям и страницам курса минимально. "
                    "Такой профиль может указывать на прохождение курса преимущественно через "
                    "контрольные активности без предварительного изучения материалов."
                    + behavior_text
            ),
        )

    # 2.2. Активная практико-тестовая стратегия с обращением к лекциям
    if (
            control_activity_share >= 0.75
            and study_material_share >= 0.10
            and active_days >= global_active_days
    ):
        return (
            "Выполнение заданий и тестов с обращением к лекциям",
            (
                    "Студенты активно выполняют практические задания и проходят тесты, "
                    "но также заметно обращаются к лекционным материалам Moodle. "
                    "Профиль отличается высоким числом событий, активных дней и сессий."
                    + behavior_text
            ),
        )

    # 2.3. Преимущественное выполнение заданий и тестов
    if (
            control_activity_share >= 0.75
            and study_material_share < 0.10
    ):
        return (
            "Преимущественное выполнение заданий и тестов",
            (
                    "Основная активность связана с тестами и практическими заданиями. "
                    "Учебные материалы используются редко и выполняют вспомогательную роль."
                    + behavior_text
            ),
        )

    # 2.4. Общая контрольно-практическая стратегия
    if control_activity_share >= 0.75:
        return (
            TARGET_CLUSTER_NAMES["practice_test"],
            (
                    "Основная активность связана с практическими заданиями и тестами. "
                    "Обращение к учебным материалам выражено слабее, чем контрольная активность."
                    + behavior_text
            ),
        )

    # ------------------------------------------------------------
    # 3. Комплексное изучение курса
    # ------------------------------------------------------------
    if (
        full_course_activity >= 0.5
        or (
            used_video >= 0.5
            and used_lecture >= 0.5
            and used_practice >= 0.5
            and used_test >= 0.5
        )
        or (
            material_diversity_count >= 2.5
            and study_material_share > 0
            and control_activity_share > 0
        )
    ):
        return (
            TARGET_CLUSTER_NAMES["complex"],
            (
                "Студенты используют разные элементы курса: видеолекции, лекции Moodle, "
                "страницы, практические задания и тесты. Такой профиль отражает "
                "комплексное изучение курса."
                + behavior_text
            ),
        )

    # ------------------------------------------------------------
    # 4. Видеолекционная стратегия
    # ------------------------------------------------------------
    if (
        used_video >= 0.5
        and video_share >= lecture_share
        and video_share >= practice_share
        and video_share >= test_share
    ):
        return (
            TARGET_CLUSTER_NAMES["video"],
            (
                "Основная доля учебной активности связана с видеолекциями. "
                "Остальные элементы курса используются как дополнительные."
                + behavior_text
            ),
        )

    # ------------------------------------------------------------
    # 5. Лекционно-ориентированная стратегия
    # ------------------------------------------------------------
    if (
        used_lecture >= 0.5
        and lecture_share >= video_share
        and lecture_share >= practice_share
        and lecture_share >= test_share
    ):
        return (
            TARGET_CLUSTER_NAMES["lecture"],
            (
                "Студенты преимущественно используют лекционные материалы Moodle. "
                "Видеолекции, практические задания и тесты не являются доминирующим "
                "типом активности."
                + behavior_text
            ),
        )

    # ------------------------------------------------------------
    # 6. Эпизодическая активность
    # ------------------------------------------------------------
    if (
        long_pauses_over_3d >= 2
        or weekly_regularity_cv >= 1.5
    ):
        return (
            TARGET_CLUSTER_NAMES["episodic"],
            (
                "Студенты взаимодействуют с курсом нерегулярно: активность проявляется "
                "отдельными всплесками или сопровождается длительными паузами."
                + behavior_text
            ),
        )

    # ------------------------------------------------------------
    # 7. Смешанный профиль
    # ------------------------------------------------------------
    return (
        TARGET_CLUSTER_NAMES["mixed"],
        (
            "Кластер не имеет одного явно доминирующего типа учебной активности. "
            "Студенты используют несколько элементов курса, но без выраженного "
            "преобладания одного ресурса."
            + behavior_text
        ),
    )


def build_cluster_names(
    result_df: pd.DataFrame,
    cluster_profiles: pd.DataFrame,
) -> pd.DataFrame:
    """
    Формирует таблицу с названиями и описаниями кластеров.
    """
    global_means = {
        col: float(result_df[col].mean())
        for col in result_df.select_dtypes(include="number").columns
        if col != "cluster"
    }

    cluster_sizes = (
        result_df.groupby("cluster")
        .size()
        .reset_index(name="cluster_size")
    )

    profiles_with_size = cluster_profiles.merge(
        cluster_sizes,
        on="cluster",
        how="left",
    )

    rows = []

    for _, row in profiles_with_size.iterrows():
        cluster_id = int(row["cluster"])
        suggested_name, description = suggest_cluster_name(row, global_means)

        rows.append(
            {
                "cluster": cluster_id,
                "cluster_size": int(row["cluster_size"]),
                "suggested_name": suggested_name,
                "description": description,

                # Доли активности
                "video_share": round(float(row.get("video_share", 0)), 3),
                "lecture_share": round(float(row.get("lecture_share", 0)), 3),
                "practice_share": round(float(row.get("practice_share", 0)), 3),
                "test_share": round(float(row.get("test_share", 0)), 3),
                "page_share": round(float(row.get("page_share", 0)), 3),
                "study_material_share": round(
                    float(row.get("study_material_share", 0)),
                    3,
                ),
                "control_activity_share": round(
                    float(row.get("control_activity_share", 0)),
                    3,
                ),

                # Факт использования ресурсов
                "used_video": round(float(row.get("used_video", 0)), 3),
                "used_lecture": round(float(row.get("used_lecture", 0)), 3),
                "used_practice": round(float(row.get("used_practice", 0)), 3),
                "used_test": round(float(row.get("used_test", 0)), 3),
                "used_page": round(float(row.get("used_page", 0)), 3),
                "material_diversity_count": round(
                    float(row.get("material_diversity_count", 0)),
                    2,
                ),
                "full_course_activity": round(
                    float(row.get("full_course_activity", 0)),
                    3,
                ),
                "practice_test_without_materials": round(
                    float(row.get("practice_test_without_materials", 0)),
                    3,
                ),

                # Общая активность
                "total_events": round(float(row.get("total_events", 0)), 2),
                "active_days": round(float(row.get("active_days", 0)), 2),
                "active_weeks": round(float(row.get("active_weeks", 0)), 2),
                "sessions_count": round(float(row.get("sessions_count", 0)), 2),
                "unique_activities": round(
                    float(row.get("unique_activities", 0)),
                    2,
                ),

                # Регулярность
                "weekly_regularity_cv": round(
                    float(row.get("weekly_regularity_cv", 0)),
                    3,
                ),
                "long_pauses_over_3d": round(
                    float(row.get("long_pauses_over_3d", 0)),
                    2,
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)