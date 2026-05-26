import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.state import init_session_state
from src.ui_styles import apply_global_styles

apply_global_styles()
st.set_page_config(page_title="Анализ курса для методистов", layout="wide")
init_session_state()


# ============================================================
# Вспомогательные функции
# ============================================================

def get_base_log() -> pd.DataFrame:
    """
    Берём human events, если они есть.
    Если их нет, используем общий лог df_sessions.
    """
    df_human = st.session_state.get("df_human_events")
    df_sessions = st.session_state.get("df_sessions")

    if df_human is not None and len(df_human) > 0:
        return df_human.copy()

    if df_sessions is not None and len(df_sessions) > 0:
        return df_sessions.copy()

    return pd.DataFrame()


def prepare_course_log(df: pd.DataFrame) -> pd.DataFrame:
    """
    Подготовка лога для анализа курса.
    Добавляет:
    - course_item: элемент курса;
    - action_group: укрупнённый тип действия;
    - date, week, hour, weekday;
    - gap_min: интервал между действиями студента.
    """
    df = df.copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])

    # Элемент курса лучше брать из context, если он есть.
    # Если context нет, используем component + activity.
    if "context" in df.columns:
        df["course_item"] = df["context"].fillna("").astype(str).str.strip()
        df.loc[df["course_item"] == "", "course_item"] = np.nan
    else:
        df["course_item"] = np.nan

    if "component" in df.columns:
        component = df["component"].fillna("").astype(str)
    else:
        component = pd.Series([""] * len(df), index=df.index)

    activity = df["activity"].fillna("").astype(str)

    df["course_item"] = df["course_item"].fillna(component + " — " + activity)
    df["course_item"] = df["course_item"].astype(str).str.strip()

    # Нормализованное действие: если human_activity есть, используем его.
    if "human_activity" in df.columns:
        df["action_group"] = df["human_activity"].fillna(df["activity"]).astype(str)
    else:
        df["action_group"] = df["activity"].astype(str)

    df["component"] = component
    df["activity"] = activity

    df["date"] = df["timestamp"].dt.date
    df["week"] = df["timestamp"].dt.isocalendar().week.astype(int)
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.weekday
    df["weekday_name"] = df["timestamp"].dt.day_name()

    df = df.sort_values(["student_id", "timestamp"])
    df["prev_timestamp"] = df.groupby("student_id")["timestamp"].shift(1)
    df["gap_min"] = (
        (df["timestamp"] - df["prev_timestamp"])
        .dt.total_seconds()
        .div(60)
    )

    # Чтобы огромные паузы между разными днями не портили анализ скорости,
    # для "скорости прохождения" оставляем разумные интервалы.
    df["gap_min_clean"] = df["gap_min"]
    df.loc[df["gap_min_clean"] < 0, "gap_min_clean"] = np.nan
    df.loc[df["gap_min_clean"] > 24 * 60, "gap_min_clean"] = np.nan

    return df


def safe_metric(value, digits=2):
    if pd.isna(value):
        return "—"
    if isinstance(value, (int, np.integer)):
        return f"{value}"
    return f"{value:.{digits}f}"


def build_popularity_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Таблица востребованности элементов курса.
    """
    agg = (
        df.groupby("course_item")
        .agg(
            events_count=("activity", "count"),
            students_count=("student_id", "nunique"),
            components_count=("component", "nunique"),
            actions_count=("action_group", "nunique"),
            first_visit=("timestamp", "min"),
            last_visit=("timestamp", "max"),
            median_gap_min=("gap_min_clean", "median"),
            avg_gap_min=("gap_min_clean", "mean"),
        )
        .reset_index()
    )

    total_students = df["student_id"].nunique()
    if total_students > 0:
        agg["student_coverage_pct"] = agg["students_count"] / total_students * 100
    else:
        agg["student_coverage_pct"] = 0

    agg = agg.sort_values(
        ["students_count", "events_count"],
        ascending=False
    )

    return agg


def build_component_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Анализ по типам элементов курса: тесты, задания, лекции, страницы и т.д.
    """
    component_df = (
        df.groupby("component")
        .agg(
            events_count=("activity", "count"),
            students_count=("student_id", "nunique"),
            unique_items=("course_item", "nunique"),
            median_gap_min=("gap_min_clean", "median"),
            avg_gap_min=("gap_min_clean", "mean"),
        )
        .reset_index()
        .sort_values("events_count", ascending=False)
    )

    return component_df


def build_assignment_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Анализ заданий, тестов и отправок.
    Берём события, которые похожи на выполнение:
    - задание;
    - тест;
    - попытка;
    - ответ;
    - файл;
    - представлена работа.
    """
    keywords = [
        "задание",
        "тест",
        "попытка",
        "ответ",
        "файл",
        "представлена",
        "сдача",
        "отправка",
        "quiz",
        "assignment",
        "attempt",
        "submit",
        "submitted",
    ]

    text = (
            df["component"].astype(str).str.lower()
            + " "
            + df["activity"].astype(str).str.lower()
            + " "
            + df["action_group"].astype(str).str.lower()
            + " "
            + df["course_item"].astype(str).str.lower()
    )

    mask = text.apply(lambda x: any(k in x for k in keywords))
    task_df = df[mask].copy()

    if task_df.empty:
        return pd.DataFrame()

    result = (
        task_df.groupby("course_item")
        .agg(
            events_count=("activity", "count"),
            students_count=("student_id", "nunique"),
            first_action=("timestamp", "min"),
            last_action=("timestamp", "max"),
            median_gap_min=("gap_min_clean", "median"),
            avg_gap_min=("gap_min_clean", "mean"),
            unique_actions=("action_group", "nunique"),
        )
        .reset_index()
        .sort_values(["students_count", "events_count"], ascending=False)
    )

    return result


def build_return_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Элементы курса, к которым студенты чаще возвращаются.
    Считаем возвратом повторное посещение одного course_item одним student_id.
    """
    item_visits = (
        df.groupby(["student_id", "course_item"])
        .agg(
            visits=("activity", "count"),
            first_visit=("timestamp", "min"),
            last_visit=("timestamp", "max"),
        )
        .reset_index()
    )

    item_visits["has_return"] = item_visits["visits"] > 1

    returns = (
        item_visits.groupby("course_item")
        .agg(
            students_count=("student_id", "nunique"),
            students_with_returns=("has_return", "sum"),
            avg_visits_per_student=("visits", "mean"),
            max_visits_by_student=("visits", "max"),
        )
        .reset_index()
    )

    returns["return_ratio_pct"] = np.where(
        returns["students_count"] > 0,
        returns["students_with_returns"] / returns["students_count"] * 100,
        0,
    )

    returns = returns.sort_values(
        ["return_ratio_pct", "avg_visits_per_student"],
        ascending=False
    )

    return returns


def build_sequence_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Частые переходы между действиями.
    Полезно для понимания того, как студенты реально проходят курс.
    """
    temp = df.sort_values(["student_id", "timestamp"]).copy()
    temp["next_item"] = temp.groupby("student_id")["course_item"].shift(-1)
    temp["next_action"] = temp.groupby("student_id")["action_group"].shift(-1)

    transitions = temp.dropna(subset=["next_item"]).copy()
    transitions["transition"] = (
            transitions["course_item"].astype(str)
            + " → "
            + transitions["next_item"].astype(str)
    )

    result = (
        transitions.groupby("transition")
        .agg(
            transitions_count=("transition", "count"),
            students_count=("student_id", "nunique"),
        )
        .reset_index()
        .sort_values("transitions_count", ascending=False)
    )

    return result


def generate_methodist_conclusions(
        popularity_df: pd.DataFrame,
        rare_df: pd.DataFrame,
        component_df: pd.DataFrame,
        assignment_df: pd.DataFrame,
        return_df: pd.DataFrame,
) -> list:
    """
    Автоматические текстовые выводы для методистов.
    """
    conclusions = []

    if not popularity_df.empty:
        top = popularity_df.iloc[0]
        conclusions.append(
            f"Самый посещаемый элемент курса: «{top['course_item']}». "
            f"Его открывали {int(top['students_count'])} студентов, "
            f"всего зафиксировано {int(top['events_count'])} событий."
        )

    if not rare_df.empty:
        rare_count = len(rare_df)
        conclusions.append(
            f"Найдено {rare_count} элементов курса с низкой посещаемостью. "
            f"Их стоит проверить: возможно, они плохо заметны студентам, "
            f"необязательны или расположены в неудобном месте курса."
        )

    if not component_df.empty:
        top_component = component_df.iloc[0]
        conclusions.append(
            f"Наиболее используемый тип элемента курса: «{top_component['component']}». "
            f"По нему зафиксировано {int(top_component['events_count'])} событий."
        )

    if not assignment_df.empty:
        most_used_task = assignment_df.iloc[0]
        conclusions.append(
            f"Самое активное задание/тест: «{most_used_task['course_item']}». "
            f"С ним взаимодействовали {int(most_used_task['students_count'])} студентов."
        )

    if not return_df.empty:
        top_return = return_df.iloc[0]
        conclusions.append(
            f"Чаще всего студенты возвращаются к элементу «{top_return['course_item']}». "
            f"Доля студентов с повторными посещениями: {top_return['return_ratio_pct']:.1f}%."
        )

    if not conclusions:
        conclusions.append(
            "Недостаточно данных для автоматических выводов. "
            "Проверьте, что лог содержит student_id, timestamp, activity, component и context."
        )

    return conclusions


# ============================================================
# Интерфейс страницы
# ============================================================

st.title("Анализ курса для методистов")

st.write(
    """
    Эта страница помогает оценить качество курса по цифровому следу студентов:
    какие элементы открывали чаще всего, какие почти не использовали,
    где студенты возвращались повторно, какие задания выполнялись активнее,
    и как распределялась активность по времени.
    """
)

df = get_base_log()

if df.empty:
    st.warning(
        "Сначала подготовьте данные на странице Data Preparation. "
        "Нужно загрузить лог и выполнить предобработку."
    )
    st.stop()

required_cols = {"student_id", "timestamp", "activity"}
missing = required_cols - set(df.columns)

if missing:
    st.error(
        "Для анализа курса не хватает обязательных колонок: "
        + ", ".join(sorted(missing))
    )
    st.stop()

df = prepare_course_log(df)

if df.empty:
    st.warning("После подготовки не осталось событий для анализа.")
    st.stop()

# ============================================================
# Фильтры
# ============================================================

st.sidebar.header("Фильтры анализа курса")

min_date = df["timestamp"].min().date()
max_date = df["timestamp"].max().date()

date_range = st.sidebar.date_input(
    "Период анализа",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
    df = df[
        (df["timestamp"].dt.date >= start_date)
        & (df["timestamp"].dt.date <= end_date)
        ].copy()

components = sorted(df["component"].dropna().astype(str).unique().tolist())
selected_components = st.sidebar.multiselect(
    "Типы элементов курса",
    options=components,
    default=components,
)

if selected_components:
    df = df[df["component"].astype(str).isin(selected_components)].copy()

search_text = st.sidebar.text_input(
    "Поиск по названию элемента курса",
    value="",
)

if search_text.strip():
    df = df[
        df["course_item"]
        .astype(str)
        .str.lower()
        .str.contains(search_text.strip().lower(), na=False)
    ].copy()

if df.empty:
    st.warning("По выбранным фильтрам нет данных.")
    st.stop()

# ============================================================
# Основные метрики
# ============================================================

st.subheader("Общая картина курса")

total_events = len(df)
total_students = df["student_id"].nunique()
total_items = df["course_item"].nunique()
total_components = df["component"].nunique()
period_days = max((df["timestamp"].max() - df["timestamp"].min()).days, 1)

m1, m2, m3, m4, m5 = st.columns(5)

m1.metric("Событий", total_events)
m2.metric("Студентов", total_students)
m3.metric("Элементов курса", total_items)
m4.metric("Типов элементов", total_components)
m5.metric("Период, дней", period_days)

st.caption(
    "Событие — это действие студента в курсе: просмотр страницы, начало теста, "
    "сдача задания, просмотр лекции и т.п."
)

# ============================================================
# Агрегированные таблицы
# ============================================================

popularity_df = build_popularity_table(df)
component_df = build_component_table(df)
assignment_df = build_assignment_table(df)
return_df = build_return_table(df)
sequence_df = build_sequence_table(df)

low_coverage_threshold = st.slider(
    "Порог низкой посещаемости элемента курса, %",
    min_value=1,
    max_value=50,
    value=20,
    step=1,
)

rare_df = popularity_df[
    popularity_df["student_coverage_pct"] <= low_coverage_threshold
    ].copy()

# ============================================================
# Вкладки
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Популярность элементов",
        "Редко используемые элементы",
        "Задания и тесты",
        "Возвраты и сложности",
        "Активность во времени",
        "Выводы для методистов",
    ]
)

# ------------------------------------------------------------
# 1. Популярность элементов
# ------------------------------------------------------------

with tab1:
    st.subheader("Куда студенты заходили чаще всего")

    top_n = st.slider(
        "Сколько элементов показать",
        min_value=5,
        max_value=50,
        value=15,
        step=5,
        key="top_items_slider",
    )

    top_items = popularity_df.head(top_n).copy()

    fig = px.bar(
        top_items.sort_values("events_count"),
        x="events_count",
        y="course_item",
        orientation="h",
        hover_data=[
            "students_count",
            "student_coverage_pct",
            "median_gap_min",
        ],
        title="Самые посещаемые элементы курса",
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        popularity_df[
            [
                "course_item",
                "events_count",
                "students_count",
                "student_coverage_pct",
                "components_count",
                "actions_count",
                "median_gap_min",
                "first_visit",
                "last_visit",
            ]
        ],
        use_container_width=True,
    )

    st.info(
        """
        Как читать таблицу:
        - events_count — сколько всего действий было связано с элементом;
        - students_count — сколько уникальных студентов открывали элемент;
        - student_coverage_pct — доля студентов, которые взаимодействовали с элементом;
        - median_gap_min — типичный интервал между действиями перед этим элементом.
        """
    )

# ------------------------------------------------------------
# 2. Редко используемые элементы
# ------------------------------------------------------------

with tab2:
    st.subheader("Куда студенты заходили реже всего")

    st.write(
        f"Элементы с охватом студентов не выше **{low_coverage_threshold}%**."
    )

    if rare_df.empty:
        st.success("Редко используемые элементы по выбранному порогу не найдены.")
    else:
        fig = px.bar(
            rare_df.sort_values("student_coverage_pct").head(30),
            x="student_coverage_pct",
            y="course_item",
            orientation="h",
            hover_data=["events_count", "students_count"],
            title="Элементы с низкой посещаемостью",
        )
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            rare_df[
                [
                    "course_item",
                    "events_count",
                    "students_count",
                    "student_coverage_pct",
                    "first_visit",
                    "last_visit",
                ]
            ].sort_values(["student_coverage_pct", "events_count"]),
            use_container_width=True,
        )

        st.warning(
            """
            Методическая интерпретация:
            если важный материал имеет низкую посещаемость, его стоит проверить.
            Возможные причины: элемент плохо виден, не связан с заданиями,
            расположен слишком поздно, не имеет понятного названия или не требуется для прохождения курса.
            """
        )

# ------------------------------------------------------------
# 3. Задания и тесты
# ------------------------------------------------------------

with tab3:
    st.subheader("Какие задания и тесты выполняли активнее")

    if assignment_df.empty:
        st.warning(
            "Не найдено событий, похожих на выполнение заданий или тестов. "
            "Проверьте названия component/activity в исходном логе."
        )
    else:
        fig = px.bar(
            assignment_df.head(20).sort_values("students_count"),
            x="students_count",
            y="course_item",
            orientation="h",
            hover_data=[
                "events_count",
                "unique_actions",
                "median_gap_min",
                "avg_gap_min",
            ],
            title="Задания и тесты по числу студентов",
        )
        fig.update_layout(height=650)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            assignment_df[
                [
                    "course_item",
                    "students_count",
                    "events_count",
                    "unique_actions",
                    "median_gap_min",
                    "avg_gap_min",
                    "first_action",
                    "last_action",
                ]
            ],
            use_container_width=True,
        )

        st.info(
            """
            Для методиста здесь важны два показателя:
            - students_count показывает, сколько студентов дошло до задания или теста;
            - median_gap_min показывает, насколько быстро обычно происходило следующее действие.

            Если задание имеет низкий охват, значит часть студентов могла до него не дойти.
            Если около задания большие интервалы, возможно, оно вызывает затруднения.
            """
        )

# ------------------------------------------------------------
# 4. Возвраты и сложности
# ------------------------------------------------------------

with tab4:
    st.subheader("К каким материалам студенты возвращались чаще всего")

    if return_df.empty:
        st.warning("Недостаточно данных для анализа возвратов.")
    else:
        fig = px.scatter(
            return_df.head(50),
            x="return_ratio_pct",
            y="avg_visits_per_student",
            size="students_count",
            hover_name="course_item",
            hover_data=[
                "students_count",
                "students_with_returns",
                "max_visits_by_student",
            ],
            title="Повторные посещения элементов курса",
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            return_df[
                [
                    "course_item",
                    "students_count",
                    "students_with_returns",
                    "return_ratio_pct",
                    "avg_visits_per_student",
                    "max_visits_by_student",
                ]
            ],
            use_container_width=True,
        )

        st.info(
            """
            Частые возвраты можно трактовать по-разному:
            - положительно: материал полезный, студенты используют его как опорный;
            - проблемно: материал сложный, непонятный или нужен для выполнения задания;
            - организационно: студенты вынуждены возвращаться из-за структуры курса.
            """
        )

    st.subheader("Частые переходы между элементами курса")

    if sequence_df.empty:
        st.warning("Недостаточно данных для анализа переходов.")
    else:
        st.dataframe(
            sequence_df.head(50),
            use_container_width=True,
        )

        st.caption(
            "Эта таблица показывает реальные переходы студентов между элементами курса. "
            "Она помогает понять, совпадает ли фактическая траектория с логикой, "
            "которую предполагал преподаватель."
        )

# ------------------------------------------------------------
# 5. Активность во времени
# ------------------------------------------------------------

with tab5:
    st.subheader("Как распределялась активность студентов во времени")

    daily_df = (
        df.groupby("date")
        .agg(
            events_count=("activity", "count"),
            students_count=("student_id", "nunique"),
        )
        .reset_index()
    )

    fig_daily = px.line(
        daily_df,
        x="date",
        y=["events_count", "students_count"],
        title="Динамика активности по дням",
        markers=True,
    )
    st.plotly_chart(fig_daily, use_container_width=True)

    hour_df = (
        df.groupby("hour")
        .agg(events_count=("activity", "count"))
        .reset_index()
    )

    fig_hour = px.bar(
        hour_df,
        x="hour",
        y="events_count",
        title="Активность по часам суток",
    )
    st.plotly_chart(fig_hour, use_container_width=True)

    weekday_df = (
        df.groupby("weekday")
        .agg(events_count=("activity", "count"))
        .reset_index()
    )

    weekday_map = {
        0: "Понедельник",
        1: "Вторник",
        2: "Среда",
        3: "Четверг",
        4: "Пятница",
        5: "Суббота",
        6: "Воскресенье",
    }

    weekday_df["weekday_name_ru"] = weekday_df["weekday"].map(weekday_map)

    fig_weekday = px.bar(
        weekday_df,
        x="weekday_name_ru",
        y="events_count",
        title="Активность по дням недели",
    )
    st.plotly_chart(fig_weekday, use_container_width=True)

    st.subheader("Активность по типам элементов курса")

    st.dataframe(component_df, use_container_width=True)

    fig_component = px.bar(
        component_df.sort_values("events_count"),
        x="events_count",
        y="component",
        orientation="h",
        hover_data=["students_count", "unique_items", "median_gap_min"],
        title="Использование типов элементов курса",
    )
    fig_component.update_layout(height=550)
    st.plotly_chart(fig_component, use_container_width=True)

# ------------------------------------------------------------
# 6. Выводы для методистов
# ------------------------------------------------------------

with tab6:
    st.subheader("Автоматические выводы")

    conclusions = generate_methodist_conclusions(
        popularity_df=popularity_df,
        rare_df=rare_df,
        component_df=component_df,
        assignment_df=assignment_df,
        return_df=return_df,
    )

    for i, conclusion in enumerate(conclusions, start=1):
        st.markdown(f"**{i}. {conclusion}**")

    st.subheader("Методическая интерпретация")

    st.markdown(
        """
        На основе анализа цифрового следа можно оценить не только активность студентов,
        но и качество организации курса.

        **Что стоит проверить методисту:**

        1. **Элементы с низкой посещаемостью.**  
        Если материал важный, но его открывало мало студентов, возможно, он плохо расположен
        или не связан с обязательными заданиями.

        2. **Элементы с большим числом возвратов.**  
        Такие материалы могут быть ключевыми для обучения, но также могут указывать
        на сложность темы или недостаточно понятное объяснение.

        3. **Задания и тесты с низким охватом.**  
        Если до задания дошла только часть студентов, нужно проверить сроки,
        видимость задания, инструкции и связь с предыдущими материалами.

        4. **Пики активности.**  
        Резкие всплески активности часто связаны с дедлайнами. Если большая часть работы
        выполняется в последний момент, можно пересмотреть структуру курса,
        добавить промежуточные контрольные точки или напоминания.

        5. **Фактические переходы между элементами.**  
        Если студенты проходят курс не в той последовательности, которая предполагалась,
        стоит улучшить навигацию, названия тем или добавить пояснения.
        """
    )

    st.subheader("Краткий отчёт")

    report_text = f"""
Анализ курса выполнен по журналу действий студентов.

Всего событий: {total_events}
Всего студентов: {total_students}
Элементов курса: {total_items}
Типов элементов курса: {total_components}

Самые посещаемые элементы курса:
{popularity_df[["course_item", "events_count", "students_count", "student_coverage_pct"]].head(10).to_string(index=False)}

Редко используемые элементы курса:
{rare_df[["course_item", "events_count", "students_count", "student_coverage_pct"]].head(10).to_string(index=False) if not rare_df.empty else "Редко используемые элементы не найдены по выбранному порогу."}

Выводы:
{chr(10).join([f"- {x}" for x in conclusions])}
"""

    st.download_button(
        label="Скачать краткий отчёт TXT",
        data=report_text,
        file_name="course_methodist_report.txt",
        mime="text/plain",
    )