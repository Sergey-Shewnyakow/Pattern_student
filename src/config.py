REQUIRED_INTERNAL_COLUMNS = ["student_id", "timestamp", "activity"]

COLUMN_ALIASES = {
    "student_id": ["ФИО", "ФИО хэш"],
    "timestamp": ["Время"],
    "activity": ["Название события"],
    "context": ["Контекст события"],
    "component": ["Компонент"],
}

DEFAULT_SESSION_GAP_MINUTES = 120