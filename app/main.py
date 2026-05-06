import streamlit as st

st.set_page_config(page_title="Student Behavior Analysis", layout="wide")

st.title("Анализ паттернов поведения студентов")
st.write(
    "Используйте боковое меню слева для перехода между этапами анализа:\n"
    "- Data Preparation\n"
    "- KMeans Clustering"
)