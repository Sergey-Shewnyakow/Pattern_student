import streamlit as st


def apply_global_styles():
    st.markdown("""
    <style>
    [data-testid="stDataFrame"] div {
        font-size: 18px !important;
    }

    [data-testid="stDataFrame"] thead tr th div {
        font-size:22px !important;
        font-weight: 600 !important;
    }

    table {
        font-size: 18px !important;
    }

    thead tr th {
        font-size: 18px !important;
        font-weight: 600 !important;
    }

    tbody tr td {
        font-size: 18px !important;
    }
    </style>
    """, unsafe_allow_html=True)