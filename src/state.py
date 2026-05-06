import streamlit as st


def init_session_state():
    defaults = {
        "last_filename": None,
        "last_gap_minutes": None,

        "df_raw": None,
        "df_norm": None,
        "df_clean": None,
        "df_sessions": None,
        "df_human_events": None,
        "df_system_events": None,
        "df_other_events": None,
        "features_df": None,

        "anomaly_df": None,
        "features_df_for_clustering": None,

        "pending_filtered_df": None,
        "applied_filtered_df": None,

        "k_scores_df": None,
        "clustering_result": None,

        "agg_k_scores_df": None,
        "agg_clustering_result": None,

        "gmm_k_scores_df": None,
        "gmm_clustering_result": None,

        "hdbscan_scores_df": None,
        "hdbscan_clustering_result": None,

        "pm_agg_k_scores_df": None,
        "pm_agg_clustering_result": None,

        "pm_gmm_k_scores_df": None,
        "pm_gmm_clustering_result": None,

        "pm_hdbscan_scores_df": None,
        "pm_hdbscan_clustering_result": None,

        "ae_k_scores_df": None,
        "ae_clustering_result": None,

        "dec_k_scores_df": None,
        "dec_clustering_result": None,

        "hybrid_features_df": None,
        "hybrid_k_scores_df": None,
        "hybrid_clustering_result": None,
        "hybrid_method": None,

        "cluster_pm_features_df": None,
        "cluster_pm_variants_df": None,
        "cluster_pm_trace_preview_df": None,
        "cluster_pm_heuristics_svg_path": None,
        "cluster_pm_inductive_svg_path": None,

        # process mining
        "pm_features_df": None,
        "pm_variants_df": None,
        "pm_trace_preview_df": None,
        "pm_clustering_result": None,
        "pm_k_scores_df": None,

        "pm_heuristics_svg_path": None,
        "pm_inductive_svg_path": None,

        "pm_student_heuristics_svg_path": None,
        "pm_student_inductive_svg_path": None,
        "pm_student_features_df": None,
        "pm_selected_student_log_df": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value