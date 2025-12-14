import streamlit as st

from ui.sidebar import render_sidebar
from utils.session import dataframe_signature, ensure_session_state


def load_dataframes(debug_mode: bool):
    """
    Load df_A/df_B/df_init from session, render sidebar, and compute change flags.

    Returns:
        tuple: (df_a, df_b, dataset_changed, df_b_changed, df_init)
    """

    ensure_session_state()
    render_sidebar(show_debug=debug_mode)

    df_a = st.session_state["df_A_data"]
    df_b = st.session_state["df_B_data"]
    df_init = st.session_state["df_init_data"]

    sig_a = dataframe_signature(df_a, st.session_state.get("csv_path", ""))
    sig_a_prev = st.session_state.get("df_A_signature", "")
    dataset_changed = sig_a != sig_a_prev
    st.session_state["df_A_signature"] = sig_a

    sig_b = dataframe_signature(df_b, st.session_state.get("csv_b_path", ""))
    sig_b_prev = st.session_state.get("df_B_signature", "")
    df_b_changed = sig_b != sig_b_prev
    st.session_state["df_B_signature"] = sig_b

    # df_init은 table loading 될 때 수정 되는 값임
    return df_a, df_b, dataset_changed, df_b_changed, df_init


__all__ = ["load_dataframes"]
