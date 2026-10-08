"""Open the supported persistent analysis agent from the Telly page URL."""

from pathlib import Path
import runpy

import streamlit as st

from ui.agent_entry import runtime_compatibility_error


error = runtime_compatibility_error()
if error:
    st.set_page_config(page_title="Telly · 실행 환경 확인", page_icon="📊")
    st.error(error)
    st.stop()

runpy.run_path(str(Path(__file__).resolve().parents[1] / "ui" / "analysis_page.py"), run_name="__main__")
