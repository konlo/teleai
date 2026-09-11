"""Use the persistent agent in the v1 environment; preserve old sessions."""
from importlib.metadata import version
from pathlib import Path
import runpy
import streamlit as st
root=Path(__file__).resolve().parents[1]
is_v1=int(version('langchain').split('.')[0])>=1
page='analysis_page.py' if is_v1 else 'legacy_telly.py'
runpy.run_path(str(root/'ui'/page),run_name='__main__')
