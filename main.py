"""
Streamlit entry point that lists available pages under app/pages in the sidebar.
"""

import runpy
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st


ROOT = Path(__file__).resolve().parent
PAGES_DIR = ROOT / "app" / "pages"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _normalise_label(stem: str) -> str:
    """
    Convert a filename stem like '06_EDA_external_location' into a readable label.
    """
    prefix, _, remainder = stem.partition("_")
    base = remainder if prefix.isdigit() and remainder else stem
    parts = [part for part in base.split("_") if part]
    pretty: List[str] = []
    for part in parts:
        pretty.append(part if part.isupper() else part.capitalize())
    return " ".join(pretty) if pretty else stem


def _discover_pages() -> List[Tuple[str, Path]]:
    """
    Return list of (label, path) for every Streamlit page module under app/pages.
    """
    pages: List[Tuple[str, Path]] = []
    if not PAGES_DIR.exists():
        return pages
    for file_path in sorted(PAGES_DIR.glob("*.py")):
        if file_path.name.startswith("_"):
            continue
        label = _normalise_label(file_path.stem)
        pages.append((label, file_path))
    return pages


if not st.session_state.get("_page_configured", False):
    st.set_page_config(page_title="DF Chatbot (Gemini)", page_icon="✨", layout="wide")
    st.session_state["_page_configured"] = True

available_pages = _discover_pages()
if not available_pages:
    st.error("No Streamlit pages found under app/pages.")
else:
    labels = [label for label, _ in available_pages]
    lookup: Dict[str, Path] = {label: path for label, path in available_pages}
    selected_label = st.sidebar.selectbox("Pages", labels, key="page_selector")
    runpy.run_path(str(lookup[selected_label]), run_name="__main__")
