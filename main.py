# """
# Compatibility entry point so `streamlit run main.py` still works.
# """

# import importlib
# import sys
# from pathlib import Path


# ROOT = Path(__file__).resolve().parent
# if str(ROOT) not in sys.path:
#     sys.path.insert(0, str(ROOT))

# importlib.import_module("app.pages.06_EDA_external_location")
import streamlit as st

st.title("Data Science")