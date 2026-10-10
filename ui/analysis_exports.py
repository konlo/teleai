"""Download only scope-owned exports published by verified tool execution."""
import streamlit as st
from functools import partial

def _payload(runtime,key):
    return runtime.db.get(key,'export')[1]

def render_exports(runtime):
    entries=runtime.db.metadata('export')
    if not entries:return
    with st.expander('저장된 분석 결과 다운로드'):
        for key,entry in list(entries.items())[-10:]:
            st.download_button('다운로드: '+entry['filename'],partial(_payload,runtime,key),
                file_name=entry['filename'],mime='application/zip',key='analysis-export-'+key,
                on_click='ignore',icon=':material/download:')
