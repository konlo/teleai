"""Explicit SQL-only connection check for environments without shared logs."""
import streamlit as st
from core.analysis_agent.connection_probe import probe_databricks
from core.analysis_agent.diagnostics import Diagnostics


def render_database_connection_check(backend, root):
    if backend.name != 'databricks':
        return
    identity = backend.config.identity()
    with st.expander('Databricks 연결 확인'):
        st.caption('모델 호출 없이 SELECT 1을 한 번 실행합니다. 분석 데이터는 변경하지 않습니다.')
        if st.button('DB 연결만 확인', key='v1_database_connection_check'):
            with st.spinner('Databricks 연결을 확인하고 있습니다…'):
                report = probe_databricks(backend.config)
            Diagnostics(root).emit('database_connection_probe', **report)
            st.session_state['v1_database_connection_report'] = (identity, report)
        saved = st.session_state.get('v1_database_connection_report')
        if saved and saved[0] == identity:
            report = saved[1]
            if report['status'] == 'PASS':
                st.success('Databricks 연결 및 SELECT 1 조회가 성공했습니다.')
            else:
                st.error('Databricks 연결 확인 실패: ' + report['stage'])
            st.json(report, expanded=False)
