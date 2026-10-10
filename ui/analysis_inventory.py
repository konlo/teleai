"""Independent sidebar inventory; reruns never resume a pending analysis."""
import streamlit as st
from core.analysis_agent.table_inventory import fetch_table_inventory


@st.fragment
def render_table_inventory(backend, diagnostics):
    identity = (backend.name, backend.config.identity())
    if st.button('테이블 목록 조회', key='v1_table_inventory', width='stretch'):
        with st.spinner('테이블 목록을 조회하고 있습니다…'):
            frame, report = fetch_table_inventory(backend, diagnostics)
        st.session_state['v1_table_inventory_result'] = (identity, frame, report)
    saved = st.session_state.get('v1_table_inventory_result')
    if not saved or saved[0] != identity:
        return
    _, frame, report = saved
    if report['status'] == 'FAIL':
        st.error(f"테이블 목록 조회 실패 ({report['error_type']}, 오류 ID: {report['error_id']})")
        st.caption('실패 단계: ' + report['stage'])
        st.caption('Databricks에서는 DATABRICKS_CATALOG·DATABRICKS_SCHEMA와 조회 권한을 확인해주세요.'
                   if backend.name == 'databricks' else 'MySQL 접속 설정과 조회 권한을 확인해주세요.')
        return
    st.caption('조회 범위: ' + report['scope'])
    st.caption('조회 시각(UTC): ' + report['observed_at'])
    if report['truncated']:
        st.warning(f"목록이 {report['limit']}건을 넘습니다. 앞의 {report['limit']}건만 표시합니다.")
    elif frame.empty:
        st.info('해당 범위에서 조회 가능한 테이블이 없습니다. 설정과 접근 권한을 확인해주세요.')
    else:
        st.caption(f"조회된 테이블 {report['rows']}개")
    if not frame.empty:
        order = ['table_name', 'table_type', 'table_schema']
        if 'table_catalog' in frame.columns:
            order.append('table_catalog')
        st.dataframe(frame[order], hide_index=True, width='stretch')
