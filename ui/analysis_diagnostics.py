"""On-demand local diagnostic view; never invokes a model or a database query."""
import re
import streamlit as st
from core.analysis_agent.support_report import summarize, brief


def _diagnostic_rerun():
    st.session_state['diagnostic_readonly_rerun']=True


def render_diagnostics(runtime):
    with st.expander('문제 진단 · 로그 원문 없이 확인'):
        st.caption('정상 문장처럼 보이는 오답도 확인할 수 있습니다. 진단 조회는 모델·Databricks를 호출하지 않습니다.')
        st.caption('서버 내부 로그 위치: '+str(runtime.diagnostics.path))
        reference=st.text_input('실행 ID 또는 오류 ID (비워두면 최근 실행)',key='diagnostic_reference',
                                on_change=_diagnostic_rerun).strip()
        if st.button('진단 요약 확인',key='diagnostic_summary'):
            if reference and not re.fullmatch(r'(?:[a-f0-9]{12}|[a-f0-9]{32})',reference):
                st.error('오류 ID는 12자리, 실행 ID는 32자리 영문 소문자/숫자입니다.')
                return
            report=summarize(runtime.diagnostics.path,
                             error_id=reference if len(reference)==12 else None,
                             run_id=reference if len(reference)==32 else None)
            if not report['found']:
                st.warning('이 대화의 보관된 로그에서 실행을 찾지 못했습니다. 대화 선택·ID·로그 보관 기간을 확인하세요.')
            else:
                st.caption('아래 항목 중 공유 가능한 코드와 상태만 알려주세요. 요청 원문·SQL·데이터·토큰은 이 요약에 포함하지 않습니다.')
                st.code(brief(report),language=None)
                st.json(report,expanded=False)
                st.caption('revision은 최초 runtime 생성 시 checkout이며 전체 로딩 코드 일치를 보증하지 않습니다. NO_TERMINAL_EVENT는 실행 중·강제 종료·로그 유실을 구분하지 못합니다.')
