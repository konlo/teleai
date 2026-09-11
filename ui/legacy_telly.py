"""Conversational analysis UI. Page rendering never queries Databricks."""
import json
from pathlib import Path
from uuid import uuid4

import streamlit as st

from core.analysis_loop import AnalysisSession
from core.analysis_runtime import CurrentAnalysisRuntime
from core.analysis_instructions import ANALYSIS_INSTRUCTIONS
from core.analysis_runtime_tools import build_runtime_tools
from utils.analysis_datasets import DatasetStore

st.set_page_config(page_title="Telly | 데이터 분석", page_icon="📊", layout="wide")
st.title("Telly · 함께 살펴보는 데이터")
st.caption("궁금한 것을 말하고, 결과를 보며 이어서 질문하세요. 추가 데이터 조회는 먼저 확인합니다.")

if "analysis_session" not in st.session_state:
    store = DatasetStore()
    session = AnalysisSession(str(uuid4()), ANALYSIS_INSTRUCTIONS, [])
    session.tools = build_runtime_tools(session, store)
    st.session_state.analysis_session = session
    st.session_state.analysis_datasets = store
    st.session_state.analysis_error = ""
    root = Path(__file__).resolve().parents[1] / ".telly_table_context" / "contexts"
    from utils.table_context import load_saved_table_context
    for path in sorted(root.glob("*.json")):
        try:
            raw = json.loads(path.read_text())
            context = load_saved_table_context(raw["table_fqn"])
            if context:
                session.reference_context.append({"table": context.table_fqn,
                    "columns": [{"name": c.name, "dtype": c.dtype,
                                 "aliases": c.aliases} for c in context.columns]})
        except (ValueError, KeyError, OSError):
            continue

session = st.session_state.analysis_session
store = st.session_state.analysis_datasets
runtime = CurrentAnalysisRuntime(session)


def model():
    if "analysis_model" not in st.session_state:
        from core.analysis_model import build_analysis_model
        st.session_state.analysis_model = build_analysis_model()
    return st.session_state.analysis_model


def execute(request):
    from utils.session import ensure_session_state, get_databricks_credentials
    from core.analysis_databricks import execute_approved
    ensure_session_state()
    return execute_approved(request, get_databricks_credentials().to_config(), store)


def run_action(action):
    st.session_state.analysis_error = ""
    try:
        with st.spinner("데이터를 살펴보고 있습니다…"):
            result = action()
        if result.get("status") == "limit_reached":
            st.session_state.analysis_error = result["text"]
    except Exception:
        st.session_state.analysis_error = "분석을 완료하지 못했습니다. 모델 연결을 확인하거나 요청을 다시 시도해주세요. 기존 결과는 유지됩니다."


with st.sidebar:
    st.subheader("분석할 자료")
    options = [item["table"] for item in session.reference_context]
    table = st.text_input("Databricks 테이블", value=options[0] if options else "",
                          placeholder="catalog.schema.table")
    if options:
        st.caption("저장된 테이블 정보: " + ", ".join(options))
    if st.button("데이터 불러오기 제안", disabled=not table.strip()):
        try:
            runtime.propose_table(table)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.rerun()
    st.divider()
    st.subheader("보유한 결과")
    if not store.metadata:
        st.caption("아직 로딩된 결과가 없습니다.")
    for info in store.metadata.values():
        with st.expander(f"{info.source} · {info.rows:,}행"):
            st.caption(f"결과 {info.id[:8]} · {info.grain} · {info.coverage}")
            st.dataframe(store.frames[info.id].head(5), hide_index=True)
    with st.expander("사용 가능한 분석 스킬"):
        from utils.analysis_skill_registry import AnalysisSkillRegistry
        for skill in AnalysisSkillRegistry().list():
            st.markdown(f"**{skill['name']}** — {skill['description']}")

if not session.history:
    st.info("예: ‘온도 분포를 추천해줘’, ‘그중 특정 모델만 비교해줘’, ‘지난달과 어떻게 달라?’")

for index, message in enumerate(runtime.events()):
    if message["role"] in {"user", "assistant"} and message.get("content"):
        with st.chat_message(message["role"]):
            content = message["content"]
            if message["role"] == "user" and content.startswith('{"approval_request_id"'):
                event = json.loads(content)
                st.write("추가 데이터 조회를 승인했습니다." if event["approved"] else "추가 조회를 취소했습니다.")
            else:
                st.markdown(content)
    elif message["role"] == "tool" and message.get("name") == "recommend_chart_images":
        result = json.loads(message["content"])
        cards = [session.artifacts[c["id"]] for c in result.get("cards", []) if c["id"] in session.artifacts]
        if cards:
            with st.chat_message("assistant"):
                st.write("이렇게 살펴볼 수 있어요")
                for container, card in zip(st.columns(len(cards)), cards):
                    with container:
                        st.markdown(f"**{card.title}**")
                        st.image(card.image, use_container_width=True)
                        st.caption(card.reason)
                        st.caption(card.scope)
                        if st.button("이 차트 선택", key=f"choose_{card.id}",
                                     disabled=session.state == "awaiting_approval"):
                            st.session_state.analysis_selected_chart = runtime.select_chart(card.id)
                            st.rerun()

selected = session.artifacts.get(st.session_state.get("analysis_selected_chart"))
if selected:
    st.subheader(selected.title)
    st.image(selected.image)
    st.caption(selected.scope)

for request in session.approvals.pending():
    with st.container(border=True):
        st.subheader("추가 데이터를 불러올까요?")
        st.write(request.reason)
        st.caption(f"조회 대상: {request.source} · 기존 결과는 유지됩니다.")
        with st.expander("실행할 조회 자세히 보기"):
            st.code(request.query, language="sql")
        yes, no = st.columns(2)
        if yes.button("불러오고 계속", key=f"approve_{request.id}", type="primary"):
            run_action(lambda: runtime.respond(request.id, execute=execute, model=model(), approved=True))
            st.rerun()
        if no.button("조회 취소", key=f"decline_{request.id}"):
            runtime.cancel(request.id)
            st.rerun()

if st.session_state.analysis_error:
    st.warning(st.session_state.analysis_error)

prompt = st.chat_input("어떤 점이 궁금하세요? 이전 결과에 이어서 질문해도 됩니다.")
if prompt:
    run_action(lambda: runtime.submit(prompt, model()))
    st.rerun()
