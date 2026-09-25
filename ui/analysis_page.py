"""New local-only desktop entrypoint. Existing conversations remain in the old app."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import json
import os
import sqlite3
from uuid import uuid4
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from ui.analysis_text import display_analysis_text
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.databricks import ConnectionConfig, make_executor
from core.analysis_agent.policy import RuntimePolicy

load_dotenv(ROOT/'.env')
st.set_page_config(page_title='Telly · 분석',page_icon='📊',layout='wide')
st.title('Telly · 함께 살펴보는 데이터')
st.caption('결과를 보며 이어서 질문하세요. 추가 데이터 조회는 실행 전에 확인합니다.')
root=Path(os.getenv('TELLY_V1_STORAGE',str(ROOT/'.telly_runtime/v1')))
root.mkdir(parents=True,exist_ok=True)
# Local desktop identity; this entrypoint binds loopback and is not a multi-user service.
owner='local-owner'
with sqlite3.connect(root/'conversations.sqlite') as db:
    db.execute('CREATE TABLE IF NOT EXISTS conversations (id TEXT PRIMARY KEY,title TEXT)')
    if 'v1_conversation' not in st.session_state:
        candidate=st.query_params.get('conversation','')
        found=db.execute('SELECT id FROM conversations WHERE id=?',(candidate,)).fetchone()
        st.session_state.v1_conversation=candidate if found else str(uuid4())
    cid=st.session_state.v1_conversation
    db.execute('INSERT OR IGNORE INTO conversations VALUES (?,?)',(cid,'분석 '+cid[:8]))
    conversations=db.execute('SELECT id,title FROM conversations ORDER BY rowid DESC').fetchall()
st.query_params['conversation']=cid
if st.session_state.get('v1_runtime_id')!=cid:
    previous=st.session_state.pop('v1_runtime',None)
    if previous:previous.close()
    policy=RuntimePolicy.from_env()
    model=ChatOllama(model=os.getenv('OLLAMA_MODEL','gemma4:e4b'),
        base_url=os.getenv('OLLAMA_BASE_URL','http://localhost:11434'),reasoning=True,
        temperature=0,num_ctx=16384,num_predict=4096,
        client_kwargs={'timeout':policy.model_timeout_seconds})
    config=ConnectionConfig.from_env()
    from core.analysis_catalog import load_saved_reference_context
    context_loader=lambda:load_saved_reference_context(ROOT/'.telly_table_context')
    runtime=GraphAnalysisRuntime(root,owner,cid,model,
        connection_identity=config.identity(),
        remote_factory=lambda d:make_executor(config,d,max_rows=policy.max_remote_rows),
        reference_context_loader=context_loader,policy=policy)
    st.session_state.v1_runtime=runtime
    st.session_state.v1_runtime_id=cid
runtime=st.session_state.v1_runtime
BUSY_NOTICE='현재 분석이 이미 실행 중입니다. 완료될 때까지 잠시 기다려주세요.'


def action(fn):
    try:
        with st.status('분석 중입니다…',expanded=True) as progress:
            runtime.on_progress=lambda message:progress.update(label=message)
            result=fn()
            state='error' if isinstance(result,dict) and result.get('status') in {'incomplete','needs_data','blocked','exhausted'} else 'complete'
            progress.update(label='확인이 필요합니다.' if state=='error' else '처리했습니다.',state=state)
        st.session_state.v1_notice=result.get('text','') if isinstance(result,dict) and result.get('status') not in {'answered','needs_data','blocked','exhausted'} else ''
    except Exception as exc:
        if isinstance(exc,RuntimeError) and str(exc)=='현재 대화가 실행 중입니다.':
            st.session_state.v1_notice=BUSY_NOTICE
        else:
            error_id=runtime.diagnostics.failure(exc, stage='ui_action')
            st.session_state.v1_notice=f'오류 ID: {error_id}. '+'작업을 완료하지 못했습니다. 기존 결과와 승인 상태를 확인해주세요. ('+type(exc).__name__+')'
    finally:
        runtime.on_progress=None


with st.sidebar:
    st.subheader('대화')
    ids=[r[0] for r in conversations];titles=dict(conversations)
    chosen=st.selectbox('저장된 대화',ids,index=ids.index(cid),format_func=lambda x:titles[x])
    if chosen!=cid:
        st.session_state.v1_conversation=chosen;st.rerun()
    if st.button('새 대화'):
        st.session_state.v1_conversation=str(uuid4());st.rerun()
    st.subheader('분석할 자료')
    table=st.text_input('Databricks 테이블',placeholder='catalog.schema.table')
    if st.button('데이터 불러오기 제안',disabled=not table.strip()):
        action(lambda:runtime.propose_table(table));st.rerun()
    if st.button('예제 데이터로 시작'):
        def example():
            fixture=json.loads((ROOT/'tests/fixtures/analysis_acceptance.json').read_text())
            info=runtime.datasets.register(pd.DataFrame(fixture['rows']),source=fixture['source'],
                coverage='complete',predicate_known=True)
            runtime.select_dataset(info.id)
            return {'text':f'합성 예제 데이터 {info.rows}행을 로컬에 준비했습니다.'}
        action(example);st.rerun()
    st.subheader('보유한 결과')
    selected_id=runtime.context.selected_dataset_id
    for index,info in enumerate(runtime.datasets.metadata.values(),1):
        marker=' · 분석 기준' if info.id==selected_id else ''
        with st.expander(f'결과 {index} · {info.source} · {info.rows:,}행{marker}'):
            st.caption(f'{info.role} · {info.grain} · {info.coverage}')
            preview=runtime.db.dataset_preview(info.id)
            if preview is None:
                st.caption('이전 형식으로 저장된 결과입니다. 전체 데이터를 화면 미리보기용으로 복원하지 않습니다.')
            elif preview:
                st.dataframe(preview,hide_index=True)
            else:
                st.caption('조회 결과에 행이 없습니다.')
            if st.button('분석 기준으로 선택',key='select-'+info.id,disabled=info.id==selected_id):
                action(lambda:runtime.select_dataset(info.id));st.rerun()
            if st.button('차트 추천',key='recommend-'+info.id):
                action(lambda:runtime.recommend_charts(info.id));st.rerun()
    with st.expander('분석 스킬'):
        from utils.analysis_skill_registry import AnalysisSkillRegistry
        for item in AnalysisSkillRegistry().list():st.write(item['name']+' — '+item['description'])
    with st.expander('현재 지원 범위와 운영 한도'):
        policy=runtime.inspect()['operational_policy']
        st.write('지원: 보유 데이터 재사용, 기본 집계, 명시적 필터, histogram/bar/line/scatter/단일 수치 boxplot')
        st.write('검증 중: 조인, 가설 검정, 고급 복합 시각화')
        st.caption(f"원격 결과 최대 {policy['max_remote_rows']:,}행 · 최대 {policy['max_dataset_columns']:,}열 · 대화별 저장공간 {policy['scope_disk_quota_bytes'] / 1024**3:.1f} GiB · 정리 후보 기준 {policy['retention_days']}일")

for message in runtime.events():
    if isinstance(message,(HumanMessage,AIMessage)) and message.content:
        with st.chat_message('user' if isinstance(message,HumanMessage) else 'assistant'):
            # Internal result IDs remain in model context, not user-facing prose.
            content=message.content
            if isinstance(content,str) and content.startswith('선택한 차트:'):content=content.split(', dataset_id=')[0]
            st.markdown(display_analysis_text(content,runtime.datasets.metadata))
    if isinstance(message,ToolMessage):
        try:
            data=json.loads(message.content)
        except (TypeError,ValueError):
            data={}
        references=data.get('cards',[]) if isinstance(data,dict) else []
        cards=[runtime.artifacts[card_id] for item in references
               if isinstance(item,dict) and isinstance(card_id:=item.get('id'),str)
               and card_id in runtime.artifacts]
        if cards:
            with st.chat_message('assistant'):
                st.write('이렇게 살펴볼 수 있어요')
                for column,card in zip(st.columns(len(cards)),cards):
                    with column:
                        st.image(card.image,caption=card.title)
                        st.caption(card.reason+' · '+card.scope)
                        if st.button('이 차트 선택',key=f'chart-{message.id}-{card.id}'):
                            action(lambda:runtime.select_chart(card.id))
                            st.session_state.v1_selected=card.id;st.rerun()
selected=None
for previous in reversed(runtime.events()):
    if isinstance(previous,HumanMessage) and isinstance(previous.content,str) and ', card_id=' in previous.content:
        selected=previous.content.rsplit(', card_id=',1)[1]
        break
if selected in runtime.artifacts:
    card=runtime.artifacts[selected];st.subheader(card.title);st.image(card.image);st.caption(card.scope)

state=runtime.inspect()
if state['state']=='idle' and st.session_state.get('v1_notice')==BUSY_NOTICE:
    # A concurrent duplicate submission can finish before this page reruns.
    # Do not leave an obsolete "already running" notice after completion.
    st.session_state.pop('v1_notice',None)
if state.get('recovery',{}).get('status')=='blocked':
    from core.analysis_agent.failure_messages import remote_failure_message
    failures=list(state['recovery'].get('failed',{}).values())
    if any(f.get('status')=='unavailable' for f in failures):
        st.error(remote_failure_message(failures))
for pending in state['requests']:
    with st.container(border=True):
        schema_probe=runtime.is_schema_probe(pending['query'])
        st.subheader('현재 컬럼을 확인할까요?' if schema_probe else '추가 데이터를 불러올까요?')
        from core.analysis_load_plan import source_plan
        plan=source_plan(pending['source'],pending['query'])
        st.write(pending['reason'])
        st.caption('대상: '+(', '.join(plan.actual_tables) if plan.actual_tables else pending['source'])
                   +' · 결과 유형: '+('컬럼 정보(0행)' if schema_probe else
                                  '집계' if plan.grain=='aggregate' else '행 데이터')
                   +' · SQL 행 제한: '+('있음' if plan.bounded_result else '없음'))
        st.code(pending['query'],language='sql')
        st.caption('이 조회에만 승인이 적용됩니다. 기존 결과는 유지됩니다.')
        yes,no=st.columns(2)
        if yes.button('컬럼 확인하고 계속' if schema_probe else '불러오고 계속',
                      key='yes-'+pending['id'],disabled=pending['status']!='proposed'):
            action(lambda:runtime.respond(pending['id'],approved=True));st.rerun()
        if no.button('조회 취소',key='no-'+pending['id'],disabled=pending['status'] not in {'proposed','invalidated'}):
            action(lambda:runtime.cancel(pending['id']));st.rerun()
if state.get('uncertain_executions'):
    st.warning('이전 원격 조회의 제출 상태를 확인할 수 없습니다. 중복 조회를 막기 위해 자동 재실행을 중단했습니다. Databricks에서 실행 상태를 먼저 확인해주세요.')
if state['state']=='incomplete' and not state.get('uncertain_executions'):
    if not st.session_state.get('v1_notice'):
        st.warning('이전 분석이 중단되었습니다. 저장된 지점부터 재개할 수 있습니다.')
    if st.button('미완료 분석 재개'):
        action(runtime.resume);st.rerun()
if st.session_state.get('v1_notice'):st.info(display_analysis_text(st.session_state.v1_notice,runtime.datasets.metadata))
prompt=st.chat_input('무엇을 살펴볼까요?')
if prompt:
    action(lambda:runtime.submit(prompt));st.rerun()
