"""Persistent analysis runtime with approval-gated Databricks tools."""
from contextlib import contextmanager
import fcntl
import json
import sqlite3
import time
from uuid import uuid4
from core.analysis_agent.diagnostics import Diagnostics, process_peak_rss_bytes
from core.analysis_agent.recovery import RecoveryMiddleware, RecoveryPlanningMiddleware

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, HumanInTheLoopMiddleware
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from core.analysis_agent.approvals import ApprovalLedger
from core.analysis_agent.memory import Transcript, memory_middleware, CompactDiscoveryMiddleware, latest_user_request, QueuedRequestMiddleware, ModelTimingMiddleware
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from core.analysis_instructions import ANALYSIS_INSTRUCTIONS
from core.analysis_tool_contract import AnalysisToolContext, normalize_tool_result
from core.analysis_runtime_tools import build_analysis_tools
from core.analysis_agent.assets import AssetDB, PersistentDatasets, PersistentCharts
from core.analysis_agent.tools import local_tools
from core.analysis_agent.progress import tool_progress
from core.analysis_agent.policy import RuntimePolicy


class GraphAnalysisRuntime:
    version='langgraph-v1'

    def __init__(self,root,owner,conversation,model,cache_bytes=64*1024*1024,max_context_chars=32000,connection_identity=None,remote_factory=None,summary_trigger_tokens=6000,summary_keep_messages=8,reference_context_loader=None,policy=None):
        self.policy=policy or RuntimePolicy(frame_cache_bytes=cache_bytes)
        self.db=AssetDB(root,owner,conversation,max_scope_bytes=self.policy.scope_disk_quota_bytes)
        self.diagnostics=Diagnostics(self.db.directory)
        self.transcript=Transcript(self.db)
        self.on_progress=None
        self.datasets=PersistentDatasets(self.db,self.policy.frame_cache_bytes,
            max_columns=self.policy.max_dataset_columns,max_frame_bytes=self.policy.max_dataset_bytes)
        self.artifacts=PersistentCharts(self.db)
        self.max_context_chars=max_context_chars
        self.reference_context_loader=reference_context_loader
        self.model=model
        self.connection_identity=connection_identity
        self.ledger=ApprovalLedger(self.db.directory/'approvals.sqlite')
        self.remote_execute=remote_factory(self.datasets) if remote_factory else None
        self.connection=sqlite3.connect(self.db.directory/'graph.sqlite',check_same_thread=False)
        self.saver=SqliteSaver(self.connection)
        # Recovery owns the user-facing model/tool budgets. Graph steps include
        # middleware, so this is only a final safety ceiling, not the work budget.
        self.config={'configurable':{'thread_id':'conversation'},'recursion_limit':256}
        def blocked(**kwargs):raise PermissionError('실 DB 실행은 아직 연결되지 않았습니다.')
        self.context=AnalysisToolContext(self.datasets,self.artifacts,[],blocked,
            max_join_rows=self.policy.max_join_rows,
            max_join_expansion_ratio=self.policy.max_join_expansion_ratio,
            selected_dataset_id=self.db.selected_dataset_id())
        self._refresh_reference_context()
        catalog=next(t.run for t in build_analysis_tools(self.context) if t.name=='list_analysis_context')
        @dynamic_prompt
        def prompt(request):
            self._refresh_reference_context()
            instructions=ANALYSIS_INSTRUCTIONS.replace('propose_databricks_query','query_databricks')
            rendered=instructions+'\n현재 분석 환경:\n'+json.dumps(catalog(),ensure_ascii=False,default=str)
            selected=self.datasets.metadata.get(self.context.selected_dataset_id)
            if selected is not None:
                rendered += ('\n사용자가 선택한 분석 기준 데이터: '
                             + json.dumps({'dataset_id':selected.id,'root_id':selected.root_id,
                                'role':selected.role,'source':selected.source,
                                'snapshot':selected.snapshot,'coverage':selected.coverage},
                                ensure_ascii=False)
                             + '\n후속 요청에 이 기준을 사용하되, 명시한 다른 출처·범위와 충돌하면 선택을 추측하지 마세요.')
            active_request = latest_user_request(request.state['messages']) or latest_user_request(self.transcript.messages())
            if active_request is not None:
                rendered += ('\n현재 사용자 요청 원문(JSON 문자열): '+json.dumps(str(active_request.content),ensure_ascii=False)
                             +'\n요약의 과거 제안은 새 요청이 아닙니다. 이 요청을 완료하고, 요청하지 않은 후속 분석은 실행하지 마세요.')
            # Conservative bound, preserve all message/tool pairs. No silent truncation.
            length=len(rendered)+sum(len(str(m.content))+len(str(getattr(m,'tool_calls',[])))
                                      for m in request.state['messages'])
            self.diagnostics.emit('context_budget', characters=length, limit=self.max_context_chars)
            if length>self.max_context_chars:
                raise ValueError('context_budget_exceeded')
            return rendered
        registered=local_tools(self.context, self.diagnostics)
        recovery=RecoveryMiddleware(self.artifacts,self.diagnostics,context=self.context,
            transcript=self.transcript,max_model_seconds=self.policy.turn_slo_seconds)
        self.recovery=recovery
        middleware=[QueuedRequestMiddleware(),RecoveryPlanningMiddleware(recovery),CompactDiscoveryMiddleware(),
                    memory_middleware(model,summary_trigger_tokens,summary_keep_messages,diagnostics=self.diagnostics),
                    ModelTimingMiddleware(self.diagnostics),prompt]
        if self.remote_execute is not None:
            if not connection_identity:raise ValueError('Connection identity required')
            @tool
            def query_databricks(source: str, query: str, reason: str, runtime: ToolRuntime) -> dict:
                """추가 원격 데이터 조회. 정확한 SQL을 제시하고 매번 사용자 승인 후 실행합니다."""
                envelope=self.ledger.envelope(source,query,reason,self.connection_identity)
                try:
                    return normalize_tool_result(
                        self.ledger.execute(runtime.tool_call_id,envelope,self.remote_execute))
                except Exception as exc:
                    self.diagnostics.failure(exc, stage='query_databricks')
                    return normalize_tool_result({'status':'unavailable','error_type':type(exc).__name__,
                            'http_status':getattr(exc,'http_status',None),
                            'error_code':'databricks_forbidden' if getattr(exc,'http_status',None)==403 else 'databricks_unavailable',
                            'retryable':False,
                            'user_action':'Databricks 연결 권한과 설정을 확인해주세요.',
                            'message':('Databricks 접근이 거부되었습니다(403). 연결 권한과 설정을 확인해주세요. 조회는 제출되지 않았습니다.' if getattr(exc,'http_status',None)==403 else '조회가 완료되지 않았습니다. 임의로 재시도하거나 수치를 추정하지 마세요. 실행 기록과 연결 상태를 확인하고, 새 조회는 새 승인을 받아야 합니다.')})
            registered.append(query_databricks)
            middleware.append(HumanInTheLoopMiddleware(interrupt_on={
                'query_databricks':{'allowed_decisions':['approve','reject']}}))
        middleware.append(recovery)
        self.agent=create_agent(model,tools=registered,checkpointer=self.saver,middleware=middleware)
        self._reconcile_completed_controller_load()

    def _reconcile_completed_controller_load(self):
        """Repair a saved load whose dataset exists but final verdict failed.

        Reconciliation uses only the approved result already persisted in this
        conversation.  It never submits another remote query.
        """
        state=self.agent.get_state(self.config)
        values=state.values or {}
        previous=values.get('recovery') or {}
        if state.next or not values or previous.get('status') == 'complete':
            return False
        current,_=self.recovery._state(values)
        if not current.get('data_load') or not self.recovery._complete(current):
            return False
        last=next((message for message in reversed(values.get('messages',[]))
            if isinstance(message,AIMessage) and message.content),None)
        finished=self.recovery._finish(current)
        additions=[]
        if last is not None and last.additional_kwargs.get('analysis_status') != 'answered':
            additions.append(SystemMessage(content='저장된 원격 결과로 완료 상태를 복구했습니다.',
                additional_kwargs={'lc_source':'recovery','invalidated_message_id':last.id}))
        additions.extend(finished['messages'])
        self.agent.update_state(self.config,{'recovery':finished['recovery'],
            'messages':additions},as_node='model')
        self.transcript.record(self.agent.get_state(self.config).values.get('messages',[]))
        outcome={'status':'answered'}
        if self.agent.get_state(self.config).next:
            outcome=self._invoke(None)
        self.diagnostics.emit('completed_load_reconciled',request_id=current.get('request_id'),
            dataset_id=current.get('load_evidence_id'),remote_reexecuted=False,
            status=outcome.get('status'))
        return outcome.get('status') == 'answered'

    def _refresh_reference_context(self):
        if self.reference_context_loader is None:
            return
        try:
            loaded = list(self.reference_context_loader() or [])
        except Exception as exc:
            if hasattr(self, 'diagnostics'):
                self.diagnostics.failure(exc, stage='reference_context_refresh')
            return
        self.context.reference_context[:] = loaded

    @contextmanager
    def _exclusive(self):
        # Separate descriptor per invocation: flock also excludes concurrent calls
        # using the same runtime instance, not only separate processes.
        with open(self.db.directory/'runtime.lock','a+') as lock_file:
            try:fcntl.flock(lock_file,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise RuntimeError('현재 대화가 실행 중입니다.')
            try:yield
            finally:fcntl.flock(lock_file,fcntl.LOCK_UN)

    def events(self):
        self.transcript.record(self.agent.get_state(self.config).values.get('messages',[]))
        return self.transcript.messages()

    def _pending(self):
        state=self.agent.get_state(self.config)
        if not any(t.interrupts for t in state.tasks):return []
        calls=next((m.tool_calls for m in reversed(state.values.get('messages',[]))
                    if isinstance(m,AIMessage) and m.tool_calls),[])
        pending=[]
        for call in calls:
            if call['name']=='query_databricks':
                envelope=self.ledger.envelope(**call['args'],connection=self.connection_identity)
                try:
                    recorded=self.ledger.get(call['id'])
                except KeyError:
                    recorded=self.ledger.propose(call['id'],envelope)
                recorded_envelope={key:recorded[key] for key in envelope}
                if self.ledger.fingerprint(recorded_envelope)!=self.ledger.fingerprint(envelope):
                    self.ledger.invalidate(call['id'])
                    recorded=self.ledger.get(call['id'])
                pending.append(recorded)
        return pending

    def inspect(self):
        self._refresh_reference_context()
        state=self.agent.get_state(self.config)
        pending=self._pending()
        selected=self.datasets.metadata.get(self.context.selected_dataset_id)
        return {'runtime_version':self.version,
                'state':'awaiting_approval' if pending else 'incomplete' if state.next else 'idle',
                'message_count':len(self.events()),
                'model_message_count':len(state.values.get('messages',[])),
                'dataset_ids':list(self.datasets.metadata),'chart_ids':list(self.artifacts),
                'selected_dataset':({'id':selected.id,'root_id':selected.root_id,
                    'role':selected.role,'source':selected.source,
                    'snapshot':selected.snapshot,'coverage':selected.coverage}
                    if selected else None),
                'requests':pending,'uncertain_executions':self.ledger.uncertain(),
                'recovery':state.values.get('recovery',{}),
                'operational_policy':self.policy.public()}

    def _invoke(self,value):
        self._refresh_reference_context()
        started=time.monotonic()
        run_id=uuid4().hex
        self.diagnostics.run_id=run_id
        self.events()
        self.diagnostics.emit('run_started', run_id=run_id,
            operation='resume' if value is None else 'submit',
            process_peak_rss_bytes=process_peak_rss_bytes())
        try:
            result={}
            completed_load_id=''
            if self.on_progress:self.on_progress('요청과 보유 데이터를 확인하고 있습니다.')
            for update in self.agent.stream(value,self.config,stream_mode='values'):
                result=update
                messages=update.get('messages',[])
                self.transcript.record(messages)
                if messages and self.on_progress:
                    last=messages[-1]
                    if isinstance(last,ToolMessage):
                        self.on_progress(tool_progress(last))
                    elif isinstance(last,AIMessage) and last.tool_calls:
                        self.on_progress('필요한 분석 도구를 실행하고 있습니다.')
                if messages and isinstance(messages[-1],ToolMessage) and messages[-1].name=='query_databricks':
                    try:
                        observation=json.loads(messages[-1].content)
                    except (ValueError,TypeError):
                        observation={}
                    if observation.get('status')=='ready':
                        completed_load_id=observation.get('dataset',{}).get('id','')
            if result.get('__interrupt__'):
                self.diagnostics.emit('run_paused', run_id=run_id, reason='approval')
                return {'status':'awaiting_approval','requests':self._pending()}
            messages=self.agent.get_state(self.config).values.get('messages',[])
            outcome=messages[-1].additional_kwargs.get('analysis_status','answered') if messages else 'answered'
            if outcome=='answered' and completed_load_id in self.datasets.metadata:
                loaded=self.datasets.metadata[completed_load_id]
                # A remote statistic is evidence for this answer, not a new
                # row-level EDA baseline. Keep the user's selected raw branch.
                if loaded.role=='root' and loaded.grain=='raw':
                    self._select_dataset_unlocked(completed_load_id)
            elapsed=round(time.monotonic()-started,3)
            self.diagnostics.emit('run_completed', run_id=run_id, status=outcome,
                elapsed_seconds=elapsed, process_peak_rss_bytes=process_peak_rss_bytes(),
                frame_cache_bytes=self.datasets.frames.bytes)
            self.diagnostics.emit('turn_slo', run_id=run_id, elapsed_seconds=elapsed,
                limit=self.policy.turn_slo_seconds,
                status='violation' if elapsed>self.policy.turn_slo_seconds else 'ok')
            return {'status':outcome,'text':messages[-1].content if messages else '',
                    'elapsed_seconds':elapsed}
        except Exception as exc:
            elapsed=round(time.monotonic()-started,3)
            error_id=self.diagnostics.failure(exc, run_id=run_id, stage='agent_stream')
            self.diagnostics.emit('turn_slo', run_id=run_id, elapsed_seconds=elapsed,
                limit=self.policy.turn_slo_seconds,
                status='violation' if elapsed>self.policy.turn_slo_seconds else 'error',
                process_peak_rss_bytes=process_peak_rss_bytes(),
                frame_cache_bytes=self.datasets.frames.bytes)
            return {'error_id':error_id, 'status':'incomplete','error_type':type(exc).__name__,
                    'text':f'분석 중 오류가 발생했습니다 ({type(exc).__name__}, 오류 ID: {error_id}). 기존 결과는 보존했습니다. 미완료 분석 재개로 다시 시도할 수 있습니다.',
                    'elapsed_seconds':elapsed}

    def submit(self,text,model=None):
        if not text.strip():raise ValueError('요청을 입력해주세요.')
        with self._exclusive():
            pending=self._pending()
            if pending:
                normalized=text.strip().rstrip('.!').strip()
                approve_words={'승인','조회 승인','승인해줘','불러와','불러와줘','네','응','진행해','진행해줘'}
                reject_words={'취소','조회 취소','취소해줘','아니','아니요','재조회하지 마'}
                if len(pending)==1 and normalized in approve_words|reject_words:
                    return self._respond_locked(pending[0]['id'],normalized in approve_words)
                # Narrow, read-only interpretation while paused; no tools or approval API.
                decision=self.model.invoke([
                    SystemMessage(content='사용자 메시지가 오직 현재 승인 상태/이유를 묻는지 판정하세요. 변경, 실행, 취소, 모호함은 change입니다. JSON만 반환: {"action":"status"} 또는 {"action":"change"}.'),
                    HumanMessage(content=text)])
                try:status_only=json.loads(decision.content).get('action')=='status'
                except (ValueError,TypeError):status_only=False
                if status_only:
                    return {'status':'awaiting_approval','requests':pending,
                            'text':'아래 조회의 승인을 기다리고 있습니다. 아직 실행하지 않았습니다.'}
                for item in pending:self.ledger.invalidate(item['id'])
                # Finish old HITL calls before appending a real new user turn.
                # Updating messages directly here would put the old rejection
                # after the new request and contaminate its recovery state.
                return self._invoke(Command(update={'queued_user_request':{
                    'id':uuid4().hex,'content':text}}, resume={'decisions':[
                    {'type':'reject','message':'사용자가 요청을 변경하여 이전 조회 승인은 취소되었습니다.'} for _ in pending]}))
            if self.agent.get_state(self.config).next:
                raise ValueError('미완료 분석이 있습니다. 먼저 resume()로 재개해주세요.')
            return self._invoke({'messages':[HumanMessage(content=text)]})

    def resume(self):
        with self._exclusive():
            if self._pending():raise ValueError('대기 중인 조회는 먼저 승인 또는 거절해주세요.')
            if self.ledger.uncertain():raise PermissionError('원격 제출 상태가 불명확합니다. 자동 재개할 수 없습니다.')
            if not self.agent.get_state(self.config).next:raise ValueError('재개할 작업이 없습니다.')
            return self._invoke(None)

    def recommend_charts(self,dataset_id):
        from uuid import uuid4
        with self._exclusive():
            if self.agent.get_state(self.config).next:raise ValueError('기존 작업을 먼저 처리해주세요.')
            definition=next(t for t in build_analysis_tools(self.context) if t.name=='recommend_chart_images')
            result=definition.run(dataset_id=dataset_id)
            call_id=str(uuid4())
            self.agent.update_state(self.config,{'messages':[
                HumanMessage(content='현재 보유 데이터의 차트를 추천해줘.'),
                AIMessage(content='',tool_calls=[{'name':definition.name,'args':{'dataset_id':dataset_id},'id':call_id}]),
                ToolMessage(content=json.dumps(result,ensure_ascii=False),name=definition.name,tool_call_id=call_id),
                AIMessage(content='보유 데이터로 만든 차트입니다. 원하는 이미지를 선택해주세요.')
            ]},as_node='model')
            self.transcript.record(self.agent.get_state(self.config).values.get('messages',[]))
            if self.agent.get_state(self.config).next:
                finished=self._invoke(None)
                if finished['status']!='answered':return finished
            return result

    def select_chart(self,card_id):
        with self._exclusive():
            if self.agent.get_state(self.config).next:raise ValueError('미완료 작업이 있습니다.')
            card=self.artifacts[card_id]
            self.agent.update_state(self.config,{'messages':[
                HumanMessage(content=f'선택한 차트: {card.title}, dataset_id={card.dataset_id}, card_id={card.id}',additional_kwargs={'selected_card':card.id}),
                AIMessage(content='선택한 차트를 표시합니다.')]},as_node='model')
            self.transcript.record(self.agent.get_state(self.config).values.get('messages',[]))
            if self.agent.get_state(self.config).next:
                finished=self._invoke(None)
                if finished['status']!='answered':raise RuntimeError('차트 선택 완료 처리에 실패했습니다.')
            self._select_dataset_unlocked(card.dataset_id)
            return card.id

    def _select_dataset_unlocked(self,dataset_id):
        info=self.datasets.metadata[dataset_id]
        self.db.select_dataset(dataset_id)
        self.context.selected_dataset_id=dataset_id
        self.diagnostics.emit('dataset_selected', dataset_id=dataset_id,
                              root_id=info.root_id, role=info.role, source=info.source)
        return info

    def select_dataset(self,dataset_id):
        with self._exclusive():
            if self.agent.get_state(self.config).next:
                raise ValueError('미완료 작업을 먼저 처리해주세요.')
            info=self._select_dataset_unlocked(dataset_id)
            return {'status':'ready',
                    'text':f'{info.rows:,}행의 저장된 데이터를 분석 기준으로 선택했습니다.',
                    'dataset_id':dataset_id,'root_id':info.root_id}

    def _respond_locked(self,request_id,approved):
        from uuid import uuid4
        pending=self._pending()
        if request_id not in [r['id'] for r in pending]:raise PermissionError('현재 대기의 승인이 아닙니다.')
        current=self.ledger.get(request_id)
        if not approved and current['status']=='invalidated':pass
        else:self.ledger.decide(request_id,approved)
        self.transcript.record([HumanMessage(id=str(uuid4()),content=(
            '추가 데이터 조회를 승인했습니다.' if approved else '추가 데이터 조회를 취소했습니다.'))])
        states=[self.ledger.get(r['id']) for r in pending]
        if any(r['status']=='proposed' for r in states):
            return {'status':'awaiting_approval','requests':states}
        return self._invoke(Command(resume={'decisions':[
            {'type':'approve'} if r['status']=='approved' else
            {'type':'reject','message':'사용자가 조회를 거절했습니다. 재조회 없이 기존 데이터로 가능한 범위를 안내하세요.'}
            for r in states]}))

    def respond(self,request_id,*,approved,execute=None,model=None):
        if execute is not None:raise PermissionError('실행기는 승인 시 교체할 수 없습니다.')
        with self._exclusive():
            return self._respond_locked(request_id,approved)

    def cancel(self,request_id):return self.respond(request_id,approved=False)

    def propose_table(self,table):
        from uuid import uuid4
        from core.analysis_sql import validate_query
        if self.remote_execute is None:raise PermissionError('원격 실행기가 없습니다.')
        parts=table.strip().split('.')
        if not 1<=len(parts)<=3 or any(not p.strip() for p in parts):raise ValueError('테이블 이름을 확인해주세요.')
        query='SELECT * FROM '+'.'.join('`'+p.replace('`','``')+'`' for p in parts)+' LIMIT 10000'
        validate_query(query)
        return self.propose_query(table,query,'데이터 구조와 예시를 살펴볼 최대 10,000행입니다. 전체 통계용 데이터가 아닙니다.')

    def propose_query(self,source,query,reason):
        from uuid import uuid4
        self.ledger.envelope(source,query,reason,self.connection_identity)
        with self._exclusive():
            if self.agent.get_state(self.config).next:raise ValueError('기존 작업을 먼저 처리해주세요.')
            if self.remote_execute is None:raise PermissionError('원격 실행기가 없습니다.')
            # Feed a controller-authored proposal through the same HITL before-model node.
            call=AIMessage(content='',tool_calls=[{'name':'query_databricks','args':{
                'source':source,'query':query,'reason':reason},'id':str(uuid4())}])
            self.agent.update_state(self.config,{'messages':[HumanMessage(content=reason,additional_kwargs={
                'request_kind':'remote_load','source':source,'query':query}),call]},as_node='model')
            return self._invoke(None)

    def close(self):
        self.connection.close();self.db.close()
