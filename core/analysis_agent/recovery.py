"""Checkpointed, bounded recovery with artifact-backed chart completion."""
import json
import re
from core.analysis_agent.failure_messages import remote_failure_message
from core.analysis_agent.memory import latest_user_request
from typing import NotRequired
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage, RemoveMessage


class RecoveryState(AgentState):
    recovery: NotRequired[dict]


class RecoveryMiddleware(AgentMiddleware):
    state_schema = RecoveryState

    def __init__(self, artifacts, diagnostics, max_attempts=2, context=None, transcript=None):
        self.artifacts, self.diagnostics, self.max_attempts = artifacts, diagnostics, max_attempts
        self.context=context
        self.transcript=transcript

    @hook_config(can_jump_to=['model'])
    def after_model(self, state, runtime):
        messages = state.get('messages', [])
        if not messages or not isinstance(messages[-1], AIMessage): return None
        last = messages[-1]
        # Persist obligations independently of summarization. A new user turn resets
        # recovery; approvals resume the same graph turn and cannot reset its budget.
        human = latest_user_request(messages)
        if human is None and self.transcript is not None:
            human = latest_user_request(self.transcript.messages())
        current = dict(state.get('recovery') or {})
        if human and current.get('request_id') != human.id:
            text = str(human.content)
            kind = 'histogram' if re.search(r'히스토그램|\bhistogram', text, re.I) else None
            chart = bool(kind or re.search(r'차트|시각화|그래프|\bchart', text, re.I))
            names=set()
            if self.context:
                names.update(c['name'] for t in self.context.reference_context for c in t.get('columns',[]) if c.get('name'))
                names.update(c for info in self.context.datasets.metadata.values() for c in info.columns)
            required=[name for name in names if re.search(r'(?<![A-Za-z0-9_])'+re.escape(name)+r'(?![A-Za-z0-9_])',text)]
            current = {'required_columns':required,'request_id': human.id, 'attempts': 0, 'chart': chart,
                       'kind': kind, 'columns': [], 'failed': {}, 'status': 'working'}
        if human and human.additional_kwargs.get('selected_card') in self.artifacts:
            card=self.artifacts[human.additional_kwargs['selected_card']]
            if card.image.startswith(b'\x89PNG\r\n\x1a\n'):
                current['artifact_ids']=[card.id]
        current.setdefault('failed', {})
        proposed_next=None
        calls = {c['id']: c for m in messages if isinstance(m, AIMessage) for c in m.tool_calls}
        # Use current-turn observations only; persistent obligations survive summary.
        start = next((i for i,m in enumerate(messages) if human and m.id == human.id), 0)
        for message in messages[start:]:
            if not isinstance(message, ToolMessage): continue
            call = calls.get(message.tool_call_id, {})
            name = message.name or call.get('name')
            try: observation = json.loads(message.content)
            except (ValueError, TypeError): continue
            if not isinstance(observation, dict): continue
            if name=='prepare_histogram' and observation.get('histogram_plan'):
                current['chart']=True
                current['kind']='histogram'
                current['plan']=observation['histogram_plan']
                current['plan_call']=message.tool_call_id
                current['columns']=[current['plan']['value_column']]
                current['failed'].pop('prepare_histogram',None)
            if name=='query_databricks' and current.get('plan'):
                if call.get('args',{}).get('query')==current['plan']['query']:
                    current['query_seen']=current['plan_call']
                    if observation.get('status')=='ready':
                        current['loaded_dataset']=observation['dataset']['id']
            if observation.get('status') in {'error','needs_data','needs_context','unavailable','no_valid_chart'}:
                current['failed'][name or message.tool_call_id] = observation
            elif name in current['failed']:
                del current['failed'][name]
            if name in {'recommend_chart_images','render_histogram'}:
                current['chart'] = True
                arguments = call.get('args', {})
                current['columns'] = arguments.get('columns') or ([arguments['value_column']] if 'value_column' in arguments else current['columns'])
                current['dataset_id'] = arguments.get('dataset_id')
                current['artifact_ids']=[]
                valid = []
                for entry in observation.get('cards', []):
                    try: card = self.artifacts[entry['id']]
                    except (KeyError, TypeError): continue
                    if not card.image.startswith(b'\x89PNG\r\n\x1a\n'): continue
                    if current.get('kind') and card.kind != current['kind']: continue
                    expected=current.get('required_columns') or current['columns']
                    if expected and not set(expected).issubset(card.columns): continue
                    if card.dataset_id != arguments.get('dataset_id'): continue
                    valid.append(card.id)
                if valid:
                    current['artifact_ids'] = valid
                    current['failed'].pop('recommend_chart_images', None)
                    current['failed'].pop('render_histogram', None)
        for call in last.tool_calls:
            if call['name'] in {'recommend_chart_images','render_histogram'}:
                current['chart'] = True
        remote_failure = any(o.get('status') == 'unavailable' for o in current['failed'].values())
        remote_rejected = any(isinstance(m, ToolMessage) and 'rejected' in str(m.content).lower() for m in messages[start:])
        if last.tool_calls and (remote_failure or remote_rejected) and any(c['name']=='query_databricks' for c in last.tool_calls):
            current['status']='blocked'
            return {'recovery':current,'messages':[last.model_copy(update={
                'content':remote_failure_message(current['failed'].values(),remote_rejected),
                'tool_calls':[], 'additional_kwargs':{**last.additional_kwargs,'analysis_status':'blocked'}})]}
        if current.get('plan') and not (remote_failure or remote_rejected):
            plan=current['plan']
            if current.get('query_seen')!=current.get('plan_call'):
                proposed_next={'name':'query_databricks','args':{k:plan[k] for k in ('source','query','reason')}}
            elif current.get('loaded_dataset') and not current.get('artifact_ids'):
                already_rendered=any(c.get('name')=='render_histogram' and c.get('args',{}).get('dataset_id')==current['loaded_dataset'] for c in calls.values())
                if not already_rendered:
                    proposed_next={'name':'render_histogram','args':{'dataset_id':current['loaded_dataset'],'value_column':plan['value_column'],'weight_column':plan['weight_column']}}
        if proposed_next:
            from uuid import uuid4
            self.diagnostics.emit('recovery_transition',tool=proposed_next['name'])
            proposed_next['id']=str(uuid4())
            proposed_next['type']='tool_call'
            return {'recovery':current,'messages':[last.model_copy(update={'content':'','tool_calls':[proposed_next]})]}
        if last.tool_calls:
            return {'recovery': current}
        missing_chart = current.get('chart') and not current.get('artifact_ids')
        # Metadata inspection is not a substitute for failed analysis. For non-chart
        # inspection, successfully reading table context is a valid alternate route.
        if not current.get('chart') and 'inspect_table_context' not in current['failed']:
            if any(isinstance(m, ToolMessage) and m.name == 'inspect_table_context' for m in messages[start:]):
                current['failed'].pop('inspect_dataset', None)
        if not missing_chart and not current['failed']:
            current['status'] = 'complete'
            return {'recovery': current}
        remote_block = any(o.get('status') == 'unavailable' for o in current['failed'].values())
        rejection = any(isinstance(m, ToolMessage) and 'rejected' in str(m.content).lower()
                        for m in messages[start:])
        if remote_block or rejection or current['attempts'] >= self.max_attempts:
            current['status'] = 'blocked' if remote_block or rejection else 'exhausted'
            text = remote_failure_message(current['failed'].values(),rejection)
            self.diagnostics.emit('recovery_stopped', reason=current['status'], attempts=current['attempts'])
            return {'recovery': current, 'messages': [last.model_copy(update={'content':text,
                    'additional_kwargs':{**last.additional_kwargs,'analysis_status':current['status']}})]}
        current['attempts'] += 1
        self.diagnostics.emit('recovery_replan', attempts=current['attempts'], missing_chart=bool(missing_chart))
        instruction = ('이전 응답은 완료 증거가 없어 채택되지 않았습니다. 원래 사용자 요청을 계속 수행하세요. '
            '데이터가 없는 히스토그램이면 inspect_table_context로 컬럼을 확인하고 prepare_histogram(source, column, where_sql) 도구를 호출하세요. 이 도구가 승인과 로딩 후 이미지 생성 계획을 연결합니다. '
            'query_databricks 호출 자체가 승인 카드를 생성하며, 승인은 시스템이 기다립니다. 먼저 승인받겠다는 말만 하고 끝내지 마세요. '
            '히스토그램은 필요한 수치값과 COUNT(*)를 GROUP BY로 집계해 가져온 뒤 render_histogram의 weight_column에 빈도 컬럼을 지정할 수 있습니다. '
            '사용자 조건을 그대로 유지하고, 실패한 동일 호출을 반복하지 마세요. 실제 차트 없이 완료라고 말하지 마세요.')
        return {'recovery': current, 'messages':[RemoveMessage(id=last.id),
                SystemMessage(content=instruction, additional_kwargs={'lc_source':'recovery','invalidated_message_id':last.id})], 'jump_to':'model'}
