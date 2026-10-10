"""Planner-visible contracts over existing durable data and execution services."""
from dataclasses import asdict
from copy import deepcopy
from core.analysis_tool_contract import ToolDefinition, normalize_tool_result
from utils.analysis_datasets import AnalysisNeed, Condition, select_reusable_dataset, full_read_preflight


def current(context):
    return (context.runtime_services or {}).get('current',lambda:{})()

def retained_input_id(context,state):
    if state.get('population_basis')=='selected_original':return context.selected_dataset_id
    if state.get('display_dataset_id'):return state['display_dataset_id']
    prior=state.get('confirmed_analysis') or {}
    if prior.get('custom_analysis_evidence'):return prior['custom_analysis_evidence'].get('output_dataset_id')
    if prior.get('advanced_eda_evidence'):return prior['advanced_eda_evidence'].get('dataset_id')
    if prior.get('export_evidence'):return prior['export_evidence'].get('dataset_id')
    for key in ('table_preview_evidence','chart_display_evidence'):
        if prior.get(key):return prior[key].get('dataset_id')
    if prior.get('artifact_ids'):
        card=context.artifacts.get(prior['artifact_ids'][-1])
        if card:return card.dataset_id
    if prior.get('evidence_ids'):return prior['evidence_ids'][-1]
    return context.selected_dataset_id


def scoped_dataset(context, dataset_id, columns, preserve_grain=False):
    state=current(context);info=context.datasets.metadata[dataset_id]
    if state.get('current_result_only') and dataset_id!=retained_input_id(context,state):
        raise ValueError('Dataset is not the verified displayed/selected result requested by this goal')
    if state.get('fresh_source_required'):
        reader=context.remote_receipt_reader
        if reader is None or not reader(dataset_id):raise ValueError('Fresh source needs a verified new remote receipt')
        from datetime import datetime
        if not info.snapshot or datetime.fromisoformat(info.snapshot).timestamp()<state.get('request_started_at',float('inf')):
            raise ValueError('Stored snapshot predates requested refresh')
    if state.get('required_sources') and info.source not in state['required_sources']:
        raise ValueError('Dataset source differs from current goal')
    scope=state.get('scope') or {}
    if any(scope.get(k) for k in ('any_conditions','measure_conditions','join_edges','ratio','unresolved')):
        raise ValueError('This local extension requires a verified AND-only population')
    need=AnalysisNeed(info.source,tuple(columns),conditions=tuple(Condition(**c) for c in scope.get('conditions',[])),
        current_result_only=bool(state.get('current_result_only')),
        grain=info.grain if state.get('current_result_only') or preserve_grain else 'raw',
        aggregation=info.aggregation if state.get('current_result_only') or preserve_grain else '')
    selected=select_reusable_dataset(context.datasets.metadata,dataset_id,need)
    if selected.decision.action=='query_source':raise ValueError('Loaded dataset does not cover requested population')
    rejected=full_read_preflight(context.datasets,[selected.dataset_id])
    if rejected:raise MemoryError('Bounded full-read budget exceeded; use database aggregation')
    return context.datasets.derive(selected.dataset_id,need)


def build_extension_tools(context):
    def tool(name,description,properties,required,run,schema_extra=None):
        return ToolDefinition(name,description,{'type':'object','additionalProperties':False,'properties':properties,'required':required,**(schema_extra or {})},
                              lambda **args:normalize_tool_result(run(**args)))
    text={'type':'string','minLength':1,'maxLength':512}
    strings={'type':'array','minItems':1,'maxItems':8,'uniqueItems':True,'items':text}
    def contract(name):
        value=(context.registered_contracts or {}).get(name)
        return {'status':'ready','contract':value} if value else {'status':'needs_context','error_code':'tool_not_registered'}
    def plan(source,columns,operation='rows'):
        state=current(context)
        if state.get('required_sources') and source not in state['required_sources']:raise ValueError('Plan source mismatch')
        conditions=(state.get('scope') or {}).get('conditions',[])
        candidates=[]
        complex_scope=any((state.get('scope') or {}).get(k) for k in ('any_conditions','measure_conditions','join_edges','ratio','unresolved'))
        for info in context.datasets.metadata.values():
            if info.source!=source or complex_scope or (state.get('current_result_only') and info.id!=retained_input_id(context,state)):continue
            need=AnalysisNeed(source,tuple(columns),conditions=tuple(Condition(**c) for c in conditions),
                current_result_only=bool(state.get('current_result_only')),
                grain=info.grain if state.get('current_result_only') else 'raw',
                aggregation=info.aggregation if state.get('current_result_only') else '')
            selection=select_reusable_dataset(context.datasets.metadata,info.id,need)
            if selection.decision.action!='query_source':
                chosen=context.datasets.metadata[selection.dataset_id]
                estimate=context.datasets.frames.estimate_full_decode_bytes(chosen.id,chosen.rows,len(columns)) if hasattr(context.datasets.frames,'estimate_full_decode_bytes') else None
                candidates.append({'dataset_id':chosen.id,'rows':chosen.rows,'decode_bytes_estimate':estimate,
                                   'selection_origin':selection.origin,'coverage':chosen.coverage})
        service=context.runtime_services or {};limits=service.get('policy',lambda:{})()
        fits=[c for c in candidates if not state.get('fresh_source_required') and c['decode_bytes_estimate'] is not None and c['decode_bytes_estimate']<=limits.get('max_full_read_bytes',128*1024*1024)]
        route='reuse_local' if fits else 'database_aggregate' if operation!='rows' else 'projected_batch_load'
        return {'status':'planned','execution_plan':{'source':source,'columns':columns,'scope':deepcopy(state.get('scope',{})),
                'operation':operation,'route':route,'local_candidates':candidates,'limits':limits,
                'remote_scan_rows_estimate':None,'remote_cost_estimate':None,
                'estimate_note':'DB execution statistics are not available from the current driver. Unknown is not zero.'},
                'message':'계획만 작성했습니다. 로컬/DB 실행 도구로 실행하고 실제 결과 범위를 검증하세요.'}
    def status(request_id=''):
        service=context.runtime_services or {}
        return service.get('status',lambda *_:{'status':'unavailable','error_code':'execution_service_unavailable'})(request_id)
    def health():
        return (context.runtime_services or {}).get('health',lambda:{'status':'unavailable','error_code':'probe_unavailable'})()
    def inspect_plan():
        return (context.runtime_services or {}).get('plan',lambda:{'status':'needs_context'})()
    def replan(edges):
        return (context.runtime_services or {}).get('dependencies',lambda *_:{'status':'needs_context'})(edges)
    def failure():
        state=current(context)
        return {'status':'ready','diagnostics':(context.runtime_services or {}).get('diagnostics',lambda:{})(),
            'failure':{k:state.get(k) for k in
            ('status','failure_stage','scope_error','goal_interpretation_error','model_calls','tool_calls','model_seconds')},
            'remaining_goal':state.get('goal'), 'available_datasets':[{'id':i.id,'source':i.source,'coverage':i.coverage} for i in context.datasets.metadata.values()][-8:],
            'message':'진단은 실행 결과가 아닙니다. 인자 오류는 계약을 확인하고, 불명확한 원격 제출은 재제출하지 마세요.'}
    def python(dataset_id,columns,code):
        from core.analysis_agent.python_analysis import execute
        return execute(context,dataset_id,columns,code)
    def chart(dataset_id,columns,kind,max_points=2000):
        from core.analysis_agent.advanced_eda import render
        return render(context,dataset_id,columns,kind,max_points)
    def export(dataset_id='',chart_id='',format='csv',columns=None):
        from core.analysis_agent.result_export import export_result
        return export_result(context,dataset_id,chart_id,format,columns)
    return [
        tool('get_analysis_tool_contract','이름으로 현재 실제 등록 도구의 입력/출력 계약을 정확히 확인합니다. 실행하지 않습니다.',{'name':text},['name'],contract),
        tool('plan_analysis_execution','현재 목표의 출처·조건을 유지하며 보유 데이터 재사용/DB 집계/컬럼 배치 적재 계획과 로컬 메모리 추정량을 반환합니다. 조회·비용 추정이 없으면 null이며 실행하지 않습니다.',{'source':text,'columns':strings,'operation':{'type':'string','enum':['rows','aggregate','chart']}},['source','columns'],plan),
        tool('inspect_execution_status','현재 대화의 조회 장부 상태와 완료 receipt를 확인합니다. unknown/submitting은 재조회하지 않습니다.',{'request_id':{'type':'string','maxLength':128}},[],status),
        tool('cancel_database_query','현재 대화에 속한 실행 중 조회만 DB 드라이버로 취소 요청합니다. 취소 요청과 실제 종료는 구분하며 재시작 후 연결이 없는 unknown 조회를 재제출하지 않습니다.',{'request_id':text},['request_id'],lambda request_id:(context.runtime_services or {}).get('cancel_query',lambda *_:{'status':'unavailable'})(request_id)),
        tool('inspect_backend_health','현재 연결의 SELECT 1 검사로 모델과 DB 장애를 분리합니다. 사용자 데이터는 적재하지 않습니다.',{},[],health),
        tool('diagnose_analysis_failure','현재 실행의 실패 단계·남은 목표·실제 데이터 핸들을 읽습니다. 원본/조건/권한은 변경하지 않습니다.',{},[],failure),
        tool('inspect_analysis_plan','현재 요청의 영속 작업별 범위·완료 근거·의존성 계획을 읽습니다.',{},[],inspect_plan),
        tool('update_analysis_dependencies','현재 계획의 미완료 작업 의존성을 지정합니다. 범위·옵션·완료 상태를 변경할 수 없으며 순환 의존성을 거절합니다.',{'edges':{'type':'array','maxItems':8,'items':{'type':'object','additionalProperties':False,'properties':{'task_id':text,'depends_on':{'type':'array','maxItems':8,'items':text}},'required':['task_id','depends_on']}}},['edges'],replan),
        tool('execute_analysis_python','제한된 pandas/numpy 코드를 별도 워커의 작업본에서 수행합니다. df/pd/np가 이미 제공됩니다. import/파일/네트워크/함수/루프/private/inplace는 금지입니다. 작업본 df[컬럼]=값 또는 assign으로 새 컬럼을 만들 수 있으며 저장된 원본은 바뀌지 않습니다. result DataFrame을 지정하거나 마지막 줄에서 한 DataFrame만 print하세요. rolling/groupby/agg/quantile/산술 등을 지원합니다. 원래 목표·범위를 유지하세요.',{'dataset_id':text,'columns':strings,'code':{'type':'string','minLength':1,'maxLength':6000}},['dataset_id','columns','code'],python),
        tool('render_advanced_eda','관측된 수치 컬럼 2~6개로 correlation_heatmap/scatter_matrix/distribution_panels PNG와 계산 digest를 만듭니다. max_points는 산점도 표본 상한이며 heatmap/분포 계산은 전체 보유 행을 사용합니다. 표본 산점도는 명시하며 원본·조건을 보존합니다.',{'dataset_id':text,'columns':strings,'kind':{'type':'string','enum':['correlation_heatmap','scatter_matrix','distribution_panels']},'max_points':{'type':'integer','minimum':1,'maximum':5000}},['dataset_id','columns','kind'],chart),
        tool('export_analysis_result','검증된 저장 dataset/chart를 CSV/Parquet/PNG와 출처·조건·범위 manifest ZIP으로 저장합니다. 실제 dataset_id 또는 chart_id 중 하나가 반드시 필요합니다. dataset columns는 요청한 출력 컬럼이며 생략하면 현재 목표의 명시 컬럼 또는 전체 저장 컬럼을 사용합니다. 임의 파일/URL은 받지 않습니다.',{'dataset_id':text,'chart_id':text,'columns':strings,'format':{'type':'string','enum':['csv','parquet','png']}},['format'],export,
            {'oneOf':[{'required':['dataset_id'],'not':{'required':['chart_id']}},{'required':['chart_id'],'not':{'required':['dataset_id']}}]}),
    ]
