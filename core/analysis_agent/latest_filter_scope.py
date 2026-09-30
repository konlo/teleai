"""Bind only explicit stage instructions; never infer filter/rank order."""
import re
from utils.analysis_latest_filters import validate

STAGE_PATTERN=r'(?:조건|필터)(?:을|를|은|는)?\s*최신\s*행\s*선택\s*(전|후)(?:에)?\s*(?:적용|필터링)(?:해줘|해주세요)?'
STAGES={'전':'before_selection','후':'after_selection'}

def bind_scope(spec,scope,text,context):
    if not spec:return
    conditions=scope.get('conditions') or []
    if not conditions and not any(scope.get(k) for k in ('unresolved','any_conditions','join_edges','measure_conditions','ratio')):
        return
    spec['filter_scope_bound']=False
    if any(scope.get(k) for k in ('unresolved','any_conditions','join_edges','measure_conditions','ratio')):
        spec.setdefault('question','최신행 필터의 조건을 확인해주세요. 현재는 명시한 단일 단계의 AND 조건을 검증할 수 있습니다.')
        return
    stages={m[1] for m in re.finditer(STAGE_PATTERN,text,re.I)}
    if len(stages)!=1 or re.search(STAGE_PATTERN+r'.{0,12}(?:말고|않|아니|하지)',text,re.I):
        spec.setdefault('question','조건을 최신행 선택 전에 적용할지, 선택 후에 적용할지 알려주세요.')
        spec['filter_stage_pending']=True
        return
    stage=STAGES[stages.pop()]
    if spec.get('remote'):
        from core.analysis_catalog import resolve_table_context
        observed=resolve_table_context(context.reference_context,context.datasets,spec['source'])
        columns=[c['name'] for c in (observed.get('table_context') or {}).get('columns',[])]
    else:
        info=context.datasets.metadata.get(spec.get('dataset_id'))
        columns=info.columns if info else []
    try:conditions=validate(conditions,stage,columns)
    except (ValueError,TypeError):
        spec.setdefault('question','현재 스키마에서 필터 컬럼과 비교값을 확인해주세요.')
        return
    spec.update(conditions=conditions,filter_stage=stage,filter_scope_bound=True)

def continue_stage(text,previous):
    match=re.fullmatch(r'\s*'+STAGE_PATTERN+r'[.!]?\s*',text,re.I)
    prior=previous.get('latest_per_key_spec') or {}
    pending=previous.get('latest_pending') or {}
    if not match or not (prior.get('conditions') or pending.get('filter_stage_pending')):
        return None
    base=pending.get('request_text') or previous.get('request_text','')
    base=re.sub(STAGE_PATTERN,'',base,flags=re.I)
    return base+'\n조건을 최신행 선택 '+match[1]+'에 적용해줘.'
