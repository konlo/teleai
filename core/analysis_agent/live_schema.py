"""Explicit model-selected schema refresh through the existing guarded SQL tool."""
from core.analysis_catalog import resolve_table_context, _quoted_table, _source_key
from copy import deepcopy


def valid_call(recovery,current,args):
    source=args.get('source','')
    wanted={_source_key(s) for s in current.get('required_sources',[])}
    if (current.get('intent_origin')!='llm' or not source or _source_key(source) not in wanted
            or current.get('catalog_discovery_source')):return False
    try:
        query='SELECT * FROM '+_quoted_table(source)+' LIMIT 0'
        return recovery._proposed_scope_valid({'name':'query_databricks','args':{
            'source':source,'query':query,'reason':'Read the actual schema without data rows'}},deepcopy(current))
    except (ValueError,TypeError,KeyError):return False


def execute(recovery,current,source,query_call,runtime):
    if not valid_call(recovery,current,{'source':source}):
        return {'status':'needs_context','error_code':'schema_source_mismatch',
            'message':'확정된 현재 분석 출처의 스키마만 확인할 수 있습니다.'}
    context=recovery.context
    observed=resolve_table_context(context.reference_context,context.datasets,source)
    exact=_source_key((observed.get('table_context') or {}).get('table',''))==_source_key(source)
    if observed.get('status')=='ready' and exact and not current.get('fresh_source_required'):
        return observed
    plan={'source':source,'query':'SELECT * FROM '+_quoted_table(source)+' LIMIT 0',
        'reason':'현재 DB의 실제 컬럼을 0행 조회로 확인합니다. 원본 데이터는 가져오지 않습니다.'}
    result=query_call(**plan,runtime=runtime)
    if result.get('status')!='ready':return {**result,'schema_query_plan':plan,'query_result':result}
    from core.analysis_agent.remote_completion import verified_receipt
    receipt=recovery.approval_ledger.get(runtime.tool_call_id)
    proof=verified_receipt(recovery.approval_ledger,context,{'id':receipt['id'],'args':plan},result)
    if not proof:return {'status':'unavailable','error_code':'remote_receipt_unverified'}
    observed=resolve_table_context(context.reference_context,context.datasets,source)
    return {**observed,'schema_query_plan':plan,'query_result':result,'receipt_id':receipt['id']}
