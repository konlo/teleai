"""A bounded, ledger-controlled source-date observation; never changes selection."""
from hashlib import sha256
from copy import deepcopy
from core.analysis_agent.relative_period import probe_query, boundaries


def observe(runtime,current,source,spec):
    if runtime.remote_execute is None:return {'status':'unavailable'}
    if runtime.policy.require_remote_approval:return {'status':'needs_approval'}
    column=spec['column']
    tables=[t for t in runtime.context.reference_context if t.get('table')==source]
    columns=[c for t in tables for c in t.get('columns',[]) if c.get('name')==column]
    if len(tables)!=1 or len(columns)!=1:return {'status':'needs_schema'}
    query=probe_query(source,column,columns[0].get('dtype',''),runtime.sql_dialect)
    envelope=runtime.ledger.envelope(source,query,'상대 기간의 데이터 기준일과 날짜 형식을 확인합니다.',runtime.connection_identity)
    for uncertain in runtime.ledger.uncertain():
        record=runtime.ledger.get(uncertain['id'])
        if runtime.ledger.fingerprint({k:record[k] for k in envelope})==runtime.ledger.fingerprint(envelope):
            return {'status':'unknown_submission'}
    key='period_'+sha256((current['request_id']+runtime.ledger.fingerprint(envelope)).encode()).hexdigest()[:32]
    recorded=runtime.ledger.propose(key,envelope)
    runtime.ledger.authorize_automatic(key,envelope)
    # This receipt proves the date anchor; it is not the requested analysis
    # result. Keeping it in analysis-result IDs creates a false completion
    # obligation even when the histogram is computed from retained raw data.
    if key not in current.setdefault('observation_query_ids',[]):current['observation_query_ids'].append(key)
    runtime.context.current_analysis=deepcopy(current)
    runtime.diagnostics.emit('remote_query_started',ledger_status=recorded['status'],
        cached_receipt=recorded['status']=='completed',purpose='period_anchor')
    try:
        result=runtime.ledger.execute(key,envelope,runtime._controlled_executor(key))
    except Exception as exc:
        runtime.diagnostics.failure(exc,stage='source_period_probe')
        runtime.diagnostics.emit('remote_query_finished',status='unavailable',ledger_status=runtime.ledger.get(key)['status'])
        # An uncertain receipt stays uncertain; the deterministic key prevents
        # goal JSON repair from submitting the same probe a second time.
        return {'status':'unavailable'}
    runtime.diagnostics.emit('remote_query_finished',status=result.get('status'),ledger_status=runtime.ledger.get(key)['status'])
    info=runtime.datasets.metadata.get((result.get('dataset') or {}).get('id'))
    if (result.get('status')!='ready' or not info or info.coverage!='complete' or info.rows!=1
            or info.source!=source or info.query!=query or info.grain!='aggregate' or not info.aggregation):
        return {'status':'invalid_receipt'}
    from utils.analysis_datasets import preview_dataset
    row=preview_dataset(runtime.datasets,info.id,limit=1).iloc[0]
    if int(row['__period_count'])==0:return {'status':'empty_time_column'}
    if int(row['__period_invalid'])!=0:return {'status':'invalid_date_format'}
    if int(row['__period_formats'])!=1:return {'status':'mixed_date_format'}
    try:lower,upper=boundaries(row['__period_max'],spec['amount'],spec['unit'])
    except (ValueError,OverflowError):return {'status':'unsupported_period'}
    conditions=[{'column':column,'op':'ge','value':lower},{'column':column,'op':'le','value':upper}]
    runtime.diagnostics.emit('goal_relative_period_resolved',request_id=current['request_id'],
        anchor=spec['anchor'],unit=spec['unit'],amount=spec['amount'],receipt_dataset_id=info.id)
    return {'status':'ready','anchor':'source_max','source':source,'column':column,
        'amount':spec['amount'],'unit':spec['unit'],'dataset_id':info.id,
        'query_id':key,'conditions':conditions}
