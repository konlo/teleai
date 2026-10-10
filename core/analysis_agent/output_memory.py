"""Bounded references to completed outputs, without copying data rows."""
from copy import deepcopy


def remember(previous):
    references=deepcopy(previous.get('output_references') or [])
    if previous.get('status')!='complete':return references
    for index,task in enumerate((previous.get('goal') or {}).get('tasks',[])):
        kind=task.get('capability')
        if not kind:continue
        reference_id=str(previous.get('request_id',''))+':'+str(index)
        entry={'reference_id':reference_id,'capability':kind,
            'request_text':previous.get('request_text','')[:240],
            'required_sources':list(previous.get('required_sources',[])),
            'required_columns':list(previous.get('required_columns',[]))[:8]}
        key={'row_preview':'table_preview_evidence','chart':'chart_display_evidence'}.get(kind)
        proof=previous.get(key) if key else None
        if proof and all(k in proof for k in ('dataset_id','source','snapshot','columns','rows','total_rows','limit')):
            entry['display_evidence']=deepcopy(proof)
            entry['scope']=deepcopy(previous.get('scope') or {})
        references=[r for r in references if not previous.get('request_id') or r.get('reference_id')!=reference_id]
        references.append(entry)
    return references[-8:]


def available(context,references):
    """Resolve identities against live metadata; stale references grant no scope."""
    from core.analysis_agent.row_preview import frame
    eligible=[]
    for reference in references[-8:]:
        proof=reference.get('display_evidence')
        if not proof:continue
        try:frame(context.datasets,proof)
        except (KeyError,ValueError,TypeError,FileNotFoundError):continue
        eligible.append(deepcopy(reference))
    return eligible


def bind(context,current):
    """Bind the LLM-selected verified display before any analysis tool runs."""
    reference=current.get('requested_output_reference')
    if not reference or not current.get('current_result_only'):return
    proof=reference['display_evidence']
    from core.analysis_agent.row_preview import frame,evidence
    displayed=frame(context.datasets,proof)
    info=context.datasets.metadata[proof['dataset_id']]
    if info.source not in current.get('required_sources',[]) or not set(current.get('required_columns',[])).issubset(proof['columns']):
        raise ValueError('Requested output reference cannot supply the goal source/columns')
    if info.rows!=proof['rows']:
        # Materialize only a bounded prefix, never read/reload the source table.
        info=context.datasets.register(displayed.copy(),source=info.source,
            parent_id=info.id,snapshot=info.snapshot,coverage='truncated',
            predicate_known=info.predicate_known,conditions=info.conditions,
            grain=info.grain,aggregation=info.aggregation)
        proof=evidence(context.datasets,info.id,proof['limit'],proof['columns'])
    current.update(display_dataset_id=info.id,chart_display_evidence=deepcopy(proof),
                   requested_result_rows=proof['rows'])
