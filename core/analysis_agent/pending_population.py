"""Validate model recovery of failed HUMAN intent, without treating it as data."""
from copy import deepcopy


def recover(step,data,scope,sources,previous_sources,context):
    raw=data.get('unparsed_user_intent') or []
    population=step.get('pending_population')
    wanted=(raw and step['goal']['source_reference']=='previous_analysis'
        and step['population']['basis'] in {'source_population','unfiltered_source'}
        and step['population_delta']['change']=='keep'
        and step['goal']['tasks'][0]['capability']!='table_list')
    if wanted and not population:
        raise ValueError('Latest human intent was not interpreted. Reconstruct its population in pending_population, or clarify; older successful filters are not its substitute.')
    if not population:return scope
    if not raw or any(r.get('truncated') for r in raw):
        raise ValueError('Pending population requires complete preserved human request text')
    from core.analysis_agent.goal_contract import canonical_sources
    from core.analysis_agent.source_references import source_mentions,qualified_mentions
    named=[]
    for r in raw:
        references=source_mentions(r['request_text'],context)+qualified_mentions(r['request_text'])
        if references:named=canonical_sources([m['source'] for m in references],context)
    intended=named or previous_sources
    if not sources or sources!=intended:
        raise ValueError('Pending population belongs to a different requested physical subject')
    text='\n'.join(r['request_text'] for r in raw)
    review={'question':'',**deepcopy(population['restriction_review'])}
    if any(c['status']!='resolved' or c['kind']=='relative_time' for c in review['clauses']):
        raise ValueError('Unresolved pending restrictions need reobservation or clarification')
    delta={k:deepcopy(population[k]) for k in ('conditions','any_conditions','change')}
    delta.update(evidence_quote=text,removed_columns=[],included_values=[])
    from core.analysis_agent.population_requirements import canonical_delta,validate
    delta=canonical_delta(delta,review,text,data.get('tables',[]))
    validate(review,text,delta,delta,data.get('tables',[]))
    declared={column for c in review['clauses'] for column in c['columns']}
    if any(c['column'] not in declared for key in ('conditions','any_conditions') for c in delta[key]):
        raise ValueError('Recovered pending predicate lacks quoted human restriction')
    from core.analysis_agent.population_audit import reconcile
    return reconcile(delta,scope,text,sources==previous_sources)
