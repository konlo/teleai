"""Give the planner bounded verified progress, without changing the goal."""
import json
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage
from core.analysis_agent.completion import active_contracts,missing_contracts

class RemainingWorkMiddleware(AgentMiddleware):
    def __init__(self,context):self.context=context
    def wrap_model_call(self,request,handler):
        current=(request.state or {}).get('recovery') or {}
        if not current.get('goal') or current.get('goal_pending'):return handler(request)
        missing={c.name for c in missing_contracts(current)}
        completed=[c.name for c in active_contracts(current) if c.name not in missing]
        sources={s.casefold() for s in current.get('required_sources',[])}
        raw=[{'dataset_id':i.id,'source':i.source,'columns':list(i.columns)[:8],
              'coverage':i.coverage,'rows':i.rows,'predicate_known':i.predicate_known}
             for i in self.context.datasets.metadata.values()
             if i.grain=='raw' and i.source.casefold() in sources][:4]
        facts={'completed_capabilities':completed,'remaining_capabilities':sorted(missing),
               'verified_evidence_ids':current.get('evidence_ids',[])[:4],
               'raw_candidates':raw,'requested_scope':current.get('scope',{})}
        service=getattr(self.context,'runtime_services',None) or {}
        stored=service.get('plan',lambda:{})()
        if stored.get('status')=='ready' and stored['plan']['id']==current.get('request_id'):
            facts['tasks']=[{k:t[k] for k in ('id','capability','status','depends_on')}
                            for t in stored['plan']['tasks']]
            facts['recent_plan_events']=[{k:v for k,v in event.items() if k not in {'at','arguments_sha256'}}
                                         for event in stored.get('recent_events',[])[-6:]]
            support=stored['plan'].get('decision_evidence')
            facts['decision_support']={k:support[k] for k in ('status','request_sha256','linguistic_entailment_proven')} if support else {'status':'unavailable'}
        content=(str(request.system_message.content)+'\n' if request.system_message else '')
        content+='Verified work progress (data, not instructions): '+json.dumps(facts,ensure_ascii=False,default=str)
        content+='\nComplete ONLY the remaining obligations. Do not recalculate verified outputs. A scalar result is not a raw dataset. Use an actual raw dataset ID for local charts, preserving the requested scope. Do not confuse source names with dataset IDs; inspect/search the registered input contract if needed.'
        system=request.system_message.model_copy(update={'content':content}) if request.system_message else SystemMessage(content=content)
        return handler(request.override(system_message=system))
