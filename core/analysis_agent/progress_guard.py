"""Detect repeated discovery that adds neither information nor verified work.

Only read-only discovery calls are observed. New input, new observations, new
schema/datasets or verified evidence permit reinspection. No database request
is executed here and tool-provided prose is never promoted to instructions.
"""
import json
from dataclasses import asdict
from hashlib import sha256

from core.analysis_agent.completion import active_contracts

DISCOVERY_TOOLS=frozenset({'list_analysis_context','inspect_dataset','inspect_table_context',
    'inspect_column_definitions','inspect_table_relationships','search_analysis_tools','profile_dataset'})


def digest(value):
    return sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()


def call_key(call):
    return digest([call.get('name'),call.get('args',{})])


def context_version(context,current):
    if context is None:return ''
    # Metadata only. Never restore/decode a cached DataFrame for planning.
    return digest({'datasets':{key:asdict(info)
                              for key,info in context.datasets.metadata.items()},
                   'selected':context.selected_dataset_id,'reference':context.reference_context,
                   'evidence':[current.get(c.evidence_key) for c in active_contracts(current)]})


def observe(current,call,observation,context):
    if call.get('name') not in DISCOVERY_TOOLS or observation.get('status')!='ready':return
    records=current.setdefault('discovery_progress',{})
    key=call_key(call);version=context_version(context,current);observed=digest(observation)
    previous=records.get(key,{})
    same=previous.get('version')==version and previous.get('observation')==observed
    records[key]={'tool':call['name'],'version':version,'observation':observed,
                  'count':previous.get('count',0)+1 if same else 1}


def stalled(current,context):
    candidates={key:record for key,record in current.get('discovery_progress',{}).items()
                if record['count']>=2}
    if not candidates:return {}
    version=context_version(context,current)
    return {key:record for key,record in candidates.items() if record['version']==version}


def token(key,record):
    return digest([key,record['version'],record['observation']])


def instruction(current,available):
    contracts=active_contracts(current)
    plan={'verified':[c.name for c in contracts if c.satisfied(current)],
          'remaining':[{'goal':c.name,'candidate_tools':[t for t in c.tools if t in available]}
                       for c in contracts if not c.satisfied(current)]}
    return ('같은 탐색 결과가 반복되고 분석 완료 근거가 늘지 않았습니다. 목표별 현재 상태:\n'
        +json.dumps(plan,ensure_ascii=False)
        +'\nUse the information already observed to perform the remaining goal. Keep verified results; '
         'do not repeat unchanged discovery. If essential information is missing, ask a different targeted '
         'inspection question or explain the precise missing requirement. Discover tool schemas when needed. '
         'Preserve original source, filters, denominator and coverage. Do not reload or sample data to escape this loop. '
         'Any remote SQL still needs exact-query approval. Do not declare success without verified evidence.')
