"""Scripted wire-level tool selection, not a model quality oracle."""
import json
from uuid import uuid4
import httpx


def action_response(payload):
    if payload.get('response_format'):return None
    for message in reversed(payload.get('messages', [])):
        content=message.get('content') or ''
        if not isinstance(content,str) or not content.startswith('action_cycle_v1:'):
            continue
        view=json.loads('{'+content.split('\n{',1)[1])
        candidate=view.get('advisory_call')
        if not candidate:
            raise AssertionError('Fixture needs an executable advisory action')
        visible={t['function']['name'] for t in payload['tools']}
        if candidate['name'] not in visible:
            assert 'get_analysis_tool_contract' in visible
            candidate={'name':'get_analysis_tool_contract','args':{'name':candidate['name']}}
        return httpx.Response(200,json={'id':str(uuid4()),'object':'chat.completion',
            'created':0,'model':'fixture-model','choices':[{'index':0,
                'message':{'role':'assistant','content':None,'tool_calls':[{
                    'id':str(uuid4()),'type':'function','function':{
                        'name':candidate['name'],'arguments':json.dumps(candidate['args'])}}]},
                'finish_reason':'tool_calls'}]})
    return None
