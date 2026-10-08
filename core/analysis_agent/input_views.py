"""Lossless schema identities and bounded, discoverable model context views.

These are projections for inference. Saved tool results and custom instructions
are never rewritten here.
"""
import json
from copy import deepcopy
from langchain_core.messages import ToolMessage


CATALOG_KEY='telly_catalog_block'


def compact_system_catalog(system,current, *, minimal=False):
    if system is None:return system
    block=system.additional_kwargs.get(CATALOG_KEY)
    if not isinstance(block,str) or str(system.content).count(block)!=1:return system
    try:catalog=json.loads(block)
    except (ValueError,TypeError):return system
    sources=current.get('required_sources') or (current.get('confirmed_analysis') or {}).get('required_sources',[])
    wanted={str(s).casefold() for s in sources}
    datasets=catalog.get('datasets',[])
    tables=catalog.get('available_tables',[])
    small={**catalog,'datasets':[d for d in datasets if str(d.get('source')).casefold() in wanted],
        'available_tables':[t for t in tables if str(t.get('table')).casefold() in wanted],
        'catalog_projection':{'total_datasets':len(datasets),'total_tables':len(tables),
            'full_catalog_tool':'list_analysis_context',
            'note':'현재 출처 관련 identity만 제공합니다. 다른 출처는 list_analysis_context로 찾으세요.'}}
    if minimal:
        # Catalog identities are discoverable, unlike protected request and
        # receipt facts. Never let many old snapshots crowd out the next step.
        small={'datasets':[], 'available_tables':[{'table':s} for s in sources],
               'catalog_projection':{'total_datasets':len(datasets),'total_tables':len(tables),
                                     'full_catalog_tool':'list_analysis_context'}}
    replacement=json.dumps(small,ensure_ascii=False,default=str)
    if len(replacement.encode())>=len(block.encode()):return system
    return system.model_copy(update={'content':str(system.content).replace(block,replacement),
        'additional_kwargs':{**system.additional_kwargs,CATALOG_KEY:replacement}})


def schema_observation(message):
    if not isinstance(message,ToolMessage) or message.name!='inspect_table_context':return message
    try:
        data=json.loads(message.content)
        table=data.get('table_context',{})
        columns=table.get('columns',[])
        if len(columns)<=24 or not all(isinstance(c,dict) and c.get('name') for c in columns):return message
        grouped={}
        details=[]
        for column in columns:
            grouped.setdefault(str(column.get('dtype') or ''),[]).append(column['name'])
            extra={k:v for k,v in column.items() if k not in {'name','dtype'} and v is not None}
            if extra:details.append({'name':column['name'],**extra})
        data=deepcopy(data)
        projected=data['table_context']
        projected.pop('columns')
        projected['columns_by_dtype']=grouped
        if details:projected['column_details']=details
        data['model_view_projection']={'column_count':len(columns),'all_column_names_preserved':True,
            'all_column_types_preserved':True,'full_schema_tool':'inspect_table_context',
            'note':'columns_by_dtype는 데이터 타입별 전체 컬럼 이름입니다. authority/freshness/scope 및 추가 컬럼 설명은 그대로 보존했습니다.'}
        content=json.dumps(data,ensure_ascii=False,default=str)
        if len(content.encode())>=len(message.content.encode()):return message
        return message.model_copy(update={'content':content})
    except (ValueError,TypeError,AttributeError):return message


def payload_components(system,messages,tools):
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from langchain_core.messages import AIMessage
    size=lambda v:len(json.dumps(v,ensure_ascii=False,separators=(',',':'),default=str).encode('utf8'))
    return {'system_bytes':len(str(system.content).encode()) if system else 0,
        'message_content_bytes':sum(len(str(m.content).encode()) for m in messages),
        'tool_calls_bytes':sum(size(m.tool_calls) for m in messages if isinstance(m,AIMessage) and m.tool_calls),
        'reasoning_bytes':sum(len(str(m.additional_kwargs.get('reasoning_content','')).encode()) for m in messages),
        'tool_schema_bytes':sum(size(convert_to_openai_tool(t)) for t in tools),
        'tool_names':[t.name for t in tools]}
