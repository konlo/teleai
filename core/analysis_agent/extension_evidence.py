"""Bind extension receipts to admitted calls and current source/output obligations."""
from copy import deepcopy
from hashlib import sha256
import json
from langchain_core.messages import ToolMessage

def valid_call(context,current,call):
    names={'execute_analysis_python':'custom_analysis_spec','render_advanced_eda':'advanced_eda_spec','export_analysis_result':'export_spec'}
    name=call.get('name')
    if name not in names:return None
    spec=current.get(names[name]);args=call.get('args') or {}
    if not spec:return False
    if name=='export_analysis_result':
        if args.get('format')!=spec['format']:return False
        if args.get('columns') and set(args['columns'])!=set(current.get('required_columns',[])):return False
        if args.get('chart_id'):
            card=context.artifacts.get(args['chart_id'])
            info=context.datasets.metadata.get(card.dataset_id) if card else None
        else:info=context.datasets.metadata.get(args.get('dataset_id'))
    else:
        info=context.datasets.metadata.get(args.get('dataset_id'))
        if set(current.get('required_columns',[]))!=set(args.get('columns',[])):return False
    if not info or info.source not in current.get('required_sources',[]):return False
    if current.get('current_result_only'):
        from core.analysis_agent.analysis_extensions import retained_input_id
        if info.id!=retained_input_id(context,current):return False
    if name=='render_advanced_eda' and args.get('kind')!=spec['kind']:return False
    return True

def collect(context,current,messages,calls):
    for message in messages:
        if not isinstance(message,ToolMessage):continue
        call=calls.get(message.tool_call_id)
        if not call or valid_call(context,current,call) is not True:continue
        try:result=json.loads(message.content)
        except (ValueError,TypeError):continue
        if result.get('status')!='ready':continue
        name=call['name']
        if name=='execute_analysis_python':
            receipt=result.get('python_receipt') or {}
            output=context.datasets.metadata.get(receipt.get('output_dataset_id'))
            parent=context.datasets.metadata.get(receipt.get('input_dataset_id'))
            if not output or not parent or output.parent_id!=parent.id or output.source!=parent.source:continue
            if parent.id!=call['args']['dataset_id'] and parent.parent_id!=call['args']['dataset_id']:continue
            code_hash=sha256(call['args']['code'].encode()).hexdigest()
            if receipt.get('code_hash')!=code_hash or output.aggregation!='restricted_python:'+code_hash:continue
            from utils.analysis_datasets import stored_dataset_digest
            if receipt.get('input_digest')!=stored_dataset_digest(context.datasets,parent.id):continue
            from core.analysis_agent.python_result_contract import verified
            if not verified({**current,'custom_analysis_evidence':receipt}):continue
            current['custom_analysis_evidence']=deepcopy(receipt)
        elif name=='render_advanced_eda':
            receipt=result.get('advanced_eda_receipt') or {}
            cards=[c['id'] for c in result.get('cards',[]) if c.get('id') in context.artifacts]
            if not cards or receipt.get('kind')!=current['advanced_eda_spec']['kind']:continue
            if receipt.get('columns')!=call['args']['columns']:continue
            parent=context.datasets.metadata.get(receipt.get('dataset_id'))
            if not parent or (parent.id!=call['args']['dataset_id'] and parent.parent_id!=call['args']['dataset_id']):continue
            if any(context.artifacts[key].dataset_id!=parent.id or context.artifacts[key].kind!=receipt['kind'] for key in cards):continue
            current['advanced_eda_evidence']=deepcopy(receipt)
            current['artifact_ids']=list(dict.fromkeys(current.get('artifact_ids',[])+cards))
        else:
            exported=result.get('export') or {}
            db=(context.runtime_services or {}).get('asset_db')
            if not db or exported.get('id') not in db.metadata('export'):continue
            if (current.get('required_columns') and exported.get('format')!='png'
                    and set(exported.get('columns',[]))!=set(current['required_columns'])):continue
            current['export_evidence']=deepcopy(exported)

def render_python(runtime,current):
    receipt=current['custom_analysis_evidence'];store=runtime.context.datasets;info=store.metadata[receipt['output_dataset_id']]
    from utils.analysis_datasets import preview_dataset
    frame=preview_dataset(store,info.id,limit=10)
    def cell(value):return str(value).replace('|','\\|').replace('\n',' ')
    table='\n'.join(['| '+' | '.join(map(cell,frame.columns))+' |','| '+' | '.join('---' for _ in frame.columns)+' |']+
                    ['| '+' | '.join(map(cell,row))+' |' for row in frame.itertuples(index=False,name=None)])
    return f'제한된 Python 분석을 실행하고 파생 결과 {info.rows:,}행을 저장했습니다. 출처: {info.source} · {info.coverage}\n\n'+table

def render_eda(runtime,current):
    proof=current['advanced_eda_evidence']
    return f"{proof['kind']} 시각화 이미지를 생성했습니다. {proof['scope']}"

def render_export(runtime,current):
    proof=current['export_evidence']
    return '분석 결과와 출처·조건 manifest를 ZIP으로 저장했습니다. 다운로드 파일: '+proof['filename']
