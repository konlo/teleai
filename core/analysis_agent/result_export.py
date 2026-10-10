"""Scope-owned exports with a provenance manifest; no arbitrary file paths."""
from dataclasses import asdict
from io import BytesIO
from uuid import uuid4
import json
import zipfile
from utils.analysis_datasets import project_dataset,full_read_preflight
from core.analysis_agent.analysis_extensions import scoped_dataset

def export_result(context,dataset_id='',chart_id='',format='csv',columns=None):
    if format not in {'csv','parquet','png'}:raise ValueError('Unsupported export format')
    if bool(dataset_id)==bool(chart_id):raise ValueError('Exactly one dataset_id or chart_id is required')
    db=(context.runtime_services or {}).get('asset_db')
    if db is None:return {'status':'unavailable','error_code':'export_store_unavailable'}
    output=BytesIO()
    if chart_id:
        if columns:raise ValueError('A saved PNG cannot change its columns; render a new chart first')
        if format!='png':raise ValueError('Charts export as PNG')
        card=context.artifacts[chart_id];info=context.datasets.metadata[card.dataset_id]
        if (context.runtime_services or {}).get('current',lambda:{})():
            scoped=scoped_dataset(context,info.id,list(card.columns),preserve_grain=True)
            if scoped.id!=info.id:raise ValueError('Saved chart population differs from requested export scope')
        payload=card.image;name='chart.png'
        manifest={'dataset':asdict(info),'chart':{k:v for k,v in asdict(card).items() if k!='image'}}
    else:
        if format=='png':raise ValueError('Dataset export requires CSV or Parquet')
        info=context.datasets.metadata[dataset_id]
        state=(context.runtime_services or {}).get('current',lambda:{})()
        chosen=list(columns or state.get('required_columns') or info.columns)
        if len(chosen)!=len(set(chosen)) or not set(chosen)<=set(info.columns):raise ValueError('Export needs distinct existing columns')
        if state.get('required_columns') and set(chosen)!=set(state['required_columns']):raise ValueError('Export columns differ from current goal')
        if (context.runtime_services or {}).get('current',lambda:{})():
            info=scoped_dataset(context,dataset_id,chosen)
            dataset_id=info.id
        if info.rows>100000 or full_read_preflight(context.datasets,[dataset_id]):
            return {'status':'rejected','error_code':'export_limit','retryable':False}
        frame=project_dataset(context.datasets,dataset_id,chosen)
        if format=='csv':payload=frame.to_csv(index=False).encode('utf-8-sig')
        else:
            buffer=BytesIO();frame.to_parquet(buffer,index=False);payload=buffer.getvalue()
        name='result.'+format;manifest={'dataset':asdict(info),'exported_columns':chosen}
    if len(payload)>16*1024*1024:raise ValueError('Export byte limit exceeded')
    manifest.update(format=format,coverage_note='Coverage and conditions describe this saved result, not necessarily the whole source table.')
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name,payload);archive.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,default=str))
    key=str(uuid4());metadata={'id':key,'dataset_id':info.id,'chart_id':chart_id,'format':format,'filename':'analysis-'+key[:8]+'.zip','bytes':len(output.getvalue())}
    if not chart_id:metadata['columns']=chosen
    db.put(key,'export',metadata,output.getvalue())
    return {'status':'ready','export':metadata,'evidence_ids':[key],'scope':asdict(info)}
