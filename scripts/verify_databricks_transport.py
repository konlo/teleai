"""Actual read-only Databricks transfer at the unchanged product row boundary.

This is adapter/storage verification, not a natural-language agent quality score.
Data stays in ignored local storage; only shape, digests and timings are reported.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve previous evidence')
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env')
    import pandas as pd
    from core.analysis_agent.assets import AssetDB,PersistentDatasets
    from core.analysis_agent.approvals import ApprovalLedger
    from core.analysis_agent.backends import load_data_backend
    from core.analysis_agent.policy import RuntimePolicy
    from scripts.evaluation_provenance import snapshot
    from utils.analysis_datasets import stored_dataset_digest
    policy=RuntimePolicy.from_env();backend=load_data_backend(ROOT,name='databricks')
    root=ROOT/'.telly_runtime/actual_transport'/uuid4().hex
    db=AssetDB(root,'transport','isolated',max_scope_bytes=policy.scope_disk_quota_bytes)
    store=PersistentDatasets(db,budget=policy.frame_cache_bytes,max_columns=policy.max_dataset_columns,
        max_frame_bytes=policy.max_dataset_bytes,max_full_read_bytes=policy.max_full_read_bytes)
    original=store.register(pd.DataFrame({'sentinel':[3,7,11]}),source='transport.sentinel',coverage='complete')
    db.select_dataset(original.id);before=stored_dataset_digest(store,original.id)
    ledger=ApprovalLedger(db.directory/'approvals.sqlite');cap=policy.max_remote_rows
    source='workspace.default.stormtrooper'
    query=f'SELECT * FROM {source} LIMIT {cap+1}'
    envelope=ledger.envelope(source,query,'Evaluate actual bounded transfer and original preservation',backend.config.identity())
    ledger.propose('one-transfer',envelope);ledger.authorize_automatic('one-transfer',envelope)
    result={'build':snapshot(),'scope':'Actual Databricks adapter/storage; no LLM/agent/browser score',
        'policy_max_rows':cap,'requested_rows':cap+1,'sql':query,'model_calls':0,
        'created_at':datetime.now(timezone.utc).isoformat(),'source':'workspace.default.stormtrooper'}
    start=time.monotonic()
    try:
        returned=ledger.execute('one-transfer',envelope,backend.executor_factory(backend.config,store,max_rows=cap))
        info=store.metadata[returned['dataset']['id']];data_digest=stored_dataset_digest(store,info.id)
        result.update(rows=info.rows,columns=list(info.columns),coverage=info.coverage,
            receipt_status=ledger.get('one-transfer')['status'],result_digest=data_digest,
            original_preserved=stored_dataset_digest(store,original.id)==before,
            selected_original_preserved=db.selected_dataset_id()==original.id,
            cache_bytes=store.frames.bytes)
        db.close();db=AssetDB(root,'transport','isolated',max_scope_bytes=policy.scope_disk_quota_bytes)
        reopened=PersistentDatasets(db,budget=policy.frame_cache_bytes)
        result['restart_preserved']=stored_dataset_digest(reopened,info.id)==data_digest and stored_dataset_digest(reopened,original.id)==before and db.selected_dataset_id()==original.id
        result['status']='PASS' if (info.rows==cap and info.coverage=='truncated' and
            result['receipt_status']=='completed' and all(result[k] for k in
            ['original_preserved','selected_original_preserved','restart_preserved'])) else 'FAIL'
    except Exception as exc:
        result.update(status='FAIL',error_type=type(exc).__name__)
    finally:
        result['elapsed_seconds']=round(time.monotonic()-start,3)
        try:
            import resource
            peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            result['peak_rss_bytes']=peak if sys.platform=='darwin' else peak*1024
        except ImportError:
            result['peak_rss_bytes']=None
        result['platform']=sys.platform
        result['limitations']=['One100k-row boundary transfer, not a full9.5m raw transfer',
            'Million-row stress uses a separate synthetic transport report',
            'No raw records or credentials are exported in this report']
        db.close();a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result.get(k) for k in ['status','error_type','rows','coverage','elapsed_seconds']}))
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
