"""Prepare a concrete smoke query; execution needs its explicit approval hash."""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from dotenv import load_dotenv
from migration.graph_runtime import GraphAnalysisRuntime
from migration.remote_backend import ConnectionConfig, make_executor
from migration.test_persistent_runtime import QuietModel


def query_spec():
    fixture=json.loads(Path('tests/fixtures/analysis_acceptance.json').read_text())
    def literal(value):
        if isinstance(value,(int,float)):return str(value)
        return "'"+str(value).replace("'","''")+"'"
    columns=list(fixture['rows'][0])
    rows=','.join('('+','.join(literal(row[c]) for c in columns)+')' for row in fixture['rows'])
    query='SELECT * FROM VALUES '+rows+' AS fixture('+','.join(columns)+')'
    return query,sha256(query.encode()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--approve-query-sha256')
    args=parser.parse_args()
    query,fingerprint=query_spec()
    if not args.approve_query_sha256:
        print(json.dumps({'query':query,'sha256':fingerprint,'rows':7,'execution':'NOT EXECUTED'},ensure_ascii=False,indent=2))
        return
    if args.approve_query_sha256!=fingerprint:raise PermissionError('Approval hash does not match query')
    load_dotenv()
    config=ConnectionConfig.from_env()
    if not all([config.server_hostname,config.http_path,config.access_token]):raise ValueError('Missing connection configuration')
    runtime=GraphAnalysisRuntime('.telly_runtime/v1','local-owner','approved-connection-smoke',QuietModel(),
        connection_identity=config.identity(),remote_factory=lambda d:make_executor(config,d))
    # Never silently create another request if this smoke run already has state.
    if runtime.events():raise RuntimeError('Existing smoke run: inspect saved state; do not resubmit')
    pending=runtime.propose_query('synthetic acceptance fixture',query,'사용자가 승인한 연결 검증용 합성 데이터 7행 조회')['requests'][0]
    result=runtime.respond(pending['id'],approved=True)
    receipt=runtime.ledger.get(pending['id'])
    checks=[]
    if receipt['status']=='completed':
        from core.analysis_sql import local_query
        fixture=json.loads(Path('tests/fixtures/analysis_acceptance.json').read_text())
        dataset_id=receipt['result']['dataset']['id']
        frame=runtime.datasets.frames[dataset_id]
        for turn in fixture['turns']:
            calculated,_,_=local_query(frame,turn['sql'])
            actual=float(calculated.iloc[0,0])
            checks.append({'expected':turn['expected'],'actual':actual,'pass':actual==turn['expected']})
        cards=runtime.recommend_charts(dataset_id)
        checks.append({'png_cards':len(cards['cards']),'pass':bool(cards['cards'])})
    Path('docs/v1_databricks_smoke.json').write_text(json.dumps({
        'query':query,'sha256':fingerprint,'result':result,'receipt':receipt,'checks':checks},ensure_ascii=False,indent=2,default=str)+'\n')
    print(json.dumps({'status':receipt['status'],'query_sha256':fingerprint},ensure_ascii=False))
    runtime.close()
    return int(receipt['status']!='completed' or not all(c['pass'] for c in checks))

if __name__=='__main__':raise SystemExit(main())
