"""One-shot evaluation queue. Failures are retained; no automatic resubmission.

This runner never deploys or grants GO. The original31 replay must already be
running; it waits for its terminal report before invoking the remaining suites.
Only one live model suite runs at a time. Expectations stay outside model input.
"""
import argparse
from hashlib import sha256
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original-report', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--eval-python', type=Path, required=True)
    p.add_argument('--spider-root', type=Path, required=True)
    args = p.parse_args()
    dest = args.output_dir.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    manifest = dest/'queue.json'
    if manifest.exists():
        raise FileExistsError('Use a new directory; previous evidence is preserved')
    from scripts.evaluation_provenance import snapshot
    report = {'started_at':datetime.now(timezone.utc).isoformat(), 'build':snapshot(),
        'status':'WAITING_ORIGINAL31', 'steps':[], 'release_decision':'NOT_GRADED',
        'scope':'Local Ollama/Databricks, public Spider SQLite, synthetic heldout; company Windows/Azure NOT_RUN'}
    official=args.spider_root.resolve()/'spider2-lite/evaluation_suite/evaluate.py'
    report['official_spider_evaluator_sha256']=sha256(official.read_bytes()).hexdigest()
    def save():
        manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    save()
    started = time.monotonic()
    while True:
        original = json.loads(args.original_report.read_text())
        if original.get('status') != 'RUNNING':
            report['original_status'] = original.get('status')
            break
        if time.monotonic()-started > 7200:
            report['status']='ORIGINAL_WAIT_TIMEOUT'; save(); return 1
        time.sleep(5)
    py = sys.executable
    jobs = [
        ('AI10', [py,'scripts/run_prompt_scenario.py','--scenario','prompt_ai_test_scenario_#1',
            '--run','--backend','databricks','--provider','ollama','--output',str(dest/'ai10.json')]),
        ('HELDOUT8', [py,'scripts/evaluate_natural_language_journeys.py','--provider','ollama',
            '--output',str(dest/'heldout8.json'), *sum((['--id',i] for i in
                ['literal_mean','korean_median','english_mean','exclude_high','followup',
                 'change_measure','chart_compound','scoped_compound']),[])]),
        ('DEEPEVAL_AGENT4',[py,'scripts/evaluate_analysis_agent.py','--live-local-model',
            '--include-final-output','--id','L1_001','--id','L1_016','--id','L1_017',
            '--id','L2_005','--output',str(dest/'deepeval_agent4.json')]),
        ('DEEPEVAL_OFFICIAL',[str(args.eval_python.absolute()),'scripts/evaluate_deepeval_real_report.py',
            '--report',str(dest/'deepeval_agent4.json'),'--output',str(dest/'deepeval_official.json'),
            '--judge-provider','ollama']),
        ('SPIDER_AGENT3',[py,'scripts/evaluate_spider2_teleai.py','--spider-root',
            str(args.spider_root.resolve()),'--id','local002','--id','local003','--id','local004',
            '--interactive','--provider','ollama','--output-dir',str(dest/'spider3')]),
        ('SPIDER_OFFICIAL',[str(args.eval_python.absolute()),str(official),'--mode','sql',
            '--result_dir',str(dest/'spider3/predictions'),'--gold_dir',str(official.parent/'gold'),
            '--max_workers','3']),
        ('ACTUAL_DATABRICKS_TRANSPORT',[py,'scripts/verify_databricks_transport.py',
            '--output',str(dest/'actual_databricks_transport.json')]),
        ('FULL_CONTRACT_SUITE',[py,'-m','pytest','tests','-q']),
    ]
    for name, command in jobs:
        if name=='SPIDER_OFFICIAL':(dest/'spider3/predictions').mkdir(parents=True,exist_ok=True)
        entry={'name':name,'status':'RUNNING','started_at':datetime.now(timezone.utc).isoformat()}
        report['steps'].append(entry); report['status']='RUNNING'; save()
        print(json.dumps(entry),flush=True)
        start=time.monotonic()
        with (dest/(name.lower()+'.log')).open('x') as log:
            process=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        entry.update(status='RECORDED_EXIT',exit_code=process.returncode,
                     elapsed_seconds=round(time.monotonic()-start,3)); save()
        print(json.dumps(entry),flush=True)
    report['official_spider_evaluator_unchanged']=sha256(official.read_bytes()).hexdigest()==report['official_spider_evaluator_sha256']
    report['status']='QUEUE_FINISHED_UNGRADED'; save()
    return 0


if __name__=='__main__':
    sys.path.insert(0,str(ROOT))
    raise SystemExit(main())
