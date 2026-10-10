"""Controller-side admission and immutable publication for restricted Python."""
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
import os
import sys
import json
import ast
from hashlib import sha256
from dataclasses import asdict
import pandas as pd
from core.analysis_agent.python_contract import validate_code
from core.analysis_agent.analysis_extensions import scoped_dataset
from utils.analysis_datasets import project_dataset,stored_dataset_digest

def execute(context,dataset_id,columns,code):
    try:tree=validate_code(code)
    except (ValueError,SyntaxError) as exc:
        return {'status':'error','error_code':'python_contract_violation','retryable':True,
                'error_type':type(exc).__name__,
                'message':'Invalid Python expression syntax' if isinstance(exc,SyntaxError) else str(exc),
                'repair_hint':'df/pd/np are supplied. Remove imports, IO, loops and private/inplace access. Worker-local column assignments are allowed. Assign a bounded DataFrame to result or print one final DataFrame.'}
    effective_code=ast.unparse(tree)
    from core.analysis_agent.analysis_extensions import current
    from core.analysis_agent.python_result_contract import validate
    contract=(current(context).get('custom_analysis_spec') or {}).get('result_contract')
    try:validate(contract,columns)
    except ValueError:
        return {'status':'needs_context','error_code':'result_contract_missing','retryable':False,
                'message':'The current goal needs independently agreed numeric result postconditions before code execution.'}
    parent=scoped_dataset(context,dataset_id,columns)
    before=stored_dataset_digest(context.datasets,parent.id)
    inputs=project_dataset(context.datasets,parent.id,columns)
    from core.analysis_agent.python_result_verification import verification_columns
    try:verification_columns(inputs,contract)
    except ValueError as exc:
        return {'status':'unavailable','error_code':'semantic_verification_budget','retryable':False,
                'message':str(exc)+'. Split the calculation into bounded verified steps without changing its population.'}
    with TemporaryDirectory(prefix='telly-analysis-') as directory:
        root=Path(directory)
        inputs.to_parquet(root/'input.parquet',index=False)
        (root/'request.json').write_text(json.dumps({'code':effective_code}),encoding='utf-8')
        # Exclude application credentials and inherited Python startup hooks.
        env={k:v for k,v in os.environ.items() if k in {'PATH','SYSTEMROOT','WINDIR','TEMP','TMP'}}
        env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        try:
            proc=subprocess.run([sys.executable,'-I',str(Path(__file__).with_name('python_worker.py')),str(root)],
                                cwd=root,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=12)
        except subprocess.TimeoutExpired:
            return {'status':'unavailable','error_code':'python_worker_timeout','retryable':False,
                    'message':'Worker stopped at the time limit; original data and goal are preserved.'}
        if proc.returncode!=0 or not (root/'response.json').is_file():
            return {'status':'error','error_code':'python_worker_terminated','retryable':False}
        response=json.loads((root/'response.json').read_text())
        if response['status']!='ready':return response
        output=root/'output.parquet'
        if output.stat().st_size>16*1024*1024:raise ValueError('Serialized output limit exceeded')
        frame=pd.read_parquet(output)
    if before!=stored_dataset_digest(context.datasets,parent.id):
        raise RuntimeError('Input integrity check failed')
    from core.analysis_agent.python_result_verification import verify,normalize_reference_columns
    frame,reference_columns=normalize_reference_columns(frame,columns,contract)
    try:verification=verify(inputs,frame,contract)
    except (ValueError,TypeError,KeyError,MemoryError):
        return {'status':'unavailable','error_code':'semantic_verification_unavailable','retryable':False,
                'message':'Independent verification could not establish the requested computation. No output was published.'}
    if not verification['semantic_verified']:
        return {'status':'error','error_code':'semantic_mismatch','retryable':True,
                'issues':verification['issues'],'expected_result_contract':contract,
                'message':'Execution succeeded but the requested values/order/output contract failed. Correct the code; keep the original goal and dataset. No output was published.'}
    info=context.datasets.register(frame,source=parent.source,parent_id=parent.id,
        snapshot=parent.snapshot,coverage=parent.coverage,predicate_known=parent.predicate_known,
        conditions=parent.conditions,grain='aggregate',aggregation='restricted_python:'+sha256(code.encode()).hexdigest())
    return {'status':'ready','dataset':asdict(info),'python_receipt':{'input_dataset_id':parent.id,
            'input_digest':before,'code_hash':sha256(code.encode()).hexdigest(),'output_dataset_id':info.id,
            'effective_code_hash':sha256(effective_code.encode()).hexdigest(),
            'working_copy_only':True,'terminal_table_output_adapted':tree.table_output_adapted,
            'omitted_input_reference_columns':reference_columns,
            **verification,
            'expression_restricted':True,'wall_time_limit_seconds':12,
            'os_memory_limit':sys.platform.startswith('linux')},
            'message':'제한된 Python 계산을 실행했습니다. 원본 모집단과 파생 결과의 행/관측 단위는 구분하세요.'}
