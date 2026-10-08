"""Read-only, bounded diagnostic summaries. Never export prompts or SQL bodies."""
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
from importlib import metadata
import json
import marshal
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
TOKEN = re.compile(r'[A-Za-z_][A-Za-z0-9_.-]{0,95}\Z')
HEX = re.compile(r'[a-f0-9]{12,64}\Z')
STATUSES = {'answered','blocked','exhausted','needs_data','incomplete','awaiting_approval',
            'ready','ok','error','unavailable','proposed','approved','auto_authorized',
            'submitting','unknown','failed','completed','rejected','invalidated'}


def token(value):
    return value if isinstance(value,str) and TOKEN.fullmatch(value) else None


def number(value):
    return value if type(value) in (int,float) and 0 <= value < 10**15 else None


def identifier(value):
    return value if isinstance(value,str) and HEX.fullmatch(value) else None


@lru_cache(maxsize=1)
def process_code_identity():
    # Captured once, not silently updated by a git pull while modules are loaded.
    try:
        result=subprocess.run(['git','-C',str(ROOT),'rev-parse','HEAD'],
                              capture_output=True,text=True,timeout=2)
        revision=identifier(result.stdout.strip()) if result.returncode==0 else None
    except (OSError,subprocess.SubprocessError):
        revision=None
    packages={}
    for name in ('langchain','langchain-core','langgraph','streamlit','sqlglot'):
        try:
            value=metadata.version(name)
            packages[name]=value if re.fullmatch(r'[0-9][a-zA-Z0-9.+-]{0,40}',value) else None
        except metadata.PackageNotFoundError:
            packages[name]=None
    return {'revision_at_first_runtime':revision,'packages':packages,
            'python_version':sys.version.split()[0],
            'identity_captured_at':datetime.now(timezone.utc).isoformat()}


def runtime_identity(route, model_class, require_approval):
    from core.analysis_agent.remote_completion import deferred_execution_claim
    from core.analysis_agent.memory import Transcript
    # Fingerprint actual loaded guard bytecode, not a reread of changed disk files.
    loaded=marshal.dumps(deferred_execution_claim.__code__)+marshal.dumps(Transcript._delivery_message.__code__)
    return {**process_code_identity(),'route':route,'model_class':model_class,
            'require_remote_approval':bool(require_approval),
            'loaded_delivery_guard':hashlib.sha256(loaded).hexdigest()[:16]}


def read_events(path):
    events=[]
    integrity={'malformed_lines':0,'bounded_read':False,'unreadable_files':0}
    # The existing logger retains 3 backups. Never follow arbitrary paths in logs.
    for file in [Path(str(path)+'.'+str(i)) for i in (3,2,1)]+[Path(path)]:
        if not file.exists():continue
        try:
            with file.open('rb') as stream:
                size=file.stat().st_size
                if size>2_100_000:
                    stream.seek(size-2_100_000);stream.readline()
                    integrity['bounded_read']=True
                for raw in stream:
                    try:
                        entry=json.loads(raw)
                        if not isinstance(entry,dict):raise ValueError()
                        if not isinstance(entry.get('event'),str):raise ValueError()
                        for key in ('status','ledger_status'):
                            if key in entry and not isinstance(entry[key],str):entry[key]=None
                        events.append(entry)
                    except (ValueError,UnicodeDecodeError):
                        integrity['malformed_lines']+=1
        except OSError:
            integrity['unreadable_files']+=1
    return events,integrity


def public_runtime(value):
    if not isinstance(value,dict):return {}
    result={key:token(value.get(key)) for key in ('route','model_class')}
    for key in ('goal_model','planner_model'):
        name=value.get(key)
        result[key]=name if isinstance(name,str) and re.fullmatch(r'[a-zA-Z0-9._:/-]{1,128}',name) else None
    result['data_backend'] = value.get('data_backend') if value.get('data_backend') in {'databricks','mysql','sqlite'} else None
    result.update({key:identifier(value.get(key)) for key in
                   ('revision_at_first_runtime','loaded_delivery_guard')})
    result['require_remote_approval']=value.get('require_remote_approval') if type(value.get('require_remote_approval')) is bool else None
    result['python_version']=value.get('python_version') if re.fullmatch(r'[0-9.]{1,20}',str(value.get('python_version'))) else None
    packages=value.get('packages',{})
    result['packages']={key:val for key,val in packages.items()
        if key in {'langchain','langchain-core','langgraph','streamlit','sqlglot'}
        and isinstance(val,str) and re.fullmatch(r'[0-9][a-zA-Z0-9.+-]{0,40}',val)} if isinstance(packages,dict) else {}
    return result


def public_input_budget(value):
    if not isinstance(value,dict):return {}
    result={k:number(value.get(k)) for k in ('payload_bytes','payload_bytes_before_projection',
        'template_headroom','input_budget_units','output_reserved','context_window','tool_count','message_count')}
    result.update({k:value.get(k) if type(value.get(k)) is bool else None for k in
        ('within_budget','older_turns_compacted','schema_compacted','system_catalog_compacted',
         'tool_menu_compacted','minimal_discovery_menu','reasoning_compacted','observations_compacted')})
    for key in ('minimal_catalog','builtin_instructions_compacted'):
        result[key]=value.get(key) if type(value.get(key)) is bool else None
    components=value.get('components')
    if isinstance(components,dict):
        result['components']={k:number(components.get(k)) for k in
            ('system_bytes','message_content_bytes','tool_calls_bytes','reasoning_bytes','tool_schema_bytes')}
        names=components.get('tool_names',[])
        result['components']['tool_names']=[token(n) for n in names[:32] if token(n)] if isinstance(names,list) else []
    result['measurement']='conservative_utf8_bytes_not_actual_token_count'
    return result


def summarize(path, *, run_id=None, error_id=None):
    """Exact selectors never silently fall back to another request."""
    if run_id and error_id:raise ValueError('Choose run_id or error_id, not both')
    events,integrity=read_events(path)
    selected_error=None
    if error_id:
        selected_error=next((e for e in reversed(events) if e.get('event')=='error' and e.get('error_id')==error_id),None)
        run_id=selected_error.get('run_id') if selected_error else None
        selected=[e for e in events if run_id and e.get('run_id')==run_id] if run_id else ([selected_error] if selected_error else [])
    else:
        if run_id is None:
            latest=next((e for e in reversed(events) if e.get('run_id')),None)
            run_id=latest.get('run_id') if latest else None
        selected=[e for e in events if run_id and e.get('run_id')==run_id]
    base={'report_version':1,'found':bool(selected),'run_id':identifier(run_id),'log_integrity':integrity}
    if not selected:return base
    counts=Counter(e.get('event') for e in selected if isinstance(e.get('event'),str))
    start=next((e for e in selected if e.get('event')=='run_started'),{})
    finish=next((e for e in reversed(selected) if e.get('event') in {'run_completed','run_paused'}),{})
    completion=next((e for e in reversed(selected) if e.get('event')=='completion_checked'),{})
    budget=next((e for e in reversed(selected) if e.get('event')=='model_payload_budget'),{})
    errors=[]
    for e in selected:
        if e.get('event')!='error':continue
        frames=e.get('frames',[])
        errors.append({'error_id':identifier(e.get('error_id')),'stage':token(e.get('stage')),
                       'error_type':token(e.get('error_type')),'http_status':number(e.get('http_status')),
                       'error_category':token(e.get('error_category')),
                       'database_errno':number(e.get('database_errno')),
                       'frames':[{'file':token(f.get('file')),'line':number(f.get('line')),
                                  'function':token(f.get('function'))} for f in frames[-3:] if isinstance(f,dict)] if isinstance(frames,list) else []})
    status='awaiting_approval' if finish.get('event')=='run_paused' else finish.get('status')
    remote=[e for e in selected if e.get('event')=='remote_query_finished']
    tools=[e for e in selected if e.get('event') in {'tool_started','tool_completed','tool_rejected','remote_query_started','remote_query_finished'}]
    flags=[]
    if not start:flags.append('RUN_START_NOT_RETAINED')
    if not finish:flags.append('NO_TERMINAL_EVENT')
    if counts['unsupported_deferred_reply_rejected'] or counts['delivery_guard_blocked']:flags.append('DEFERRED_REPLY_BLOCKED')
    if status in {'blocked','exhausted','needs_data','incomplete'}:flags.append('AGENT_NOT_COMPLETED')
    if counts['tool_started']==0 and counts['remote_query_started']==0:flags.append('NO_TOOL_EXECUTION_OBSERVED')
    if any(e.get('ledger_status') in {'unknown','submitting'} for e in remote):flags.append('REMOTE_SUBMISSION_UNCERTAIN')
    if errors:flags.append('ERROR_OBSERVED')
    if any(e.get('stage') in {'chart_display','selected_chart_display'} for e in errors):flags.append('CHART_DISPLAY_FAILED')
    def tally(field,event):
        return dict(Counter(token(e.get(field)) or 'other' for e in selected if e.get('event')==event))
    # Time comes from our logger, but still validate old/untrusted lines.
    rawtime=selected[-1].get('time','')
    timestamp=rawtime if isinstance(rawtime,str) and re.fullmatch(r'[0-9T:.+Z-]{10,40}',rawtime) else None
    return {**base,'time_utc':timestamp,'instance_id':identifier(start.get('instance_id')),
            'pid':number(start.get('pid')),'runtime':public_runtime(start.get('runtime')),
            'status':status if status in STATUSES else 'not_recorded',
            'elapsed_seconds':number(finish.get('elapsed_seconds')),
            'last_event':token(selected[-1].get('event')),'flags':flags,
            'model_calls':counts['model_call_started'],
            'input_budget':public_input_budget(budget) if budget else {},
            'proposed_tool_calls':sum(number(e.get('tool_call_count')) or 0 for e in selected if e.get('event')=='model_call_finished'),
            'local_tools_started':tally('tool','tool_started'),
            'local_tools_rejected':tally('reason','tool_rejected'),
            'last_tool_events':[{'event':token(e.get('event')),'tool':token(e.get('tool')),
                                'status':e.get('status') if e.get('status') in STATUSES else None,
                                'error_code':token(e.get('error_code')),
                                'ledger_status':e.get('ledger_status') if e.get('ledger_status') in STATUSES else None}
                               for e in tools[-6:]],
            'remote':{'started':counts['remote_query_started'],'finished':len(remote),
                      'states':dict(Counter(e.get('ledger_status') if e.get('ledger_status') in STATUSES else 'other' for e in remote)),
                      'cached_receipts':sum(e.get('cached_receipt') is True for e in remote)},
            'completion':{'status':completion.get('status') if completion.get('status') in STATUSES else None,
                          'reason':token(completion.get('reason')),
                          'required_capabilities':[token(x) for x in completion.get('required_capabilities',[]) if token(x)] if isinstance(completion.get('required_capabilities',[]),list) else [],
                          'missing_capabilities':[token(x) for x in completion.get('missing_capabilities',[]) if token(x)] if isinstance(completion.get('missing_capabilities',[]),list) else [],
                          'attempts':number(completion.get('attempts'))},
            'recovery_replans':counts['recovery_replan'],
            'deferred_reply_blocks':counts['unsupported_deferred_reply_rejected']+counts['delivery_guard_blocked'],
            'error_count':len(errors),'errors':errors[-5:]}


def brief(report):
    """Small enough to transcribe manually when files cannot leave the server."""
    if not report.get('found'):return '진단: 기록 없음 (대화/ID/보관 기간 확인 필요)'
    runtime=report['runtime']; completion=report['completion']; remote=report['remote']
    last=report['errors'][-1] if report['errors'] else {}
    budget=report.get('input_budget') or {}
    budget_text=(f" / 입력 추정(bytes): {budget.get('payload_bytes')}+{budget.get('template_headroom')}/{budget.get('input_budget_units')}" if budget else '')
    return '\n'.join([
        f"시각(UTC): {report['time_utc']}",
        f"실행 ID: {report['run_id']} / 오류 ID: {last.get('error_id')}",
        f"버전: {runtime.get('revision_at_first_runtime')} / 경로: {runtime.get('route')}",
        f"상태: {report['status']} / 마지막 이벤트: {report['last_event']}",
        '진단 코드: '+', '.join(report['flags']),
        f"주 모델 호출: {report['model_calls']} / 로컬 도구: {report['local_tools_started']}"+budget_text,
        f"원격 도구 시작: {remote['started']} / 조회 장부: {remote['states']} / 저장 결과 재사용: {remote['cached_receipts']}",
        f"완료 판정: {completion['status']} / 이유: {completion['reason']} / 미충족: {completion['missing_capabilities']}",
        f"복구 재계획: {report['recovery_replans']} / 미래 안내 차단 관측: {report['deferred_reply_blocks']}",
        f"최근 오류: 단계={last.get('stage')}, 유형={last.get('error_type')}, HTTP={last.get('http_status')}",
        f"오류 위치: {last.get('frames',[])}",
        f"로그 완전성: {report['log_integrity']}",
    ])
