"""Summarize observed UI runs, not a synthetic pass/fail judge."""
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SCOPE = ROOT / '.telly_runtime/v1/mysql_eval/44bc5ed80f4a4583ada9f905/0d8cdaafece22d76ffe25df91dc688cf8f4c01a6561f9af6583881cdc33ceae9'
load = lambda name: json.loads((HERE / name).read_text())
baseline, final, journal = load('baseline.json'), load('final.json'), load('ui.json')
logs = [json.loads(line) for line in (SCOPE / 'runtime.jsonl').read_text().splitlines()]
first_id = '232235853a1d44f484cd2a04e418d51d'
first_run = [row for row in logs if row.get('run_id') == first_id]
(HERE / '01_inventory_run.json').write_text(json.dumps(first_run, ensure_ascii=False, indent=2) + '\n')
files = ['01_inventory_run', 'early_failures', '03_age_histogram', '04_education_values',
         '05_histogram_ellipsis', '06_filtered_primary', '07_filtered_multiple',
         '08_columns_followup', '09_preview', '10_scatter', '11_scatter_full',
         '12_alibaba_columns', '13_schema_types', '14_storm_rows', '15_ncr_columns',
         '16_ncr_rows_ellipsis', '17_no_chart_columns', '18_storm_types_ellipsis',
         '19_alibaba_return', '20_large_distribution', '21_y_axis_followup']
outcomes = [
 ('FAIL', 'inventory 목표 mode와 task 불일치. 두 번의 해석 모두 거절; 목록 없음.'),
 ('PASS', '은행 대출이라는 별칭을 bank_loan으로 연결, 실제 18컬럼 출력.'),
 ('FAIL', 'chart 목표 options 계약 오류 두 번. 이미지/SQL 없음.'),
 ('FAIL', '값 목록을 범주형 컬럼 목록으로 오해. longtext를 numeric으로 분류해 0개라고 잘못 완료.'),
 ('FAIL', 'education 대상은 유지했으나 planner 두 번째 호출 전 입력 예산 차단, 이미지 없음.'),
 ('FAIL', 'primary 및 30~40 조건부 시각화 목표 계약 오류 두 번, 실행/이미지 없음.'),
 ('FAIL', 'primary/secondary 및 30~40 조건부 시각화 목표 계약 오류 두 번, 실행/이미지 없음.'),
 ('PASS', 'bank_loan 컬럼 18개를 다시 표시.'),
 ('PASS', '저장된 bank_loan 10행·18열을 실제 표로 표시.'),
 ('PASS', 'age x / balance y 실제 PNG와 전체 750000행·132613좌표 근거 표시, 저장 차트 재사용.'),
 ('PASS', '명시적 전체 산점도 요청도 같은 유효 전체 차트를 표시; 최신 히스토그램 오표시 없음.'),
 ('PASS', 'alibaba_ssd 스키마 107컬럼 표시.'),
 ('PASS', '타입 후속에서 alibaba_ssd 유지. 107개 DB 타입은 독립 schema oracle과 일치.'),
 ('PASS', 'stormtrooper 10행·13열 실제 표, 테이블 목록으로 오분류하지 않음.'),
 ('PASS', 'ncr_ride 21컬럼 표시.'),
 ('PASS', '테이블명을 생략한 후속에도 ncr_ride 유지, MySQL에서 10행·21열 조회·저장·표시.'),
 ('PASS', '추가 부정 지시 probe: stormtrooper 컬럼 13개만 출력, 최신 결과에 차트 없음.'),
 ('PASS', '추가 생략 지시 probe: stormtrooper DB 타입 유지, 독립 oracle 일치.'),
 ('PASS', '대규모 분석 준비: alibaba_ssd 107컬럼으로 다시 전환.'),
 ('FAIL', '분포 대신 distinct profile로 재해석; 생성 SQL의 MySQL 1064 후 unknown으로 차단, 이미지 없음. MySQL인데 Databricks 안내.'),
 ('BLOCKED_DEPENDENCY', '직전 분포 차트 미생성으로 축 변경 여정 불완전. 별도로 chart kind/options 계약 오류도 발생; 범위 변경 이미지 없음.'),
]
cases = []
for index, (entry, file, outcome) in enumerate(zip(journal, files, outcomes), 1):
    raw = load(file + '.json')
    events = raw if isinstance(raw, list) else raw['run']
    terminal = next((row for row in reversed(events) if row['event'] == 'run_completed'), {})
    cases.append({'case': index, 'id': entry['id'], 'prompt': entry['prompt'],
        'verdict': outcome[0], 'reason': outcome[1],
        'run_id': terminal.get('run_id'), 'runtime_status': terminal.get('status'),
        'elapsed_seconds': terminal.get('elapsed_seconds'),
        'model_calls': (raw['current']['model_calls'] if isinstance(raw, dict) else
                        sum(row['event'] == 'goal_contract_error' for row in events)),
        'data_query_completions': sum(row['event'] == 'remote_query_finished' for row in events),
        'evidence': file + '.json',
        'errors': [row for row in events if row['event'] in ('goal_contract_error', 'error')]})
old, new = baseline['assets'], final['assets']
missing = sorted(set(old) - set(new))
changed = [ident for ident in old if ident in new and any(
    old[ident][key] != new[ident][key] for key in ('metadata_sha256', 'payload_sha256'))]
added = sorted(set(new) - set(old))
with sqlite3.connect(f"file:{SCOPE / 'approvals.sqlite'}?mode=ro", uri=True) as db:
    row = db.execute('SELECT id,envelope,status FROM requests ORDER BY rowid DESC LIMIT 1').fetchone()
    envelope = json.loads(row[1])
    last_query = {'id': row[0], 'status': row[2], 'source': envelope['source'], 'query': envelope['query']}
preservation = {'original_assets': len(old), 'final_assets': len(new), 'missing': missing,
    'changed': changed, 'new_asset_ids': added,
    'selection_unchanged': baseline['selection'] == final['selection'],
    'old_assets_sha256_unchanged': not missing and not changed}
assert len(cases) == 21
assert preservation['old_assets_sha256_unchanged'] and preservation['selection_unchanged']
oracles = load('oracles.json')
schema_checks = {}
for filename in ('13_schema_types', '18_storm_types_ellipsis'):
    actual = load(filename + '.json')['current']['metadata_evidence']
    expected = oracles['schemas'][actual['table'].rsplit('.', 1)[-1]]
    assert actual['schema'] == expected
    schema_checks[filename] = {'table': actual['table'], 'columns': len(expected), 'exact_match': True}
report = {'generated_at': datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
 'verdict': 'NO-GO',
 'scope': '실제 브라우저의 기존 대화에 21턴 순서대로 입력. actual Ollama gemma4:e4b + MySQL 개발8504/PID15962. 제품 코드 변경 없음.',
 'method': 'UI 입력으로만 요청 실행. 완료 로그·goal·receipt·PNG·DOM과 독립 MySQL oracle 대조. capture.py는 읽기 전용.',
 'counts': {'requests': 21, 'correct_outputs': 13, 'failed': 7, 'blocked_after_failed_prerequisite': 1,
            'main_cases_correct_rate': '13/20 = 65%; case21 is a dependent journey, not an independent language score'},
 'limitations': ['전체 이력의 반복 요청을 모두 실행한 것이 아닌 대표 여정. 17/18은 추가 probe, 19는 준비 반복.',
   '한 번의 기존 대화·기존 cache 결과. 새 대화/cold start/재시작/다른 표현의 안정성을 증명하지 않는다.',
   '회사 Databricks 및 공식 DeepEval/Spider2 점수는 이번에 실행하지 않았다. 대규모 최초 DB 성능 재측정과 구분.',
   'case10은 명시적 sample 제한 문구 없이 전체750000행으로 해석. 표시10행을 의도하는 경우의 분석 모집단은 별도 수용검사 필요.',
   '불허 goal options 키·값은 현재 로그에 저장되지 않아 generic error에서 정확한 항목명을 확정할 수 없다.'],
 'preservation': preservation, 'schema_oracle_checks': schema_checks, 'cases': cases, 'last_failed_sql': last_query,
 'root_causes': [
  {'priority': 'P0', 'issue': 'LLM goal JSON grammarはoptionsを汎用objectとして許可し、手動validatorだけがcapability別キーを後から拒否。フィードバックは不正キー・許容キーなし。2回の再解釈で基本chartが復旧しない。', 'evidence': 'core/analysis_agent/goal_contract.py:44,97; case1/3/6/7/21'},
  {'priority': 'P0', 'issue': '意味が違うが構造が正しいgoalがそのまま実行/完了。値一覧→列一覧、分布→distinct profile。', 'evidence': 'case4/20; request-goal semantic acceptance未十分'},
  {'priority': 'P0', 'issue': 'longtextはnumeric regexのlongに先に一致。MySQLの文字列schemaをnumericと誤判定。隔離実行でも再現。', 'evidence': 'core/analysis_agent/recovery.py:44; _dtype_family(longtext)=numeric; 独立bank schema'},
  {'priority': 'P0', 'issue': 'モデルビューを最小化した後も12295>12288でHTTP前停止。教育histogramのモデル実行loopが結果へ到達しない。', 'evidence': '05_histogram_ellipsis.json; error c1d2626636d7; second planner budget'},
  {'priority': 'P0', 'issue': '生成SQLの方言検査/MySQLエラー1064の確定分類・安全なSQL修復が不足。ProgrammingErrorをunknown ledgerにして停止、UIはDatabricks案内。', 'evidence': '20_large_distribution.json; error6cb38ea7e59d; last_failed_sql'},
 ],
 'next_acceptance_gate': [
  'capability別options grammarと精密な修正フィードバック; 表inventory・chart・filter・axis全件で実モデル構造失敗なし。',
  '元依頼とgoalの意味照合、schema SQL型正規化、値一覧と頻度分布の出力義務を独立oracleで検証。',
  '入力予算超過をループ開始前に予防/安全な探索・段階分割で自律復旧。',
  'バックエンドSQL構文事前検査、確定1064は接続障害/実行不明と分離して再生成。',
  '既存cache/cold start/新会話、条件継続・変更・失敗後復旧・legend/stack・大規模再使用を同じoracleで再評価。',
 ]}
(HERE / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'counts': report['counts'], 'preservation': preservation, 'verdict': report['verdict']}, ensure_ascii=False))
