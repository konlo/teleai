"""Loopback-only, read-only progress view; no model, query or secret access."""
import argparse
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path

HTML=r'''<!doctype html><meta charset="utf-8"><title>Telly 검증 진행</title>
<style>body{font:17px system-ui;background:#101827;color:#e4edf8;margin:40px;max-width:1100px}h1{font-size:28px}.card{background:#1c293d;padding:22px;border-radius:12px;margin:15px 0}p{line-height:1.6}pre{white-space:pre-wrap;font:15px system-ui}strong{color:#f1b861}progress{width:100%;height:18px}</style>
<h1>Telly · 실제 에이전트 검증 진행</h1><p>LLM 판단 → 도구 실행 → 결과 관찰 → 검증. 실행 완료와 검증 통과를 구분합니다.</p>
<div class="card"><strong>현재 GO 판정: 미충족 · NO_GO</strong><p>회사 Windows/Azure 실환경은 여기서 검증하지 못했습니다.</p></div>
<div class="card"><h2>1. 추론 호출 효율</h2><p>outline + 작업별 검토 구조 구현. 실제 모델에서는 실행 전 시간 소진이 남아 있습니다. 구조 구현과 사용자 요청 성공은 별도로 평가합니다.</p></div>
<div class="card"><h2>2. 독립 분석 범위와 작업 의존성</h2><p>작업별 범위·의존성·검증 결과를 영속 저장. 계약 테스트 통과. 실제 복합·후속 요청은 미충족이며 수정이 필요합니다.</p></div>
<div class="card"><h2>3. 실제 사용자 프롬프트</h2><progress id="bar" max="31" value="0"></progress><pre id="original">불러오는 중</pre><pre id="grade"></pre><pre id="extra"></pre></div>
<div class="card"><h2>4. 운영·대규모·공식 평가</h2><p>Databricks 실제 100,000행 전송·원본 보존 검증 통과. 전체 회귀 952 통과 / 4 제외. DeepEval은 보조 평가이며, Spider는 3문항 부분 평가입니다.</p><pre id="scores"></pre><pre id="queue"></pre></div>
<p id="clock"></p><script>
async function update(){try{let x=await(await fetch('/status',{cache:'no-store'})).json();
document.getElementById('bar').value=x.original.recorded;
document.getElementById('original').textContent=`원본31: ${x.original.recorded}/31 실행 완료 · 현재 ${x.original.active||'-'}번 · ${x.original.status}\n실행 상태: ${JSON.stringify(x.original.counts)}`;
document.getElementById('grade').textContent=`독립 채점: ${x.graded_turns}개 판정 · ${JSON.stringify(x.grade)} (실행 완료 수와 별도)`;
document.getElementById('extra').textContent=`AI10: ${x.ai.recorded}/10 실행 완료 · ${x.ai.status}\nAI10 독립 채점: ${JSON.stringify(x.ai_grade)}\n별도8개 여정: ${JSON.stringify(x.heldout)}`;
document.getElementById('queue').textContent=x.queue.map(s=>`${s.name}: ${s.status}${s.exit_code!==undefined?' / exit='+s.exit_code:''}`).join('\n');
document.getElementById('scores').textContent=`DeepEval 독립 판정: ${JSON.stringify(x.deepeval.product)}\n보조 GEval 점수: ${JSON.stringify(x.deepeval.scores)}\nSpider 검증된 최종 SQL: ${x.spider.verified}/${x.spider.attempted} (공식 전체 점수 아님)`;
document.getElementById('clock').textContent='최근 화면 갱신: '+new Date().toLocaleString();
}catch(e){document.getElementById('clock').textContent='진행 정보 로딩 중';}}
update();setInterval(update,5000);</script>'''


def load(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {}


def progress(root):
    active=load(root/'active_evaluation.json')
    def replay(path):
        x=load(path);ts=x.get('turns',[])
        return {'recorded':len(ts),'status':x.get('status','대기'),
            'active':x.get('active_turn',{}).get('id') if x.get('status')=='RUNNING' else None,
            'counts':dict(Counter(t.get('agent_status') for t in ts))}
    queue=root/active.get('remaining','remaining_suites_v2')
    heldout=load(queue/'heldout8.json').get('results',[])
    graded=load(root/active.get('grade','konlo31_partial_grade.json'))
    ai_graded=load(queue/'ai10_grade.json') or load(queue/'ai10_grade_partial.json')
    judged=load(queue/'deepeval_official_native.json').get('results',[])
    spider=load(queue/'spider3/proposals.json').get('results',[])
    return {'original':replay(root/active.get('original','konlo31_live_v2.json')),
        'ai':replay(queue/'ai10.json'),
        'grade':graded.get('counts',{}),'graded_turns':graded.get('recorded_turns',0),
        'ai_grade':ai_graded.get('counts',{}),
        'heldout':dict(Counter(c.get('status') for c in heldout)),
        'deepeval':{'product':dict(Counter(c.get('product_oracle_status') for c in judged)),
                   'scores':[c.get('judge_score') for c in judged]},
        'spider':{'attempted':len(spider),
                  'verified':sum(c.get('status')=='SQL_COMPLETED' for c in spider)},
        'queue':load(queue/'queue.json').get('steps',[])
                +load(queue/'postqueue_qualification.json').get('steps',[])}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report-root',type=Path,required=True)
    p.add_argument('--port',type=int,default=8768);a=p.parse_args()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path=='/':body=HTML.encode();kind='text/html; charset=utf-8'
            elif self.path=='/status':body=json.dumps(progress(a.report_root),ensure_ascii=False).encode();kind='application/json; charset=utf-8'
            else:self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store')
            self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def log_message(self,*args):pass
    print(f'Progress view: http://127.0.0.1:{a.port}/',flush=True)
    ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()


if __name__=='__main__':main()
