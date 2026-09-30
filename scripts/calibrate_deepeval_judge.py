"""Bounded local GEval calibration with explicitly labelled synthetic controls."""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model')
    p.add_argument('--controls', type=Path, help='Independent labelled controls JSON')
    p.add_argument('--provider',choices=['ollama','databricks'],default='ollama')
    a = p.parse_args()
    os.environ['DEEPEVAL_TELEMETRY_OPT_OUT'] = '1'
    os.environ['DEEPEVAL_RETRY_MAX_ATTEMPTS'] = '1'
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    from evaluation_judge import make_judge, GROUNDING_STEPS
    model,judge_name=make_judge(a.provider,a.model)
    cases = [
        ('correct_groups', '그룹별 평균을 보여줘.', 'A 평균 1.5, B 평균 3.5입니다.', 'A=1.5; B=3.5', True),
        ('missing_goal', '그룹별 평균을 보여줘.', '결측값은 0개입니다. 분석을 완료했습니다.', 'A=1.5; B=3.5', False),
        ('wrong_number', '평균을 보여줘.', '평균은 25.0입니다.', 'mean=2.5', False),
        ('wrong_bins', '5개 구간 히스토그램을 보여줘.', '히스토그램 생성 완료. 실제 렌더 설정: bins=20.', 'Rendered histogram must have bins=5.', False),
    ]
    cases.extend([
        ('correct_scalar','평균을 보여줘.','평균은 2.5입니다.','The arithmetic mean is 2.5.',True),
        ('correct_percent','전환율을 백분율로 알려줘.','전환율은 25%입니다.','25 percent (ratio 0.25).',True),
        ('correct_equivalent_format','그룹별 평균을 보여줘.','B: 3.5, A: 1.5','A=1.5; B=3.5. Order is unspecified.',True),
        ('correct_rounded','평균을 소수점 둘째 자리까지 알려줘.','평균은 1.23입니다.','Exact mean=1.234; round to two decimal places.',True),
        ('wrong_scope','2025년 평균을 알려줘.','2024년 평균은 2.5입니다.','2025 population mean=8.5.',False),
        ('wrong_denominator','전체 100명 중 성공 25명의 성공률을 알려줘.','성공률은 100%입니다.','25 / 100 = 25%.',False),
        ('future_promise','테이블 목록을 보여줘.','결과가 도착하면 보여드리겠습니다.','Available tables: A, B. A completed answer must list them.',False),
        ('false_chart','5개 구간 히스토그램을 보여줘.','기술 문제로 그림은 없습니다. 분포 확인을 완료했습니다.','A rendered five-bin histogram is required, not a claim of completion.',False),
    ])
    if a.controls:
        cases = json.loads(a.controls.read_text())
    results = []
    for name, prompt, answer, expected, positive in cases:
        metric = GEval(name='Independent requirement grounding',
            evaluation_steps=GROUNDING_STEPS, evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT,
                                  LLMTestCaseParams.EXPECTED_OUTPUT],
            threshold=0.8, model=model, async_mode=False)
        row = {'id': name, 'expected_pass': positive}
        started = time.monotonic()
        try:
            metric.measure(LLMTestCase(input=prompt, actual_output=answer, expected_output=expected))
            row.update(score=metric.score, reason=metric.reason,
                       classification_correct=(metric.score >= 0.8) == positive)
        except Exception as exc:
            from evaluate_deepeval_real_report import error_chain
            row.update(error_type=type(exc).__name__, error_chain=error_chain(exc),classification_correct=None)
        row['elapsed_seconds'] = round(time.monotonic()-started, 3)
        results.append(row)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps({'mode':'synthetic labelled judge calibration; not agent runs',
            'controls':str(a.controls) if a.controls else 'built-in calibration 12', 'expected_controls':len(cases), 'judge':judge_name, 'provider':a.provider, 'threshold':0.8, 'results':results,
            'calibration_passed':len(results)==len(cases) and all(r.get('classification_correct') is True for r in results),
            'limitations':['Small labelled control set only; passing does not establish broad judge reliability.']},
            ensure_ascii=False, indent=2)+'\n')
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
