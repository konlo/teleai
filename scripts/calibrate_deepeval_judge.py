"""Bounded local GEval calibration with explicitly labelled synthetic controls."""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    os.environ['DEEPEVAL_TELEMETRY_OPT_OUT'] = '1'
    os.environ['DEEPEVAL_RETRY_MAX_ATTEMPTS'] = '1'
    from deepeval.metrics import GEval
    from deepeval.models import OpenAIModel
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    model = OpenAIModel(model='gemma4:e4b', base_url='http://localhost:11434/v1',
                        api_key='ollama', timeout=45, max_retries=0)
    cases = [
        ('correct_groups', '그룹별 평균을 보여줘.', 'A 평균 1.5, B 평균 3.5입니다.', 'A=1.5; B=3.5', True),
        ('missing_goal', '그룹별 평균을 보여줘.', '결측값은 0개입니다. 분석을 완료했습니다.', 'A=1.5; B=3.5', False),
        ('wrong_number', '평균을 보여줘.', '평균은 25.0입니다.', 'mean=2.5', False),
        ('wrong_bins', '5개 구간 히스토그램을 보여줘.', '히스토그램 생성 완료. 실제 렌더 설정: bins=20.', 'Rendered histogram must have bins=5.', False),
    ]
    results = []
    for name, prompt, answer, expected, positive in cases:
        metric = GEval(name='Independent requirement grounding',
            evaluation_steps=[
                'Compare the actual answer to every requested requirement and reference fact.',
                'Do not impose JSON or other formatting unless requested.',
                'Missing requested calculations, incorrect values or chart settings must fail even if the answer claims completion.',
            ], evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT,
                                  LLMTestCaseParams.EXPECTED_OUTPUT],
            threshold=0.8, model=model, async_mode=False)
        row = {'id': name, 'expected_pass': positive}
        started = time.monotonic()
        try:
            metric.measure(LLMTestCase(input=prompt, actual_output=answer, expected_output=expected))
            row.update(score=metric.score, reason=metric.reason,
                       classification_correct=(metric.score >= 0.8) == positive)
        except Exception as exc:
            row.update(error_type=type(exc).__name__, classification_correct=None)
        row['elapsed_seconds'] = round(time.monotonic()-started, 3)
        results.append(row)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps({'mode':'synthetic labelled judge calibration; not agent runs',
            'judge':'gemma4:e4b', 'threshold':0.8, 'results':results,
            'limitations':['Four controls only; passing does not establish judge reliability.']},
            ensure_ascii=False, indent=2)+'\n')
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
