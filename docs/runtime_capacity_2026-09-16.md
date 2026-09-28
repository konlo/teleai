# TeleAI 단일 프로세스 용량 게이트 — 2026-09-16

## 판정

현재 1인용 제한 배포 후보의 로컬 용량 게이트는 **READY**다. 저장된 10,000행 DataFrame을 사용했으며 모델과 Databricks 호출은 모두 0회였다.

| 검사 | 기준 | 실측 | 결과 |
|---|---:|---:|---:|
| 단일 결정적 분석 p95 | 1.000초 이하 | 0.162초 | PASS |
| 4 worker·20건 동시 분석 p95 | 1.000초 이하 | 0.549초 | PASS |
| 동시 성공률 | 20/20, 실패 0 | 20/20, 실패 0 | PASS |
| process peak RSS | 경고 768 MiB 미만 | 약 361 MiB | PASS |

초기 단일 프로세스 메모리 한도는 1 GiB, RSS 경고는 768 MiB, 위험 기준은 896 MiB다. `core.runtime_capacity.CapacityPolicy`와 `TELLY_*` 환경 변수로 배포별 조정이 가능하다. preflight는 양수 여부와 `경고 < 위험 < 메모리 한도` 순서를 검사한다.

`scripts/check_runtime_capacity.py`는 성능 보고서 자체에 모델·Databricks 호출 0회와 원본 미변경 증거가 있는지 먼저 확인하고, 최소 요청 수·worker 수, 전건 성공, 단일·동시 p95와 peak RSS를 검사한다. 안전 증거가 없거나 원격·모델 호출이 포함되면 실패한다. 경고 구간은 운영 검토가 필요한 READY, 위험 구간은 NOT READY다.

```bash
.telly_runtime/v1-venv/bin/python scripts/check_runtime_capacity.py \
  --input docs/runtime_performance_2026-09-16.json \
  --output docs/runtime_capacity_2026-09-16.json
```

원본은 [성능 측정 JSON](runtime_performance_2026-09-16.json)과 [용량 판정 JSON](runtime_capacity_2026-09-16.json)에 보존했다.

## 적용 범위와 남은 조건

이 결과는 한 프로세스 안의 격리된 대화에서 결정적 로컬 분석을 동시에 실행한 표본이다. 실제 배포 호스트, 컨테이너 제한, 여러 프로세스, Ollama 동시 추론, Databricks 지연을 대표하지 않는다. 배포 플랫폼이 정해지면 같은 명령을 배포 환경에서 실행하고 경고·위험 기준을 그 환경의 메모리 한도와 관측 결과로 다시 확정해야 한다.
