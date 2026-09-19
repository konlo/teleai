# TeleAI runtime 성능·메모리 측정 — 2026-09-14

## 판정

현재 DataFrame을 재사용하는 결정적 histogram 경로는 출시 점검 기준을 충족했다. 최신 반복 30회는 p50 0.074초, p95 0.146초, 최대 0.185초였다. 모든 실행이 실제 PNG를 만들었고 모델 및 Databricks 호출은 0회였다.

4개 worker의 동시 로컬 histogram 20건은 20/20 성공, p95 0.562초, 처리량 6.864 req/s였다. 실제 모델 상관계수 경로는 정확성 5/5였지만 p95 80.704초였고, 이를 결정적 로컬 계산으로 바꾼 뒤 30회 p95가 0.046초로 줄었다. 모델이 필요한 다른 질문과 배포 환경 동시 부하는 별도 관측이 필요하다.

## 측정 결과

| 구간 | 표본 | p50 | p95 | 최대 | 메모리 |
|---|---:|---:|---:|---:|---:|
| 보존된 v1 운영 로그 전체 | 23 turns | 40.983초 | 215.301초 | 277.124초 | 이전 로그는 RSS 미계측 |
| 최신 서버 실제 화면 turn | 1 turn | 0.114초 | 표본 부족 | 0.114초 | peak RSS 259,063,808 bytes, frame cache 6,421,172 bytes |
| 현재 코드·실제 10,000행 복제본 | 30 turns | 0.074초 | 0.146초 | 0.185초 | benchmark process peak 296,861,696 bytes |
| 동시 로컬 histogram, 4 workers | 20 turns | 0.463초 | 0.562초 | 0.572초 | benchmark process peak 367,771,648 bytes |
| 실제 모델 상관계수, 개선 전 | 5 turns | 77.451초 | 80.704초 | 80.704초 | peak RSS 218,464,256 bytes |
| 결정적 로컬 상관계수, 개선 후 | 30 turns | 0.037초 | 0.046초 | 0.061초 | 위 로컬 benchmark process에 포함 |
| 최신 서버 단일 RSS snapshot | 1회 | - | - | - | 41,025,536 bytes |

보존 로그 23건에는 이번에 수정한 과거 실패와 모델 재시도 실행이 포함된다. 따라서 p95 215.301초는 과거 운영 기준선이며 현재 코드만의 성능 수치가 아니다. 반대로 30회 반복 결과는 현재 표본 histogram 경로만 검증하며 모델 질문 전체를 대표하지 않는다. 백분위는 nearest-rank 방식으로 계산했다.

## 적용한 계측

`run_started`와 `run_completed` 이벤트에 표준 라이브러리 기반 process peak RSS를 기록한다. 완료 이벤트에는 현재 DataFrame LRU cache 바이트도 함께 남긴다. 예외 turn의 SLO 이벤트에도 같은 메모리 필드를 기록한다. 프롬프트, 행 데이터, 자격 증명은 로그에 포함하지 않는다.

재현 명령은 다음과 같다.

```bash
.telly_runtime/v1-venv/bin/python scripts/report_runtime_performance.py \
  --asset-db <conversation>/assets.sqlite \
  --server-pid <streamlit-pid> \
  --iterations 30 --concurrent-requests 20 --workers 4 \
  --output docs/runtime_performance_2026-09-14.json
```

원본 결과는 [runtime_performance_2026-09-14.json](runtime_performance_2026-09-14.json)에 있다. 이 실행은 원본 asset DB를 읽기 전용으로 열며 Databricks와 모델을 호출하지 않는다.

## 다음 운영 게이트

- 모델이 필요한 다른 실제 질문의 성공·실패 상태별 p50/p95를 최신 코드에서 수집한다. streaming HTTP timeout은 전체 벽시계 deadline이 아님을 운영 문서에 명시한다.
- 동시 사용자 수와 배포 인스턴스 메모리 용량을 정한 뒤 RSS 경보·재시작 기준을 확정한다.
- 최소 운영 표본을 확보할 때까지 180초는 turn SLO 경보와 모델 호출 사이의 누적 예산으로 사용한다. 실행 중인 단일 streaming 호출을 강제 종료하는 deadline은 아니다. 결정적 로컬 경로는 p95 1초를 회귀 기준으로 사용한다.
