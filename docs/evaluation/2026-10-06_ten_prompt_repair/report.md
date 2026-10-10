# 실제 10문항 미완료 수정 및 재검증 — 2026-10-06

이전 독립 새 대화는 3/10 완료였다. 공통 실행 계약을 수정한 뒤 **최종빌드 새 대화에서 동일10문항 최초 제출10회, 사용자 재제출0회로 10/10 완료**했다. 수정 도중의 실패·오답은 삭제하거나 최종 성공으로 소급하지 않았다. 이 결과는 고정된 로컬 MySQL 여정의 수용 결과이며 범용 운영 GO나 공식 DeepEval/Spider 점수가 아니다.

## 원인과 수정

| 원인 | 수정 | 근거 |
| --- | --- | --- |
| 좁은 출처 판독 모델의 오판이 명시된 테이블까지 덮어씀 | 실제 catalog에서 확인한 명시 identity와 audit를 대조. audit는 오류 가능한 관찰로 취급 | 최종2번 bank_loan18컬럼 |
| 실패한 사용자 주제와 마지막 성공한 저장 결과를 같은 맥락으로 취급 | 요청 주제와 실행 증거 분리·최근 사용자4턴을 응답과 독립 보존 | 3~7번 연결, 9→10번 테이블 전환 |
| DISTINCT 조회 완료 후 완료 의무로 연결하지 못해 불필요한 추론·예산 종료 | 현재 턴의 검증된 조회 영수증만 대상으로 출처·조건·projection·LIMIT 확인 후 bounded 값 목록 완료 | 3번4값·조회1회 |
| 기존 제안을 본 자기검토가 secondary 누락에 동의 | 제안 조건을 보지 않는 별도 LLM 모집단 판독, 이전 검증 조건과 현재 요청만 제공 | 6번 독립DB 합계208699, 7번 그대로 유지 |
| 보조 판독이 row10을 ID=10 필터로 발명 | 새 무필터 preview의 출력 제한과 모집단 분리·근거 없는 새 필터 컬럼 거절 | 9번 무필터10행13열 |
| 차트 금지를 metadata 실행 금지로 해석·names/dtypes 중복 | 부분 금지의 실행 모드 피드백, 동일 metadata 의무만 dtypes로 병합 | 10번 DB타입13개·차트 추가0 |
| 다른 오류를 수정하면서 진행해도 누적2회 실패로 중단 | 동일 오류 반복 한도와 전체4시도/추론시간 예산 분리 | 중간 실패 보존, 최종9·10번 자동복구 |

자연어 의도·조건은 LLM이 해석한다. 이 변경은 특정 테이블/표현을 production 라우팅에 하드코딩하는 방식이 아니다. catalog identity/조건 구조/영수증/실제 결과는 실행 계약으로 검증한다. `source_references.py`, `population_audit.py`, `goal_normalization.py`로 책임을 나눴다.

## 최종 실제 브라우저 결과

| 번호 | 실제 확인한 결과 | 판정 | 초 |
| --- | --- | --- | --- |
| 1 | 테이블 목록7개 | PASS | 62.7 |
| 2 | bank_loan 컬럼18개 | PASS | 80.4 |
| 3 | education 값4종 | PASS | 91.9 |
| 4 | 전체750000 빈도 막대PNG | PASS | 114.1 |
| 5 | primary·나이30~40: 27439 | PASS | 128.6 |
| 6 | secondary 추가·범위 유지: 208699 | PASS | 116.7 |
| 7 | 동일 모집단·5구간·추가SQL0 | PASS | 137.7 |
| 8 | 전체750000·132613좌표 산점도 | PASS | 110.8 |
| 9 | stormtrooper 실제10행13열 | PASS | 103.6 |
| 10 | 직전 stormtrooper DB타입13개·새차트0 | PASS | 121.4 |

9번의 잘못된 current_result_only와10번의 explain/tasks 충돌은 실행 전 검증으로 감지해 에이전트가 원래 요청을 보존하고 스스로 재계획했다. 첫 LLM 계획까지 모두 정답이었다는 뜻이 아니다. 실제 UI 제출은10회, 모델 호출은30회, 분석 조회는7회였다. 7번 구간 변경은 저장 집계를 재사용했다.

## 검증 및 한계

- 최종 회귀640건 수행: **636PASS/4SKIP**, 186.971초. 건수만으로 release를 판정하지 않는다.
- 각 턴의 request ID/run ID·완료 checkpoint·실제 parquet·DB metadata·비어 있지 않은 PNG·UI를 대조했다. 브라우저의 차트5개 모두 이미지 로딩 완료와 실제 픽셀 크기를 확인했다.
- histogram 구간별 빈도를 독립 DB oracle와 일치시켰다. 전체 산점도도 독립 read-only SQL의 모든 좌표/빈도와 exact frame 비교했고750000합계/132613좌표를 확인했다.
- 최종 대화12자산이 후속 동작에서 변경/누락되지 않았고 선택은 유지됐다. 이전3대화64자산의 metadata/payload SHA256 및 선택도 최종 실행 후 다시 검증해 변경/누락0이다.
- 턴 지연 **62.7~137.7초**, 총 실행1067.85초다. 출력 오류를 수정했지만 로컬 추론 지연 개선은 남았다.
- 회사 Databricks, 미채점 새 표현/새 schema·장기 대화/cold start, 고급 EDA와 생성Python 자율 실행, 공식 평가를 이 결과로 대체하지 않는다. 범용 운영 **NO-GO** 유지.

## 근거 파일

- [최종 독립 채점](report.json), [고정10문항](plan.json), [독립 DB oracle](oracles.json), [실제 UI10턴](acceptance_ui.json)
- `acceptance_01.json`~`acceptance_10.json`: 완료 영수증·조건·출처·실행 로그·자산지문
- [전체 회귀 로그](acceptance_unit_tests.log), [이전 데이터 불변성](older_data_preservation.json), [이미지 로딩](acceptance_rendered_images.json)
- `acceptance_07_histogram.png`, `acceptance_08_scatter.png`, `acceptance_09_rows.png`, `acceptance_10_types.png`: 최종 브라우저 화면
- report.json의 intermediate_failures는 수정 도중 실패를 별도로 보존한다. commit/push는 이번 요청에서 수행하지 않았다. 서버8504와 최종 대화는 사용자 테스트를 위해 유지했다.
