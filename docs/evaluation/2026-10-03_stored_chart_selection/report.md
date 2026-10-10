# 저장된 차트 선택 실패 수정 — 2026-10-03

현재 웹에서 `이 차트 선택` 뒤 오류 `13e79131eba1`가 발생했다. 이미지 자체는 정상 PNG(12,208bytes,660×385)였고 저장된 모집단은 age30~40 및 education primary/secondary,208,699건이다.

## 원인

`select_chart`가 저장된 차트 ID를 자연어 HumanMessage와 함께 model node에 추가했다. 일반 의도 파서는 이 controller 동작을 새 분석으로 인식해 기존 범위를 빈 조건으로 바꿨다. 필터가 있는 집계 차트를 전체 모집단과 비교하면서 완료 증거를 거절했고, 동일한 차트를 다시 생성/표시하도록 모델을 호출했다.

실패 run `2434d04d2ff54f12a9da1530460bf6f5`에서 대화 요약32.901초, 모델 호출 중 ReadTimeout과 재시도, 최종183.132초 뒤 중단을 확인했다. SQL은 제출되지 않았다. 이후 UI에서 별도 RuntimeError13e79131eba1가 발생해 원래의 모델 대기 실패보다 차트 선택 자체의 오류로 보였다. 저장된 이미지를 표시하는 controller 동작을 추론과 연결한 agent 결함이다.

## 수정

- 선택한 card ID를 구조화된 controller 요청으로 처리한다. 제목·UUID·표시문을 새 필터/통계 요청으로 해석하지 않는다.
- 실제 PNG와 연결된 dataset 존재를 확인하고 정확한 저장 이미지를 반환한다. 다른 데이터나 전체 원본으로 모집단을 승격하지 않는다.
- 저장 차트의 범위·출처·선택 데이터 및 matching confirmed scope를 보존한다. 새 SQL·모델·대화 요약을 호출하지 않고 완료 상태를 저장한다.
- 이전 버전에서 실패한 차트 선택 checkpoint도 동일 asset으로 재개한다. 일반 미완료 분석/승인 대기/불명확한 원격 제출을 이 경로로 완료하지 않는다.
- 선택 응답은 “저장된 이미지를 표시했습니다”로 변경했다. 대화에 이미 표시한 선택 이미지는 하단에 중복 표시하지 않는다.
- 선택에도 실행 ID·완료 시간·추가 모델/SQL 미실행을 진단에 기록한다. 기존 실패 로그는 보존한다.

## 검증

관련31 PASS·7subtests: 필터 있는 COUNT 집계 이미지·다른 원본 선택 상태·재선택·재시작·후속·manual/auto 정책·summary trigger1·legacy 모델예산 소진·손상/없는 PNG·실제 AppTest 선택 버튼·중복 이미지·원본 digest 보존. 모델/SQL이 호출되면 즉시 실패하는 모델/실행기를 사용했다. 실제 LLM의 자유형 분석 성능 검사는 아니다.

추가 로그/지원진단10 PASS·2subtests. 첫 전체675 PASS/0 FAIL/4 SKIP·473subtests,94.50초. 로그/중복 표시 최종 변경을 포함한 전체 회귀도 **675 PASS/0 FAIL/4 SKIP·473subtests,93.59초**였다. 상세는 [validation.json](validation.json).

실제 기존 웹 대화의 실패를 resume 버튼으로 복구하여 complete/이미지/분석 기준 선택, 추가 모델·SQL0회를 확인했다. 당시 local control 동작의 run_id가 null이었던 로그도 보존했고 새 선택에는 run ID를 부여했다. 최종 서버 PID14578에서 실제 재선택 run b51ef9f10db84bb4ac572cd88cc8a8d9는 **0.010초·모델/요약/SQL0회**, answered/complete였다. 기존 dataset6개와 같은 card/208699건 이미지가 유지됐고 오류·재개 버튼 및 하단 중복 이미지가 없다. [화면](fixed_web.png).

전체 범용 GO 판정은 갱신하지 않는다. 기존 고급 SQL/독립 실모델/일반 추론 지연 문제는 별도다. 이번 변경은 commit/push하지 않았다.
