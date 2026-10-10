# 최신 전체 데이터 산점도 실패 진단

2026-10-04 KST18:19:53~18:22:12. 화면 요청: “전체 데이타를 이용해서 age와 balance의 관계를 다시 scatter plot으로 그려줘”. 오류11c321334f37, 실행ea527c0b5ccf40bab265ffcb6732fa5b. [실행/체크포인트 근거](diagnosis.json).

## 실행 경로

1. 출처 teleai_default.bank_loan, 컬럼 age/balance, scatter와 전체 범위를 인식했다. 필터 없음, current_result_only=false. 대화 요약은 실제로 실행하지 않았다(summarized=false).
2. 첫 모델 호출60초 ReadTimeout 후1초 뒤1회 재시도. 재시도 응답은 도구 호출0이었다. 묶음 모델 호출 시간78.766초.
3. agent가 chart 미완료를 감지해 재계획 요청(attempts1). 이어지는 모델 호출이60.067초 ReadTimeout으로 끝났다. 전체139.144초에 incomplete.

SQL 제출0, 도구 실행0, 로딩 계획None, 새 데이터None, 이미지ID없음. 따라서 이번은 SQL을 생성해 MySQL/Databricks에 보냈는데 결과를 받지 못한 장애가 아니다. 모델 HTTP 응답 대기와 전체 출처 로딩 계획을 만들지 못하는 agent 경로가 함께 관여한다. 큐/prefill/추론 중 어느 단계가 모델 지연의 근본원인인지는 이 로그만으로 확정하지 않는다.

## 보유 데이터 및 구조

출처와 두 축 컬럼이 모두 맞는 raw asset은 SELECT * LIMIT10의10행, coverage unknown/predicate_known=false다. 데이터셋8개 중 나머지는 전체 원본 보장을 하는 age/balance raw가 아니다. 전체 요청에 이10행을 쓰면 잘못된 완료가 되므로 코드의 차트 결합/완료 guard는 이를 거절한다.

core/analysis_agent/chart_binding.py의bind는 전체 요청을10행 표시 근거에 연결하지 않으며, raw_scatter_eligible은 complete/predicate 근거를 요구한다. recovery.py의 로컬 차트 경로는 적격 raw를 찾으면 그릴 수 있지만 현재는 후보가 없다. 그 뒤 전체 출처의 필요한 두 컬럼·모집단·행수/byte·한도·coverage를 실행 가능한 조회/적재 계획으로 연결하는 경로가 만들어지지 않고 모델 재계획에 의존한다. 이번은 이전 실패의 잘못된dataset_id 거절과 달리 실제 도구 호출 자체가 없었다.

## 현재 상태와 필요한 보강

8504 서버health ok. dataset8개/chart11개와 선택cac7c3b3-165f-4c3f-9965-9278fad98bf5 보존. 마지막 요청은 실패checkpoint에 있고 현재 추론 중이 아니다. 이번 확인에서는 재개·종료·새 쿼리·제품 수정 없이 실패 증거를 보존했다.

전체 출처 발견/투영/자원 계획/적재/시각화 완료를 controller/tool 계약으로 연결하고, 모델이 도구를 호출하지 않거나 지연되면 같은 추론 반복 외의 복구를 실행해야 한다. 모델문맥/도구량/추론 지연도 별도로 측정해야 한다. 기존10행 성공을 전체 데이터 시각화 성공으로 계산하지 않는다.

## 모델 호출 횟수·추가 지연 근거

실제 Ollama HTTP 요청은3회: 최초60초timeout→동일요청재시도17.723초응답(도구0)→agent재계획60.067초timeout. 앱의model_call span은재시도를묶어2개이며,대화요약모델은0회다. 전체139.144초.

Ollama server.log의각api/chat 시간과slot작업을대조했다. 최초/재시도입력16083토큰,재계획16310토큰,context한도16384다. 첫입력처리14848토큰에53.84초,재계획입력처리14848토큰에57.64초를소비했다. 재시도는prompt캐시를이용해추가입력723토큰3.453초·생성301토큰12.725초,끝n_tokens16383/truncated1. 즉입력·도구·맥락이모델한도거의전체를차지했고출력공간도부족했다. 어떤SQL도구를반드시호출하지못한의미적이유를한도만으로단정하지는않는다.

현재reasoningTrue/num_ctx16384/num_predict4096. 앱의context_budget는rendered system/messages와과거toolcall인자를문자수로만센다. 제공하는tool schema와실제토큰/출력공간예산은포함하지않는다. 단계별필요한tool/schema/확정맥락축소와출력공간을예약하는전체payload토큰관리,전체출처실행계획보강이필요하다. [호출/토큰기록](model_calls.json). 이번확인은코드·설정변경/재추론없이수행했다.
