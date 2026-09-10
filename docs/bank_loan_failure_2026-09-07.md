# bank_loan 테이블 질문 실패 진단

- 대화: c1022998-2a9c-43f6-ac8a-6a903f5e21b7
- checkpoint: 2026-09-07T13:12:37.774496+00:00 (한국 시간 22:12:37)
- 사용자 질문: bank_loan의 테이블은 어떤 것들이 있지 ?
- 모델 호출: inspect_dataset(dataset_id="workspace.default.bank_loan")
- checkpoint pending write: __error__ = KeyError('workspace.default.bank_loan')

저장된 TableContext에는 테이블 설명이 있었지만 로딩된 datasets는 비어 있었습니다. 모델이 테이블명을 결과 ID로 사용했고 도구의 metadata 인덱싱에서 실패했습니다. 이 요청의 실패는 Databricks 조회 이전의 로컬 도구 오류입니다.

기존에는 checkpoint 예외와 원본 대화만 남고 별도 실행 진단 파일은 없었습니다. runtime의 예외 catch는 종류만 반환했고 UI는 이를 보여주지 않아 원인 없는 안내가 중복되었습니다. 이전 오류의 전체 traceback은 복원했다고 주장하지 않습니다.

수정: 저장된 설명을 읽는 inspect_table_context 추가, dataset ID와 테이블명 계약 분리, 잘못된 ID를 복구 가능한 도구 관찰로 반환, 중복 안내 제거. 새 runtime.jsonl은 실행 ID/도구명/완료/실패 단계/예외 종류/파일·함수·줄 번호를 기록합니다. 원본 질문·데이터·예외 문자열은 기록하지 않습니다. 대화별 로그는 최대 2MB와 백업 3개, 파일 권한 0600입니다. 전체 실행 메트릭/분산 tracing 시스템을 의미하지는 않습니다.

검증: 새 환경 20개와 기존 32개 통과. 잘못된 ID → 테이블 정보 도구 → 정상 종료, 로그 오류 위치/ID 및 비밀 문자열 미기록 검사 포함. 후속 logging 보완 후 해당 2개 재통과. 서버 재시작 완료. 사용자의 기존 대화는 변경하지 않았으며 실제 모델의 해당 요청 재개는 아직 실행하지 않았습니다.
