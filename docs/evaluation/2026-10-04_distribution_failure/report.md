# n_1 分布 요청 실패 진단 — 2026-10-04

실제 화면 요청: `n_1 column의 데이타 분포를 보여줘`. 오류 `3193b28cf8e7`, 실행 `3295f182368f4df9b3bf530018af6bd9`.

- 요약 모델 37.232초 후 다음 분석 모델 호출 전 ModelContextBudgetExceeded로 중단. 전체37.534초, 현재 요청의 분석 SQL/도구/차트0.
- chart=False/kind=None/metadata_kind=columns. “분포를 보여줘”는 chart signal에 포함되지 않고 column+보여줘는 목록 요청으로 분류된다.
- 직전10행 미리보기 required_sources와 실제table_preview_evidence는 alibaba_ssd이지만 confirmed scope는 bank_loan. 현재required_sources=[]이며 scope에는bank_loan이 남아 있다. 실제계산하지않았으므로잘못된분포결과가출력된것은아니다.
- source가비어prompt_catalog가모든저장dataset의columns를전달한다. 이전턴과tool메뉴를축소한후에도payload16108bytes+templateheadroom1216=17324보수단위>입력예산12288. Context16384/출력예약4096. 이값은실제token수가아니므로실제model한도초과확정근거가아니다.

필요한수정은목적에따른목록/분포구분,검증된마지막미리보기출처와scope일관성,요청별catalog선별/대규모스키마온디맨드조회,요약전후의동일입력예산/복구설계다. 원본10행미리보기로전체분포를추정해서는안된다.

이번턴은읽기전용진단이다. 재개/DB재조회/새모델호출/제품코드변경을하지않았다. 상세필드와단계별로그는diagnosis.json에저장했다.
