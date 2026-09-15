# Telly 제한 배포 계약 — 2026-09-15

현재 revision은 `local-owner`로 모든 대화와 데이터 자산을 저장하고 Streamlit을 loopback에 바인딩한다. 따라서 허용 범위는 로컬 데스크톱 또는 외부 접근제어로 한 사람만 접속하는 사설 배포다. 인증 없이 여러 사용자가 직접 접속하는 배포는 대화·DataFrame·승인 기록의 소유권을 분리하지 못하므로 차단한다.

## 배포 전 자동 검사

검사는 환경 변수의 존재와 형식, 영속 저장소, 운영 한도, 사용자 범위를 확인한다. Ollama나 Databricks에 연결하지 않고 SQL도 실행하지 않으며 토큰 값을 출력하지 않는다.

```sh
.telly_runtime/v1-venv/bin/python scripts/deployment_preflight.py --profile local-desktop

# 인증된 reverse proxy 또는 사설 접근제어 뒤에서 한 사람만 사용할 때
TELLY_EXTERNAL_ACCESS_CONTROL=confirmed \
  .telly_runtime/v1-venv/bin/python scripts/deployment_preflight.py \
  --profile private-single-user
```

`multi-user` profile은 현재 revision에서 항상 실패한다. 이를 허용하려면 Streamlit OIDC(`st.login`, `st.user`)를 적용하고 검증된 사용자 claim을 runtime owner에 바인딩하며, 사용자별 저장소 접근제어와 세션 격리 회귀를 먼저 통과해야 한다.

## 필요한 운영 자원

- Python 3.11과 `requirements-agent.txt`의 고정 의존성
- 앱 프로세스에서 접근 가능한 Ollama endpoint와 명시적 모델 이름
- Databricks host, HTTP path, token, catalog, schema를 주입하는 secret manager
- 코드 checkout과 분리된 `TELLY_V1_STORAGE` 영속 볼륨
- 한 프로세스 기준 최소 1 GiB 메모리의 초기 할당과 RSS 관측. 실제 동시 부하 표본을 수집하기 전에는 자동 확장·경보 임계값을 확정하지 않는다.

현재 로컬 실측 peak RSS는 약 247 MiB였고 4 worker·20개 결정적 분석은 20/20 성공했다. 이 수치는 컨테이너나 여러 사용자 배포의 용량 보장이 아니다.

## smoke test와 롤백

배포 후 다음 순서로 확인한다.

1. preflight가 `READY`인지 확인한다.
2. 기존 영속 저장소를 읽어 대화와 로컬 DataFrame이 보이는지 확인한다.
3. 보유 데이터의 histogram과 상관계수를 실행해 모델 호출과 Databricks 조회가 0회인지 로그에서 확인한다.
4. 새 Databricks SQL은 승인 카드가 나타나기 전 실행 0회인지 확인한다.
5. 사용자가 승인한 SQL 한 건만 실행하고 실패 시 자동 재조회하지 않는지 확인한다.

롤백은 코드와 `requirements-agent.txt` 환경을 함께 `agentic-analysis-rc3-2026-09-14`로 되돌린다. `TELLY_V1_STORAGE`는 교체하거나 삭제하지 않는다. 제출 여부가 불명확한 원격 요청은 구형 runtime에서도 자동 재개하지 않는다.

## 아직 필요한 결정

실제 배포 플랫폼, 한 사람만 사용할지 여러 사용자가 사용할지, Ollama 배치 위치, 영속 볼륨 경로와 secret manager가 정해져야 배포 manifest와 실제 환경 smoke test를 확정할 수 있다. 운영 TableContext 네 건의 갱신은 각각 화면에서 사용자가 SQL을 승인한 뒤에만 수행한다.
