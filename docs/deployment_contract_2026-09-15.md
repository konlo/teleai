# Telly 제한 배포 계약 — 2026-09-15

현재 revision은 `local-owner`로 모든 대화와 데이터 자산을 저장하고 Streamlit을 loopback에 바인딩한다. 따라서 허용 범위는 로컬 데스크톱 또는 외부 접근제어로 한 사람만 접속하는 사설 배포다. 인증 없이 여러 사용자가 직접 접속하는 배포는 대화·DataFrame·승인 기록의 소유권을 분리하지 못하므로 차단한다.

## 배포 전 자동 검사

검사는 환경 변수의 존재와 형식, 영속 저장소, 운영 한도, 사용자 범위를 확인한다. Ollama나 Databricks에 연결하지 않고 SQL도 실행하지 않으며 토큰 값을 출력하지 않는다.

```sh
.telly_runtime/v1-venv/bin/python scripts/deployment_preflight.py --profile local-desktop

# SSH 터널로 한 사람만 접근하는 호스트에서, 실제 접근제어를 확인한 뒤
TELLY_EXTERNAL_ACCESS_CONTROL=confirmed TELLY_ACCESS_MODE=ssh-tunnel \
  .telly_runtime/v1-venv/bin/python scripts/deployment_preflight.py \
  --profile private-single-user
```

`private-single-user` preflight는 저장소가 미리 생성된 실디렉터리이고 그룹/타인 권한이 없는지 검사한다. SSH 터널 설정값은 자기 선언이므로 preflight `READY`만으로 운영 GO가 아니다. 실행 중 리스너와 영속 저장소는 `scripts/private_host_smoke.py`로, 외부 차단과 SSH 계정 제한은 별도 네트워크 위치에서 확인한다. [1인용 Linux 배포 절차](private_single_user_deployment_2026-09-24.md)를 따른다.

`multi-user` profile은 현재 revision에서 항상 실패한다. 이를 허용하려면 Streamlit OIDC(`st.login`, `st.user`)를 적용하고 검증된 사용자 claim을 runtime owner에 바인딩하며, 사용자별 저장소 접근제어와 세션 격리 회귀를 먼저 통과해야 한다.

## 필요한 운영 자원

- Python 3.11과 `requirements-agent.txt`의 고정 의존성
- 앱 프로세스에서 접근 가능한 Ollama endpoint와 명시적 모델 이름
- Databricks host, HTTP path, token, catalog, schema를 주입하는 secret manager
- 코드 checkout과 분리된 `TELLY_V1_STORAGE` 영속 볼륨
- 한 프로세스 기준 1 GiB 메모리의 초기 할당과 RSS 관측. 초기 경고는 768 MiB, 위험 기준은 896 MiB이며 배포 호스트 실측 후 조정한다.

최신 로컬 실측은 단일 30회 p95 0.162초, 4 worker·20개 결정적 분석 p95 0.549초·20/20 성공, peak RSS 약 361 MiB다. `scripts/check_runtime_capacity.py`의 초기 용량 게이트를 통과했다. 이 수치는 컨테이너나 여러 사용자 배포의 용량 보장이 아니다.

배포 전후에는 성능 보고서를 만든 뒤 용량 게이트를 실행한다. 최소 20건·4 worker, 실패 0건, 결정적 로컬 p95 1초 이하를 요구한다. preflight는 메모리 한도와 RSS 기준의 설정 순서도 검사한다.

## smoke test와 롤백

배포 후 다음 순서로 확인한다.

1. preflight가 `READY`인지 확인한다.
2. 기존 영속 저장소를 읽어 대화와 로컬 DataFrame이 보이는지 확인한다.
3. 보유 데이터의 histogram과 상관계수를 실행해 모델 호출과 Databricks 조회가 0회인지 로그에서 확인한다.
4. 새 Databricks SQL은 승인 카드가 나타나기 전 실행 0회인지 확인한다.
5. 사용자가 승인한 SQL 한 건만 실행하고 실패 시 자동 재조회하지 않는지 확인한다.

롤백은 코드와 `requirements-agent.txt` 환경을 함께 최신 검증 tag `agentic-analysis-rc5-2026-09-15` 또는 배포 직전 승인된 tag로 되돌린다. `TELLY_V1_STORAGE`는 교체하거나 삭제하지 않는다. 제출 여부가 불명확한 원격 요청은 구형 runtime에서도 자동 재개하지 않는다.

## 아직 필요한 결정

실제 배포 플랫폼, 한 사람만 사용할지 여러 사용자가 사용할지, Ollama 배치 위치, 영속 볼륨 경로와 secret manager가 정해져야 배포 manifest와 실제 환경 smoke test를 확정할 수 있다. 운영 TableContext 네 건의 갱신은 각각 화면에서 사용자가 SQL을 승인한 뒤에만 수행한다.
