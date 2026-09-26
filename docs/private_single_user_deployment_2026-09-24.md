# 1인용 Linux 호스트 배포 및 검증 절차

이 문서는 **준비된 배포안**이다. 대상 Linux 호스트가 아직 제공되지 않아 이 절차를 그 호스트에서 실행하거나 운영 GO로 판정하지 않았다. 현재 앱의 모든 상태는 `local-owner`이므로 SSH 계정 한 명만 접속하는 범위에 한정한다. 새 Databricks 조회는 여기서 실행하지 않으며, 실제 SQL은 매 건 사용자 승인이 필요하다.

## 호스트 계약

- Linux + systemd, Python 3.11, SSH를 사용할 수 있어야 한다. 서비스는 `127.0.0.1:8502`로만 바인딩한다. 호스트/클라우드 방화벽에서 8502 외부 인바운드를 허용하지 않는다.
- 지정 사용자 한 명의 SSH 공개키만 이 앱으로의 터널 접근을 허용한다. 여러 사람에게 같은 SSH 자격증명이나 포트 포워딩 권한을 공유하면 이 프로필의 격리가 성립하지 않는다. 실제 SSH 설정과 외부 네트워크 차단을 검증한다.
- `/opt/telly/releases/<revision>`은 코드와 각 릴리스의 `.venv`, `/opt/telly/current`는 현재 릴리스로의 symlink다. `/var/lib/telly/v1`은 코드와 분리된 영속 볼륨이며 `telly:telly`, mode `0700`이다. 파일시스템이 재부팅 후에도 같은 볼륨으로 다시 마운트되는지 운영자가 확인한다.
- `/etc/telly/telly.env`에는 `.env.example`의 변수 이름을 환경 파일 형식으로 넣는다. 소유자는 `root:telly`, mode `0640`이며 Git/릴리스 디렉터리로 복사하지 않는다. `TELLY_V1_STORAGE=/var/lib/telly/v1`, `TELLY_ACCESS_MODE=ssh-tunnel`, `TELLY_EXTERNAL_ACCESS_CONTROL=confirmed`를 포함한다. `confirmed`는 실제 SSH·방화벽 확인 후에만 설정한다. Ollama와 Databricks 변수도 이 파일 또는 동일 보안 수준의 서비스 주입 경로에서 제공한다.

## 설치 순서

1. 운영자가 `telly` 서비스 계정을 만들고, 영속 볼륨을 마운트한 후 `/var/lib/telly/v1`을 `telly:telly`/`0700`으로 생성한다. 미리 데이터가 있다면 권한을 바꾸기 전에 백업과 소유권을 확인한다.
2. 검증된 Git revision을 `/opt/telly/releases/<revision>`에 배치하고 Python 3.11로 해당 디렉터리의 `.venv`를 만든 후 `pip install -r requirements-agent.txt`를 실행한다. 릴리스별 venv를 유지한다. 의존성 설치가 끝나기 전에는 current를 전환하지 않는다.
3. `deploy/private-single-user/telly.service`를 systemd 서비스로 설치한다. 이 템플릿은 `/opt/telly`, `/etc/telly/telly.env`, `/var/lib/telly` 경로를 가정하므로 실제 호스트 경로와 서비스 사용자에 맞춰 검토한다. `ReadWritePaths`가 실제 영속 볼륨을 포함하는지 확인한다.
4. 서비스 사용자 컨텍스트에서 `scripts/deployment_preflight.py --profile private-single-user --json`을 실행해 설정·권한을 검사한다. 이 명령은 원격 연결/SQL을 실행하지 않는다.
5. `scripts/private_host_release.py --releases /opt/telly/releases --current /opt/telly/current --target <revision>`으로 대상 전환을 미리 보고, 같은 명령에 `--apply`를 붙여 코드를 전환한다. `systemctl daemon-reload` 후 `systemctl restart telly`로 이전 프로세스의 세션 상태를 종료한다.

## 배포 후 검증

서비스 호스트에서 `python scripts/private_host_smoke.py --storage /var/lib/telly/v1 --checkout /opt/telly/current --port 8502`를 실행한다. `ss`의 모든 8502 리스너가 loopback이고 Streamlit 헬스가 `ok`이며 저장소가 코드 밖 0700 실디렉터리여야 통과한다. 이 검사는 SSH 계정/방화벽을 증명하지 않는다.

허가된 사용자 장비에서는 `ssh -N -L 18502:127.0.0.1:8502 <single-user-ssh-host>`로 터널을 열고 `http://127.0.0.1:18502`의 실제 챗봇을 확인한다. 별도 외부 네트워크 위치에서는 호스트의 8502 포트 직접 연결이 거절/차단되는지 확인한다. SSH 키/계정 목록에서 승인된 한 명만 터널을 열 수 있는지 확인한다. 재부팅 후 영속 볼륨의 동일 데이터가 남아 있는지도 확인한다. 검사 결과에는 접속 주소/계정 식별자와 성공·실패만 기록하고 token이나 원본 데이터는 남기지 않는다.

그다음 보유한 **로컬** 원본으로 histogram·후속 분석·재시작 복원을 확인하고, 모델·Databricks 조회가 0회인지 진단 로그를 대조한다. 승인 없는 원격 요청은 SQL 승인 대기까지만 확인한다. 실제 Databricks 승인→1회 실행은 정확한 SQL별 별도 사용자 승인 후 J22~J24로 검증한다. 이 단계가 끝나지 않으면 전체 운영 판정은 계속 NO-GO다.

## 실패와 롤백

배포 직전의 `current` symlink가 가리킨 이전 revision을 기록한다. 새 릴리스가 실패하면 `scripts/private_host_release.py`에 이전 revision을 `--target`으로 넣어 preview한 뒤 `--apply`로 symlink를 원자적으로 되돌리고 `systemctl restart telly`를 실행한다. 이전 버전의 의존성은 이전 릴리스 `.venv`에 남아 있어야 한다. 롤백 후 preflight, host smoke, 터널 실제 화면, 기존 대화/원본의 재시작 복원을 다시 확인한다. **`/var/lib/telly/v1`을 삭제·교체·역마이그레이션하지 않는다.** 새 코드가 저장 포맷을 변경한다면 이전 코드의 읽기 호환성을 릴리스 전에 검증해야 하며, 불가능하면 백업 복원 계획을 별도 승인받는다.

## 판정 기록

운영 GO 근거로는 대상 호스트 ID/revision, 영속 볼륨 마운트/권한, SSH 한 명 제한과 외부 직접 연결 차단, preflight JSON, host smoke JSON, 로컬 보유 원본 여정, 실제 모델/용량 실측, rollback/restart 후 동일 원본 확인, 승인된 원격 SQL별 결과가 필요하다. 설정값 하나나 로컬 회귀 통과만으로 GO로 갱신하지 않는다.
