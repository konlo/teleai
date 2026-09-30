"""Require the supported agent runtime at every Streamlit entrypoint."""

from importlib import metadata


def runtime_compatibility_error() -> str:
    try:
        installed = metadata.version("langchain")
        major = int(installed.split(".", 1)[0])
    except (metadata.PackageNotFoundError, ValueError):
        installed = "not installed"
        major = 0
    if major >= 1:
        return ""
    return (
        "이 Python 환경은 현재 Telly agent를 실행할 수 없습니다 "
        f"(LangChain {installed}; 1 이상 필요). "
        "기존 8501 서버를 종료하고 저장소에서 "
        "`python3 scripts/run_telly.py --port 8501`로 지원 환경을 실행하세요."
    )
