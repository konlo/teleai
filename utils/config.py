"""
utils/config.py
───────────────
프로젝트 전역 설정 상수를 한 곳에서 관리합니다.
값을 변경할 때 이 파일만 수정하면 됩니다.
"""

# ── SQL LIMIT ─────────────────────────────────────────────────────────────────

# 사용자가 %limit 명령으로 설정 가능한 최솟값 / 최댓값
SQL_LIMIT_MIN: int = 1
SQL_LIMIT_MAX: int = 10_000_000

# 세션에 별도 설정이 없을 때 사용하는 기본값
SQL_LIMIT_DEFAULT: int = 1_000_000

# Streamlit session_state 키 이름
SQL_LIMIT_SESSION_KEY: str = "sql_limit"
