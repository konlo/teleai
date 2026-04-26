from typing import Dict, List


DEFAULT_SQL_LIMIT_MIN = 1
DEFAULT_SQL_LIMIT_MAX = 10_000_000
BASE_CHAT_PLACEHOLDER = "지원되는 명령을 확인하려면 `%help` 를 입력하세요."

CHAT_COMMAND_SPECS: List[Dict[str, str]] = [
    {
        "name": "debug",
        "trigger": "%debug",
        "usage": "`%debug on|off`",
        "description": "Debug 모드를 켜거나 끄는 명령입니다.",
    },
    {
        "name": "limit",
        "trigger": "%limit",
        "usage": "`%limit <정수>`",
        "description": "SQL LIMIT 값을 {limit_range} 범위의 정수로 설정합니다.",
    },
    {
        "name": "sql",
        "trigger": "%sql",
        "usage": "`%sql <질문>`",
        "description": "SQL Builder 에이전트를 호출해 질문에 맞는 SQL을 생성합니다.",
    },
    {
        "name": "table",
        "trigger": "%table",
        "usage": "`%table training`",
        "description": "현재 선택된 Databricks 테이블의 안전한 TableContext profile을 JSON 파일로 저장합니다.",
    },
    {
        "name": "help",
        "trigger": "%help",
        "usage": "`%help`",
        "description": "지원되는 명령 목록과 사용법을 표시합니다.",
    },
    {
        "name": "example",
        "trigger": "%example",
        "usage": "`%example`",
        "description": "자주 사용하는 명령 예시를 보여줍니다.",
    },
]

AUTO_SQL_KEYWORDS = (
    "데이타 로딩", "조회", "본포", "분포", "통계", "평균", "최대", "최소",
    "가져와", "보여줘", "리스트", "찾아", "데이터", "테이블", "전체"
)

VIZ_KEYWORDS = (
    "분포", "본포", "그려", "그래프", "시각화", "차트", "플롯", "plot", "chart", "graph", "추이", "상관관계"
)

COMMAND_EXAMPLE_LINES = [
    "1. %sql cluster가 Huahai 인 것을 보고 싶어",
    "2. cluster에 대한 histogram을 보여줘",
    "3. balance 에 대한 이상점을 찾아줘",
    "4. isolation 기법을 이용해서 이상점이 있는지 봐줘",
]


def build_command_help_message() -> str:
    limit_range = f"{DEFAULT_SQL_LIMIT_MIN}~{DEFAULT_SQL_LIMIT_MAX}"
    lines = ["**사용 가능한 명령**"]
    for spec in CHAT_COMMAND_SPECS:
        description = spec["description"].format(limit_range=limit_range)
        lines.append(f"- {spec['usage']}: {description}")
    return "\n".join(lines)


def build_command_example_message() -> str:
    lines = ["**예시 명령**"]
    lines.extend(COMMAND_EXAMPLE_LINES)
    return "\n".join(lines)
