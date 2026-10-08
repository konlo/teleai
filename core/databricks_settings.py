"""Shared Databricks setting parsing; no credentials, files or network calls."""
from urllib.parse import urlsplit


def env_value(environ, name, *aliases):
    """Accept historical casing and trim values without guessing conflicts.

    An empty primary setting allows an explicitly supported alias. Conflicting
    case variants are rejected rather than choosing another workspace or token.
    Error messages contain setting names only, never their values.
    """
    for candidate in (name, *aliases):
        values = {str(value).strip() for key, value in environ.items()
                  if key.casefold() == candidate.casefold() and value is not None
                  and str(value).strip()}
        if len(values) > 1:
            raise ValueError(candidate + '의 대소문자별 설정 값이 충돌합니다. 변수명을 대문자로 통일해주세요.')
        if values:
            return values.pop()
    return ''


def normalize_hostname(host):
    """Convert a workspace root URL or bare hostname to SQL connector format."""
    value = (host or '').strip()
    if not value:
        return ''
    parsed = urlsplit(value if '://' in value else '//' + value)
    if (parsed.scheme not in {'', 'http', 'https'} or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.path not in {'', '/'} or parsed.query or parsed.fragment
            or any(character.isspace() for character in value)):
        raise ValueError('DATABRICKS_HOST에는 workspace 호스트명 또는 루트 URL을 설정해주세요.')
    return parsed.netloc
