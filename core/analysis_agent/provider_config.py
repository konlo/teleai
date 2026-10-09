"""Provider configuration shared by model construction, UI and preflight."""
import os
from urllib.parse import urlsplit
from core.databricks_settings import env_value

PROVIDERS = ('ollama', 'azure', 'databricks')


def configured_provider(environ=None):
    config = os.environ if environ is None else environ
    value = (env_value(config, 'TELLY_ANALYSIS_MODEL_PROVIDER')
             or env_value(config, 'LLM_PROVIDER') or 'ollama').casefold()
    if value not in PROVIDERS:
        raise ValueError('분석 모델 제공자는 ollama, azure, databricks 중 하나여야 합니다.')
    return value


def pinned_provider(environ=None):
    config = os.environ if environ is None else environ
    # A service-wide override is authoritative. Existing Azure service .env
    # also wins over historical conversation preferences from local development.
    if (env_value(config, 'TELLY_ANALYSIS_MODEL_PROVIDER')
            or env_value(config, 'LLM_PROVIDER').casefold() == 'azure'):
        return configured_provider(config)
    return None


def azure_options(environ):
    values = {name:env_value(environ, name) for name in
              ('AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_DEPLOYMENT')}
    missing = [name for name,value in values.items() if not value]
    if missing:
        raise ValueError('Azure OpenAI 설정이 누락되었습니다: ' + ', '.join(missing))
    endpoint = values['AZURE_OPENAI_ENDPOINT']
    parsed = urlsplit(endpoint)
    if (parsed.scheme != 'https' or not parsed.hostname or parsed.username is not None
            or parsed.password is not None or parsed.query or parsed.fragment
            or any(character.isspace() for character in endpoint)):
        raise ValueError('AZURE_OPENAI_ENDPOINT에는 유효한 HTTPS 리소스 endpoint를 설정해주세요.')
    token_parameter = env_value(environ, 'TELLY_AZURE_TOKEN_PARAMETER') or 'max_tokens'
    if token_parameter not in {'max_tokens', 'max_completion_tokens'}:
        raise ValueError('TELLY_AZURE_TOKEN_PARAMETER는 max_tokens 또는 max_completion_tokens여야 합니다.')
    return dict(azure_endpoint=endpoint, azure_deployment=values['AZURE_OPENAI_DEPLOYMENT'],
                api_key=values['AZURE_OPENAI_API_KEY'],
                api_version=env_value(environ, 'AZURE_OPENAI_API_VERSION') or '2024-02-15-preview',
                token_parameter=token_parameter)
