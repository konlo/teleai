"""Explicit model-provider selection for the persistent analysis graph.

The Databricks model endpoint is opt-in because it is a metered service. Its
token is shared with the existing workspace connection but never logged.
"""
from __future__ import annotations

import os
from copy import deepcopy

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool


def _databricks_schema(schema):
    """Keep tool arguments while removing JSON-schema features FMAPI rejects."""
    if not isinstance(schema, dict):
        return schema
    result = {}
    for key, value in schema.items():
        if key == 'properties':
            result[key] = {name: _databricks_schema(item) for name, item in value.items()}
        elif key in {'uniqueItems', 'pattern', 'anyOf', 'oneOf', 'allOf',
                     'prefixItems', '$ref'}:
            continue
        elif key == 'items':
            result[key] = _databricks_schema(value)
        elif key == 'additionalProperties' and isinstance(value, dict):
            result[key] = _databricks_schema(value)
        else:
            result[key] = value
    if 'oneOf' in schema or 'anyOf' in schema:
        # The serving API disallows mixed-type unions. The graph tool still
        # validates its actual arguments; advertise the simple scalar variant.
        choices = schema.get('oneOf') or schema.get('anyOf') or []
        scalar = next((item for item in choices if isinstance(item, dict)
                       and isinstance(item.get('type'), str)
                       and item['type'] not in {'array', 'object', 'null'}), None)
        if scalar:
            result.update(_databricks_schema(scalar))
    return result


class DatabricksChatModel(ChatOpenAI):
    """Adapt OpenAI chat payloads to Databricks serving's max_tokens field."""

    def _get_request_payload(self, *args, **kwargs):
        payload = super()._get_request_payload(*args, **kwargs)
        if 'max_completion_tokens' in payload:
            payload['max_tokens'] = payload.pop('max_completion_tokens')
        return payload

    def bind_tools(self, tools, **kwargs):
        compatible = []
        for tool in tools:
            item = deepcopy(convert_to_openai_tool(tool))
            item['function']['parameters'] = _databricks_schema(
                item['function'].get('parameters', {}))
            compatible.append(item)
        return super().bind_tools(compatible, **kwargs)


def build_analysis_chat_model(policy, *, provider=None, environ=None):
    """Construct one graph-compatible tool-calling model without a network call."""
    config = os.environ if environ is None else environ
    selected = (provider or config.get('TELLY_ANALYSIS_MODEL_PROVIDER') or 'ollama').casefold()
    if selected == 'ollama':
        return ChatOllama(
            model=config.get('OLLAMA_MODEL', 'gemma4:e4b'),
            base_url=config.get('OLLAMA_BASE_URL', 'http://localhost:11434'),
            reasoning=True, temperature=0, num_ctx=16384, num_predict=4096,
            client_kwargs={'timeout': policy.model_timeout_seconds},
        )
    if selected != 'databricks':
        raise ValueError('Unsupported analysis model provider')
    host = str(config.get('DATABRICKS_HOST') or '').rstrip('/')
    token = str(config.get('DATABRICKS_TOKEN') or '')
    if not host or not token:
        raise ValueError('Databricks model serving requires DATABRICKS_HOST and DATABRICKS_TOKEN')
    if not host.startswith(('https://', 'http://')):
        host = 'https://' + host
    if not host.startswith('https://'):
        raise ValueError('Databricks model serving requires HTTPS')
    return DatabricksChatModel(
        model=config.get('TELLY_DATABRICKS_MODEL', 'databricks-qwen3-next-80b-a3b-instruct'),
        base_url=host + '/serving-endpoints', api_key=token,
        temperature=0, max_tokens=2048,
        timeout=policy.model_timeout_seconds, max_retries=0,
    )
