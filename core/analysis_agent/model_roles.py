"""Bound declarative JSON roles without switching the configured provider."""
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI


def json_role(model, schema, tokens):
    if isinstance(ChatOllama, type) and isinstance(model, ChatOllama):
        return model.model_copy(update={'format':schema, 'reasoning':False, 'num_predict':tokens})
    if isinstance(model, AzureChatOpenAI):
        # JSON object mode supports existing Azure deployments without making
        # the richer schema a new API-version requirement. Validate locally.
        from core.analysis_agent.json_contract import compact_schema
        return model.model_copy(update={'max_tokens':tokens,
            '_telly_output_schema':compact_schema(schema),
            'model_kwargs':{**model.model_kwargs, 'response_format':{'type':'json_object'}}})
    return None
