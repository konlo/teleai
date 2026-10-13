"""Model-specific context limits; deployment names are not model identities."""
from functools import lru_cache
import json
from pathlib import Path
from core.databricks_settings import env_value


@lru_cache(maxsize=1)
def profiles():
    return json.loads(Path(__file__).with_name('model_context_limits.json').read_text())


def positive(config, name, default):
    text=env_value(config,name)
    if not text:return default
    if not text.isdecimal() or int(text)<=0:
        raise ValueError(name+' must be a positive integer')
    return int(text)


def resolve(config, provider, model, prefix):
    profile=profiles().get(provider,{}).get(model,{})
    maximum=profile.get('context_tokens')
    context=positive(config,prefix+'_CONTEXT_TOKENS',maximum)
    output=positive(config,prefix+'_MAX_OUTPUT_TOKENS',4096)
    if maximum and context>maximum:
        raise ValueError(prefix+'_CONTEXT_TOKENS exceeds the declared model maximum')
    if context and output>=context:
        raise ValueError(prefix+'_MAX_OUTPUT_TOKENS must be smaller than the context window')
    if profile.get('max_output_tokens') and output>profile['max_output_tokens']:
        raise ValueError(prefix+'_MAX_OUTPUT_TOKENS exceeds the declared model maximum')
    return context,output,profile.get('token_encoding')


def ollama_limits(config, model, *, prefix='OLLAMA'):
    context,output,_=resolve(config,'ollama',model,prefix)
    # Unknown models retain the previous window until explicitly configured.
    context=context or 16384
    if output>=context:raise ValueError(prefix+'_MAX_OUTPUT_TOKENS must be smaller than the context window')
    return context,output


def azure_limits(config):
    model=env_value(config,'AZURE_OPENAI_MODEL')
    context,output,encoding=resolve(config,'azure',model,'TELLY_AZURE')
    # An opaque existing deployment without a declared identity keeps the old
    # bounded admission policy; never assume it is the requested GPT model.
    return model,context,output,encoding


def ollama_role_model(model, name, config):
    if name==model.model and not env_value(config,'TELLY_GOAL_CONTEXT_TOKENS'):return model
    context,_=ollama_limits(config,name,prefix='TELLY_GOAL')
    if model.num_predict and model.num_predict>=context:
        raise ValueError('Goal output reservation must be smaller than its context window')
    return model.model_copy(update={'model':name,'num_ctx':context})
