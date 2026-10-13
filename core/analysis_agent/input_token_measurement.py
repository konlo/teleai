"""Local token estimates; a different provider's tokenizer is never exact."""
from functools import lru_cache


@lru_cache(maxsize=4)
def encoder(name):
    import tiktoken
    return tiktoken.get_encoding(name)


def measure(text, model):
    name=getattr(model,'input_token_encoding',None)
    if isinstance(name,str) and name:
        try:
            return len(encoder(name).encode(text,disallowed_special=())), 'tiktoken_estimate'
        except (ImportError,ValueError,OSError):
            pass
    # Without the matching tokenizer, UTF-8 bytes are an intentionally loose
    # upper bound for byte-fallback text tokenizers, never a measured token count.
    return len(text.encode('utf8')), 'conservative_utf8_token_upper_bound'


def limits(model, fallback):
    context=getattr(model,'context_window_tokens',None) or getattr(model,'num_ctx',None)
    reserve=getattr(model,'num_predict',None) or getattr(model,'max_tokens',None)
    context=context if type(context) is int and context>0 else None
    reserve=reserve if type(reserve) is int and reserve>0 else 2048
    if context and reserve>=context:raise ValueError('Model output reservation exceeds context window')
    return context,reserve,context-reserve if context else fallback
