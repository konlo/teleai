"""Recognize narrowly defined provider failures without recording response bodies."""
import re


def model_error_category(error):
    status = getattr(error, 'status_code', getattr(error, 'http_status', None))
    body = getattr(error, 'body', None)
    if status != 400 or not isinstance(body, dict) or body.get('error_code') != 'BAD_REQUEST':
        return None
    message = body.get('message')
    # Databricks serving has returned this temporary service failure as HTTP400.
    # An arbitrary 400, authentication failure or "try again" substring is not retryable.
    if isinstance(message, str) and re.fullmatch(
            r'(?:BAD_REQUEST:\s*)?Cannot create or query foundation model endpoints, '
            r'please try again later\.?', message.strip()):
        return 'model_provider_temporarily_unavailable'
    return None
