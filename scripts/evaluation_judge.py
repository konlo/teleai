"""Explicit judge provider selection; remote use is opt-in and credentials stay private."""
import os
from pathlib import Path


def make_judge(provider, model=None, timeout=45):
    from deepeval.models import OpenAIModel
    if provider=='ollama':
        return OpenAIModel(model=model or 'gemma4:e4b',base_url='http://localhost:11434/v1',
            api_key='ollama',timeout=timeout,max_retries=0),model or 'gemma4:e4b'
    if provider!='databricks':raise ValueError('Unknown judge provider')
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1]/'.env')
    host=os.environ.get('DATABRICKS_HOST','').strip().rstrip('/')
    if host and '://' not in host:
        host='https://'+host
    token=os.environ.get('DATABRICKS_TOKEN','')
    if not host.startswith('https://') or not token:raise ValueError('Configured HTTPS Databricks judge is required')
    name=model or os.environ.get('TELLY_DATABRICKS_MODEL','databricks-qwen3-next-80b-a3b-instruct')
    return OpenAIModel(model=name,base_url=host+'/serving-endpoints',api_key=token,
        timeout=timeout,max_retries=0,generation_kwargs={'max_tokens':2048}),name


GROUNDING_STEPS = [
    "INPUT determines what was requested and the requested presentation. EXPECTED_OUTPUT is the authoritative factual answer: use it to verify all numerical values and populations; it adds no presentation requirements.",
    "Compare ACTUAL_OUTPUT directly with the authoritative values in EXPECTED_OUTPUT. A numerical disagreement is a factual error even when INPUT does not state the answer. Scale or decimal-point differences are errors unless an explicitly stated unit conversion explains them. Also check requested population, calculations and delivered artifacts. Wrong numbers, missing requested results, wrong scope or undelivered charts must score below 8.",
    "Semantic equivalents are fully correct: prose versus key=value or JSON, equivalent percentages and ratios, translation into the user's language, harmless rounding and reordered groups when no order was requested. Never demand reference wording, punctuation, language or explanatory extras.",
    "If every INPUT requirement is satisfied and every requested fact agrees with the reference, award 10 even when formatting differs. Reduce the score only for a concrete unmet INPUT requirement or factual contradiction; cite it. A promise to deliver later does not complete a current request.",
]
