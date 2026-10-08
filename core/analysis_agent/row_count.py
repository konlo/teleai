"""Table-neutral row-count intent and exact scalar evidence."""
import re
from sqlglot import exp


def requested(text):
    return bool(re.search(r'(?<![A-Za-z0-9_])rows?\s*(?:수|개수|몇|count)|'
        r'(?<![A-Za-z0-9_])(?:행|레코드)\s*(?:수|개수)|'
        r'\b(?:number|count)\s+of\s+(?:rows|records)\b',text,re.I))


def exact_scalar(tree):
    if len(tree.expressions)!=1 or tree.args.get('group') or tree.args.get('having'):
        return False
    value=tree.expressions[0]
    if isinstance(value,exp.Alias):value=value.this
    if not isinstance(value,exp.Count):return False
    return bool(isinstance(value.this,exp.Star) or isinstance(value.this,exp.Literal)
                and value.this.is_int and value.this.this=='1')
