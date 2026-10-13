"""Compare planned execution constraints without equating explanatory prose."""
from sqlglot import parse_one
from sqlglot.errors import SqlglotError


def matches_plan(context,executed,planned):
    def source_key(value):return str(value).replace('`','').casefold()
    if source_key(executed.get('source'))!=source_key(planned.get('source')):return False
    try:
        def canonical(query):
            return parse_one(query,read=context.sql_dialect).sql(
                dialect=context.sql_dialect,normalize=True,identify=True)
        return bool(planned.get('query') and canonical(executed['query'])==canonical(planned['query']))
    except (KeyError,ValueError,TypeError,SqlglotError):return False
