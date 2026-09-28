"""Conservative provenance shared by remote loading and local analysis."""
from sqlglot import exp
from utils.analysis_datasets import Condition


def query_coverage(tree, *, truncated=False):
    """Completeness is about the source scope, not just a successful fetch."""
    if truncated:
        return 'truncated'
    if tree.find(exp.TableSample):
        return 'sampled'
    if any(node.args.get('limit') is not None or node.args.get('offset') is not None
           for node in tree.walk()):
        return 'unknown'
    return 'complete'


def table_identity(table):
    """Aliases, quoting and sampling syntax are not part of a table identity."""
    return '.'.join(part for part in (table.catalog, table.db, table.name) if part)


def single_table(tree):
    if not isinstance(tree, exp.Select) or tree.args.get('joins') or tree.args.get('with_'):
        return None
    source=tree.args.get('from_')
    if source is None or not isinstance(source.this,exp.Table):return None
    tables=list(tree.find_all(exp.Table))
    return source.this if len(tables) == 1 and len(list(tree.find_all(exp.Select))) == 1 else None


def query_conditions(tree):
    """Return a simple source WHERE conjunction, independently of projection.

    Unknown predicates remain unknown; in particular OR is never turned into AND.
    A HAVING/QUALIFY filter is not a raw source predicate.
    """
    if single_table(tree) is None or tree.args.get('having') or tree.args.get('qualify'):
        return None
    def literal(node):
        if isinstance(node, exp.Neg):
            return -literal(node.this)
        if isinstance(node, exp.Boolean):
            return node.this
        if not isinstance(node,exp.Literal):raise ValueError('Unsupported literal')
        if node.is_string:return node.this
        value=node.this
        return float(value) if any(c in value.lower() for c in ('.','e')) else int(value)
    def parse(node):
        if isinstance(node,exp.Paren):return parse(node.this)
        if isinstance(node,exp.And):return parse(node.this)+parse(node.expression)
        ops={exp.EQ:'eq',exp.NEQ:'ne',exp.GT:'gt',exp.GTE:'ge',exp.LT:'lt',exp.LTE:'le'}
        if type(node) in ops and isinstance(node.this,exp.Column):
            return [Condition(node.this.name,ops[type(node)],literal(node.expression))]
        if isinstance(node,exp.In) and isinstance(node.this,exp.Column) and not node.args.get('query'):
            return [Condition(node.this.name,'in',[literal(item) for item in node.expressions])]
        if isinstance(node,exp.Between) and isinstance(node.this,exp.Column):
            return [Condition(node.this.name,'ge',literal(node.args['low'])),Condition(node.this.name,'le',literal(node.args['high']))]
        raise ValueError('Unsupported predicate')
    where=tree.args.get('where')
    try:return tuple(parse(where.this)) if where else ()
    except (ValueError,TypeError,KeyError):return None


def raw_conditions(tree):
    """Only identity projections and complete filters preserve raw lineage."""
    if query_coverage(tree) != 'complete' or single_table(tree) is None:
        return None
    if tree.args.get('distinct') or tree.args.get('group') or tree.find(exp.AggFunc):
        return None
    for item in tree.expressions:
        if isinstance(item, exp.Star) and (item.args.get('replace') or item.args.get('rename')):
            return None
        if isinstance(item, exp.Alias) and isinstance(item.this, exp.Column) and item.alias == item.this.name:
            continue
        if not isinstance(item, (exp.Star, exp.Column)):
            return None
    return query_conditions(tree)


def count_frequency_columns(tree):
    """Attest the supported value + COUNT(*) GROUP BY value contract.

    Integer-looking SUM/balance fields, grouped subsets and joined/derived tables
    are not evidence of raw-row frequency.
    """
    if (single_table(tree) is None or query_coverage(tree) != 'complete'
            or tree.args.get('distinct') or tree.args.get('having') or tree.args.get('qualify')
            or tree.find(exp.Window) or len(tree.expressions) != 2):
        return None
    group = tree.args.get('group')
    if not group or len(group.expressions) != 1 or not isinstance(group.expressions[0], exp.Column):
        return None
    if any(value for key, value in group.args.items() if key != 'expressions'):
        return None
    value, frequency = None, None
    for item in tree.expressions:
        expression = item.this if isinstance(item, exp.Alias) else item
        if isinstance(expression, exp.Column) and expression.name == group.expressions[0].name:
            value = item.alias_or_name
        elif (isinstance(item, exp.Alias) and isinstance(expression, exp.Count)
              and isinstance(expression.this, exp.Star) and not expression.expressions):
            frequency = item.alias
        else:
            return None
    return (value, frequency) if value and frequency and value != frequency else None
