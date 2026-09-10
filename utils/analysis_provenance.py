"""Conservative provenance for complete, single-table raw SELECT results."""
from sqlglot import exp
from utils.analysis_datasets import Condition


def raw_conditions(tree):
    """Return a conjunction or None. Never flatten OR/NOT into AND."""
    if not isinstance(tree,exp.Select) or tree.args.get('joins') or tree.args.get('distinct') or tree.args.get('group') or tree.find(exp.AggFunc):return None
    source=tree.args.get('from_')
    if source is None or not isinstance(source.this,exp.Table):return None
    tables=list(tree.find_all(exp.Table))
    if len(tables)!=1 or tree.args.get('with_'):return None
    if any(not isinstance(item,(exp.Star,exp.Column)) for item in tree.expressions):return None
    def literal(node):
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
