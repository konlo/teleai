"""A bounded expression language, not a general Python REPL or OS sandbox."""
import ast

ATTRIBUTES={'copy','assign','groupby','agg','aggregate','mean','sum','median','min','max','count','size','std','var',
            'quantile','corr','rolling','expanding','diff','pct_change','shift','fillna','dropna','isna','notna',
            'astype','round','abs','clip','sort_values','reset_index','rename','value_counts','nunique',
            'head','tail','iloc','loc','columns','index','shape','values','T','DataFrame','Series','to_numeric',
            'to_datetime','cut','qcut','where','select','sqrt','log','log1p','exp','isfinite','nan','dt','year','month','day'}
FORBIDDEN=(ast.Import,ast.ImportFrom,ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef,ast.Lambda,
           ast.For,ast.AsyncFor,ast.While,ast.With,ast.AsyncWith,ast.Try,ast.Raise,ast.Delete,
           ast.Global,ast.Nonlocal,ast.ListComp,ast.SetComp,ast.DictComp,ast.GeneratorExp,
           ast.Await,ast.Yield,ast.YieldFrom,ast.NamedExpr)
AGGREGATIONS={'mean','sum','median','min','max','count','size','std','var','nunique','first','last','prod','sem','skew','kurt'}

def validate_aggregation(node):
    if isinstance(node,ast.Constant) and isinstance(node.value,str):
        if node.value not in AGGREGATIONS:raise ValueError('Unsupported aggregation function')
    elif isinstance(node,(ast.List,ast.Tuple)):
        for item in node.elts:validate_aggregation(item)
    elif isinstance(node,ast.Dict):
        for item in node.values:validate_aggregation(item)
    else:raise ValueError('Aggregation functions must be declared literal statistical names')

def validate_code(code):
    if not isinstance(code,str) or not 1<=len(code)<=6000:raise ValueError('Code length must be 1..6000')
    tree=ast.parse(code)
    tree.table_output_adapted=False
    # Generated pandas scripts commonly print their final table. Adapt only
    # the terminal single-argument output carrier; this does not change math.
    terminal=tree.body[-1] if tree.body else None
    if (isinstance(terminal,ast.Expr) and isinstance(terminal.value,ast.Call)
            and isinstance(terminal.value.func,ast.Name) and terminal.value.func.id=='print'
            and len(terminal.value.args)==1 and not terminal.value.keywords):
        tree.body[-1]=ast.copy_location(ast.Assign(targets=[ast.Name(id='result',ctx=ast.Store())],
            value=terminal.value.args[0]),terminal)
        tree.table_output_adapted=True
        ast.fix_missing_locations(tree)
    if len(list(ast.walk(tree)))>600:raise ValueError('Code is too complex')
    assigned={'df','pd','np','len','abs','min','max','sum','round'}
    for node in ast.walk(tree):
        if isinstance(node,FORBIDDEN):raise ValueError('Only bounded expressions and variable assignments are allowed')
        if isinstance(node,ast.Name):
            if node.id.startswith('_'):raise ValueError('Private names are forbidden')
            if isinstance(node.ctx,ast.Store):assigned.add(node.id)
        if isinstance(node,ast.Attribute) and node.attr not in ATTRIBUTES:raise ValueError('Unsupported attribute: '+node.attr)
        if isinstance(node,ast.Call):
            if not isinstance(node.func,(ast.Name,ast.Attribute)):raise ValueError('Dynamic callable is forbidden')
            if isinstance(node.func,ast.Name) and node.func.id not in {'len','abs','min','max','sum','round'}:
                raise ValueError('Unsupported function')
            if any(k.arg is None for k in node.keywords):raise ValueError('Expanded keyword arguments are forbidden')
            if isinstance(node.func,ast.Attribute) and node.func.attr in {'agg','aggregate'}:
                for argument in node.args:validate_aggregation(argument)
                for keyword in node.keywords:validate_aggregation(keyword.value)
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name):
                    if target.id in {'pd','np','len','abs','min','max','sum','round'}:raise ValueError('Library and builtin names are read-only')
                elif (isinstance(target,ast.Subscript) and isinstance(target.value,ast.Name)
                        and target.value.id not in {'pd','np'}):
                    pass # Worker-local DataFrame/variable only; never persisted originals.
                else:raise ValueError('Write only worker-local variables or DataFrame columns')
        if isinstance(node,(ast.AugAssign,ast.AnnAssign)):raise ValueError('Mutation is forbidden')
        if isinstance(node,ast.keyword) and node.arg=='inplace':raise ValueError('Inplace operations are forbidden')
        if isinstance(node,ast.Constant) and isinstance(node.value,str) and len(node.value)>1000:raise ValueError('Literal too long')
        if isinstance(node,ast.Constant) and type(node.value) is int and abs(node.value)>1000000:raise ValueError('Literal allocation bound exceeded')
    if any(isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load) and n.id not in assigned for n in ast.walk(tree)):
        raise ValueError('Unknown variable')
    if not any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='result' for t in n.targets) for n in tree.body):
        raise ValueError('Assign a pandas DataFrame to result')
    return tree
