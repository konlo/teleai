"""A small, deterministic scope contract, grounded in supplied table metadata.

This is deliberately not a general natural-language parser. It recognizes
explicit column predicates, grounded ISO dates/months and a few reference
follow-ups. Unknown or ambiguous recognized constraints remain unresolved.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import date
import json
import re

from utils.analysis_datasets import Condition, full_read_preflight, project_dataset


_ISO = re.compile(r'(?<![A-Za-z0-9_-])(\d{4}-\d{2}(?:-\d{2})?)(?![A-Za-z0-9_-])')
_REFERENCE = re.compile(r'그중|같은|아까|앞선|이어서|말고|대신|바꿔')
_PREVIOUS_MONTH = re.compile(r'이전\s*달|지난\s*달|전월')
_LITERAL = r'''(?:'(?:[^']|'')*'|"(?:[^"]|"")*"|\d{4}-\d{2}(?:-\d{2})?|[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[+-]?\d+(?:\.\d+)?|true\b|false\b|[A-Za-z_][A-Za-z0-9_-]*)'''
_NUMBER = r'[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'
_OPS = {'=':'eq', '==':'eq', '!=':'ne', '<>':'ne', '>':'gt', '>=':'ge', '<':'lt', '<=':'le'}


def _mentioned(text, name):
    # A Korean unit alias such as "월" may directly follow a digit ("5월").
    # Latin identifier prefixes remain excluded.
    if not name:
        return False
    # Korean aliases may carry particles, but must not match the beginning of
    # an unrelated word (e.g. a time-unit alias inside a currency word).
    suffix = (r'(?=(?:이나|이거나|거나|은|는|이|가|을|를|의|과|와|별|간|에서|으로|에|도)?(?:[^가-힣A-Za-z_]|$))'
              if re.search(r'[가-힣]$', name) else r'(?![A-Za-z0-9_])')
    return bool(re.search(r'(?<![A-Za-z_가-힣])' + re.escape(name) + suffix, text, re.I))


def _scalar(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, 'item'):
        try: return _scalar(value.item())
        except (ValueError, TypeError): return None
    if isinstance(value, date): return value.isoformat()
    return None


def _literal(value):
    if value[:1] in {'"', "'"}:
        return value[1:-1].replace(value[0] * 2, value[0])
    if re.fullmatch(r'\d{4}-\d{2}(?:-\d{2})?', value): return value
    if value.lower() in {'true', 'false'}: return value.lower() == 'true'
    numeric = value.replace(',', '')
    if re.fullmatch(r'[+-]?\d+(?:\.\d+)?', numeric):
        return float(numeric) if '.' in numeric else int(numeric)
    return value


def _grounding(text, context):
    """Read bounded metadata or small raw frames, never sample a large frame."""
    columns, unresolved = {}, []
    if context is None: return columns, unresolved
    reference = list(context.reference_context)
    metadata = list(context.datasets.metadata.values())
    def source_key(value):
        # Quoted persisted IDs and catalog IDs identify the same source.
        import sqlglot
        from utils.analysis_provenance import single_table, table_identity
        try:
            table = single_table(sqlglot.parse_one('SELECT * FROM ' + value, read='databricks'))
            return table_identity(table).casefold() if table is not None else value.casefold()
        except (TypeError, ValueError, sqlglot.errors.SqlglotError):
            return str(value).casefold()
    sources = {source_key(item.get('table', '')) for item in reference} | {source_key(info.source) for info in metadata}
    selected = {source for source in sources if _mentioned(text, source)}
    if not selected:
        selected = {source for source in sources if _mentioned(text, source.split('.')[-1])}
    if len(selected) > 1: unresolved.append('ambiguous_source')

    def add(name, dtype='', aliases=(), values=()):
        if not isinstance(name, str) or not name: return
        if name not in columns and len(columns) >= 64: return
        item = columns.setdefault(name, {'aliases':set(), 'dtypes':set(), 'values':[]})
        item['aliases'].update(alias for alias in aliases if isinstance(alias, str))
        item['dtypes'].add(str(dtype).lower())
        for raw in values:
            value = _scalar(raw)
            if isinstance(value, str) and len(value) > 256:
                continue  # Large cell values are neither safe nor useful as NL scope labels.
            if value is not None and value not in item['values'] and len(item['values']) < 32:
                item['values'].append(value)

    from core.analysis_catalog import grounded_reference_columns
    for table in reference[:16]:
        if selected and source_key(table.get('table', '')) not in selected: continue
        for column in grounded_reference_columns(table, context.datasets)[:64]:
            values = [entry.get('value') if isinstance(entry, dict) else entry
                      for entry in column.get('top_values', [])[:16]]
            add(column.get('name'), column.get('dtype', ''), column.get('aliases', []), values)
    for info in list(reversed(metadata))[:8]:
        if selected and source_key(info.source) not in selected: continue
        for name in info.columns[:64]: add(name)
        if info.grain != 'raw' or info.rows > 256 or len(info.columns) > 64: continue
        try:
            # A small row count does not imply a small frame: a few rows may
            # contain large strings across many columns. Ground values from
            # narrow projections only after the persisted-size preflight.
            if full_read_preflight(context.datasets, [info.id]):
                unresolved.append('small_frame_unavailable')
                continue
            if hasattr(context.datasets, 'inspect'):
                dtypes = context.datasets.inspect(info.id).get('dtypes', {})
                for name in info.columns:
                    add(name, dtypes.get(name, ''))
            for start in range(0, len(info.columns), 8):
                names = info.columns[start:start + 8]
                frame = project_dataset(context.datasets, info.id, names)
                if len(frame) > 256:
                    raise ValueError('Small-frame grounding exceeded its row bound')
                for name in names:
                    add(name, str(frame[name].dtype), values=frame[name].dropna().tolist())
        except (KeyError, OSError, ValueError, TypeError, MemoryError):
            unresolved.append('small_frame_unavailable')
    return columns, unresolved


def resolve_request_scope(text, context, previous=None):
    """Return supported request predicates and explicit unresolved reason codes.

    ``previous`` is a previous return value of this function. Only reference
    follow-ups inherit it. A newly supplied predicate replaces prior predicates
    on that column; other inherited predicates remain unchanged.
    """
    text = str(text)
    grounding, unresolved = _grounding(text, context)
    previous = previous or {}
    inherited = bool(_REFERENCE.search(text) or _PREVIOUS_MONTH.search(text))
    conditions = [dict(c) for c in previous.get('conditions', [])] if inherited else []
    inherited_measure = [dict(c) for c in previous.get('measure_conditions', [])] if inherited else []
    inherited_ratio = dict(previous.get('ratio', {})) if inherited and previous.get('ratio') else None
    if inherited: unresolved.extend(previous.get('unresolved', []))
    found, spans, mentioned_columns = [], [], []
    numeric_ranges = 0
    for name, metadata in grounding.items():
        aliases = sorted({name, *metadata['aliases']}, key=len, reverse=True)
        names = '(?:' + '|'.join(re.escape(alias) for alias in aliases) + ')'
        column_pattern = (r'(?<![A-Za-z_.])[`\'\"]?' + names
                          + r'[`\'\"]?(?![A-Za-z0-9_])\)?(?:이|가|은|는|을|를)?')
        if any(_mentioned(text, alias) for alias in aliases): mentioned_columns.append(name)
        # A user may explicitly qualify a column in a multi-table request.
        # Preserve that role instead of merging two equalities on the same
        # base column into an IN predicate. The qualifier must be supplied by
        # the user; it is never invented from a model-generated SQL alias.
        qualified_pattern = (r'(?<![A-Za-z0-9_])([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*'
                             + names + r'\s*(>=|<=|!=|<>|==|=|>|<)\s*(' + _LITERAL + r')(?!\s*\.)')
        for match in re.finditer(qualified_pattern, text, re.I):
            qualified = match[1] + '.' + name
            found.append({'column': qualified, 'op': _OPS[match[2]],
                          'value': _literal(match[3])})
            mentioned_columns.append(qualified)
            spans.append(match.span())
        for alias in aliases:
            decade = re.fullmatch(r'(\d{1,3})대', alias)
            if decade and _mentioned(text, alias):
                start = int(decade[1])
                found.extend([{'column':name, 'op':'ge', 'value':start},
                              {'column':name, 'op':'le', 'value':start + 9}])
        for match in re.finditer(column_pattern + r'\s*(>=|<=|!=|<>|==|=|>|<)\s*(' + _LITERAL + ')', text, re.I):
            found.append({'column':name, 'op':_OPS[match[1]], 'value':_literal(match[2])})
            spans.append(match.span())
        # Bind a bounded numeric interval to the explicitly named column.
        # In a chart follow-up this changes the population, so a cached chart
        # of the unfiltered source cannot satisfy the request.
        range_prefix = column_pattern + r'[^0-9+\-\n]{0,32}(' + _NUMBER + r')\s*'
        intervals = (
            range_prefix + r'부터\s*(' + _NUMBER + r')\s*까지',
            range_prefix + r'[~∼–—]\s*(' + _NUMBER + r')\s*(?:범위|구간)',
        )
        for interval in intervals:
            for match in re.finditer(interval, text, re.I):
                lower, upper = _literal(match[1]), _literal(match[2])
                if lower > upper:
                    unresolved.append('invalid_range_bounds')
                    continue
                found.extend([{'column':name, 'op':'ge', 'value':lower},
                              {'column':name, 'op':'le', 'value':upper}])
                spans.append(match.span())
                numeric_ranges += 1
        paired_bounds = (column_pattern + r'\s*(' + _NUMBER
                         + r')\s*(이상|초과)\s*(' + _NUMBER + r')\s*(이하|미만)')
        for match in re.finditer(paired_bounds, text, re.I):
            lower, upper = _literal(match[1]), _literal(match[3])
            if lower > upper:
                unresolved.append('invalid_range_bounds')
                continue
            found.extend([{'column':name, 'op':'ge' if match[2] == '이상' else 'gt',
                           'value':lower},
                          {'column':name, 'op':'le' if match[4] == '이하' else 'lt',
                           'value':upper}])
            spans.append(match.span())
        for match in re.finditer(column_pattern + r'\s*(' + _LITERAL
                                 + r')\s*(?:[A-Za-z가-힣%]+(?:을|를)?\s*)?(이상|이하|초과|미만|넘는|넘은)', text, re.I):
            found.append({'column':name, 'op':{'이상':'ge','이하':'le','초과':'gt','미만':'lt',
                          '넘는':'gt','넘은':'gt'}[match[2]],
                          'value':_literal(match[1])})
            spans.append(match.span())
        for match in re.finditer(column_pattern + r'\s*(\d{1,3})대', text, re.I):
            start = int(match[1])
            found.extend([{'column':name, 'op':'ge', 'value':start},
                          {'column':name, 'op':'le', 'value':start + 9}])
            spans.append(match.span())
        # A list written beside a grounded column is one IN predicate. The
        # literal values come from the request and need not all appear in a
        # capped TableContext profile.
        for match in re.finditer(column_pattern + r'''\s*\(\s*((?:''' + _LITERAL
                                 + r'''\s*,\s*)+''' + _LITERAL + r''')\s*\)''', text, re.I):
            values=[_literal(item.group(0)) for item in re.finditer(_LITERAL, match[1], re.I)]
            if len(values)>1:
                found.append({'column':name,'op':'in','value':values})
                spans.append(match.span())
        # Adjacent equality is allowed only for a value observed in this column.
        for value in metadata['values']:
            if not isinstance(value, (str, int, float, bool)): continue
            token = re.escape(str(value))
            patterns = [
                column_pattern + r'''\s+(?:['"]?''' + token + r'''['"]?)(?=$|[\s,.;)]|만|인|의|에서|이고|이며)''',
                column_pattern + r'''\s*\(\s*(?:['"]?''' + token + r'''['"]?)\s*\)''',
                # A grounded canonical value may follow a short human label,
                # e.g. "혼인 상태가 이혼(divorced)". Keep the window bounded
                # and stop at list/conjunction punctuation.
                column_pattern + r'''[^,;\n]{0,24}?\(\s*(?:['"]?''' + token + r'''['"]?)\s*\)''']
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.I):
                    if any(a <= match.start() < b for a, b in spans): continue
                    found.append({'column':name, 'op':'eq', 'value':value})
                    spans.append(match.span())

        # Ground common possession/affirmation wording only when the external
        # context proves this is a two-valued boolean-like column. The words do
        # not invent domain meanings or values for arbitrary categories.
        binary = {}
        for value in metadata['values']:
            normalized = str(value).strip().casefold()
            if normalized in {'yes', 'true', 'y', '1'}:
                binary.setdefault('positive', value)
            elif normalized in {'no', 'false', 'n', '0'}:
                binary.setdefault('negative', value)
        if set(binary) == {'positive', 'negative'}:
            polarities = set()
            for alias in aliases:
                for alias_match in re.finditer(
                        r'(?<![A-Za-z0-9_])' + re.escape(alias) + r'(?![A-Za-z0-9_])', text, re.I):
                    window = text[alias_match.end():alias_match.end() + 32]
                    negative = re.search(r'없(?:는|음)|안\s*받은|받지\s*않은|가입하지\s*않은|하지\s*않은|미보유|아닌', window)
                    positive = re.search(r'^\s*한\b|있(?:는|음)|받은|가입한|보유한|참인', window)
                    if negative:
                        polarities.add('negative')
                    elif positive:
                        polarities.add('positive')
            if len(polarities) == 1:
                polarity = next(iter(polarities))
                found.append({'column':name, 'op':'eq', 'value':binary[polarity]})
            elif len(polarities) > 1:
                unresolved.append('ambiguous_boolean_polarity')

    for match in _ISO.finditer(text):
        if any(start <= match.start() < end for start, end in spans): continue
        literal = match[1]
        if (re.match(r'\s*(?:이전|이후|전까지|후부터|부터|까지|보다|제외)', text[match.end():])
                or re.search(r'\b(?:before|after|since|until)\s*$', text[:match.start()], re.I)):
            unresolved.append('unsupported_date_relation')
            continue
        try: date.fromisoformat(literal + '-01' if len(literal) == 7 else literal)
        except ValueError:
            unresolved.append('invalid_date_literal')
            continue
        candidates = {name for name, meta in grounding.items() if literal in meta['values']}
        if not candidates:
            candidates = {name for name, meta in grounding.items()
                          if any(isinstance(value, str) and len(value) == len(literal)
                                 and re.fullmatch(r'\d{4}-\d{2}(?:-\d{2})?', value)
                                 for value in meta['values'])}
        if len(candidates) == 1:
            found.append({'column':next(iter(candidates)), 'op':'eq', 'value':literal})
        else:
            unresolved.append('ambiguous_date_column' if candidates else 'ungrounded_date_column')

    disjunction = bool(re.search(r'\bOR\b|또는|혹은|아니면|이나|거나|중\s*하나라도', text, re.I))
    if found or _ISO.search(text):
        if re.search(r'부터.*까지|\bbetween\b', text, re.I) and not numeric_ranges:
            unresolved.append('unsupported_range_syntax')
    changed = {item['column'] for item in found}
    conditions = [item for item in conditions if item['column'] not in changed] + found
    if _PREVIOUS_MONTH.search(text):
        months = [item for item in conditions if item['op'] == 'eq' and isinstance(item['value'], str)
                  and re.fullmatch(r'\d{4}-\d{2}', item['value'])]
        if len(months) != 1 or months[0]['column'] in changed:
            unresolved.append('ambiguous_previous_month')
        else:
            year, month = map(int, months[0]['value'].split('-'))
            if 1 <= month <= 12:
                year, month = (year - 1, 12) if month == 1 else (year, month - 1)
                months[0]['value'] = f'{year:04d}-{month:02d}'
            else: unresolved.append('invalid_date_literal')
    unique = list({json.dumps(item, sort_keys=True, ensure_ascii=False):item for item in conditions}.values())
    # Multiple equalities on one column are equivalent to a single IN.
    for column in {item['column'] for item in unique}:
        equalities=[item for item in unique if item['column']==column and item['op']=='eq']
        if len(equalities)>1:
            unique=[item for item in unique if item not in equalities]
            unique.append({'column':column,'op':'in','value':[item['value'] for item in equalities]})
    any_conditions=[]
    if disjunction:
        candidates=[item for item in unique if item['op']=='eq'
                    and item['column'] in mentioned_columns]
        marker=re.compile(r'\bOR\b|또는|혹은|아니면|이나|거나|중\s*하나라도',re.I)
        positioned=[]
        for item in candidates:
            base=item['column'].rsplit('.',1)[-1]
            names=({item['column']} if '.' in item['column'] else
                   {base,*grounding[base]['aliases']})
            matches=[match for name in names
                     for match in re.finditer(re.escape(name),text,re.I)]
            if matches:
                match=min(matches,key=lambda value:value.start())
                positioned.append((match.start(),match.end(),item))
        positioned.sort(key=lambda value:value[0])
        pairs=[(left[2],right[2]) for left,right in zip(positioned,positioned[1:])
               if 0<=right[0]-left[1]<=64 and marker.search(text[left[1]:right[0]])]
        if len(pairs)==1:
            any_conditions=list(pairs[0])
            unique=[item for item in unique if item not in any_conditions]
        else:
            unresolved.append('unsupported_disjunction')
    # A joined SQL proposal may be checked only when the user explicitly
    # supplied the join edges. A model-chosen relationship is not independent
    # evidence of the user's intended population.
    join_edges=[list(edge) for edge in previous.get('join_edges', [])] if inherited else []
    edge_pattern = (r'(?<![A-Za-z0-9_])([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)'
                    r'\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)')
    for match in re.finditer(edge_pattern, text):
        if match[2] not in grounding or match[4] not in grounding or match[1] == match[3]:
            unresolved.append('ungrounded_join_edge')
            continue
        join_edges.append(sorted([match[1]+'.'+match[2], match[3]+'.'+match[4]]))
    join_edges=list({tuple(edge):edge for edge in join_edges}.values())
    for column in {item['column'] for item in unique}:
        equality = {json.dumps(item['value'], sort_keys=True) for item in unique if item['column'] == column and item['op'] == 'eq'}
        if len(equality) > 1: unresolved.append('conflicting_equalities')
    # A predicate written beside "ratio/rate" defines the numerator rather
    # than the population. Keep those roles separate so y='yes' in a conversion
    # rate is required inside the calculation but is not incorrectly forced
    # into the dataset WHERE clause.
    ratio_match = re.search(r'비율|성공률|생존율|전환율|[가-힣A-Za-z]+[율률]|\b(?:ratio|rate|percentage|percent)\b', text, re.I)
    ratio_column = None
    if ratio_match:
        ranked = []
        for name, metadata in grounding.items():
            for alias in {name, *metadata['aliases']}:
                for mention in re.finditer(r'(?<![A-Za-z0-9_])' + re.escape(alias)
                                           + r'(?![A-Za-z0-9_])', text, re.I):
                    if mention.start() <= ratio_match.end() and ratio_match.start() - mention.end() <= 24:
                        gap = max(0, ratio_match.start() - mention.end())
                        ranked.append((gap, -len(alias), name))
        if ranked: ratio_column = min(ranked)[2]
    measure_conditions = inherited_measure
    ratio = inherited_ratio
    if ratio_column:
        ratio = {'column': ratio_column}
        explicit_measure = [item for item in unique if item['column'] == ratio_column]
        unique = [item for item in unique if item['column'] != ratio_column]
        measure_conditions = explicit_measure
        if not measure_conditions:
            binary = {}
            for value in grounding[ratio_column]['values']:
                normalized = str(value).strip().casefold()
                if normalized in {'yes', 'true', 'y', '1'}: binary.setdefault('positive', value)
                elif normalized in {'no', 'false', 'n', '0'}: binary.setdefault('negative', value)
            if set(binary) == {'positive', 'negative'}:
                measure_conditions = [{'column':ratio_column, 'op':'eq', 'value':binary['positive']}]
                values = {value for value in grounding[ratio_column]['values']
                          if isinstance(value, (int, float)) and not isinstance(value, bool)}
                if values == {0, 1}: ratio['aggregation'] = 'mean_zero_one'
            else:
                unresolved.append('ungrounded_ratio_numerator')
    return {'conditions':unique, 'any_conditions':any_conditions,
            'join_edges':join_edges,
            'measure_conditions':measure_conditions,
            'ratio':ratio, 'unresolved':sorted(set(unresolved)),
            'columns':sorted(set(mentioned_columns))}


def scope_matches(executed, requested, *, histogram_column=None, dialect='databricks'):
    """Require the exact supported Boolean population, including inherited filters.

    ``executed`` may be a DatasetInfo, a SQL query, or a sequence of Conditions
    (or their dict representations). This does not certify coverage or lineage;
    those remain separate runtime checks. Unsupported SQL predicates fail closed.
    """
    if requested.get('unresolved'): return False
    import sqlglot
    from sqlglot import exp
    from utils.analysis_provenance import single_table

    def literal(node):
        if isinstance(node, exp.Neg): return -literal(node.this)
        if isinstance(node, exp.Boolean): return node.this
        if not isinstance(node, exp.Literal): raise ValueError('unsupported literal')
        if node.is_string: return node.this
        return float(node.this) if any(c in node.this.lower() for c in ('.','e')) else int(node.this)

    def column_name(node, joined):
        if not isinstance(node, exp.Column): raise ValueError('unsupported column')
        if joined:
            if not node.table: raise ValueError('ambiguous joined column')
            return node.table + '.' + node.name
        return node.name

    def formula(node, joined=False):
        """Return a bounded disjunctive-normal-form list of conjunctions."""
        if isinstance(node, exp.Paren): return formula(node.this, joined)
        if isinstance(node, exp.Or): return formula(node.this, joined)+formula(node.expression, joined)
        if isinstance(node, exp.And):
            left,right=formula(node.this, joined),formula(node.expression, joined)
            if len(left)*len(right)>16: raise ValueError('boolean formula too large')
            return [a+b for a in left for b in right]
        ops={exp.EQ:'eq',exp.NEQ:'ne',exp.GT:'gt',exp.GTE:'ge',exp.LT:'lt',exp.LTE:'le'}
        if type(node) in ops and isinstance(node.this,exp.Column):
            return [[Condition(column_name(node.this, joined),ops[type(node)],literal(node.expression))]]
        if isinstance(node,exp.In) and isinstance(node.this,exp.Column) and not node.args.get('query'):
            return [[Condition(column_name(node.this, joined),'in',
                               [literal(item) for item in node.expressions])]]
        if isinstance(node,exp.Between) and isinstance(node.this,exp.Column):
            name=column_name(node.this, joined)
            return [[Condition(name,'ge',literal(node.args['low'])),
                     Condition(name,'le',literal(node.args['high']))]]
        raise ValueError('unsupported predicate')

    def join_sources(tree):
        """Accept only flat inner equijoins with qualified, distinct aliases."""
        source=tree.args.get('from_')
        joins=tree.args.get('joins') or []
        if (not isinstance(tree, exp.Select) or not joins or tree.args.get('with_')
                or len(list(tree.find_all(exp.Select))) != 1
                or source is None or not isinstance(source.this, exp.Table)):
            raise ValueError('unsupported joined query')
        aliases={source.this.alias_or_name}
        edges=[]
        for join in joins:
            if (not isinstance(join.this, exp.Table) or join.args.get('side')
                    or str(join.args.get('kind') or '').upper() not in {'', 'INNER'}):
                raise ValueError('unsupported join kind')
            alias=join.this.alias_or_name
            if not alias or alias in aliases: raise ValueError('duplicate join alias')
            predicate=join.args.get('on')
            if (not isinstance(predicate, exp.EQ)
                    or not isinstance(predicate.this, exp.Column)
                    or not isinstance(predicate.expression, exp.Column)):
                raise ValueError('unsupported join condition')
            left,right=predicate.this,predicate.expression
            if (not left.table or not right.table or left.table == right.table
                    or {left.table,right.table} - (aliases | {alias})
                    or alias not in {left.table,right.table}):
                raise ValueError('unbound join condition')
            edges.append(sorted([left.table+'.'+left.name, right.table+'.'+right.name]))
            aliases.add(alias)
        return edges, aliases

    def base_population_select(tree):
        """Follow a linear, filter-free CTE chain to its raw source SELECT."""
        with_clause = tree.args.get('with_')
        if with_clause is None:
            return tree
        definitions = {cte.alias.casefold(): cte.this
                       for cte in with_clause.expressions if cte.alias}
        if len(definitions) != len(with_clause.expressions):
            raise ValueError('ambiguous CTE names')
        current, seen = tree, set()
        while True:
            if not isinstance(current, exp.Select):
                raise ValueError('unsupported CTE query')
            source = current.args.get('from_')
            table = source.this if source else None
            name = table.name.casefold() if isinstance(table, exp.Table) else ''
            if name not in definitions:
                break
            if (name in seen or current.args.get('where') or current.args.get('joins')
                    or current.args.get('having') or current.args.get('qualify')
                    or current.args.get('group')):
                raise ValueError('filtered or branching CTE chain')
            seen.add(name)
            current = definitions[name]
        if seen != set(definitions):
            raise ValueError('unused or branching CTE')
        return current

    def sql_conditions(query, dialect):
        tree = sqlglot.parse_one(query, read=dialect)
        tree = base_population_select(tree)
        if tree.args.get('having') or tree.args.get('qualify'):
            raise ValueError('unsupported query scope')
        joined = single_table(tree) is None
        edges, aliases = join_sources(tree) if joined else ([], set())
        if histogram_column and tree.args.get('where'):
            # Only a top-level AND leaf on the charted value may be omitted.
            # Never simplify OR, nested queries, or another column's null filter.
            def without_chart_null(node):
                if isinstance(node, exp.Paren):
                    child = without_chart_null(node.this)
                    return exp.Paren(this=child) if child is not None else None
                if isinstance(node, exp.And):
                    left, right = without_chart_null(node.this), without_chart_null(node.expression)
                    if left is None: return right
                    if right is None: return left
                    return exp.And(this=left, expression=right)
                if (isinstance(node, exp.Not) and isinstance(node.this, exp.Is)
                        and isinstance(node.this.this, exp.Column)
                        and node.this.this.name == histogram_column
                        and isinstance(node.this.expression, exp.Null)):
                    return None
                return node
            predicate = without_chart_null(tree.args['where'].this)
            tree.set('where', exp.Where(this=predicate) if predicate is not None else None)
        where=tree.args.get('where')
        if joined and where and any(column.table and column.table not in aliases
                                for column in where.find_all(exp.Column)):
            raise ValueError('unbound filter alias')
        return (formula(where.this, joined) if where else [[]], edges)

    try:
        if isinstance(executed, str):
            condition_sets, join_edges=sql_conditions(executed, dialect)
        elif hasattr(executed, 'conditions'):
            stored=list(executed.conditions)
            if executed.query:
                parsed, join_edges=sql_conditions(
                    executed.query, 'duckdb' if executed.parent_id else 'databricks')
                condition_sets=[stored+items for items in parsed]
            elif not executed.predicate_known:
                return False
            else:
                condition_sets=[stored]
                join_edges=[]
        else:
            condition_sets=[list(executed)]
            join_edges=[]

        expected_edges=requested.get('join_edges', [])
        if joined := bool(join_edges):
            if not expected_edges or sorted(join_edges) != sorted(expected_edges):
                return False
        elif expected_edges:
            return False

        def canonical(values):
            result = set()
            for item in values:
                item = asdict(item) if isinstance(item, Condition) else dict(item)
                condition = Condition(**item)
                value = condition.value
                if condition.op == 'in': value = sorted(value, key=repr)
                result.add(json.dumps([condition.column, condition.op, value], sort_keys=True, ensure_ascii=False))
            return result
        common=list(requested.get('conditions', []))
        alternatives=list(requested.get('any_conditions', []))
        wanted=[common+[item] for item in alternatives] if alternatives else [common]
        return ({frozenset(canonical(items)) for items in condition_sets}
                == {frozenset(canonical(items)) for items in wanted})
    except (ValueError, TypeError, KeyError, sqlglot.errors.SqlglotError):
        return False


def measure_scope_matches(query, requested, *, dialect='duckdb'):
    """Verify numerator predicates used inside a ratio calculation.

    Population predicates may also appear in SQL WHERE, so remove their exact
    canonical forms before comparing the remaining simple comparisons. Unknown
    comparison shapes fail closed when a measure contract is present.
    """
    expected = requested.get('measure_conditions', [])
    if not expected: return True
    import sqlglot
    from sqlglot import exp

    def literal(node):
        if isinstance(node, exp.Neg): return -literal(node.this)
        if isinstance(node, exp.Boolean): return node.this
        if not isinstance(node, exp.Literal): raise ValueError('unsupported literal')
        if node.is_string: return node.this
        return float(node.this) if any(c in node.this.lower() for c in ('.', 'e')) else int(node.this)

    try:
        tree = sqlglot.parse_one(query, read=dialect)
        ops = {exp.EQ:'eq', exp.NEQ:'ne', exp.GT:'gt', exp.GTE:'ge', exp.LT:'lt', exp.LTE:'le'}
        found = []
        for node in tree.walk():
            if type(node) in ops:
                if not isinstance(node.this, exp.Column): return False
                found.append({'column':node.this.name, 'op':ops[type(node)],
                              'value':literal(node.expression)})
        def canonical(values):
            return {json.dumps([item['column'], item['op'], item['value']],
                               sort_keys=True, ensure_ascii=False) for item in values}
        population=list(requested.get('conditions', []))+list(requested.get('any_conditions', []))
        remaining = canonical(found) - canonical(population)
        ratio = requested.get('ratio') or {}
        if ratio.get('aggregation') == 'mean_zero_one':
            averages = [node for node in tree.find_all(exp.Avg)
                        if isinstance(node.this, exp.Column) and node.this.name == ratio.get('column')]
            scaled = any(isinstance(node, exp.Mul) and any(
                isinstance(side, exp.Literal) and not side.is_string and float(side.this) == 100.0
                for side in (node.this, node.expression)) for node in tree.find_all(exp.Mul))
            return len(averages) == 1 and scaled and not remaining
        return remaining == canonical(expected)
    except (ValueError, TypeError, KeyError, sqlglot.errors.SqlglotError):
        return False
