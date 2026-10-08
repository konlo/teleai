"""Bind row-level chart intent to verified conversation evidence and axis roles."""
import re
from core.analysis_agent.row_preview import frame, key


def axis_roles(text, columns):
    """Resolve explicit roles against runtime column identity, never schema order."""
    roles = {}
    for axis in ('x', 'y'):
        matches = []
        role = rf'{axis}\s*(?:축|axis)?(?![A-Za-z0-9_])'
        for column in columns:
            name = r'(?<![A-Za-z0-9_])[`"]?' + re.escape(column) + r'[`"]?(?![A-Za-z0-9_])'
            if (re.search(name + r'\s*(?:을|를|은|는)?\s*(?:as\s+)?' + role, text, re.I)
                    or re.search(role + r'\s*(?:은|는|에|=|:|is)?\s*' + name, text, re.I)):
                matches.append(column)
        if len(matches) == 1:
            roles[axis] = matches[0]
    return roles if len(roles) == 2 and roles['x'] != roles['y'] else {}


def bind(context, current):
    text = current.get('text', '')
    relationship = not current.get('kind') and bool(re.search(
        r'관계.{0,30}그려|(?:draw|plot).{0,30}relationship', text, re.I))
    if not context or (current.get('kind') != 'scatter' and not relationship):
        return {}
    columns = list(dict.fromkeys(c for info in context.datasets.metadata.values() for c in info.columns))
    axes = axis_roles(current.get('text', ''), columns)
    # The request text is supplied explicitly by the recovery hook.
    patch = {'chart_axes': axes}
    original_current = current.get('display_explicit_current', current.get('current_result_only', False))
    original_rows = current.get('display_explicit_rows', current.get('requested_result_rows'))
    if current.get('display_dataset_id'):
        patch.update(display_dataset_id=None, chart_display_evidence=None,
                     current_result_only=original_current, requested_result_rows=original_rows)
    previous = current.get('confirmed_analysis') or {}
    proof = previous.get('table_preview_evidence') or previous.get('chart_display_evidence')
    scope = current.get('scope') or {}
    if (not proof or current.get('fresh_source_required') or current.get('plan')
            or any(scope.get(k) for k in ('conditions', 'any_conditions', 'unresolved', 'ratio', 'measure_conditions'))
            or re.search(r'전체|모든|모집단|\b(?:all|whole|entire|population)\b', current.get('text',''), re.I)
            or previous.get('selection_at_confirmation') != context.selected_dataset_id):
        return patch
    sources = current.get('required_sources', []) or (scope.get('sources', []) if relationship else [])
    if set(map(key, sources)) != {key(proof.get('source'))}:
        return patch
    requested = set(axes.values()) if axes else set(current.get('required_columns', []))
    if len(requested) != 2 or not requested.issubset(proof.get('columns', [])):
        return patch
    try:
        info = context.datasets.metadata[proof['dataset_id']]
        # A prefix of a larger asset needs its own bounded transformation first;
        # do not silently widen the displayed population to the whole asset.
        if info.grain != 'raw' or proof['rows'] != info.rows:
            return patch
        displayed = frame(context.datasets, proof)
        if relationship:
            from pandas.api.types import is_numeric_dtype
            if current.get('operations') or not all(is_numeric_dtype(displayed[c]) for c in requested):
                return patch
            ordered = sorted(requested, key=lambda c: text.casefold().find(c.casefold()))
            patch.update(kind='scatter', chart=True, calculation=False, chart_spec_requested=True,
                         required_sources=[proof['source']], chart_axes={'x':ordered[0], 'y':ordered[1]})
            axes = patch['chart_axes']
    except (KeyError, ValueError, TypeError):
        return patch
    return {**patch, 'required_columns': list(axes.values()) if axes else current['required_columns'],
            'current_result_only': True, 'requested_result_rows': info.rows,
            'display_dataset_id': info.id, 'chart_display_evidence': dict(proof),
            'display_explicit_current': original_current, 'display_explicit_rows': original_rows}


def axes_match(current, spec):
    return all(spec.get(axis) == column for axis, column in current.get('chart_axes', {}).items())


def raw_scatter_eligible(info, current):
    if info is None or info.grain != 'raw':
        return False
    anchor = current.get('display_dataset_id')
    if anchor and info.id != anchor:
        return False
    scoped = any((current.get('scope') or {}).get(k) for k in ('conditions', 'any_conditions'))
    return bool(current.get('current_result_only') or
                info.coverage == 'complete' and info.predicate_known
                and (scoped or not info.conditions))
