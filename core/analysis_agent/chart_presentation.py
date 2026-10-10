"""Enforce typed presentation requests on tools and their actual image cards."""
DEFAULTS={'legend':False,'stacked':False,'palette':'default'}


def options(current):
    return current.get('chart_presentation_spec') or {}


def matches(spec,current, *, evidence=True):
    requested=options(current)
    if not requested or current.get('chart_group_spec'):return True
    if any(spec.get(k,default)!=requested.get(k,default) for k,default in DEFAULTS.items()):return False
    return not evidence or not requested.get('legend') or bool(spec.get('legend_labels'))
