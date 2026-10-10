"""Report chart input disagreements as presentation errors, not SQL filters."""
from core.analysis_agent.chart_presentation import DEFAULTS, matches, options


def input_error(call, current):
    if call.get('name') not in {'prepare_histogram','render_histogram','render_chart_spec'}:
        return None
    current=current or {}
    if current.get('chart_group_spec') or not options(current):return None
    arguments=call.get('args') or {}
    if matches(arguments,current,evidence=False):return None
    requested=options(current)
    mismatches={key:{'expected':requested.get(key,default),'provided':arguments.get(key,default)}
                for key,default in DEFAULTS.items()
                if arguments.get(key,default)!=requested.get(key,default)}
    return {'error_code':'chart_presentation_mismatch','failure_stage':'chart_input_validation',
            'tool':call['name'],'mismatched_fields':mismatches,
            'message':'Correct the mismatched chart arguments explicitly. Preserve the original dataset, source, axes and population; do not rewrite SQL or recalculate completed statistics.'}
