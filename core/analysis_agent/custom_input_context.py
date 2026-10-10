"""Ground Python inputs in schema/originals and the referenced visible result.

Historical derived columns from another result are not database columns and
must not become input candidates merely because they share a source name.
"""
def names(context,data,wanted):
    columns=[c['name'] for table in context.reference_context
             if table.get('table') in wanted for c in table.get('columns',[])]
    columns.extend(c for info in context.datasets.metadata.values()
                   if info.source in wanted and not info.parent_id for c in info.columns)
    if not data.get('literal_table_subjects'):
        from core.analysis_agent.analysis_extensions import retained_input_id
        dataset_id=retained_input_id(context,{'confirmed_analysis':data.get('verified_previous') or {}})
        info=context.datasets.metadata.get(dataset_id)
        if info and info.source in wanted:columns.extend(info.columns)
    return list(dict.fromkeys(columns))
