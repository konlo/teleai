"""Reject invented local handles without changing the requested population."""

def unknown_dataset_error(call, context):
    if context is None:
        return None
    arguments = call.get('args') or {}
    invalid = {key: value for key, value in arguments.items()
               if (key == 'dataset_id' or key.endswith('_dataset_id'))
               and isinstance(value, str) and value
               and value not in context.datasets.metadata}
    if not invalid:
        return None
    candidates = [{'dataset_id': info.id, 'source': info.source,
                   'columns': list(info.columns)[:12], 'grain': info.grain,
                   'coverage': info.coverage}
                  for info in context.datasets.metadata.values()][-8:]
    return {'error_code': 'unknown_dataset_id', 'invalid_handles': invalid,
            'available_datasets': candidates,
            'message': 'The proposed dataset handle does not exist. Use an exact registered ID with the required source, columns and scope; do not invent an ID or change the requested result.'}
