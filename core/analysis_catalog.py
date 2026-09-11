"""Compact discovery metadata; detailed profiles remain available on demand."""
def compact_catalog(catalog):
    return {**catalog, 'available_tables': [
        {'table': item.get('table'), 'training_status': item.get('training_status'),
         'column_count': len(item.get('columns', []))}
        for item in catalog.get('available_tables', [])]}
