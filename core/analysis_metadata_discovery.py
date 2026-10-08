"""Plan approved column-comment discovery and reuse only its exact stored result."""
from core.analysis_catalog import _source_key, _quoted_table, table_context_freshness
from utils.analysis_datasets import project_dataset, full_read_preflight

FIELDS=('table_catalog','table_schema','table_name','column_name','data_type','comment')


def column_metadata_plan(context, table):
    wanted=_source_key(table)
    known={_source_key(c.get('table','')) for c in context.reference_context}
    known.update(_source_key(c.source) for c in context.datasets.metadata.values())
    if wanted not in known:
        return {'status':'needs_context','message':'현재 문맥에서 확인된 전체 테이블 이름이 필요합니다.'}
    return make_plan(wanted, dialect=context.sql_dialect)


def make_plan(table, *, dialect='databricks'):
    parts=table.split('.')
    if dialect == 'mysql':
        if len(parts)!=2 or any(not p or any(ch in p for ch in "`\"'\n;") for p in parts):
            return {'status':'needs_context','message':'database.table 전체 이름이 필요합니다.'}
        database,name=parts
        source='information_schema.columns'
        # Stable result fields across SQL engines; MySQL has no catalog level.
        query=("SELECT '' AS table_catalog, table_schema AS table_schema, "
            "table_name AS table_name, column_name AS column_name, "
            "data_type AS data_type, column_comment AS comment FROM " + _quoted_table(source)
            + " WHERE table_schema = '"+database+"' AND table_name = '"+name
            + "' ORDER BY ordinal_position LIMIT 65")
        return {'status':'planned','metadata_plan':{'source':source,'query':query,
            'reason':'분석 대상 테이블의 컬럼 자료형과 업무 설명(comment)을 확인합니다.'},
            'target_table':table,'scope':'MySQL 컬럼 메타데이터만 최대 65건. 아직 조회하지 않았습니다.',
            'user_action':'metadata_plan을 query_databricks에 전달하여 현재 실행 정책에 따라 조회하세요.'}
    if len(parts)!=3 or any(not p or any(ch in p for ch in "`\"'\n;") for p in parts):
        return {'status':'needs_context','message':'catalog.schema.table 전체 이름이 필요합니다.'}
    catalog,schema,name=parts
    source=catalog+'.information_schema.columns'
    query=('SELECT '+', '.join(FIELDS)+' FROM '+_quoted_table(source)
        + " WHERE table_catalog = '"+catalog+"' AND table_schema = '"+schema
        + "' AND table_name = '"+name+"' ORDER BY ordinal_position LIMIT 65")
    return {'status':'planned','metadata_plan':{'source':source,'query':query,
        'reason':'분석 대상 테이블의 컬럼 자료형과 업무 설명(comment)을 확인합니다.'},
        'target_table':table,'scope':'Unity Catalog 컬럼 메타데이터만 최대 65건. 아직 조회하지 않았습니다.',
        'user_action':'metadata_plan을 query_databricks에 전달하여 현재 실행 정책에 따라 조회하세요.'}


def stored_column_definitions(datasets, table, *, dialect='databricks'):
    plan=make_plan(_source_key(table), dialect=dialect)
    if plan['status']!='planned':return None
    query=plan['metadata_plan']['query'];source=plan['metadata_plan']['source']
    candidates=[d for d in datasets.metadata.values() if _source_key(d.source)==source
                and d.query==query]
    if not candidates:return None
    info=candidates[-1]
    # A full page may have additional columns and must not be treated as exhaustive.
    if (not 0<info.rows<65 or set(info.columns)!=set(FIELDS)
            or table_context_freshness({'observed_at':info.snapshot})!='fresh'
            or full_read_preflight(datasets,[info.id])):return None
    frame=project_dataset(datasets,info.id,list(FIELDS))
    columns=[];seen=set();parts=_source_key(table).split('.')
    if dialect == 'mysql':parts=['', *parts]
    for row in frame.to_dict('records'):
        if [str(row[k]).casefold() for k in FIELDS[:3]]!=parts:return None
        name,dtype,comment=row['column_name'],row['data_type'],row['comment']
        if not isinstance(name,str) or not name or name in seen or not isinstance(dtype,str) or not dtype:return None
        seen.add(name)
        if not isinstance(comment,str):comment=''
        if len(comment)>800:return None
        columns.append({'name':name,'dtype':dtype,'description':comment})
    return {'table':_source_key(table),'columns':columns,'observed_at':info.snapshot,
            'training_status':'approved_column_metadata','definition_dataset_id':info.id}


def inspect_column_definitions(context, table):
    plan=column_metadata_plan(context,table)
    if plan['status']!='planned':return plan
    found=stored_column_definitions(context.datasets,table, dialect=context.sql_dialect)
    if found:return {'status':'ready','table_context':found,
        'scope':'승인 후 저장된 컬럼 메타데이터입니다. 원본 데이터나 계산 결과를 대체하지 않습니다.'}
    return plan


def compatible_storage_type(sql_type, pandas_type):
    accepted={'tinyint':{'int8'},'smallint':{'int16'},'int':{'int32','int64'},
        'integer':{'int32','int64'},'bigint':{'int64'},'float':{'float32'},
        'double':{'float64'},'boolean':{'bool'},'string':{'object','string'},
        'varchar':{'object','string'},'date':{'object','datetime64[ns]'},
        'timestamp':{'datetime64[ns]','datetime64[us]'}}
    return str(pandas_type).casefold() in accepted.get(str(sql_type).casefold(),set())
