"""Exact bounded grouped reductions over retained Arrow batches, without reloads."""
from itertools import chain
from tempfile import TemporaryDirectory
from threading import Timer

import duckdb
import pyarrow as pa

from utils.analysis_datasets import Condition, filter_frame

QUERY_TIMEOUT_SECONDS=30
MAX_BATCH_BYTES=8*1024*1024


def reduce_groups(store,info,columns,groups,metrics,conditions,max_groups):
    from utils.analysis_group_summary import _validated_metrics
    batches=iter(store.frames.batches(info.id,columns,expected_rows=info.rows,batch_size=1024))
    counters={'source_rows':0,'filtered_rows':0,'group_input_rows':0}
    try:
        first=next(batches,None)
        if first is None:raise ValueError('그룹 요약할 행이 없습니다.')
        if first.nbytes>MAX_BATCH_BYTES:raise MemoryError('그룹 요약 batch 크기 한도 초과')
        if any(pa.types.is_nested(field.type) for field in first.schema):
            raise ValueError('그룹 요약 컬럼은 단일 값 자료형이어야 합니다.')
        normalized=_validated_metrics(first.to_pandas(),list(metrics))
        if set(groups)&{metric['name'] for metric in normalized}:
            raise ValueError('지표 이름은 그룹 컬럼명과 달라야 합니다.')
        if any(condition.column not in columns for condition in conditions):
            raise ValueError('전체 필터 컬럼은 현재 dataset의 실제 컬럼이어야 합니다.')
        # Internal identifiers are generated here; no user values enter SQL text.
        group_aliases=['g'+str(i) for i in range(len(groups))]
        fields=[pa.field(alias,first.schema.field(column).type)
                for alias,column in zip(group_aliases,groups)]
        fields.append(pa.field('ordinal',pa.int64()))
        expressions=[];metric_counts={}
        for index,metric in enumerate(normalized):
            kind=metric['aggregation'];column=metric['value_column'];alias='m'+str(index)
            if column:fields.append(pa.field('v'+str(index),first.schema.field(column).type))
            if metric['condition']:
                fields.append(pa.field('p'+str(index),pa.bool_()))
                metric_counts[metric['name']]={'denominator_rows':0,'selected_rows':0}
                if kind=='conditional_count':expr=f'COUNT(*) FILTER (WHERE p{index})'
                elif kind=='conditional_percent':expr=f'100.0 * COUNT(*) FILTER (WHERE p{index}) / COUNT(*)'
                else:expr=f'AVG(v{index}) FILTER (WHERE p{index})'
            else:
                metric_counts[metric['name']]={'input_rows':0,'non_null_rows':0}
                function={'mean':'AVG','median':'MEDIAN','sum':'SUM','min':'MIN','max':'MAX','count':'COUNT'}[kind]
                expr='COUNT(*)' if kind=='count' else f'{function}(v{index})'
                if kind=='sum':
                    expr=f'COALESCE({expr},0)'
                    dtype=first.schema.field(column).type
                    # DuckDB HUGEINT -> pandas float conversion can round exact
                    # integer sums. Preserve integers or fail on overflow.
                    if pa.types.is_integer(dtype):
                        target='UBIGINT' if pa.types.is_unsigned_integer(dtype) else 'BIGINT'
                        expr=f'CAST({expr} AS {target})'
            expressions.append(expr+' AS '+alias)
        schema=pa.schema(fields)

        def prepared():
            for batch in chain([first],batches):
                if batch.nbytes>MAX_BATCH_BYTES:raise MemoryError('그룹 요약 batch 크기 한도 초과')
                frame=batch.to_pandas().reset_index(drop=True)
                offset=counters['source_rows'];counters['source_rows']+=len(frame)
                if counters['source_rows']>info.rows:raise ValueError('원본 행 수 불일치')
                filtered=filter_frame(frame,conditions) if conditions else frame
                counters['filtered_rows']+=len(filtered)
                grouped=filtered.dropna(subset=groups)
                counters['group_input_rows']+=len(grouped)
                arrays=[pa.array(grouped[column],type=schema.field(alias).type,from_pandas=True)
                        for alias,column in zip(group_aliases,groups)]
                arrays.append(pa.array(grouped.index.to_numpy(dtype='int64')+offset))
                for index,metric in enumerate(normalized):
                    count=metric_counts[metric['name']]
                    if metric['value_column']:
                        arrays.append(pa.array(grouped[metric['value_column']],
                                               type=schema.field('v'+str(index)).type,from_pandas=True))
                    if metric['condition']:
                        selected=filter_frame(grouped,(Condition(**metric['condition']),))
                        arrays.append(pa.array(grouped.index.isin(selected.index)))
                        count['denominator_rows']+=len(grouped);count['selected_rows']+=len(selected)
                    else:
                        count['input_rows']+=len(grouped)
                        count['non_null_rows']+=int(grouped[metric['value_column']].notna().sum()) if metric['value_column'] else len(grouped)
                yield pa.RecordBatch.from_arrays(arrays,schema=schema)

        with TemporaryDirectory(prefix='telly-group-summary-') as temporary:
            with duckdb.connect(config={'enable_external_access':False,'threads':2,
                    'memory_limit':'128MB','temp_directory':temporary,
                    'max_temp_directory_size':'512MB','preserve_insertion_order':False}) as conn:
                timer=Timer(QUERY_TIMEOUT_SECONDS,conn.interrupt);timer.daemon=True;timer.start()
                reader=pa.RecordBatchReader.from_batches(schema,prepared())
                try:
                    conn.register('input_batches',reader)
                    conn.execute('CREATE TEMP TABLE data AS SELECT * FROM input_batches')
                    if counters['source_rows']!=info.rows:raise ValueError('원본 행 수 불일치')
                    if not counters['group_input_rows']:raise ValueError('조건 적용 후 그룹 요약할 행이 없습니다.')
                    keys=', '.join(group_aliases)
                    result=conn.execute('SELECT '+keys+', '+', '.join(expressions)
                        +' FROM data GROUP BY '+keys+' ORDER BY MIN(ordinal) LIMIT '+str(int(max_groups)+1)).fetchdf()
                    if len(result)>max_groups:raise ValueError('그룹 수가 허용 범위를 벗어났습니다.')
                except duckdb.InterruptException as exc:
                    raise TimeoutError('그룹 요약 시간 한도 초과') from exc
                finally:
                    timer.cancel();timer.join();reader.close()
        result.columns=[*groups,*[metric['name'] for metric in normalized]]
        for metric in normalized:
            if metric['aggregation']=='conditional_mean' and metric['empty_value'] is not None:
                result[metric['name']]=result[metric['name']].fillna(metric['empty_value'])
        return result,normalized,metric_counts,counters
    finally:
        batches.close()
