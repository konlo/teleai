"""Bounded numeric projections with explicit missing-value policy and lineage."""
from dataclasses import asdict
from decimal import Decimal
import numpy as np
import pandas as pd
from utils.analysis_datasets import project_dataset


def prepare_numeric_dataset(store, dataset_id, columns, missing_values=None, preserve_columns=None):
    info=store.metadata[dataset_id]
    missing_values=[] if missing_values is None else missing_values
    preserve_columns=[] if preserve_columns is None else preserve_columns
    if (not isinstance(columns,list) or not 1<=len(columns)<=8
            or len(set(columns))!=len(columns) or not set(columns)<=set(info.columns)
            or not isinstance(missing_values,list) or len(missing_values)>16
            or any(not isinstance(v,str) or len(v)>64 for v in missing_values)):
        raise ValueError('Use 1–8 known columns and at most 16 explicit missing strings')
    if (not isinstance(preserve_columns,list) or len(preserve_columns)>8
            or len(set(preserve_columns))!=len(preserve_columns)
            or set(preserve_columns)&set(columns) or not set(preserve_columns)<=set(info.columns)):
        raise ValueError('Preserved columns must be known, unique and separate from numeric measures')
    source=project_dataset(store,dataset_id,[*columns,*preserve_columns])
    output=pd.DataFrame(index=source.index);audit=[];projections=[]
    for column in columns:
        series=source[column]
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
            raise ValueError('Boolean and datetime columns are not numeric measures')
        missing=series.isna() | series.isin(missing_values)
        values=pd.to_numeric(series.mask(missing),errors='coerce')
        invalid=~missing & values.isna()
        finite=values.isna() | np.isfinite(values)
        stats={'column':column,'source_dtype':str(series.dtype),'rows':len(series),
               'source_nulls':int(series.isna().sum()),'declared_missing':int((missing & series.notna()).sum()),
               'invalid_values':int(invalid.sum()),'non_finite_values':int((~finite).sum())}
        audit.append(stats)
        if invalid.any() or not finite.all():
            return {'status':'needs_context','error_code':'numeric_conversion_unresolved','retryable':False,
                    'conversion':audit,'missing_values':missing_values,
                    'message':'수치로 변환할 수 없는 값이 있습니다. 원본과 모든 행은 보존했습니다. '
                              '결측 문자열의 의미를 확인하여 missing_values를 명시하세요. 임의 값 삭제나 재조회는 필요하지 않습니다.'}
        # Explicit floating representation for analysis; do not turn large
        # identifiers/integer magnitudes into rounded values silently.
        if any(abs(Decimal(str(value)))>2**53 for value in series[~missing]):
            return {'status':'rejected','error_code':'numeric_precision_risk','retryable':False,
                    'conversion':audit,'message':'Float64의 정확한 정수 범위를 넘습니다. Decimal 분석이 필요합니다.'}
        output[column]=values.astype('float64')
        quoted='"'+column.replace('"','""')+'"'
        expression=quoted
        if missing_values and pd.api.types.is_string_dtype(series):
            literals=','.join("'"+v.replace("'","''")+"'" for v in missing_values)
            expression=f'CASE WHEN CAST({quoted} AS VARCHAR) IN ({literals}) THEN NULL ELSE {quoted} END'
        projections.append(f'CAST({expression} AS DOUBLE) AS {quoted}')
    for column in preserve_columns:
        output[column]=source[column]
        projections.append('"'+column.replace('"','""')+'"')
    query='SELECT '+', '.join(projections)+' FROM data'
    # Reuse the same immutable branch for repeated preparation of one snapshot.
    existing=next((d for d in store.metadata.values() if d.parent_id==dataset_id
                   and d.query==query and d.snapshot==info.snapshot and d.rows==info.rows),None)
    result=existing or store.register(output,source=info.source,coverage=info.coverage,
        conditions=info.conditions,predicate_known=info.predicate_known,grain=info.grain,
        aggregation=info.aggregation,parent_id=dataset_id,query=query,snapshot=info.snapshot)
    return {'status':'ready','dataset':asdict(result),'conversion':audit,'missing_values':missing_values,
            'reused':existing is not None,'scope':'원본과 같은 모든 행을 유지하는 수치 컬럼 projection입니다. '
            '명시한 문자열만 NULL로 변환하며 Float64를 사용합니다. 원본은 변경하지 않습니다.'}
