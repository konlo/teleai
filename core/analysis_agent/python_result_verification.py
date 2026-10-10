"""Independent numeric reference calculations; never execute model verifier code."""
import numpy as np
import pandas as pd
from core.analysis_agent.python_result_contract import validate,digest

def normalize_reference_columns(frame,input_columns,contract):
    """Project fixed outputs; extra input reference columns are presentation only.

    Missing outputs or unexpected derived columns remain verification errors.
    Never alter values, rows, sorting, filters or the agreed computation.
    """
    wanted=contract['output_columns'];extra=[c for c in frame if c not in wanted]
    if set(wanted)<=set(frame.columns) and set(extra)<=set(input_columns):
        return frame[wanted],extra
    return frame,[]

def verification_columns(frame,contract):
    """Bound the controller's independent work before starting the worker."""
    validate(contract,list(frame.columns))
    names=set(contract['output_columns'])
    for item in contract['steps']:
        names.update(item['columns'] if item['op']=='sort' else [item['column']])
    columns=[c for c in frame.columns if c in names]
    # Copies, numeric arrays, masks, sort indices and rolling intermediates.
    estimated=int(frame[columns].memory_usage(index=True,deep=True).sum())*3+len(frame)*160
    if estimated>64*1024*1024:raise ValueError('Independent verification memory budget exceeded')
    for item in contract['steps']:
        if item['op']=='rolling' and item['aggregation'] not in {'mean','sum','count'}:
            if len(frame)*item['window']>2000000:raise ValueError('Independent window verification budget exceeded')
    return columns

def aggregate(values,kind):
    valid=values[~np.isnan(values)]
    if kind=='count':return float(len(valid))
    if kind=='sum':return float(valid.sum())
    if not len(valid) or (kind=='std' and len(valid)<2):return np.nan
    return float({'mean':np.mean,'min':np.min,'max':np.max,'median':np.median,
                  'std':lambda a:np.std(a,ddof=1)}[kind](valid))

def expected_frame(frame,contract):
    columns=verification_columns(frame,contract)
    expected=frame[columns].copy(deep=True)
    for item in contract['steps']:
        op=item['op']
        if op=='sort':
            expected=expected.sort_values(item['columns'],ascending=item['ascending'],kind='stable').reset_index(drop=True)
            continue
        column=item['column']
        if not pd.api.types.is_numeric_dtype(expected[column]) or pd.api.types.is_bool_dtype(expected[column]):
            raise ValueError('Numeric result contract requires an observed numeric input')
        values=expected[column].to_numpy(dtype=float,na_value=np.nan)
        if np.isinf(values).any():raise ValueError('Nonfinite input needs explicit preparation before numeric verification')
        if op=='aggregate':
            expected=pd.DataFrame({item['output_column']:[aggregate(values,item['aggregation'])]})
        elif op=='diff':
            offset=item['periods'];out=np.full(len(values),np.nan)
            if offset<len(values):out[offset:]=values[offset:]-values[:-offset]
            expected[item['output_column']]=out
        else:
            # Linear mean/sum/count reference avoids a second expensive rolling
            # computation for large retained inputs. Other windows are bounded.
            window=item['window'];minimum=item['min_periods'];kind=item['aggregation']
            starts=np.maximum(np.arange(len(values))+1-window,0);ends=np.arange(len(values))+1
            valid=~np.isnan(values);counts=np.r_[0,np.cumsum(valid)][ends]-np.r_[0,np.cumsum(valid)][starts]
            if kind in {'mean','sum','count'}:
                sums=np.r_[np.longdouble(0),np.cumsum(np.where(valid,values,0.),dtype=np.longdouble)]
                total=sums[ends]-sums[starts]
                out=counts.astype(float) if kind=='count' else total if kind=='sum' else np.divide(total,counts,out=np.full(len(values),np.nan),where=counts>0)
                # pandas rolling count min_periods counts window observations,
                # whereas other aggregations count valid numeric observations.
                admitted=(ends-starts>=minimum) if kind=='count' else (counts>=minimum)
                out[~admitted]=np.nan
            else:
                if len(values)*window>2000000:raise ValueError('Independent window verification budget exceeded')
                out=np.array([aggregate(values[start:end],kind) if count>=minimum else np.nan
                              for start,end,count in zip(starts,ends,counts)])
            expected[item['output_column']]=out
    return expected[contract['output_columns']].reset_index(drop=True)

def verify(frame,result,contract):
    expected=expected_frame(frame,contract);issues=[]
    if list(result.columns)!=contract['output_columns']:issues.append('output_columns_mismatch')
    if len(result)!=len(expected):issues.append('output_row_count_mismatch')
    if not issues:
        for column in expected:
            actual=result[column]
            if pd.api.types.is_numeric_dtype(expected[column]):
                good=pd.api.types.is_numeric_dtype(actual) and np.allclose(
                    actual.to_numpy(dtype=float,na_value=np.nan),expected[column].to_numpy(dtype=float,na_value=np.nan),
                    rtol=1e-7,atol=1e-9,equal_nan=True)
            else:good=actual.reset_index(drop=True).equals(expected[column])
            if not good:issues.append('output_values_or_order_mismatch:'+column)
    return {'semantic_verified':not issues,'result_contract_hash':digest(contract),
            'verification_method':'independent_declarative_numeric_reference','issues':issues}
