"""Standalone worker; reads only controller-owned temporary input/output files."""
import json
import sys
import os

def main():
    root=sys.argv[1]
    request=json.loads(open(os.path.join(root,'request.json'),encoding='utf-8').read())
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU,(8,8))
        resource.setrlimit(resource.RLIMIT_FSIZE,(16*1024*1024,16*1024*1024))
        # macOS RLIMIT_AS is not a reliable RSS limit. Report this distinction.
        if sys.platform.startswith('linux'):
            resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
    except ImportError:
        pass
    import pandas as pd
    import numpy as np
    df=pd.read_parquet(os.path.join(root,'input.parquet'))
    namespace={'df':df.copy(deep=True),'pd':pd,'np':np,'__builtins__':
               {'len':len,'abs':abs,'min':min,'max':max,'sum':sum,'round':round}}
    try:
        exec(compile(request['code'],'<analysis>','exec'),namespace,namespace)
        result=namespace.get('result')
        if not isinstance(result,pd.DataFrame):raise ValueError('result must be a DataFrame')
        if len(result)>100000 or len(result.columns)>32 or result.columns.has_duplicates:
            raise ValueError('Output row/column limit exceeded')
        if any(not isinstance(c,str) or not c for c in result.columns):raise ValueError('Output needs unique string columns')
        if result.memory_usage(index=True,deep=True).sum()>16*1024*1024:raise ValueError('Output byte limit exceeded')
        result.to_parquet(os.path.join(root,'output.parquet'),index=False)
        response={'status':'ready','rows':len(result),'columns':list(result.columns)}
    except Exception as exc:
        # Do not echo arbitrary data, source code or secrets in errors.
        response={'status':'error','error_code':'python_execution_failed','error_type':type(exc).__name__,
                  'message':'Review pandas types and the declared result DataFrame contract.'}
    open(os.path.join(root,'response.json'),'w',encoding='utf-8').write(json.dumps(response))

if __name__=='__main__':main()
