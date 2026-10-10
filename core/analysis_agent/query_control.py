"""Runtime-owned active query cancellation; never infer provider state after restart."""
from threading import RLock,Timer
import time

class QueryResultDiscarded(RuntimeError):
    """Cancel/deadline prevents publication; provider termination is unconfirmed."""
    error_code='query_result_discarded'

class QueryControl:
    def __init__(self):
        self.lock=RLock();self.key='';self.deadline=None;self.callback=None;self.timer=None;self.state='idle'
    def bind(self,key,deadline=None):
        with self.lock:
            if self.callback is not None:raise RuntimeError('A query is already active')
            self.key=key;self.deadline=deadline;self.state='connecting'
    def start(self,callback):
        with self.lock:
            self.callback=callback;self.state='running'
            if self.deadline is not None and callback is not None:
                key=self.key
                self.timer=Timer(max(.001,self.deadline-time.time()),lambda:self.cancel(key))
                self.timer.daemon=True;self.timer.start()
    def finish(self):
        with self.lock:
            if self.timer:self.timer.cancel()
            self.timer=None;self.callback=None
            if self.state=='running':self.state='returned'
    def inspect(self,key):
        with self.lock:
            if key!=self.key:return {'status':'unavailable','error_code':'provider_operation_not_attached','retryable':False}
            return {'status':'ready','operation_id':key,'driver_state':self.state,
                    'cancel_available':self.callback is not None,'provider_polling_available':False}
    def validate_result(self):
        with self.lock:
            if self.state in {'cancel_requested','cancel_failed'} or (self.deadline is not None and time.time()>=self.deadline):
                raise QueryResultDiscarded('Cancelled/expired query output cannot be published')
    def cancel(self,key):
        with self.lock:
            if key!=self.key or self.callback is None:
                return {'status':'unavailable','error_code':'query_not_active','retryable':False}
            if self.state=='cancel_requested':
                return {'status':'ready','operation_id':key,'execution_state':'cancel_requested'}
            callback=self.callback;self.state='cancel_requested'
            try:callback()
            except Exception as exc:
                self.state='cancel_failed'
                return {'status':'unavailable','error_code':'cancel_failed','error_type':type(exc).__name__,'retryable':False}
            return {'status':'ready','operation_id':key,'execution_state':'cancel_requested',
                    'message':'취소 요청이 전달됐습니다. 실제 종료/결과 여부는 조회 실행 장부에서 별도로 확인해야 합니다.'}

class TrackedConnection:
    def __init__(self,connection,control):self.connection=connection;self.control=control
    def __enter__(self):
        self.connection=self.connection.__enter__()
        return self
    def __exit__(self,*args):
        try:return self.connection.__exit__(*args)
        finally:self.control.finish()
    def cursor(self):
        return TrackedCursor(self.connection.cursor(),self.control)
    def __getattr__(self,name):return getattr(self.connection,name)

class TrackedCursor:
    def __init__(self,cursor,control):self.cursor=cursor;self.control=control
    def __enter__(self):
        self.cursor=self.cursor.__enter__()
        self.control.start(getattr(self.cursor,'cancel',None))
        return self
    def __exit__(self,*args):return self.cursor.__exit__(*args)
    def execute(self,*args,**kwargs):
        result=self.cursor.execute(*args,**kwargs)
        self.control.validate_result()
        return result
    def fetchmany(self,*args,**kwargs):
        self.control.validate_result()
        rows=self.cursor.fetchmany(*args,**kwargs)
        self.control.validate_result()
        return rows
    def __getattr__(self,name):return getattr(self.cursor,name)
