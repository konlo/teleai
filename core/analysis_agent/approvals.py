"""Durable one-submission grants. Checkpoint replay cannot grant DB execution."""
from hashlib import sha256
import json
import sqlite3
from core.analysis_load_plan import source_plan


class QueryNotSubmitted(RuntimeError):
    """Backend proved failure occurred before SQL submission."""
    def __init__(self,http_status=None):
        super().__init__('Databricks 세션을 열지 못했습니다. 조회는 제출되지 않았습니다.')
        self.http_status=http_status


class ApprovalLedger:
    def __init__(self,path):
        self.path=path
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS requests (id TEXT PRIMARY KEY, fingerprint TEXT, envelope TEXT, status TEXT, result TEXT)')

    def connect(self):return sqlite3.connect(self.path,timeout=30)

    @staticmethod
    def envelope(source,query,reason,connection):
        source_plan(source, query)
        return dict(source=source,query=query,reason=reason,connection=connection)

    @staticmethod
    def fingerprint(envelope):
        return sha256(json.dumps(envelope,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

    def propose(self,key,envelope):
        fingerprint=self.fingerprint(envelope)
        with self.connect() as db:
            db.execute('INSERT OR IGNORE INTO requests VALUES (?,?,?,?,NULL)',
                       (key,fingerprint,json.dumps(envelope,ensure_ascii=False),'proposed'))
            row=db.execute('SELECT fingerprint FROM requests WHERE id=?',(key,)).fetchone()
            if row[0]!=fingerprint:raise PermissionError('조회 내용 또는 연결이 변경되었습니다. 재승인이 필요합니다.')
        return self.get(key)

    def get(self,key):
        with self.connect() as db:
            row=db.execute('SELECT envelope,status,result FROM requests WHERE id=?',(key,)).fetchone()
        if row is None:raise KeyError('승인 요청이 없습니다.')
        return dict(id=key,**json.loads(row[0]),status=row[1],result=json.loads(row[2]) if row[2] else None)

    def decide(self,key,approved):
        with self.connect() as db:
            changed=db.execute('UPDATE requests SET status=? WHERE id=? AND status=?',
                ('approved' if approved else 'rejected',key,'proposed')).rowcount
            if not changed:raise PermissionError('이미 처리된 승인입니다.')

    def uncertain(self):
        with self.connect() as db:
            rows=db.execute("SELECT id,status FROM requests WHERE status IN ('submitting','unknown')").fetchall()
        return [{'id':key,'status':status} for key,status in rows]

    def invalidate(self,key):
        with self.connect() as db:
            db.execute("UPDATE requests SET status='invalidated' WHERE id=? AND status IN ('proposed','approved')",(key,))

    def execute(self,key,envelope,executor):
        fingerprint=self.fingerprint(envelope)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT fingerprint,status,result FROM requests WHERE id=?',(key,)).fetchone()
            if row is None or row[0]!=fingerprint:raise PermissionError('정확한 조회의 승인 기록이 필요합니다.')
            if row[1]=='completed':return json.loads(row[2])
            if row[1]!='approved':raise PermissionError('미승인 또는 제출 상태 불명인 조회는 실행할 수 없습니다.')
            db.execute("UPDATE requests SET status='submitting' WHERE id=?",(key,))
        try:
            result=executor(envelope)
            encoded=json.dumps(result,ensure_ascii=False,default=str)
        except BaseException as exc:
            with self.connect() as db:
                db.execute('UPDATE requests SET status=? WHERE id=?',
                           ('failed' if isinstance(exc,QueryNotSubmitted) else 'unknown',key))
            raise
        with self.connect() as db:
            db.execute("UPDATE requests SET status='completed',result=? WHERE id=?",(encoded,key))
        return result
