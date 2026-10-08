"""Environment-based connection, no Streamlit/legacy imports and no auto query."""
from dataclasses import dataclass, field
from hashlib import sha256
import json
import os
from types import SimpleNamespace
from core.analysis_databricks import execute_approved
from core.analysis_agent.approvals import QueryNotSubmitted


@dataclass(frozen=True)
class ConnectionConfig:
    server_hostname: str
    http_path: str
    access_token: str = field(repr=False)
    catalog: str
    schema: str

    @classmethod
    def from_env(cls):
        return cls(os.getenv('DATABRICKS_HOST','').removeprefix('https://').rstrip('/'),
                   os.getenv('DATABRICKS_HTTP_PATH',''),os.getenv('DATABRICKS_TOKEN') or os.getenv('DATABRICKS_ACCESS_TOKEN',''),
                   os.getenv('DATABRICKS_CATALOG',''),os.getenv('DATABRICKS_SCHEMA',''))

    def validate(self):
        if not all((self.server_hostname, self.http_path, self.access_token)):
            raise ValueError('Databricks 설정 DATABRICKS_HOST, DATABRICKS_HTTP_PATH, DATABRICKS_TOKEN(또는 DATABRICKS_ACCESS_TOKEN)을 확인해주세요.')

    def identity(self):
        # Bind credentials too without writing them to checkpoints or the ledger.
        return sha256(json.dumps([self.server_hostname,self.http_path,self.catalog,self.schema,
                                  sha256(self.access_token.encode()).hexdigest()]).encode()).hexdigest()


def make_executor(config,datasets,*,max_rows=100_000,max_coordinate_rows=None):
    def execute(envelope):
        if not config.server_hostname or not config.http_path or not config.access_token:
            raise QueryNotSubmitted()
        if envelope['connection']!=config.identity():raise PermissionError('연결 설정이 변경되어 기존 조회를 실행할 수 없습니다.')
        request=SimpleNamespace(status='executing',query=envelope['query'],source=envelope['source'])
        try:
            from core.analysis_agent.source_scatter import fetch_limit
            row_limit=fetch_limit(envelope['query'],'databricks',max_rows,max_coordinate_rows)
            return execute_approved(request,config,datasets,max_rows=row_limit)
        except Exception as exc:
            context=getattr(exc,'context',{})
            if context.get('method')=='OpenSession':
                raise QueryNotSubmitted(context.get('http-code')) from exc
            raise
    return execute
