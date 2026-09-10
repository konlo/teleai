"""Environment-based connection, no Streamlit/legacy imports and no auto query."""
from dataclasses import dataclass
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
    access_token: str
    catalog: str
    schema: str

    @classmethod
    def from_env(cls):
        return cls(os.getenv('DATABRICKS_HOST','').removeprefix('https://').rstrip('/'),
                   os.getenv('DATABRICKS_HTTP_PATH',''),os.getenv('DATABRICKS_TOKEN',os.getenv('DATABRICKS_ACCESS_TOKEN','')),
                   os.getenv('DATABRICKS_CATALOG',''),os.getenv('DATABRICKS_SCHEMA',''))

    def identity(self):
        # Bind credentials too without writing them to checkpoints or the ledger.
        return sha256(json.dumps([self.server_hostname,self.http_path,self.catalog,self.schema,
                                  sha256(self.access_token.encode()).hexdigest()]).encode()).hexdigest()


def make_executor(config,datasets):
    def execute(envelope):
        if not config.server_hostname or not config.http_path or not config.access_token:
            raise QueryNotSubmitted()
        if envelope['connection']!=config.identity():raise PermissionError('연결 설정 변경: 재승인이 필요합니다.')
        request=SimpleNamespace(status='executing',query=envelope['query'],source=envelope['source'])
        try:
            return execute_approved(request,config,datasets)
        except Exception as exc:
            context=getattr(exc,'context',{})
            if context.get('method')=='OpenSession':
                raise QueryNotSubmitted(context.get('http-code')) from exc
            raise
    return execute
