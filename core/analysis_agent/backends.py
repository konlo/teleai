"""Select one data adapter without importing or falling back to the other."""
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class DataBackend:
    name: str
    config: object = field(repr=False)
    executor_factory: Callable = field(repr=False)
    context_loader: Callable = field(repr=False)
    preflight: Callable = field(repr=False)

    @property
    def dialect(self):
        return self.name

    @property
    def namespace(self):
        return self.config.database if self.name == 'mysql' else self.config.catalog

    @property
    def label(self):
        return '로컬 MySQL 평가 DB' if self.name == 'mysql' else 'Databricks'

    def storage_root(self, base):
        # Retain existing Databricks data/checkpoints. MySQL never uses this root.
        return (Path(base) if self.name == 'databricks' else
                Path(base)/'mysql_eval'/self.config.identity()[:24])

    def qualify_table(self, value):
        value = value.strip()
        return self.config.database+'.'+value if self.name=='mysql' and '.' not in value else value


def load_data_backend(project_root, *, name=None):
    selected = (name if name is not None else os.getenv('TELLY_DATA_BACKEND', 'databricks')).strip().lower()
    root = Path(project_root)
    if selected == 'databricks':
        from core.analysis_agent.databricks import ConnectionConfig, make_executor
        from core.analysis_catalog import load_saved_reference_context
        config = ConnectionConfig.from_env()
        config.validate()
        return DataBackend(selected, config, make_executor,
            lambda:load_saved_reference_context(root/'.telly_table_context'), lambda:None)
    if selected == 'mysql':
        try:
            from core.analysis_agent.mysql import MySQLConfig, make_executor, reference_context
        except ModuleNotFoundError as error:
            if error.name and (error.name=='mysql' or error.name.startswith('mysql.')):
                raise ValueError('MySQL 평가 모드에는 requirements-mysql-eval.txt 설치가 필요합니다.') from error
            raise
        config = MySQLConfig.from_env(root)
        config.identity()
        def preflight():
            with config.connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute('SELECT 1')
                    if cursor.fetchone() != (1,):
                        raise ValueError('MySQL 연결 확인 결과가 유효하지 않습니다.')
        return DataBackend(selected, config, make_executor, lambda:reference_context(config), preflight)
    raise ValueError('TELLY_DATA_BACKEND는 databricks 또는 mysql이어야 합니다.')
