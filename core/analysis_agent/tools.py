"""Connect runtime-independent Telly tools to LangChain."""
from langchain_core.tools import StructuredTool
import duckdb
from sqlglot.errors import SqlglotError
from core.analysis_runtime_tools import build_analysis_tools
from core.analysis_tool_contract import normalize_tool_result
from utils.analysis_datasets import InvalidConditionValue
import time


def local_tools(context, diagnostics=None):
    def bind(definition):
        def execute(**arguments):
            started = time.monotonic()
            if diagnostics: diagnostics.emit('tool_started', tool=definition.name)
            dataset_ids = {key: arguments.get(key) for key in (
                'dataset_id', 'left_dataset_id', 'right_dataset_id',
                'baseline_dataset_id', 'cohort_dataset_id') if arguments.get(key) is not None}
            missing_ids = {key: value for key, value in dataset_ids.items()
                           if value not in context.datasets.metadata}
            dataset_id = arguments.get('dataset_id')
            if missing_ids:
                if diagnostics: diagnostics.emit('tool_rejected', tool=definition.name, reason='dataset_not_loaded')
                return normalize_tool_result({'status': 'error', 'error_code': 'dataset_not_loaded',
                        'retryable': False,
                        'missing_dataset_arguments': sorted(missing_ids),
                        'message': '이 ID의 로딩된 결과가 없습니다. 테이블명과 dataset ID는 다릅니다. 테이블 컬럼/설명은 inspect_table_context로 확인하세요. 실제 행이 필요하면 승인형 조회를 제안하세요.'}
                )
            try:
                result = definition.run(**arguments)
            except Exception as exc:
                if diagnostics:
                    diagnostics.failure(exc, stage=definition.name)
                    diagnostics.emit('tool_completed', tool=definition.name, status='error',
                                     elapsed_seconds=round(time.monotonic()-started, 3))
                if isinstance(exc, InvalidConditionValue):
                    return normalize_tool_result({'status':'error', 'error_code':'invalid_condition_value',
                            'retryable': False,
                            'column':exc.column, 'dtype':exc.dtype, 'value_examples':exc.examples,
                            'message':'조건값 자료형이 실제 컬럼과 다릅니다. 이 결과를 0건으로 해석하지 마세요. 관측된 문자열 값과 사용자 의미를 확인해 조건을 수정하세요. 값 예시는 전체 허용값 목록이 아닙니다.'})
                if isinstance(exc, (duckdb.Error, SqlglotError)):
                    return normalize_tool_result({'status':'error','error_code':'local_sql_error',
                            'retryable': False,
                            'error_type':type(exc).__name__,
                            'available_columns':list(context.datasets.metadata[dataset_id].columns) if dataset_id in context.datasets.metadata else [],
                            'message':'로컬 SQL을 실행하지 못했습니다. FROM data와 실제 컬럼명·자료형을 확인해 SQL을 수정하세요. 원래 필터와 집계를 유지하고 같은 실패 호출을 반복하지 마세요. 대규모 중간 결과는 집계/필터로 줄이고 표본을 전체 결과로 대신하지 마세요.'})
                if isinstance(exc, (ValueError, KeyError, TypeError)):
                    return normalize_tool_result({'status':'error','error_code':'invalid_tool_input',
                            'retryable': False,
                            'message':'도구 입력이나 데이터 범위가 유효하지 않습니다. 스키마/범위를 확인하고 다른 유효한 방법을 선택하세요.'})
                raise
            if diagnostics: diagnostics.emit('tool_completed', tool=definition.name,
                status=result.get('status') if isinstance(result,dict) else None,
                elapsed_seconds=round(time.monotonic()-started, 3),
                applied_corrections=result.get('applied_corrections',[]) if isinstance(result,dict) else [])
            return result
        return execute
    return [StructuredTool(name=t.name, description=t.description,
                           args_schema=t.parameters, func=bind(t))
            for t in build_analysis_tools(context)
            if t.name != 'propose_databricks_query']
