"""Connect runtime-independent Telly tools to LangChain."""
from langchain_core.tools import StructuredTool
from core.analysis_runtime_tools import build_analysis_tools


def local_tools(context, diagnostics=None):
    def bind(definition):
        def execute(**arguments):
            if diagnostics: diagnostics.emit('tool_started', tool=definition.name)
            dataset_id = arguments.get('dataset_id')
            if dataset_id is not None and dataset_id not in context.datasets.metadata:
                if diagnostics: diagnostics.emit('tool_rejected', tool=definition.name, reason='dataset_not_loaded')
                return {'status': 'error', 'error_code': 'dataset_not_loaded',
                        'message': '이 ID의 로딩된 결과가 없습니다. 테이블명과 dataset ID는 다릅니다. 테이블 컬럼/설명은 inspect_table_context로 확인하세요. 실제 행이 필요하면 승인형 조회를 제안하세요.'}
            try:
                result = definition.run(**arguments)
            except Exception as exc:
                if diagnostics: diagnostics.failure(exc, stage=definition.name)
                if isinstance(exc, (ValueError, KeyError, TypeError)):
                    return {'status':'error','error_code':'invalid_tool_input',
                            'message':'도구 입력이나 데이터 범위가 유효하지 않습니다. 스키마/범위를 확인하고 다른 유효한 방법을 선택하세요.'}
                raise
            if diagnostics: diagnostics.emit('tool_completed', tool=definition.name)
            return result
        return execute
    return [StructuredTool(name=t.name, description=t.description,
                           args_schema=t.parameters, func=bind(t))
            for t in build_analysis_tools(context)
            if t.name != 'propose_databricks_query']
