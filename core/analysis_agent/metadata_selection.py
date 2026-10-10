"""Focused model review of names versus names-and-types output obligations."""
import json
from types import SimpleNamespace
from langchain_core.messages import HumanMessage,SystemMessage
from core.analysis_agent.model_roles import json_role

SCHEMA={'type': 'object', 'additionalProperties': False, 'required': ['schema_request', 'include_database_types', 'column_family'], 'properties': {'schema_request': {'type': 'boolean'}, 'include_database_types': {'type': 'boolean'}, 'column_family': {'type': 'string', 'enum': ['all', 'numeric', 'categorical']}}}
PROMPT="현재 사용자 요청에서 원하는 스키마 출력 내용을 확인하세요. JSON만 반환하세요.\n실제 테이블의 열/컬럼/스키마를 요청하면 schema_request=true입니다. 개념 설명, 데이터 행 조회, 집계/시각화 요청, 도구 규칙 확인은 false입니다.\ninclude_database_types는 DB 데이터 타입/자료형/dtype/type의 실제 값도 표시해 달라고 요청했으면 true입니다. 컬럼 이름과 자료형을 함께 요청했으면 반드시 true입니다. 이름만 요청하거나 타입을 제외하라고 요청하면 false입니다. 차트를 만들지 말라는 조건은 데이터 타입 표시와 무관합니다.\ncolumn_family는 전체=all, 수치형 컬럼만=numeric, 범주형 컬럼만=categorical입니다.\n예: '스키마의 열 이름 및 자료형을 정리해줘' => true,true,all.\n예: '그림 말고 필드명과 자료형을 표로 정리해줘' => true,true,all.\n예: '열 이름만 알려줘' => true,false,all.\n테이블명, 조건, 결과를 생성하지 마세요."

def review(interpreter,current,data,selection):
    if selection.get('mode')!='explain' and selection.get('capabilities')!=['metadata']:
        return selection
    model=json_role(interpreter.selection_model,SCHEMA,128)
    if model is None:return selection
    req=SimpleNamespace(state={'recovery':current},system_message=SystemMessage(content=PROMPT),
        messages=[HumanMessage(content=current['request_text'])],tools=[])
    def override(**kw):
        r=SimpleNamespace(**{**vars(req),**kw});r.override=override;return r
    req.override=override
    def invoke():return interpreter.budget.wrap_model_call(req,lambda r:model.invoke([r.system_message,*r.messages]))
    response=interpreter.model_recovery.auxiliary_call(current,invoke) if interpreter.model_recovery else invoke()
    value=json.loads(response.content)
    if (not isinstance(value,dict) or set(value)!=set(SCHEMA['required'])
            or type(value['schema_request']) is not bool or type(value['include_database_types']) is not bool
            or value['column_family'] not in {'all','numeric','categorical'}):
        raise ValueError('invalid metadata output review')
    if not value['schema_request']:return selection
    kind=({'numeric':'numeric_columns','categorical':'categorical_columns'}.get(value['column_family'])
          or ('dtypes' if value['include_database_types'] else 'columns'))
    result={**selection,'mode':'execute','capabilities':['metadata'],'metadata_kind':kind,
        'chart_kind':'','output_columns':[],'group_column':'','scalar_operations':[],'chart_edit_fields':[]}
    interpreter.diagnostics.emit('goal_metadata_obligation_reviewed',request_id=current['request_id'],
        metadata_kind=kind,include_database_types=value['include_database_types'],changed=result!=selection)
    return result
