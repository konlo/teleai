import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT/'.env')
from core.analysis_agent.model_provider import build_analysis_chat_model
from core.analysis_agent.policy import RuntimePolicy
from core.analysis_agent.model_roles import json_role
from langchain_core.messages import SystemMessage,HumanMessage
SCHEMA={'type':'object','additionalProperties':False,'required':['schema_request','include_database_types','column_family'],'properties':{'schema_request':{'type':'boolean'},'include_database_types':{'type':'boolean'},'column_family':{'type':'string','enum':['all','numeric','categorical']}}}
PROMPT="""현재 사용자 요청에서 원하는 스키마 출력 내용을 확인하세요. JSON만 반환하세요.
실제 테이블의 열/컬럼/스키마를 요청하면 schema_request=true입니다. 개념 설명, 데이터 행 조회, 집계/시각화 요청, 도구 규칙 확인은 false입니다.
include_database_types는 DB 데이터 타입/자료형/dtype/type의 실제 값도 표시해 달라고 요청했으면 true입니다. 컬럼 이름과 자료형을 함께 요청했으면 반드시 true입니다. 이름만 요청하거나 타입을 제외하라고 요청하면 false입니다. 차트를 만들지 말라는 조건은 데이터 타입 표시와 무관합니다.
column_family는 전체=all, 수치형 컬럼만=numeric, 범주형 컬럼만=categorical입니다.
예: '스키마의 열 이름 및 자료형을 정리해줘' => true,true,all.
예: '그림 말고 필드명과 자료형을 표로 정리해줘' => true,true,all.
예: '열 이름만 알려줘' => true,false,all.
테이블명, 조건, 결과를 생성하지 마세요."""
CASES=[('이 테이블의 컬럼 이름과 database 데이터 타입을 함께 표로 보여줘. 히스토그램은 만들지 마.',True,True),('alibaba_ssd 테이블의 컬럼 이름과 database 데이터 타입을 함께 표로 보여줘.',True,True),('이전 자료의 자료형과 필드 이름을 함께 정리해 줘',True,True),('테이블 열 이름만 알려줘',True,False),('데이터 타입이 무엇인지 설명해줘',False,False)]
model=json_role(build_analysis_chat_model(RuntimePolicy.from_env(),provider='ollama'),SCHEMA,128)
results=[]
for text,request,types in CASES:
 response=model.invoke([SystemMessage(content=PROMPT),HumanMessage(content=text)])
 value=json.loads(response.content);results.append({'prompt':text,'actual':value,'pass':value['schema_request']==request and value['include_database_types']==types});print(results[-1],flush=True)
Path(__file__).with_name('metadata_probe.json').write_text(json.dumps({'mode':'actual model role probe, not complete agent execution','cases':results},ensure_ascii=False,indent=2))
