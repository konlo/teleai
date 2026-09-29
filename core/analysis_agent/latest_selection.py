"""Bind explicit key/order/value roles to observed schema, never to table names."""
import re
from langchain_core.messages import AIMessage


def name_pattern(name):
    return r'(?<![A-Za-z0-9_])[`\"]?'+re.escape(name)+r'[`\"]?(?![A-Za-z0-9_])'


def bind(text,context):
    if not context or not re.search(r'마지막|가장\s*(?:늦|최근)|최신|\blatest\b|\blast\b',text,re.I):return None
    if not re.search(r'중복|unique|고유|(?:별|마다).{0,55}(?:행|기록)|\bper\b',text,re.I):return None
    if not re.search(r'분포|histogram|히스토그램|차트|그려|distribution',text,re.I):return None
    selected=context.datasets.metadata.get(context.selected_dataset_id)
    if selected is None:
        candidates=[info for info in context.datasets.metadata.values() if info.role=='root']
        if len(candidates)==1:selected=candidates[0]
    if selected is None:return {'question':'어떤 데이터에서 키별 최신 행을 선택할지 알려주세요.'}
    dtypes=context.datasets.inspect(selected.id).get('dtypes',{})
    mentioned=[c for c in selected.columns if re.search(name_pattern(c),text,re.I)]
    targets=[c for c in mentioned if re.search(name_pattern(c)+r'(?:에\s*대한|의|별)?\s*(?:장치\s*수\s*)?(?:histogram|히스토그램|분포)',text,re.I)]
    keys=[c for c in mentioned if c not in targets and re.search(name_pattern(c)+r'\s*(?:컬럼(?:을)?\s*)?(?:을|를)?\s*(?:기준|별|마다)',text,re.I)
          and not re.search(r'datetime|timestamp|date',dtypes.get(c,''),re.I)]
    orders=[c for c in mentioned if c not in keys and c not in targets and (
        re.search(r'datetime|timestamp|date',dtypes.get(c,''),re.I) or
        re.search(name_pattern(c)+r'(?:이|가|을|를)?\s*(?:기준|가장\s*큰)',text,re.I))]
    result={'dataset_id':selected.id,'key_columns':keys,'value_column':targets[0] if len(targets)==1 else '',
            'order_column':orders[0] if len(orders)==1 else '', 'categorical':True}
    if re.search(r'평균|중앙값|표준편차|상관|합계|\b(?:mean|average|median|correlation)\b', text, re.I):
        result['question']='키별 최신 행의 분포와 추가 통계를 함께 요청하셨습니다. 추가 통계의 대상 컬럼과 적용 범위를 알려주세요. 아직 분석을 완료하지 않았습니다.'
    elif re.search(r'\bbins?\b|구간\s*(?:수|개수)|\d+\s*개\s*구간|제목|축\s*라벨', text, re.I):
        result['question']='키별 최신 행 분포의 별도 구간·표시 설정은 아직 함께 검증할 수 없습니다. 최신행 선택과 기본 분포부터 진행할지 알려주세요.'
    elif len(targets)!=1 or len(keys)!=1:
        result['question']='중복을 구분할 키 컬럼과 분포를 그릴 대상 컬럼을 각각 알려주세요.'
    elif len(orders)!=1:
        clocks=[c for c in selected.columns if re.search(r'datetime|timestamp|date',dtypes.get(c,''),re.I)]
        result['question']='마지막 행을 결정할 정렬 기준은 어느 컬럼인가요? '+(', '.join(clocks) if clocks else '시간 또는 순번 컬럼을 알려주세요.')
        result['order_choices']=clocks
    else:
        numeric=bool(re.search(r'int|float|double|decimal|numeric',dtypes.get(targets[0],''),re.I))
        identifier=bool(re.search(name_pattern(targets[0])+r'.{0,20}(?:식별자|identifier|category|범주)',text,re.I))
        result['categorical']=not numeric or identifier
    return result


def continue_order(text,previous,context):
    pending=previous.get('latest_pending') or {}
    if not pending or context.selected_dataset_id!=pending.get('dataset_id'):return None
    # A new analysis question must not be merged into this pending request.
    for column in pending.get('order_choices',[]):
        if re.fullmatch(r'\s*[`\"]?'+re.escape(column)+r'[`\"]?\s*(?:기준(?:으로)?|으로|로)?\s*(?:해줘|선택|사용해줘)?[.!]?\s*',text,re.I):
            return pending['request_text']+'\n마지막 정렬 컬럼은 '+column+' 기준으로 사용해줘.'
    return None


def ask(current, question):
    current['status']='needs_context'
    current['stop_reason']='latest_selection_context'
    current['latest_pending']={**current['latest_per_key_spec'],'request_text':current['request_text']}
    return {'recovery':current,'messages':[AIMessage(content=question,
        additional_kwargs={'analysis_status':'answered','analysis_complete':False,'analysis_artifact_ids':[]})], 'jump_to':'end'}


def accepted(context,artifacts,spec,args,observation):
    if observation.get('status')!='ready':return False
    if any(args.get(k)!=spec.get(k) for k in ('dataset_id','key_columns','order_column','value_column','categorical')):return False
    chosen=context.datasets.metadata.get(observation.get('dataset',{}).get('id'))
    distribution=context.datasets.metadata.get(observation.get('distribution',{}).get('id'))
    if chosen is None or distribution is None or chosen.parent_id!=spec['dataset_id'] or distribution.parent_id!=chosen.id:return False
    expected={'kind':'latest_per_key','key_columns':spec['key_columns'],
        'order_columns':[spec['order_column']],'descending':True,'null_policy':'reject','tie_policy':'reject',
        'input_dataset_id':spec['dataset_id'],'input_snapshot':context.datasets.metadata[spec['dataset_id']].snapshot}
    if chosen.row_selection!=expected or distribution.row_selection!=expected:return False
    if chosen.rows!=observation.get('selected_keys') or chosen.rows<1:return False
    from utils.analysis_image_validation import validate_chart_image
    try:
        cards=observation['cards']
        if len(cards)!=1:return False
        card=artifacts[cards[0]['id']]
        if card.dataset_id!=(distribution.id if spec['categorical'] else chosen.id):return False
        if card.kind!=('bar' if spec['categorical'] else 'histogram') or card.columns[0]!=spec['value_column']:return False
        validate_chart_image(card.image)
    except (KeyError,ValueError,TypeError,IndexError):return False
    return True


def render(runtime,current):
    proof=current['latest_selection_evidence']
    spec=current['latest_per_key_spec']
    return ('키 '+', '.join(spec['key_columns'])+'마다 '+spec['order_column']+'이 가장 큰 행을 선택했습니다. '
        +f"원본 {proof['input_rows']:,}행 중 {proof['selected_keys']:,}개의 키를 각각 한 번 계산했습니다. 원본은 보존했습니다."
        +(' 식별자·범주 값은 값별 빈도 막대그래프로 표시합니다.' if spec['categorical'] else ' 수치 구간별 히스토그램으로 표시합니다.'))
