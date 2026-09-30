"""Bind explicit key/order/value roles to observed schema, never to table names."""
import re
from langchain_core.messages import AIMessage

NULL_BEFORE_PATTERN=(r'(?:키[·/\s,]*정렬[·/\s,]*분포\s*컬럼의\s*)?'
    r'결측\s*(?:값이\s*있는\s*)?행(?:을|은)?\s*최신\s*행\s*선택\s*전에\s*제외(?:해줘|해주세요)?(?=$|[\s.!?])')


def name_pattern(name):
    return r'(?<![A-Za-z0-9_])[`\"]?'+re.escape(name)+r'[`\"]?(?![A-Za-z0-9_])'


def bind(text,context, *, sources=(), remote_available=False):
    if not context or not re.search(r'마지막|가장\s*(?:늦|최근)|최신|\blatest\b|\blast\b',text,re.I):return None
    if not re.search(r'중복|unique|고유|(?:별|마다).{0,55}(?:행|기록)|\bper\b',text,re.I):return None
    if not re.search(r'분포|histogram|히스토그램|차트|그려|distribution',text,re.I):return None
    selected=context.datasets.metadata.get(context.selected_dataset_id)
    remote_source = ''
    if remote_available and len(sources) <= 1:
        from core.analysis_catalog import _source_key, resolve_table_context
        source = next(iter(sources), '') or (selected.source if selected else '')
        compatible = (selected and _source_key(selected.source) == _source_key(source)
            and selected.coverage == 'complete' and selected.grain == 'raw' and selected.predicate_known)
        refresh = re.search(r'재조회|다시\s*(?:로딩|조회|불러)|새로\s*(?:조회|로딩|불러)|\brefresh\b', text, re.I)
        if source and (not compatible or refresh):
            observed = resolve_table_context(context.reference_context, context.datasets, source)
            schema = observed.get('table_context') or {}
            if schema.get('columns'):
                from types import SimpleNamespace
                remote_source = schema['table']
                selected = SimpleNamespace(id='', columns=[c['name'] for c in schema['columns']])
                remote_dtypes = {c['name']: c.get('dtype', '') for c in schema['columns']}
    if selected is None:
        candidates=[info for info in context.datasets.metadata.values() if info.role=='root']
        if len(candidates)==1:selected=candidates[0]
    if selected is None:return {'question':'어떤 데이터에서 키별 최신 행을 선택할지 알려주세요.'}
    dtypes=remote_dtypes if remote_source else context.datasets.inspect(selected.id).get('dtypes',{})
    tie_columns=[]
    tie_matches=[]
    for column in selected.columns:
        match=re.search(r'(?:동률(?:일\s*때|이면|인\s*경우)?|동일하면|ties?\s*(?:use|by)?)\s*[, :]*'+name_pattern(column)
            +r'\s*(?:이|가|을|를)?\s*(?:가장\s*큰|내림차순|descending|DESC\b)(?:\s*행)?',text,re.I)
        if match:
            tie_matches.append(match)
            tie_columns.append(column)
    role_text=text
    for match in sorted(tie_matches,key=lambda m:m.start(),reverse=True):
        role_text=role_text[:match.start()]+role_text[match.end():]
    mentioned=[c for c in selected.columns if re.search(name_pattern(c),role_text,re.I)]
    targets=[c for c in mentioned if re.search(name_pattern(c)+r'(?:에\s*대한|의|별)?\s*(?:장치\s*수\s*)?(?:histogram|히스토그램|분포)',text,re.I)]
    keys=[c for c in mentioned if c not in targets and re.search(name_pattern(c)+r'\s*(?:컬럼(?:을)?\s*)?(?:을|를)?\s*(?:기준|별|마다)',text,re.I)
          and not re.search(r'datetime|timestamp|date',dtypes.get(c,''),re.I)]
    # A composite key is an explicitly connected list immediately before its role.
    for match in re.finditer(r'((?:[`"\w]+\s*(?:,|와|과|and|&)\s*)+[`"\w]+)\s*(?:의\s*)?(?:조합|복합\s*키|기준)',text,re.I):
        candidates=[c for c in mentioned if c not in targets and re.search(name_pattern(c),match[1],re.I)]
        if len(candidates)>=2:
            keys=list(dict.fromkeys([*keys,*candidates]))
    orders=[c for c in mentioned if c not in keys and c not in targets and c not in tie_columns and (
        re.search(r'datetime|timestamp|date',dtypes.get(c,''),re.I) or
        re.search(name_pattern(c)+r'(?:이|가|을|를)?\s*(?:기준|가장\s*큰)',text,re.I))]
    result={'dataset_id':selected.id,'key_columns':keys,'value_column':targets[0] if len(targets)==1 else '',
            'order_column':orders[0] if len(orders)==1 else '', 'categorical':True}
    anchor=context.datasets.metadata.get(context.selected_dataset_id)
    result.update(context_dataset_id=context.selected_dataset_id,
        context_snapshot=anchor.snapshot if anchor else None)
    if tie_columns:result['tie_break_columns']=tie_columns
    null_before=re.search(NULL_BEFORE_PATTERN,text,re.I)
    if null_before:result['null_policy']='drop_before_selection'
    if remote_source:
        result.update(remote=True, source=remote_source)
    if re.search(r'(?:동률|결측).{0,90}(?:말고|말아|않|아니|사용하지|제외하지)',text,re.I):
        result['question']='최신행의 동률·결측 처리 정책에 부정 또는 변경 조건이 있습니다. 사용할 정책을 명확히 지정해주세요.'
    elif re.search(r'결측|\b(?:null|missing)\b',text,re.I) and (not null_before or re.search(r'선택\s*후',text)):
        result['question']='결측 행을 최신행 선택 전에 제외할지, 최신행을 선택한 뒤 제외할지 확인해주세요. 현재는 선택 전에 키·정렬·분포 컬럼의 결측 행을 제외하는 정책을 지원합니다.'
    elif re.search(r'동률|동일하면|\bties?\b',role_text,re.I) or len(tie_columns)>1:
        result['question']='동률일 때 사용할 추가 정렬 컬럼 하나와 가장 큰 값(내림차순)을 선택할지 확인해주세요.'
    elif set(tie_columns)&set([*keys,*orders,*targets]):
        result['question']='동률 해소 컬럼은 키·기본 정렬·분포 컬럼과 다른 컬럼이어야 합니다.'
    elif re.search(r'평균|중앙값|표준편차|상관|합계|\b(?:mean|average|median|correlation)\b', text, re.I):
        result['question']='키별 최신 행의 분포와 추가 통계를 함께 요청하셨습니다. 추가 통계의 대상 컬럼과 적용 범위를 알려주세요. 아직 분석을 완료하지 않았습니다.'
    elif re.search(r'제목|축\s*라벨', text, re.I):
        result['question']='키별 최신 행 분포의 별도 구간·표시 설정은 아직 함께 검증할 수 없습니다. 최신행 선택과 기본 분포부터 진행할지 알려주세요.'
    elif len(targets)!=1 or not 1 <= len(keys) <= 4:
        result['question']='중복을 구분할 키 컬럼과 분포를 그릴 대상 컬럼을 각각 알려주세요.'
    elif len(orders)!=1:
        clocks=[c for c in selected.columns if re.search(r'datetime|timestamp|date',dtypes.get(c,''),re.I)]
        result['question']='마지막 행을 결정할 정렬 기준은 어느 컬럼인가요? '+(', '.join(clocks) if clocks else '시간 또는 순번 컬럼을 알려주세요.')
        result['order_choices']=clocks
    else:
        from core.analysis_agent.eda_contract import histogram_bins
        requested_bins = histogram_bins(text)
        numeric=bool(re.search(r'int|long|short|byte|float|double|decimal|numeric',dtypes.get(targets[0],''),re.I))
        identifier=bool(re.search(name_pattern(targets[0])+r'.{0,20}(?:식별자|identifier|category|범주)',text,re.I))
        result['categorical']=not numeric or identifier
        if requested_bins is not None:
            if result['categorical']:
                result['question']='범주별 빈도와 수치 구간 히스토그램 중 어떤 방식인지 확인해주세요.'
            elif not 2 <= requested_bins <= 100:
                result['question']='히스토그램 구간 수는 2~100으로 지정해주세요.'
            else:
                result['bins'] = requested_bins
    return result


def continue_order(text,previous,context):
    prior=previous.get('latest_per_key_spec') or {}
    anchor=context.datasets.metadata.get(context.selected_dataset_id) if context else None
    same_anchor=(context is not None and context.selected_dataset_id==prior.get('context_dataset_id',prior.get('dataset_id'))
        and ('context_snapshot' not in prior or prior['context_snapshot']==(anchor.snapshot if anchor else None)))
    if prior and same_anchor:
        from core.analysis_agent.latest_filter_scope import continue_stage
        continued=continue_stage(text,previous)
        if continued:return continued
    if prior and same_anchor and previous.get('latest_selection_evidence') and not prior.get('categorical',True):
        from core.analysis_agent.eda_contract import histogram_bins
        bins=histogram_bins(text)
        if bins is not None and re.fullmatch(
                r'\s*(?:그\s*(?:차트|히스토그램)\s*)?(?:구간\s*(?:수|개수)\s*(?:는|를|을|만|:|=)?\s*\d+\s*(?:개)?|\d+\s*개\s*구간|bins?\s*[=:]?\s*\d+)'
                r'\s*(?:으로|로)?\s*(?:바꿔줘|변경해줘|그려줘|해줘)?[.!]?\s*',text,re.I):
            base=previous.get('request_text','')
            patterns=(r'구간\s*(?:수|개수)\s*(?:는|를|을|만|:|=)?\s*\d+',
                r'\d+\s*(?:개(?:의)?\s*)?(?:구간|bins?\b)',r'\bbins?\s*(?:count\s*)?(?:=|:|of)?\s*\d+')
            for pattern in patterns:base=re.sub(pattern,'',base,flags=re.I)
            return base+'\n'+str(bins)+'개 구간으로 그려줘.'
    pending=previous.get('latest_pending') or {}
    if not pending:return None
    anchor=context.datasets.metadata.get(context.selected_dataset_id)
    if (context.selected_dataset_id!=pending.get('context_dataset_id',pending.get('dataset_id'))
            or ('context_snapshot' in pending and pending['context_snapshot']!=(anchor.snapshot if anchor else None))):return None
    if pending.get('error_code')=='latest_null_policy' and re.fullmatch(r'\s*'+NULL_BEFORE_PATTERN+r'[.!]?\s*',text,re.I):
        return pending['request_text']+'\n키·정렬·분포 컬럼의 결측행을 최신행 선택 전에 제외해줘.'
    if pending.get('error_code')=='latest_order_tie':
        if pending.get('remote'):
            from core.analysis_catalog import resolve_table_context
            observed=resolve_table_context(context.reference_context,context.datasets,pending['source'])
            columns=[c['name'] for c in (observed.get('table_context') or {}).get('columns',[])]
        else:
            info=context.datasets.metadata.get(pending.get('dataset_id'))
            columns=info.columns if info else []
        forbidden={*pending.get('key_columns',[]),pending.get('order_column'),pending.get('value_column'),*pending.get('tie_break_columns',[])}
        for column in columns:
            if column in forbidden:continue
            if re.fullmatch(r'\s*(?:동률(?:이면|일\s*때)?\s*)?'+name_pattern(column)
                    +r'\s*(?:(?:이|가)\s*가장\s*큰\s*(?:행)?|내림차순|기준(?:으로)?|으로|로)?\s*(?:해줘|선택|사용해줘)?[.!]?\s*',text,re.I):
                return pending['request_text']+'\n동률이면 '+column+'이 가장 큰 행을 사용해줘.'
    # A new analysis question must not be merged into this pending request.
    for column in pending.get('order_choices',[]):
        if re.fullmatch(r'\s*[`\"]?'+re.escape(column)+r'[`\"]?\s*(?:기준(?:으로)?|으로|로)?\s*(?:해줘|선택|사용해줘)?[.!]?\s*',text,re.I):
            return pending['request_text']+'\n마지막 정렬 컬럼은 '+column+' 기준으로 사용해줘.'
    return None


def ask(current, question):
    current['status']='needs_context'
    current['stop_reason']='latest_selection_context'
    current['latest_pending']={**current['latest_per_key_spec'],'request_text':current['request_text'],
        'error_code':current.get('latest_error_code')}
    if current.get('latest_error_code')=='latest_order_tie':
        question+=' 추가 컬럼의 가장 큰 값(내림차순)을 선택하려면 컬럼명을 알려주세요.'
    if current.get('latest_error_code')=='latest_null_policy':
        question+=' 키·정렬·분포 컬럼의 결측행을 최신행 선택 전에 제외하려면 “결측행을 최신행 선택 전에 제외해줘”라고 지정할 수 있습니다. 원본은 보존됩니다.'
    return {'recovery':current,'messages':[AIMessage(content=question,
        additional_kwargs={'analysis_status':'answered','analysis_complete':False,'analysis_artifact_ids':[]})], 'jump_to':'end'}


def accepted(context,artifacts,spec,args,observation):
    from utils.analysis_latest_filters import matches, lineage
    if not matches(args,spec):return False
    if observation.get('status')!='ready':return False
    if any(args.get(k)!=spec.get(k) for k in ('dataset_id','key_columns','order_column','value_column','categorical')):return False
    if args.get('bins', 20) != spec.get('bins', 20):return False
    if (args.get('tie_break_columns') or []) != spec.get('tie_break_columns',[]):return False
    if args.get('null_policy','reject') != spec.get('null_policy','reject'):return False
    chosen=context.datasets.metadata.get(observation.get('dataset',{}).get('id'))
    distribution=context.datasets.metadata.get(observation.get('distribution',{}).get('id'))
    if chosen is None or distribution is None or chosen.parent_id!=spec['dataset_id'] or distribution.parent_id!=chosen.id:return False
    expected={'kind':'latest_per_key','key_columns':spec['key_columns'],
        'order_columns':[spec['order_column'],*spec.get('tie_break_columns',[])],'descending':True,'null_policy':spec.get('null_policy','reject'),'tie_policy':'reject',
        'input_dataset_id':spec['dataset_id'],'input_snapshot':context.datasets.metadata[spec['dataset_id']].snapshot,**lineage(spec.get('conditions'),spec.get('filter_stage'))}
    if chosen.row_selection!=expected or distribution.row_selection!=expected:return False
    if chosen.rows!=observation.get('selected_keys') or chosen.rows<1:return False
    from utils.analysis_image_validation import validate_chart_image
    try:
        cards=observation['cards']
        if len(cards)!=1:return False
        card=artifacts[cards[0]['id']]
        if card.dataset_id!=(distribution.id if spec['categorical'] else chosen.id):return False
        if card.kind!=('bar' if spec['categorical'] else 'histogram') or card.columns[0]!=spec['value_column']:return False
        if not spec['categorical'] and card.render_spec.get('bins') != spec.get('bins',20):return False
        validate_chart_image(card.image)
    except (KeyError,ValueError,TypeError,IndexError):return False
    return True


def accepted_remote(context, artifacts, spec, args, observation, receipts):
    from utils.analysis_latest_filters import matches, lineage
    if not matches(args,spec) or not matches(observation.get('remote_latest_plan',{}),spec):return False
    if observation.get('status') != 'ready' or any(args.get(k) != spec.get(k)
            for k in ('source', 'key_columns', 'order_column', 'value_column')):
        return False
    if args.get('categorical',True) != spec.get('categorical',True) or args.get('bins',20) != spec.get('bins',20):
        return False
    planned = observation.get('remote_latest_plan',{})
    if (args.get('null_policy','reject') != spec.get('null_policy','reject')
            or planned.get('null_policy','reject') != spec.get('null_policy','reject')):return False
    if ((args.get('tie_break_columns') or []) != spec.get('tie_break_columns',[])
            or planned.get('tie_break_columns',[]) != spec.get('tie_break_columns',[])):return False
    if planned.get('categorical',True) != spec.get('categorical',True) or planned.get('bins',20) != spec.get('bins',20):
        return False
    result_id = observation.get('input_result_id')
    query = observation.get('remote_latest_plan', {}).get('query')
    if not result_id or args.get('result_dataset_id') != result_id or not any(
            r.get('dataset_id') == result_id and r.get('query') == query for r in receipts.values()):
        return False
    try:
        info = context.datasets.metadata[observation['distribution']['id']]
        parent=context.datasets.metadata[result_id]
        expected={'kind':'remote_latest_per_key_distribution','key_columns':spec['key_columns'],
            'order_columns':[spec['order_column'],*spec.get('tie_break_columns',[])],'descending':True,'null_policy':spec.get('null_policy','reject'),'tie_policy':'reject',
            'input_result_id':result_id,'input_snapshot':parent.snapshot,**lineage(spec.get('conditions'),spec.get('filter_stage'))}
        if info.parent_id != result_id or info.row_selection != expected or info.snapshot != parent.snapshot:
            return False
        card = artifacts[observation['cards'][0]['id']]
        if len(observation['cards']) != 1 or card.kind != ('bar' if spec.get('categorical',True) else 'histogram') or card.dataset_id != info.id or card.columns[0] != spec['value_column']:
            return False
        if not spec.get('categorical',True) and card.render_spec.get('bins') != spec.get('bins',20):
            return False
        from utils.analysis_image_validation import validate_chart_image
        validate_chart_image(card.image)
    except (KeyError, ValueError, TypeError, IndexError):
        return False
    return True


def reuse_verified(context, artifacts, ledger, spec, previous, text):
    if ledger is None or re.search(r'재조회|다시\s*(?:로딩|조회|불러)|새로\s*(?:조회|로딩|불러)|\brefresh\b', text, re.I):
        return None
    old = previous.get('latest_per_key_spec') or {}
    proof = previous.get('latest_selection_evidence') or {}
    if not old.get('remote') or any(old.get(k) != spec.get(k) for k in
            ('source', 'key_columns', 'order_column', 'value_column', 'categorical', 'bins', 'tie_break_columns', 'null_policy', 'conditions', 'filter_stage')):
        return None
    from core.analysis_agent.remote_completion import verified_receipt
    verified = {}
    for identity in previous.get('remote_query_evidence', {}):
        try:
            receipt = ledger.get(identity)
        except KeyError:
            continue
        call = {'id': identity, 'args': {k: receipt[k] for k in ('source', 'query', 'reason')}}
        evidence = verified_receipt(ledger, context, call, receipt.get('result') or {})
        if evidence:
            verified[identity] = evidence
    arguments = {k: spec[k] for k in ('source', 'key_columns', 'order_column', 'value_column')}
    arguments.update(categorical=spec.get('categorical',True),bins=spec.get('bins',20))
    arguments['tie_break_columns']=spec.get('tie_break_columns',[])
    arguments['null_policy']=spec.get('null_policy','reject')
    arguments.update(conditions=spec.get('conditions',[]),filter_stage=spec.get('filter_stage',''))
    arguments['result_dataset_id'] = proof.get('input_result_id')
    return proof if accepted_remote(context, artifacts, spec, arguments, proof, verified) else None


def render(runtime,current):
    proof=current['latest_selection_evidence']
    spec=current['latest_per_key_spec']
    tie_note=('동률은 '+', '.join(spec['tie_break_columns'])+'의 큰 값 순서로 해소했습니다. ' if spec.get('tie_break_columns') else '')
    if spec.get('conditions'):
        stage='전' if spec['filter_stage']=='before_selection' else '후'
        tie_note+=f'필터 조건을 최신행 선택 {stage}에 적용했습니다. '
    if spec.get('null_policy')=='drop_before_selection':
        tie_note+=f"최신행 선택 전에 키·정렬·분포 컬럼의 결측 행 {proof['excluded_rows']:,}개를 제외했습니다. "
    if spec.get('remote'):
        return (('저장된 집계 결과를 재사용했습니다. ' if current.get('remote_latest_reused') else '')
            + f"{spec['source']}에서 키 {', '.join(spec['key_columns'])}마다 {spec['order_column']}이 가장 큰 행을 검증했습니다. "
            + tie_note + f"전체 {proof['input_rows']:,}행의 {proof['selected_keys']:,}개 키를 DB에서 집계했습니다. "
            + ('범주별 빈도 막대그래프로 표시했습니다. ' if spec.get('categorical',True) else f"{spec.get('bins',20)}개 수치 구간의 히스토그램으로 표시했습니다. " )
            + '원본 전체를 로딩하지 않았으며 기존 보유 데이터는 보존했습니다.')
    return ('키 '+', '.join(spec['key_columns'])+'마다 '+spec['order_column']+'이 가장 큰 행을 선택했습니다. '
        +tie_note+f"원본 {proof['input_rows']:,}행 중 {proof['selected_keys']:,}개의 키를 각각 한 번 계산했습니다. 원본은 보존했습니다."
        +(' 식별자·범주 값은 값별 빈도 막대그래프로 표시합니다.' if spec['categorical'] else ' 수치 구간별 히스토그램으로 표시합니다.'))
