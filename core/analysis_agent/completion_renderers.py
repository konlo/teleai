"""Render only validated capability evidence; never use model prose as analysis output."""
from utils.analysis_datasets import preview_dataset


def render_preview(runtime, current):
    parts = []
    if current.get('preview_evidence'):
        evidence = current['preview_evidence']
        values = evidence['values']
        parts.append(
            f"보유된 {evidence['source']} 데이터의 `{evidence['column']}` 앞 "
            f"{len(values)}개 값입니다 (전체 {evidence['total_rows']:,}행 중 저장된 미리보기).\n"
            + ('\n'.join(f'{index}. {value}' for index, value in enumerate(values, 1))
               if values else '보유 데이터에 행이 없습니다.')
            + '\n문자열 값은 미리보기 저장 시 표시 길이가 제한될 수 있습니다.')
    return '\n\n'.join(parts)


def render_data_load(runtime, current):
    parts = []
    if current.get('data_load') and current.get('load_evidence_id') and runtime.context:
        info=runtime.context.datasets.metadata[current['load_evidence_id']]
        parts.append(f'승인한 조회로 {info.source} 데이터 {info.rows:,}행, {len(info.columns):,}열을 불러와 저장했습니다. '
            '이 결과는 구조와 예시 확인용 범위이며 전체 통계로 간주하지 않습니다.\n'
            +'컬럼: '+', '.join(info.columns))
    return '\n\n'.join(parts)


def render_metadata(runtime, current):
    parts = []
    if current.get('metadata_evidence'):
        metadata = current['metadata_evidence']
        origin = ('승인 후 로딩된 실제 결과' if metadata.get('authority') == 'approved_select_star_result'
                  else '확인된 스키마 스냅샷')
        changed = ' 이전 스냅샷과 컬럼 구성이 달라 새 스키마를 사용했습니다.' if metadata.get('schema_changed') else ''
        kind = metadata.get('kind', 'columns')
        if kind == 'dtypes':
            parts.append(f"{origin} 기준으로 {metadata['table']}의 컬럼별 데이터 타입입니다.{changed}\n"
                         + '\n'.join(f"{column['name']}: {column['dtype']}" for column in metadata['schema']))
        elif kind in {'numeric_columns', 'categorical_columns'}:
            label = '수치형' if kind == 'numeric_columns' else '문자열/범주형'
            selected = metadata.get('selected_columns', [])
            parts.append(f"{origin} 기준으로 {metadata['table']}의 {label} 컬럼은 {len(selected)}개입니다.{changed}\n"
                         + (', '.join(selected) if selected else '해당 컬럼이 없습니다.'))
        else:
            parts.append(f"{origin} 기준으로 {metadata['table']}에는 컬럼이 {len(metadata['columns'])}개 있습니다.{changed}\n"
                         + ', '.join(metadata['columns']))
    return '\n\n'.join(parts)


def render_profile(runtime, current):
    parts = []
    if current.get('profile_evidence'):
        evidence = current['profile_evidence']
        profile = evidence['profile']
        kind = current.get('profile_kind')
        columns = profile.get('columns', [])
        parts.append(f"보유 데이터 프로파일 결과입니다. 출처: {profile.get('source')}\n분석 범위: {evidence.get('scope')}")
        if kind == 'missing':
            parts.append('\n'.join(
                f"{column['name']}: 결측 {column['null_count']:,}건 ({column['null_ratio_pct']}%)"
                for column in columns))
        elif kind == 'distinct':
            parts.append('\n'.join(
                f"{column['name']}: 고유값 {column['distinct_count']:,}개"
                for column in columns))
        else:
            lines = [f"행 {profile.get('rows', 0):,}개, 컬럼 {profile.get('column_count', 0):,}개"]
            for column in columns:
                summary = column.get('numeric_summary')
                if summary:
                    lines.append(
                        f"{column['name']}: 평균 {summary['mean']}, 중앙값 {summary['median']}, "
                        f"최솟값 {summary['min']}, 최댓값 {summary['max']}, 결측 {column['null_count']:,}건")
                else:
                    lines.append(
                        f"{column['name']}: 고유값 {column['distinct_count']:,}개, 결측 {column['null_count']:,}건")
            parts.append('\n'.join(lines))
        if profile.get('column_page', {}).get('has_more'):
            parts.append('컬럼이 많아 이번 응답에는 일부 컬럼만 포함했습니다.')
    return '\n\n'.join(parts)


def render_join(runtime, current):
    parts = []
    if current.get('join_query_evidence') and not current.get('join_evidence'):
        evidence = current['join_query_evidence']
        return (f"{evidence['source']}를 승인된 SQL에서 조인하여 요청한 통계를 계산했습니다. "
                '결과는 해당 조인 조건으로 결합된 행 기준입니다. 키의 실제 유일성·미일치 행 수는 별도로 측정하지 않았습니다.')
    if current.get('join_evidence') and runtime.context:
        evidence = current['join_evidence']
        summary = evidence['summary']
        info = runtime.context.datasets.metadata[evidence['dataset_id']]
        parts.append(
            f"보유 dataset 두 개를 {summary['how']} join해 {info.rows:,}행, {len(info.columns):,}열의 결과를 저장했습니다.\n"
            f"cardinality: {summary['relationship']}; key 일치 {summary['matched_distinct_keys']:,}개; "
            f"미일치 왼쪽 {summary['unmatched_left_rows']:,}행, 오른쪽 {summary['unmatched_right_rows']:,}행; "
            f"NULL key 왼쪽 {summary['left_null_key_rows']:,}행, 오른쪽 {summary['right_null_key_rows']:,}행.\n"
            f"분석 범위: {evidence.get('scope')}"
        )
    return '\n\n'.join(parts)


def render_time_series(runtime, current):
    parts = []
    if current.get('time_series_evidence'):
        evidence = current['time_series_evidence']
        result = evidence['time_series_result']
        line = (
            f"{result['time_column']}을 {result['frequency']} 단위로 준비했습니다. "
            f"{result['aggregation']} 집계 {result['output_rows']:,}행, "
            f"timezone {result['timezone']}입니다.\n"
            f"사용 {result['complete_rows']:,}행, 제외 {result['dropped_rows']:,}행, "
            f"중복 시각 관측 {result['duplicate_time_rows']:,}행, "
            f"추가한 빈 구간 {result['gap_rows_added']:,}행(gap policy: {result['gap_policy']})."
        )
        if result.get('group_column'):
            line += f"\n{result['group_column']} 기준 {result['group_count']:,}개 series를 분리했습니다."
        line += f"\n기간: {result['start']} ~ {result['end']}.\n분석 범위: {evidence.get('scope')}"
        parts.append(line)
    return '\n\n'.join(parts)


def render_statistics(runtime, current):
    parts = []
    if current.get('statistical_evidence'):
        evidence = current['statistical_evidence']
        result = evidence['test_result']
        sample = result['sample']
        line = (
            f"{result['method']} 결과입니다. 사용 {sample['complete_rows']:,}행, "
            f"결측 제외 {sample['dropped_rows']:,}행."
        )
        if result.get('statistic') is not None:
            line += f"\n통계량: {result['statistic']}; 자유도: {result.get('degrees_of_freedom')}; p-value: {result.get('p_value')}."
            line += (f" alpha={result['alpha']} 기준으로 귀무가설을 기각합니다."
                     if result.get('significant') else
                     f" alpha={result['alpha']} 기준으로 귀무가설을 기각할 근거가 부족합니다.")
        if result.get('estimate'):
            line += f"\n추정값({result['estimate']['name']}): {result['estimate']['value']}."
        if result.get('effect_size'):
            effect = result['effect_size']
            line += f"\n효과크기({effect['name']}): {effect.get('value')}."
        if result.get('confidence_intervals'):
            intervals = result['confidence_intervals']
            first = intervals[0]
            line += (f"\n{first.get('level', 1-result['alpha']):.1%} 신뢰구간"
                     f"({first['parameter']}): [{first['lower']}, {first['upper']}].")
            if len(intervals) > 1:
                line += f" 추가 신뢰구간 {len(intervals)-1}개는 구조화 결과에 보존했습니다."
        if result.get('warnings'):
            line += "\n주의: " + " ".join(result['warnings'])
        if result.get('kind') == 'paired_t':
            line += "\n행 단위 쌍이 동일 관측 단위인지 데이터만으로 검증할 수 없습니다."
        else:
            line += "\n관측치 독립성은 데이터만으로 검증할 수 없습니다."
        line += f"\n분석 범위: {evidence.get('scope')}"
        parts.append(line)
    return '\n\n'.join(parts)


def render_winsorization(runtime, current):
    parts = []
    if current.get('winsor_evidence'):
        evidence = current['winsor_evidence']
        result = evidence['winsorization_result']
        sample = result['sample']
        clipped = result['clipped_counts']
        line = (
            f"{result['column']}에 하위 {result['parameters']['lower_quantile']:.2%}, "
            f"상위 {1-result['parameters']['upper_quantile']:.2%} 윈저화를 적용해 비교했습니다. "
            f"유효값 {sample['valid_rows']:,}개, 결측 제외 {sample['missing_rows']:,}개.\n"
            f"경계: {result['thresholds']['lower']} ~ {result['thresholds']['upper']}; "
            f"하한 clip {clipped['lower']:,}개, 상한 clip {clipped['upper']:,}개.\n"
            f"원본 평균 {result['original']['mean']}, 보정 평균 {result['winsorized']['mean']}, "
            f"평균 변화 {result['mean_change']}.\n분석 범위: {evidence.get('scope')}"
        )
        if result.get('warnings'):
            line += "\n주의: " + " ".join(result['warnings'])
        parts.append(line)
    return '\n\n'.join(parts)


def render_pivot(runtime, current):
    parts = []
    if current.get('pivot_evidence') and runtime.context:
        evidence = current['pivot_evidence']
        result = evidence['pivot_result']
        dataset_id = evidence['dataset']['id']
        preview = preview_dataset(runtime.context.datasets, dataset_id)
        result_rows = runtime.context.datasets.metadata[dataset_id].rows
        line = (
            f"{', '.join(result['index_columns'])} 행 축과 "
            f"{', '.join(result['column_columns'])} 열 축으로 "
            f"{result['aggregation']} 피벗 표를 만들었습니다. "
            f"조건 적용 {result['filtered_rows']:,}행, 완전한 관측값 "
            f"{result['complete_rows']:,}행, 결과 {result['output_rows']:,}행 × "
            f"{result['output_columns']:,}열입니다."
        )
        if result.get('margins'):
            line += f" 행·열 총계는 {result['margins_name']}으로 표시했습니다."
        line += ('\n```csv\n' + preview.to_csv(index=False).strip()
                 + '\n```\n분석 범위: ' + str(evidence.get('scope', '')))
        if result_rows > 15:
            line += f"\n총 {result_rows:,}행 중 앞 15행입니다. 전체 결과는 저장된 데이터에서 확인할 수 있습니다."
        parts.append(line)
    return '\n\n'.join(parts)


def render_group_summary(runtime, current):
    parts = []
    if current.get('group_summary_evidence') and runtime.context:
        evidence = current['group_summary_evidence']
        result = evidence['group_summary_result']
        dataset_id = evidence['dataset']['id']
        preview = preview_dataset(runtime.context.datasets, dataset_id)
        line = (
            f"{', '.join(result['group_columns'])}별 {len(result['metrics'])}개 지표를 계산했습니다. "
            f"원본 {result['source_rows']:,}행, 조건 적용 {result['filtered_rows']:,}행, "
            f"그룹 키 결측 제외 {result['group_input_rows']:,}행, "
            f"결과 {result['output_rows']:,}행입니다.\n"
            + '```csv\n' + preview.to_csv(index=False).strip() + '\n```'
            + '\n분석 범위: ' + str(evidence.get('scope', '')))
        if result['output_rows'] > 15:
            line += f"\n총 {result['output_rows']:,}행 중 앞 15행입니다. 전체 결과는 저장된 데이터에서 확인할 수 있습니다."
        parts.append(line)
    return '\n\n'.join(parts)


def render_outliers(runtime, current):
    parts = []
    if current.get('outlier_evidence'):
        evidence = current['outlier_evidence']
        result = evidence['outlier_result']
        sample, counts = result['sample'], result['counts']
        thresholds, distribution = result['thresholds'], result['distribution']
        line = (
            f"{result['column']}에 {result['method']} {result['tail']} 기준을 적용했습니다. "
            f"유효값 {sample['valid_rows']:,}개, 결측 제외 {sample['missing_rows']:,}개.\n"
            f"하한 {thresholds['lower']}, 상한 {thresholds['upper']}; "
            f"선택된 이상치 {counts['selected']:,}개 ({counts['selected_percent']:.2f}%).\n"
            f"전체 유효값 범위: {distribution['minimum']} ~ {distribution['maximum']}; "
            f"하한 미만 {counts['lower']:,}개, 상한 초과 {counts['upper']:,}개."
        )
        if result['method'] == 'iqr':
            parameters = result['parameters']
            line += (f"\nQ1: {parameters['q1']}; Q3: {parameters['q3']}; "
                     f"IQR(Q3 − Q1): {parameters['iqr']}.")
        if result.get('warnings'):
            line += "\n주의: " + " ".join(result['warnings'])
        line += f"\n분석 범위: {evidence.get('scope')}"
        parts.append(line)
    return '\n\n'.join(parts)


def render_outlier_aggregate(runtime, current):
    parts = []
    for evidence_key in ('overall', 'grouped'):
        evidence = current.get('outlier_aggregate_evidence', {}).get(evidence_key)
        if not evidence:
            continue
        result = evidence['aggregation_result']
        label = 'cohort 전체 집계' if evidence_key == 'overall' else 'cohort 그룹 집계'
        parts.append(
            f"{label}: {result['aggregation']}"
            + (f"({result['value_column']})" if result.get('value_column') else "(*)")
            + (f" by {result['group_column']}" if result.get('group_column') else "")
            + f" · 완전한 관측값 {result['complete_rows']:,}행 · 제외 {result['dropped_rows']:,}행\n"
            + '```csv\n'
            + preview_dataset(runtime.context.datasets, evidence['dataset']['id']).to_csv(index=False).strip()
            + '\n```\n분석 범위: ' + str(evidence.get('scope', ''))
        )
    comparison = current.get('outlier_aggregate_evidence', {}).get('comparison')
    if comparison:
        result = comparison['comparison_result']
        parts.append(
            f"원본 전체와 cohort의 그룹 집계 비교: {result['aggregation']}"
            + (f"({result['value_column']})" if result.get('value_column') else "(*)")
            + f" by {result['group_column']} · 기준 {result['baseline_complete_rows']:,}행 · "
            + f"cohort {result['cohort_complete_rows']:,}행\n"
            + '```csv\n'
            + preview_dataset(runtime.context.datasets, comparison['dataset']['id']).to_csv(index=False).strip()
            + '\n```\n분석 범위: ' + str(comparison.get('scope', ''))
        )
    return '\n\n'.join(parts)


def render_chart(runtime, current):
    parts = []
    for card_id in current.get('artifact_ids', []):
        card = runtime.artifacts[card_id]
        parts.append(f'{card.title} 이미지를 생성했습니다.\n분석 범위: {card.scope}')
    return '\n\n'.join(parts)


def render_calculation(runtime, current):
    parts = []
    if current.get('calculation') and current.get('evidence_ids'):
        info = runtime.context.datasets.metadata[current['evidence_ids'][-1]]
        preview = preview_dataset(runtime.context.datasets, info.id)
        scope = '요청 조건에 포함된 보유 데이터 전체' if info.coverage == 'complete' else '현재 보유한 일부 데이터'
        parts.append(f'보유 데이터로 계산한 결과입니다. 출처: {info.source}\n분석 범위: {scope}')
        requested_conditions=current.get('scope',{}).get('conditions',[])
        any_conditions=current.get('scope',{}).get('any_conditions',[])
        if requested_conditions or any_conditions:
            ops = {'eq':'=', 'ne':'≠', 'gt':'>', 'ge':'≥', 'lt':'<', 'le':'≤', 'in':'포함'}
            conjunction=' AND '.join(f"{c['column']} {ops[c['op']]} {c['value']}" for c in requested_conditions)
            disjunction=' OR '.join(f"{c['column']} {ops[c['op']]} {c['value']}" for c in any_conditions)
            rendered=' AND '.join(item for item in (conjunction, '('+disjunction+')' if disjunction else '') if item)
            parts.append('적용 조건: '+rendered)
        # Never publish unchecked numbers from the model as computed results.
        if info.rows == 1 and len(info.columns) == 1:
            labels = {'AVG':'평균', 'MEDIAN':'중앙값', 'SUM':'합계', 'COUNT':'건수', 'MIN':'최솟값', 'MAX':'최댓값', 'CORR':'피어슨 상관계수',
                      'RATIO':'비율(%)'}
            operations = current.get('operations', [])
            label = labels.get(operations[0], preview.columns[0]) if len(operations) == 1 else preview.columns[0]
            parts.append(f'{label}: {preview.iloc[0, 0]}')
        else:
            parts.append('```csv\n' + preview.to_csv(index=False).strip() + '\n```')
        if info.rows > 15: parts.append(f'총 {info.rows}행 중 앞 15행입니다. 전체 결과는 저장된 데이터에서 확인할 수 있습니다.')
    return '\n\n'.join(parts)
