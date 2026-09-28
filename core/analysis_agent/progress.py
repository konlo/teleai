"""User-visible progress comes from tool observations, never implied success."""
import json


def tool_progress(message):
    try:
        observation = json.loads(message.content)
    except (ValueError, TypeError):
        observation = {}
    if not isinstance(observation, dict):
        observation = {}
    status = observation.get('status')
    if message.name == 'query_databricks' and 'rejected' in str(message.content).lower():
        return '추가 조회가 취소되었습니다. 기존 결과를 확인합니다.'
    if status in {'error', 'needs_data', 'needs_context', 'unavailable', 'no_valid_chart'}:
        return '도구가 결과를 완성하지 못했습니다. 원인과 복구 방법을 확인하고 있습니다.'
    if status == 'planned':
        return '필요한 데이터 범위와 조회 계획을 준비했습니다.'
    if status == 'ready':
        if message.name in {'recommend_chart_images', 'render_histogram', 'prepare_histogram'}:
            return ('생성된 차트의 요청 조건을 검증하고 있습니다.' if observation.get('cards')
                    else '시각화에 사용할 데이터를 확인하고 있습니다.')
        if message.name == 'local_analysis_sql':
            return '계산 결과의 기간·조건과 정확성을 검증하고 있습니다.'
        if message.name == 'query_databricks':
            return '데이터를 불러왔습니다. 요청한 분석을 이어갑니다.'
    return '도구 결과를 확인하고 있습니다.'
