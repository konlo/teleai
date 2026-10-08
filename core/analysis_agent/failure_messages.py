"""User-visible causes derived from structured observations, never model guesses."""
def remote_blocked(current):
    return bool(current.get('remote_rejected') or
                (current.get('failed', {}).get('query_databricks', {}).get('status') == 'unavailable'
                 and not current.get('failed', {}).get('query_databricks', {}).get('repairable')))


def remote_failure_message(observations, rejected=False):
    failures=[o for o in observations if o.get('status')=='unavailable']
    if failures:
        failure=failures[-1]
        if failure.get('execution_state')=='rejected' and failure.get('repairable'):
            return 'DB가 SQL 문법 또는 스키마 참조를 거절했습니다. 제한된 수정으로도 요청을 완료하지 못했습니다. 결과는 생성되지 않았으며 기존 데이터는 보존했습니다.'
        if failure.get('error_code') in {'mysql_query_timeout','mysql_query_interrupted'}:
            return ('MySQL이 조회 실행을 중단했습니다. '
                    + ('설정된 조회 시간 제한에 도달했습니다. ' if failure['error_code']=='mysql_query_timeout'
                       else '서버에서 조회가 취소되었습니다. ')
                    + '집계 결과와 차트는 생성되지 않았으며 기존 데이터는 보존했습니다. '
                    '전체 범위 조회에는 인덱스·집계 테이블 또는 조회 시간 설정을 점검해야 합니다. 자동 재조회하지 않습니다.')
        if failure.get('error_code') == 'remote_receipt_unverified':
            return ('조회 실행 기록과 저장된 결과의 일치를 확인하지 못해 완료로 처리하지 않았습니다. '
                    '기존 데이터는 보존했으며 중복 실행을 막기 위해 자동 재조회하지 않습니다.')
        not_submitted=failure.get('error_type')=='QueryNotSubmitted'
        if failure.get('http_status')==403:
            text='사용자 승인은 정상 처리됐지만 Databricks가 접근을 거부했습니다(HTTP 403). '
            text+=('연결 단계에서 실패하여 SQL은 제출되지 않았고 데이터도 로딩되지 않았습니다. ' if not_submitted else
                   '조회 완료 여부를 확인할 수 없습니다. ')
            return text+'연결 주소·HTTP Path·토큰의 유효성과 SQL Warehouse 접근 권한을 확인해야 합니다. 요청한 결과는 생성되지 않았으며 자동 재조회하지 않습니다.'
        if not_submitted:
            return 'Databricks 세션을 열지 못해 SQL이 제출되지 않았습니다. 연결 설정과 인증 상태를 확인해야 합니다. 분석은 완료되지 않았습니다.'
        return '원격 조회가 실패하여 완료 여부를 확인해야 합니다. 중복 실행을 막기 위해 자동 재조회하지 않습니다. 분석은 완료되지 않았습니다.'
    if rejected:
        return '사용자가 조회를 취소하여 데이터를 불러오지 않았습니다. 요청한 분석은 완료되지 않았습니다.'
    return '복구를 시도했지만 요청한 결과를 확인하지 못했습니다. 기존 데이터는 보존했습니다.'
