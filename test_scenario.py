import os
import sys

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.llm import load_llm
from core.llm_router import route_query

def run_tests():
    print("🚀 LLM 라우터 테스트 시나리오 실행 시작...\n")
    
    try:
        llm = load_llm()
        print("✅ LLM 로드 성공\n")
    except Exception as e:
        print(f"❌ LLM 로드 실패: {e}")
        sys.exit(1)

    test_cases = [
        {
            "description": "1. 단순 데이터 로딩 요청 (미리보기 상태)",
            "query": "나이가 42세인 사람들을 찾아줘",
            "is_preview_state": True,
            "expected_agents": ["SQL Builder"]
        },
        {
            "description": "2. 시각화가 필요한 데이터 요청 (미리보기 상태 -> 연쇄 실행 필요)",
            "query": "나이가 42세인 사람들의 직업에 대해서 파이 차트로 그려줘",
            "is_preview_state": True,
            "expected_agents": ["SQL Builder", "EDA Analyst"]
        },
        {
            "description": "3. 이미 데이터가 로드된 상태에서의 시각화 요청 (EDA 단독 실행)",
            "query": "나이가 42세인 사람들의 직업에 대해서 파이 차트로 그려줘",
            "is_preview_state": False,
            "expected_agents": ["EDA Analyst"]
        },
        {
            "description": "4. 이미 데이터가 로드된 상태에서의 단순 요약 요청 (EDA 단독 실행)",
            "query": "데이터의 요약 통계를 보여줘",
            "is_preview_state": False,
            "expected_agents": ["EDA Analyst"]
        },
        {
            "description": "5-1. [다단계-1] 20~30대 데이터 로딩 요청",
            "query": "나이가 20대에서 30대 사이의 데이타를 로딩해줘",
            "is_preview_state": True,
            "expected_agents": ["SQL Builder"]
        },
        {
            "description": "5-2. [다단계-2] 데이터 로드 후 시각화 요청",
            "query": "age에 따른 job 분포를 시각화 해줘",
            "is_preview_state": False,
            "expected_agents": ["EDA Analyst"]
        },
        {
            "description": "5-3. [다단계-3] 필터링 추가 시각화 요청",
            "query": "여기에서 unknown은 제거 해서 그려줘",
            "is_preview_state": False,
            "expected_agents": ["EDA Analyst"]
        }
    ]

    passed = 0
    failed = 0

    for idx, tc in enumerate(test_cases, 1):
        print(f"[{idx}] {tc['description']}")
        print(f"   질문: {tc['query']}")
        print(f"   미리보기 상태(is_preview_state): {tc['is_preview_state']}")
        
        try:
            plan = route_query(llm, tc["query"], tc["is_preview_state"])
            print(f"   => 분류된 의도(Intent): {plan.intent_type}")
            print(f"   => 추론(Reasoning): {plan.reasoning}")
            print(f"   => 제안된 에이전트(Agents): {plan.suggested_agents}")
            
            if plan.suggested_agents == tc["expected_agents"]:
                print("   ✅ PASS\n")
                passed += 1
            else:
                print(f"   ❌ FAIL (Expected: {tc['expected_agents']}, Got: {plan.suggested_agents})\n")
                failed += 1
        except Exception as e:
            print(f"   ❌ ERROR 수행 중 예외 발생: {e}\n")
            failed += 1

    print("-" * 50)
    print(f"🎯 테스트 결과: {passed} 통과, {failed} 실패")
    if failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    run_tests()
