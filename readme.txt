2025.11.03
1. UI를 개선하고 싶어 아래와 같이 개선 해줘 
   - %sql 로 prompt가 시작하면 이건 새로운 SQL문을 새롭게 LLM을 통해서 만들고 수행하고 Data Preview를 업데이트 해달라는 거야 
   - %eda 로 시작하면 이건 loading 된 데이타를 가지고 EDA를 수행한다는 의미 이고 loading data dataframe으로만 작업을 진행한다는 거야 
     %eda 가 있으면 input 장에 이 값을 계속 default로 넣어줘서 계속해서 eda를 진행할 수 있도록 해줘 
     --> %eda는 구지 넣을 필요 없을 것 같음 그냥 %sql 일 때만 새로운 SQL을 생성하고 실행하는 것으로 

2025.11.02
1. 기존에 있는 databricks에서 table를 선택하는 UI는 제거하고 효과적인 table 선택을 하고 싶어.
   왼쪽 메뉴는 Access 할 수 있는 table list를 보여주고 사용자가 선택할 수 있도록 하고 싶어. 
   catalog와 schema는 default에 .env에 있는 값을 가지고 사용하고 이 값을 읽어서 접근할 수있는 table list를 보여주 
2. 여기에 선택한 table name을 session으로 저장하고 있다가 사용자가 prompt에 입력하는 data 관련 요청 사항은 이 값을 활용할 수 있도록 해줘

3. prompt에서 이야기 하는 데이타에 대한 이야기는 table 에 대한 이야기야 
   그래서 catalog나 schema를 물어 보는 경우는 없어. 그래서 catalog난 schema는 side menu에서 table list를 찾는 것으로만 사용해줘
   
4. UI 확장을 위해서 화면에 tab를 추가 하고 싶어. 
   1. data가 loading 되었으면 지금 화면에 보이는 데이타를 여기 tab으로 옮겨줘 
   2. 사용자 입력 및 대답에 대한 history를 볼 수 있는 화면이면 좋겠어. 
   3. 실시간 실행 로그도 tab으로 해줘

2025.10.28
1. 현재 sql를 생성하는 것과 EDA를 위한 prompt가 각각 나눠져 있는데 이것을 하나의 prompt에서 입력해서 모든 것을 처리할 수 있도록 수정 해줘
2. 두 개의 agent를 만드는데 첫번째는 입력된 prompt를 통해서 SQL문을 만들어서 화면에 보여주고 사용자에게 수행할까요 ? 수행을 원할 때만 데이타를 loading 해줘
   두번째 agent는 지금 처럼 사용자가 EDA를 요청하면 이를 수행하여 결과를 화면에 보여주는 것으로 해줘. 
3. 당연하게 두 개의 agent를 구별해서 필요한 agent를 수행할 수 있도록 해줘.

2025.10.27
지금 좀 데이타를 선택하는 방법을 좀 변경하고 싶어.  
1. 조회하고자 하는 table를 입력한다. 
   Query 는 아래와 같은 구조로 되어 있어 사용자는 반드시 table 정보를 입력해야하고 
   SELECT * FROM {table} 
   입력되어 있지 않을 경우 아래와 같이 내가 선택할 수 있는 테이블 이름을 보여준다. 
   ['samples.bakehouse.sales_franchises', 'samples.bakehouse.sales_customers', 'samples.bakehouse.sales_transactions'
    'samples.accuweather.forecast_daily_calendar_imperial', 'samples.accuweather.forecast_hourly_imperial '
    'v_msc_online_pm9a3'
   ]
2. 테이블이 잘 선택되었을 때 SQL문을 생성하고 생성된 SQL문을 확인하고 편지할 수 있도록 해줘.
   그리고 내가 loading 버튼을 수행하면 테이블 limit 10으로 데이타를 loading 하고 10개에 대해서는 table로 보여줘

3. EDA를 위한 UI는 그대로 두고 데이타 loading 방식만 수정해줘