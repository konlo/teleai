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
   ]
2. 테이블이 잘 선택되었을 때 SQL문을 생성하고 생성된 SQL문을 확인하고 편지할 수 있도록 해줘.
   그리고 내가 loading 버튼을 수행하면 테이블 limit 10으로 데이타를 loading 하고 10개에 대해서는 table로 보여줘

3. EDA를 위한 UI는 그대로 두고 데이타 loading 방식만 수정해줘