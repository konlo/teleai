"""Goal, tool filtering and SQL provenance agree on null/exclusion semantics."""
import unittest
from unittest.mock import Mock
import pandas as pd
from pandas.testing import assert_frame_equal
from core.analysis_runtime_tools import AnalysisToolContext,build_analysis_tools
from core.analysis_agent.goal_contract import compile_goal
from utils.analysis_datasets import DatasetStore,Condition,DatasetInfo,filter_frame,_implies
from utils.analysis_provenance import query_conditions
from core.analysis_sql import validate_query
from core.analysis_agent.intent_scope import scope_matches
from tests.test_llm_goal import goal


class NullPredicateContractTests(unittest.TestCase):
    def test_local_sql_filters_preserve_nulls_and_original(self):
        frame=pd.DataFrame({'measure':[1.,None,4.],'label':['a',None,'b']})
        store=DatasetStore();raw=store.register(frame,source='lab.nullable',coverage='complete',predicate_known=True)
        context=AnalysisToolContext(store,{},[],Mock())
        tools={t.name:t for t in build_analysis_tools(context)}
        cases=[('measure','not_null',None,2),('label','is_null',None,1),
               ('label','not_in',['a'],1),('label','not_in',['a',None],0)]
        for column,op,value,count in cases:
            with self.subTest(op=op,column=column,value=value):
                args={'dataset_id':raw.id,'query':'SELECT COUNT(*) AS n FROM data',
                    'requested_conditions':[{'column':column,'op':op,'value':value}]}
                output=tools['local_analysis_sql'].run(**args)
                self.assertEqual(output['status'],'ready',output)
                self.assertEqual(output['preview'][0]['n'],count)
                assert_frame_equal(store.frames[raw.id],frame)
        again=tools['local_analysis_sql'].run(raw.id,'SELECT COUNT(*) AS n FROM data')
        self.assertEqual(again['preview'][0]['n'],3)

    def test_sql_provenance_and_compiled_scope_share_canonical_null_conditions(self):
        for sql,op,value in [('label IS NULL','eq',None),('label IS NOT NULL','ne',None),
                             ("label NOT IN ('a')",'not_in',['a'])]:
            parsed=query_conditions(validate_query('SELECT * FROM data WHERE '+sql,dialect='duckdb'))
            self.assertEqual(parsed,(Condition('label',op,value),))
            scope={'conditions':[{'column':'label','op':op,'value':value}]}
            self.assertTrue(scope_matches('SELECT COUNT(*) FROM data WHERE '+sql,scope,dialect='duckdb'))
            self.assertFalse(scope_matches('SELECT COUNT(*) FROM data',scope,dialect='duckdb'))
        context=Mock();context.reference_context=[]
        context.datasets.metadata={};context.sql_dialect='mysql'
        plan=goal('calculation',{'operations':['SUM']},sources=['lab.nullable'],columns=['measure'],
            conditions=[{'column':'measure','op':'not_null','value':None}])
        state=compile_goal({},plan,context,False)
        self.assertEqual(state['scope']['conditions'],[{'column':'measure','op':'ne','value':None}])
        self.assertFalse(_implies(Condition('label','eq',None),Condition('label','ne','a')))
        self.assertFalse(_implies(Condition('label','eq','b'),Condition('label','not_in',['a'])))
        sql='SELECT * FROM data WHERE label = NULL'
        self.assertIsNone(query_conditions(validate_query(sql,dialect='duckdb')))
        self.assertFalse(scope_matches(sql,{'conditions':[{'column':'label','op':'eq','value':None}]},dialect='duckdb'))

    def test_implicit_chart_null_is_ignored_consistently_in_sql_and_lineage(self):
        query='SELECT measure, COUNT(*) AS n FROM data WHERE measure IS NOT NULL AND measure >= 3 GROUP BY measure'
        info=DatasetInfo(id='derived',source='lab.nullable',columns=('measure','n'),rows=2,
            query=query,parent_id='raw',conditions=(Condition('measure','ne',None),Condition('measure','ge',3)),predicate_known=True)
        scope={'conditions':[{'column':'measure','op':'ge','value':3}]}
        self.assertTrue(scope_matches(info,scope,histogram_column='measure',dialect='duckdb'))
        explicit={'conditions':scope['conditions']+[{'column':'measure','op':'ne','value':None}]}
        self.assertTrue(scope_matches(info,explicit,histogram_column='measure',dialect='duckdb'))
        null_only={'conditions':[{'column':'measure','op':'eq','value':None}]}
        self.assertFalse(scope_matches(info,null_only,histogram_column='measure',dialect='duckdb'))


if __name__=='__main__':unittest.main()
