"""Generic scalar SQL keeps observed names, dialect and exact predicates."""
import unittest
from types import SimpleNamespace
import pandas as pd
import duckdb
from sqlglot import parse_one,exp
from core.analysis_agent.source_scalar import next_call
from utils.analysis_datasets import DatasetStore

class SourceScalarContractTests(unittest.TestCase):
    def test_mysql_and_databricks_sql_have_same_independent_result(self):
        for dialect,source in [('mysql','different_schema.measurements'),('databricks','catalog.other.measurements')]:
            with self.subTest(dialect=dialect):
                context=SimpleNamespace(sql_dialect=dialect,datasets=DatasetStore(),reference_context=[
                    {'table':source,'columns':[{'name':'metric','dtype':'DOUBLE'},{'name':'cohort','dtype':'VARCHAR'}]}])
                current={'intent_origin':'llm','calculation':True,'required_sources':[source],
                    'required_columns':['metric'],'operations':['SUM'],
                    'scope':{'conditions':[{'column':'metric','op':'ge','value':3}],
                        'any_conditions':[{'column':'cohort','op':'eq','value':'red'},{'column':'cohort','op':'eq','value':'blue'}]}}
                call=next_call(context,current,True,False)
                self.assertEqual(call['args']['source'],source)
                tree=parse_one(call['args']['query'],read=dialect)
                for table in list(tree.find_all(exp.Table)):table.replace(exp.to_table('data'))
                data=pd.DataFrame({'metric':[2.,4.,10.,20.],'cohort':['red','red','blue','green']})
                with duckdb.connect() as db:
                    db.register('data',data)
                    self.assertEqual(db.execute(tree.sql(dialect='duckdb')).fetchone(),(14.,))
                self.assertEqual(current['scope']['conditions'][0]['value'],3)
                for overrides in [{'current_result_only':True},{'operations':['MEDIAN']},
                                  {'required_columns':['not_observed']},{'group_columns':['cohort']}]:
                    self.assertIsNone(next_call(context,{**current,**overrides},True,False))
                self.assertIsNone(next_call(context,current,True,True))
                self.assertIsNone(next_call(context,current,False,False))
