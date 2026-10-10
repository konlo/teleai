"""Grouped charts must execute the complete population, not only AND terms."""
from copy import deepcopy
from dataclasses import asdict
import sqlite3,tempfile,unittest
import pandas as pd
from core.analysis_agent.runtime import GraphAnalysisRuntime
from tests.test_llm_goal import goal,GoalModel

DATA=pd.DataFrame({'reading':[29,30,35,40,35,41,30,40],
                   'cohort':['A','A','A','A','B','A','B','B']})

class GroupedBooleanPopulationTests(unittest.TestCase):
    def test_or_population_renders_and_rebins_cached_frequencies_without_extra_sql(self):
        variants=[([{'column':'cohort','op':'in','value':['A','B']}],6),
                  ([{'column':'cohort','op':'eq','value':'A'},
                    {'column':'cohort','op':'eq','value':'B'}],6),
                  ([{'column':'cohort','op':'eq','value':'A'},
                    {'column':'reading','op':'ge','value':40}],4)]
        for alternatives,expected in variants:
            with self.subTest(alternatives=alternatives),tempfile.TemporaryDirectory() as root:
                db=sqlite3.connect(':memory:',check_same_thread=False);db.execute("ATTACH DATABASE ':memory:' AS lab")
                db.execute('CREATE TABLE lab.observations(reading REAL,cohort TEXT)')
                db.executemany('INSERT INTO lab.observations VALUES (?,?)',DATA.itertuples(index=False,name=None))
                calls=[]
                def factory(store):
                    def execute(envelope):
                        calls.append(envelope['query'])
                        frame=pd.read_sql_query(envelope['query'],db)
                        info=store.register(frame,source='lab.observations',query=envelope['query'],
                            grain='aggregate',aggregation=envelope['query'],coverage='complete',predicate_known=True)
                        return {'status':'ready','dataset':asdict(info)}
                    return execute
                plan=goal('chart',{'kind':'histogram','axes':{'x':'reading'},'category':'cohort',
                    'bins':8,'legend':True},columns=['reading','cohort'],
                    conditions=[{'column':'reading','op':'between','value':[30,40]}])
                plan['any_conditions']=alternatives
                rebin=deepcopy(plan);rebin['tasks'][0]['options']['bins']=5
                model=GoalModel(goals=[plan,plan,rebin,rebin])
                r=GraphAnalysisRuntime(root,'owner','boolean',model,sql_dialect='mysql',
                    connection_identity='fixture',remote_factory=factory,
                    reference_context_loader=lambda:[{'table':'lab.observations',
                        'columns':[{'name':'reading','dtype':'double'},{'name':'cohort','dtype':'varchar'}],
                        'observed_at':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()}])
                try:
                    for text,bins in [('first chart',8),('same population with five bins',5)]:
                        result=r.submit(text);self.assertEqual(result['status'],'answered',result)
                        c=r.inspect()['recovery'];card=r.artifacts[c['artifact_ids'][0]]
                        self.assertEqual(card.render_spec['bins'],bins)
                        self.assertEqual(card.render_spec['total_count'],expected)
                        self.assertTrue(card.image.startswith(b'\x89PNG'))
                    self.assertEqual(len(calls),1,calls)
                    self.assertEqual(len(r.datasets.metadata),1)
                finally:r.close();db.close()
