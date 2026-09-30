"""Do not score intermediate exploration as a completed benchmark answer."""
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration,ChatResult
from uuid import uuid4
from scripts.spider_interactive import sqlite_executor,evaluate_interactive
from utils.analysis_datasets import DatasetStore


class ExplorationThenAggregate(BaseChatModel):
    calls:int=0
    @property
    def _llm_type(self):return 'scripted-public-sqlite-exploration'
    def bind_tools(self,tools,**kwargs):return self
    def _generate(self,messages,**kwargs):
        self.calls+=1
        query='SELECT measurement FROM events LIMIT 2' if self.calls==1 else 'SELECT AVG(measurement) AS average FROM events'
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content='',tool_calls=[
            {'name':'query_databricks','id':str(uuid4()),'args':{'query':query,'source':'events','reason':'Inspect then calculate'}}]))])


class InteractiveSpiderTests(unittest.TestCase):
    def test_readonly_bounded_result_and_source_verification(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'fixture.sqlite'
            with sqlite3.connect(path) as db:
                db.executescript('CREATE TABLE events(measurement INTEGER); INSERT INTO events VALUES(1),(3),(8);')
            calls=[];store=DatasetStore()
            execute=sqlite_executor(path,store,calls,max_rows=2)
            result=execute({'source':'events','query':'SELECT * FROM events'})
            self.assertEqual(result['dataset']['coverage'],'truncated')
            self.assertEqual(result['dataset']['rows'],2)
            for q in ['DELETE FROM events','PRAGMA writable_schema=ON']:
                with self.assertRaises(Exception):execute({'source':'events','query':q})
            with self.assertRaises(ValueError):execute({'source':'different','query':'SELECT * FROM events'})
            self.assertEqual(len(calls),1)

    def test_preview_is_not_final_calculation(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'fixture.sqlite'
            with sqlite3.connect(path) as db:
                db.executescript('CREATE TABLE events(measurement INTEGER); INSERT INTO events VALUES(1),(3),(8);')
            model=ExplorationThenAggregate()
            with patch('scripts.evaluate_spider2_teleai.database_path',return_value=path),patch('scripts.evaluate_spider2_teleai.task_document',return_value=''):
                result=evaluate_interactive(Path(root),{'instance_id':'synthetic','db':'fixture',
                    'question':'events 테이블에서 measurement 평균을 계산해줘'},model,Path(root)/'predictions')
            self.assertEqual(result['status'],'SQL_COMPLETED',result)
            query=Path(result['prediction']).read_text()
            self.assertIn('AVG(measurement)',query)
            self.assertNotIn('LIMIT 2',query)
            self.assertEqual(result['remote_executions'],0)

    def test_provider_block_counts_failed_inference_attempts_not_sql_accuracy(self):
        from tests.test_model_recovery import IntermittentModel
        from tests.test_model_service_unavailable import error
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'fixture.sqlite'
            with sqlite3.connect(path) as db:db.execute('CREATE TABLE events(measurement INTEGER)')
            model=IntermittentModel(failures_left=3,calls=[])
            with patch('scripts.evaluate_spider2_teleai.database_path',return_value=path),patch('scripts.evaluate_spider2_teleai.task_document',return_value=''),patch('tests.test_model_recovery.rate_error',side_effect=error),patch('core.analysis_agent.model_recovery.time.sleep'):
                result=evaluate_interactive(Path(root),{'instance_id':'outage','db':'fixture',
                    'question':'events 테이블에서 measurement 평균을 계산해줘'},model,Path(root)/'predictions')
            self.assertEqual(result['status'],'BLOCKED_PROVIDER',result)
            self.assertEqual(result['model_calls'],3)
            self.assertEqual(result['model_retries'],2)
            self.assertEqual(result['local_sql_executions'],[])
            self.assertNotIn('prediction',result)
