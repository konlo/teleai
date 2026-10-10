"""A focused LLM role preserves requested type output without prose routing."""
import json,unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch
from core.analysis_agent.metadata_selection import review

class MetadataReviewTests(unittest.TestCase):
    def test_names_and_types_repair_explain_but_do_not_change_source(self):
        model=Mock();model.invoke.return_value=SimpleNamespace(content=json.dumps({'schema_request':True,'include_database_types':True,'column_family':'all'}))
        interpreter=SimpleNamespace(selection_model=object(),model_recovery=None,diagnostics=Mock(),
            budget=SimpleNamespace(wrap_model_call=lambda req,handler:handler(req)))
        selection={'mode':'explain','capabilities':[],'source_reference':'previous_analysis','source_mentions':[]}
        with patch('core.analysis_agent.metadata_selection.json_role',return_value=model):
            result=review(interpreter,{'request_id':'r','request_text':'Names with database types; do not chart'},
                {'tables':[{'table':'unfamiliar.measurements'}]},selection)
            self.assertEqual(result['capabilities'],['metadata'])
            self.assertEqual(result['metadata_kind'],'dtypes')
            self.assertEqual(result['source_reference'],'previous_analysis')
            self.assertEqual(selection['mode'],'explain')
            model.invoke.return_value=SimpleNamespace(content=json.dumps({'schema_request':False,'include_database_types':False,'column_family':'all'}))
            self.assertIs(review(interpreter,{'request_id':'r','request_text':'What is a data type?'},{},selection),selection)
        with patch('core.analysis_agent.metadata_selection.json_role') as role:
            compound={'mode':'execute','capabilities':['calculation','chart']}
            self.assertIs(review(interpreter,{}, {},compound),compound);role.assert_not_called()
