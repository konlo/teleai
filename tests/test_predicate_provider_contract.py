"""Provider grammar prevents scalar/range shape confusion before execution."""
import unittest
from jsonschema import Draft202012Validator
from core.analysis_agent.goal_contract import response_schema,predicates

class PredicateProviderContractTests(unittest.TestCase):
    def test_operations_and_literal_shapes_are_consistent(self):
        schema=response_schema()['properties']['conditions']['items']
        validator=Draft202012Validator(schema)
        valid=[{'column':'reading','op':'ge','value':10},
               {'column':'reading','op':'between','value':[10,20]},
               {'column':'label','op':'not_in','value':['red']},
               {'column':'reading','op':'is_null','value':None}]
        for item in valid:
            with self.subTest(valid=item):validator.validate(item);predicates([item])
        invalid=[{'column':'reading','op':'between','value':10},
                 {'column':'reading','op':'between','value':[10]},
                 {'column':'reading','op':'ge','value':[10,20]},
                 {'column':'label','op':'in','value':'red'},
                 {'column':'reading','op':'not_null','value':10},
                 {'column':'reading','op':'ge','value':None}]
        for item in invalid:
            with self.subTest(invalid=item):
                self.assertTrue(list(validator.iter_errors(item)))
                with self.assertRaises(ValueError):predicates([item])
        alternative=response_schema()['properties']['any_conditions']['items']
        self.assertTrue(list(Draft202012Validator(alternative).iter_errors(valid[1])))
