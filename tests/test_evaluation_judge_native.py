"""Local judge transport contracts; no model, API or database calls."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock, Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


@unittest.skipUnless(importlib.util.find_spec('deepeval'), 'Separate evaluation environment required')
class NativeJudgeContracts(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from evaluation_judge import make_bounded_native_judge
        from pydantic import BaseModel

        class Score(BaseModel):
            score: int

        self.schema = Score
        self.judge, _ = make_bounded_native_judge()

    def test_sync_validated_output_and_bounded_settings(self):
        client = Mock()
        client.chat.return_value.message.content = '{"score":10}'
        self.judge.load_model = Mock(return_value=client)
        answer, cost = self.judge.generate('Judge a recorded answer', self.schema)
        self.assertEqual((answer.score, cost), (10, 0))
        args = client.chat.call_args.kwargs
        self.assertFalse(args['think'])
        self.assertEqual(args['options']['num_ctx'], 8192)
        self.assertEqual(args['options']['num_predict'], 1024)
        self.assertFalse(self.judge.supports_log_probs())

    async def test_async_uses_same_schema_contract(self):
        client = Mock()
        client.chat = AsyncMock()
        client.chat.return_value.message.content = '{"score":2}'
        self.judge.load_model = Mock(return_value=client)
        answer, cost = await self.judge.a_generate('Judge another recorded answer', self.schema)
        self.assertEqual((answer.score, cost), (2, 0))
        self.judge.load_model.assert_called_once_with(async_mode=True)
        self.assertEqual(client.chat.call_args.kwargs['format'], self.schema.model_json_schema())

    def test_oversized_or_invalid_output_is_not_a_score(self):
        client = Mock()
        self.judge.load_model = Mock(return_value=client)
        with self.assertRaises(ValueError):
            self.judge.generate('x' * 8192, self.schema)
        client.chat.assert_not_called()
        client.chat.return_value.message.content = '{"score":"invalid"}'
        with self.assertRaises(ValueError):
            self.judge.generate('Judge a recorded answer', self.schema)


if __name__ == '__main__':
    unittest.main()
