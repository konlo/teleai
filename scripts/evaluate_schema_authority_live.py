"""Live warehouse schema-authority regression with a scripted exploration step.

The schema report must come from an actual preceding schema observation.
No actual row values, credentials or provider exception text are saved.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from core.analysis_agent.databricks import ConnectionConfig, make_executor
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_catalog import _quoted_table, resolve_table_context


class ProbeThenAggregate(BaseChatModel):
    source: str
    column: str
    calls: int = 0

    @property
    def _llm_type(self):
        return 'scripted-live-schema-regression'

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, **kwargs):
        self.calls += 1
        if self.calls > 2:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content='두 조회 결과를 검증해주세요.'))])
        expression = 'COUNT(*) AS n' if self.calls == 1 else 'AVG(`' + self.column.replace('`', '``') + '`) AS average'
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content='', tool_calls=[{
            'name': 'query_databricks', 'id': str(uuid4()), 'args': {
                'source': self.source, 'query': 'SELECT ' + expression + ' FROM ' + _quoted_table(self.source),
                'reason': 'Verify source schema survives a count probe'}}]))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--schema-report', type=Path, required=True)
    parser.add_argument('--column', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--live-model', action='store_true', help='Use the configured Databricks model instead of the scripted count probe')
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
    previous = json.loads(args.schema_report.read_text())
    refs, source = previous['observed_schema'], previous['source']
    if not any(c['name'] == args.column for c in refs[0]['columns']):
        raise ValueError('Column absent from observed schema')
    config = ConnectionConfig.from_env()
    queries = []
    model = ProbeThenAggregate(source=source, column=args.column)
    if args.live_model:
        from core.analysis_agent.model_provider import build_analysis_chat_model
        from core.analysis_agent.policy import RuntimePolicy
        model = build_analysis_chat_model(RuntimePolicy(), provider='databricks')

    def factory(store):
        execute = make_executor(config, store, max_rows=5)
        def tracked(envelope):
            queries.append(envelope['query'])
            return execute(envelope)
        return tracked

    with tempfile.TemporaryDirectory(prefix='teleai-schema-live-') as root:
        runtime = GraphAnalysisRuntime(root, 'live-eval', 'schema-authority',
            model,
            connection_identity=config.identity(), remote_factory=factory,
            reference_context_loader=lambda: refs)
        try:
            outcome = runtime.submit(source + '의 ' + args.column + ' 평균을 계산해줘')
            context = resolve_table_context(refs, runtime.datasets, source)
            state = runtime.inspect()['recovery']
            preserved = context.get('table_context', {}).get('columns') == refs[0]['columns']
            diagnostics = [json.loads(line) for line in runtime.diagnostics.path.read_text().splitlines()]
            query_contract = bool(queries) if args.live_model else len(queries) == 2
            report = {'status': 'PASS' if outcome['status'] == 'answered' and preserved and query_contract else 'FAIL',
                'agent_status': outcome['status'], 'data': 'actual Databricks',
                'model': 'configured Databricks model' if args.live_model else 'scripted probe then aggregate; no LLM inference',
                'model_calls': state.get('model_calls'), 'queries': queries,
                'source_schema_preserved': preserved, 'schema_authority': context.get('authority'),
                'source_column_count': len(refs[0]['columns']),
                'contract': {k: state.get(k) for k in ('required_columns','required_sources','scope','scope_error','stop_reason','operations','evidence_ids')},
                'datasets': [asdict(info) for info in runtime.datasets.metadata.values()],
                'diagnostics': diagnostics, 'error_type': outcome.get('error_type')}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
            print({k: report[k] for k in ('status','agent_status','source_schema_preserved','error_type')})
            return 0 if report['status'] == 'PASS' else 1
        finally:
            runtime.close()


if __name__ == '__main__':
    raise SystemExit(main())
