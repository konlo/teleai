"""Secret-safe company server check; --database executes SELECT 1 once, no LLM."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def check(environ, *, database=False, connect=None):
    from core.analysis_agent.provider_config import configured_provider, azure_options
    from core.analysis_agent.databricks import ConnectionConfig
    from core.analysis_agent.connection_probe import probe_databricks
    selected = environ.get('TELLY_DATA_BACKEND', 'databricks').strip().lower()
    report = {'model_called':False, 'model_api_checked':False,
              'data_backend':selected if selected in {'databricks', 'mysql'} else 'unsupported'}
    try:
        provider = configured_provider(environ)
        report['model_provider'] = provider
        if provider == 'azure':
            azure_options(environ)
        report['model_configuration'] = 'PASS'
    except ValueError:
        report['model_configuration'] = 'FAIL'
    if report['data_backend'] != 'databricks':
        report['database'] = {'status':'FAIL', 'stage':'database_backend_selection'}
        return report
    try:
        config = ConnectionConfig.from_env(environ=environ)
        config.validate()
    except ValueError:
        report['database'] = {'status':'FAIL', 'stage':'database_configuration'}
    else:
        report['database'] = (probe_databricks(config, connect=connect) if database else
                              {'status':'CONFIGURED', 'stage':'database_configuration',
                               'network_checked':False})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', action='store_true', help='Execute SELECT 1 once; no model API call')
    args = parser.parse_args()
    from dotenv import load_dotenv
    import os
    import logging
    load_dotenv(ROOT / '.env')
    # This standalone command outputs safe metadata only. SDK exception logs
    # must not append private URLs/messages to the shareable diagnostic result.
    logging.getLogger().handlers[:] = [logging.NullHandler()]
    for name in ('databricks', 'urllib3', 'httpx', 'httpcore'):
        logging.getLogger(name).disabled = True
    report = check(os.environ, database=args.database)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(report['model_configuration'] == 'FAIL' or report['database']['status'] == 'FAIL')


if __name__ == '__main__':
    raise SystemExit(main())
