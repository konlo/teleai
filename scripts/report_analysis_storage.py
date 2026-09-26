"""Report analysis storage quota and retention candidates without deleting data."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from core.analysis_agent.policy import RuntimePolicy
from core.analysis_agent.storage_policy import storage_report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT/'.telly_runtime/v1')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    policy=RuntimePolicy.from_env()
    report=storage_report(args.root,retention_days=policy.retention_days,
                          scope_quota_bytes=policy.scope_disk_quota_bytes)
    rendered=json.dumps(report,ensure_ascii=False,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(rendered)
    print(rendered,end='')
    return 1 if report['over_quota_scopes'] else 0


if __name__=='__main__':raise SystemExit(main())
