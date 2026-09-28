"""Read-only storage lifecycle reporting for persisted conversation scopes."""
from datetime import datetime, timezone
from pathlib import Path
import re


_SCOPE = re.compile(r'^[0-9a-f]{64}$')


def storage_report(root, *, retention_days, scope_quota_bytes, now=None):
    root=Path(root)
    now=now or datetime.now(timezone.utc).timestamp()
    cutoff=now-retention_days*86400
    scopes=[]
    if root.exists():
        for directory in root.iterdir():
            if not directory.is_dir() or not _SCOPE.fullmatch(directory.name):continue
            files=[path for path in directory.rglob('*') if path.is_file()]
            size=sum(path.stat().st_size for path in files)
            modified=max((path.stat().st_mtime for path in files),default=directory.stat().st_mtime)
            scopes.append({'scope':directory.name,'bytes':size,
                           'last_modified':datetime.fromtimestamp(modified,timezone.utc).isoformat(),
                           'expired_candidate':modified<cutoff,
                           'over_quota':size>scope_quota_bytes})
    return {'root':str(root.resolve()),'retention_days':retention_days,
            'scope_quota_bytes':scope_quota_bytes,'scope_count':len(scopes),
            'total_bytes':sum(item['bytes'] for item in scopes),
            'expired_candidates':sum(item['expired_candidate'] for item in scopes),
            'over_quota_scopes':sum(item['over_quota'] for item in scopes),
            'destructive_action_performed':False,'scopes':scopes}
