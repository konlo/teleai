"""Fingerprint the executable agent without reading configuration or secrets."""
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
import sys


def snapshot(root=None):
    root=Path(root) if root else Path(__file__).resolve().parents[1]
    paths={root/'main.py',root/'pages/Telly.py'}
    for directory in ('core','utils','ui'):
        paths.update((root/directory).rglob('*.py'))
    files={str(p.relative_to(root)):sha256(p.read_bytes()).hexdigest()
           for p in sorted(paths) if p.is_file()}
    try:
        revision=subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,
                                text=True,check=True).stdout.strip()
    except (OSError,subprocess.CalledProcessError):revision=None
    return {'source_sha256':sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),
            'source_files':files,'git_head':revision,'python':sys.version.split()[0],
            'platform':platform.system(),
            'scope':'Executable source fingerprint; no environment/configuration/credentials'}
