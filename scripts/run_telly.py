"""Launch the production persistent agent using its pinned environment."""
from pathlib import Path
import os
import sys
root=Path(__file__).resolve().parents[1]
python=root/'.telly_runtime/v1-venv/bin/python'
if not python.exists():raise SystemExit('Install requirements-agent.txt in .telly_runtime/v1-venv first.')
os.chdir(root)
os.execv(str(python),[str(python),'-m','streamlit','run','main.py',
                      '--server.address=127.0.0.1','--server.port=8502',*sys.argv[1:]])
