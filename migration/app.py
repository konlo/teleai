"""Compatibility entrypoint. Production UI lives in ui/analysis_page.py."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parents[1]/'ui/analysis_page.py'),run_name='__main__')
