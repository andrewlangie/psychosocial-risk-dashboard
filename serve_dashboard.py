"""Launcher kept so the old command still works: runs backend/serve_dashboard.py."""
import runpy, sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent / "backend"
sys.path.insert(0, str(BACKEND))
runpy.run_path(str(BACKEND / "serve_dashboard.py"), run_name="__main__")
