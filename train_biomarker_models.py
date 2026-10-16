"""Launcher kept so the old command still works: runs backend/train_biomarker_models.py."""
import runpy, sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent / "backend"
sys.path.insert(0, str(BACKEND))
runpy.run_path(str(BACKEND / "train_biomarker_models.py"), run_name="__main__")
