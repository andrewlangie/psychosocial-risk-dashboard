"""Launcher kept so the old command still works: runs backend/avro_stream_poller_with_ml.py."""
import runpy, sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent / "backend"
sys.path.insert(0, str(BACKEND))
runpy.run_path(str(BACKEND / "avro_stream_poller_with_ml.py"), run_name="__main__")
