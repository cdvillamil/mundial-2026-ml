"""Rutas centrales del proyecto."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
EXTERNAL_DIR = DATA_DIR / "external"
PROCESSED_DIR = DATA_DIR / "processed"
DB_PATH = PROJECT_ROOT / "db" / "worldcup.sqlite"
CONFIGS_DIR = PROJECT_ROOT / "configs"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
