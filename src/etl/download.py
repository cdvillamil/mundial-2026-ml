"""Descarga de fuentes publicas con copia cruda fechada."""
import datetime as dt
from pathlib import Path

import requests

from src.config import RAW_DIR

_BASE = "https://raw.githubusercontent.com/martj42/international_results/master"
SOURCES = {
    "results": f"{_BASE}/results.csv",
    "shootouts": f"{_BASE}/shootouts.csv",
    "former_names": f"{_BASE}/former_names.csv",
}


def fetch(url: str, timeout: int = 60) -> bytes:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def save_raw(name: str, content: bytes, raw_dir: Path = RAW_DIR,
             today: dt.date | None = None) -> Path:
    today = today or dt.date.today()
    raw_dir.mkdir(parents=True, exist_ok=True)
    dated = raw_dir / f"{name}_{today.isoformat()}.csv"
    dated.write_bytes(content)
    (raw_dir / f"{name}_latest.csv").write_bytes(content)
    return dated


def download_all() -> dict[str, Path]:
    return {name: save_raw(name, fetch(url)) for name, url in SOURCES.items()}


if __name__ == "__main__":
    for name, path in download_all().items():
        print(f"{name}: {path}")
