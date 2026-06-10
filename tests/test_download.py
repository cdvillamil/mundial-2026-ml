import datetime as dt

from src.etl.download import SOURCES, save_raw


def test_sources_define_the_three_files():
    assert set(SOURCES) == {"results", "shootouts", "former_names"}


def test_save_raw_writes_dated_and_latest_copies(tmp_path):
    content = b"date,home_team\n2022-12-18,Argentina\n"
    dated = save_raw("results", content, raw_dir=tmp_path, today=dt.date(2026, 6, 10))
    assert dated.name == "results_2026-06-10.csv"
    assert dated.read_bytes() == content
    assert (tmp_path / "results_latest.csv").read_bytes() == content
