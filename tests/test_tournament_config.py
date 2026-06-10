import yaml

from src.config import CONFIGS_DIR


def test_tournament_2026_has_16_venues_with_coords_and_altitude():
    cfg = yaml.safe_load((CONFIGS_DIR / "tournament_2026.yaml").read_text(encoding="utf-8"))
    venues = cfg["venues"]
    assert len(venues) == 16
    for v in venues:
        assert {"name", "city", "country", "lat", "lon", "altitude_m"} <= set(v)
    countries = {v["country"] for v in venues}
    assert countries == {"United States", "Mexico", "Canada"}
