"""Genera un reporte HTML interactivo autonomo (sin servidor)."""
import pandas as pd

from src.config import PROJECT_ROOT
from src.dashboard.figures import champion_bar, phase_heatmap


def build_html_report(sim_results: pd.DataFrame, preds: pd.DataFrame) -> str:
    out = PROJECT_ROOT / "outputs" / "reports" / "f7_dashboard.html"
    out.parent.mkdir(parents=True, exist_ok=True)

    parts = ["<html><head><meta charset='utf-8'>",
             "<title>Mundial 2026 - Predicciones</title></head><body>",
             "<h1>Prediccion del Mundial FIFA 2026</h1>",
             "<p><b>Campo de ejemplo sembrado por Elo (no es el sorteo oficial).</b></p>",
             champion_bar(sim_results).to_html(full_html=False, include_plotlyjs="cdn"),
             phase_heatmap(sim_results).to_html(full_html=False, include_plotlyjs=False),
             "<h2>Predicciones de la fase de grupos (marcador mas probable)</h2>",
             preds.to_html(index=False)]
    parts.append("</body></html>")
    html = "\n".join(parts)
    out.write_text(html, encoding="utf-8")
    return str(out)


def main():
    from src.dashboard.data import build_rate_provider, load_predictions, run_simulation
    field, rp = build_rate_provider()
    sim = run_simulation(field, rp, n_sims=50000)
    preds = load_predictions()
    path = build_html_report(sim, preds)
    print(f"Reporte HTML generado: {path}")


if __name__ == "__main__":
    main()
