"""Dashboard Streamlit del Mundial 2026 (corre si streamlit esta instalado:
   streamlit run src/dashboard/app.py)."""
import streamlit as st

from src.dashboard.data import build_rate_provider, load_predictions, run_simulation
from src.dashboard.figures import champion_bar, phase_heatmap, score_matrix_heatmap


@st.cache_resource
def _setup():
    field, rp = build_rate_provider()
    sim = run_simulation(field, rp, n_sims=50000)
    return field, rp, sim


def main():
    st.set_page_config(page_title="Mundial 2026 ML", layout="wide")
    st.title("Prediccion del Mundial FIFA 2026")

    field, rp, sim = _setup()

    if field.get("official"):
        st.caption("Sorteo OFICIAL FIFA (grupos A-L) + cuadro oficial.")
    else:
        st.caption("Campo de ejemplo sembrado por Elo (no es el sorteo oficial).")

    st.header("Probabilidad de ser campeon")
    st.plotly_chart(champion_bar(sim), use_container_width=True)

    st.header("Probabilidad de avance por fase")
    st.plotly_chart(phase_heatmap(sim), use_container_width=True)

    st.header("Marcador mas probable de un partido")
    teams = sorted(field["elos"])
    c1, c2 = st.columns(2)
    home = c1.selectbox("Local", teams, index=0)
    away = c2.selectbox("Visitante", teams, index=1)
    if home != away:
        st.plotly_chart(score_matrix_heatmap(rp.matrix(home, away), home, away),
                        use_container_width=True)

    st.header("Predicciones de la fase de grupos")
    st.dataframe(load_predictions(), use_container_width=True)


if __name__ == "__main__":
    main()
