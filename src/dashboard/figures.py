"""Figuras Plotly del dashboard (puras, sin estado/servidor)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def champion_bar(sim_results: pd.DataFrame, top: int = 12) -> go.Figure:
    """Barra horizontal de P(campeon) para los 'top' equipos."""
    d = sim_results.nlargest(top, "p_champion").iloc[::-1]
    fig = go.Figure(go.Bar(
        x=(d["p_champion"] * 100), y=d["team"], orientation="h",
        text=[f"{v*100:.1f}%" for v in d["p_champion"]], textposition="auto"))
    fig.update_layout(title="Probabilidad de ser campeon (%)",
                      xaxis_title="%", yaxis_title="", height=500)
    return fig


def phase_heatmap(sim_results: pd.DataFrame, top: int = 16) -> go.Figure:
    """Heatmap de probabilidades por fase para los 'top' equipos."""
    cols = ["p_qualify", "p_r16", "p_qf", "p_sf", "p_final", "p_champion"]
    labels = ["Clasifica", "Octavos", "Cuartos", "Semis", "Final", "Campeon"]
    d = sim_results.nlargest(top, "p_champion")
    z = (d[cols].to_numpy() * 100)
    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=d["team"].tolist(),
        colorscale="Blues", text=np.round(z, 1), texttemplate="%{text}"))
    fig.update_layout(title="Probabilidad de avance por fase (%)",
                      height=600, yaxis=dict(autorange="reversed"))
    return fig


def score_matrix_heatmap(matrix: np.ndarray, home: str, away: str,
                         max_goals: int = 5) -> go.Figure:
    """Heatmap de la matriz de probabilidades de marcador (recortada)."""
    m = matrix[:max_goals + 1, :max_goals + 1]
    z = m * 100
    fig = go.Figure(go.Heatmap(
        z=z, x=[str(j) for j in range(max_goals + 1)],
        y=[str(i) for i in range(max_goals + 1)],
        colorscale="YlOrRd", text=np.round(z, 1), texttemplate="%{text}"))
    fig.update_layout(title=f"Prob. de marcador: {home} (filas) vs {away} (col)",
                      xaxis_title=f"Goles {away}", yaxis_title=f"Goles {home}",
                      height=450)
    return fig
