import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.dashboard.figures import champion_bar, score_matrix_heatmap


def test_champion_bar_returns_figure():
    df = pd.DataFrame({"team": ["A", "B", "C"], "p_champion": [0.3, 0.2, 0.1]})
    fig = champion_bar(df, top=3)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


def test_score_matrix_heatmap_returns_figure():
    m = np.full((6, 6), 1 / 36)
    fig = score_matrix_heatmap(m, "A", "B")
    assert isinstance(fig, go.Figure)
