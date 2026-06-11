# Comparacion de modelos con GBM (corte temporal 2018-01-01)

Entrenamiento: 2010-2018. Test: 2018+. GBM usa Elo point-in-time.

| model       |   n_test |    RPS |   log_loss |   Brier |   accuracy |    ECE |
|:------------|---------:|-------:|-----------:|--------:|-----------:|-------:|
| Poisson     |     8038 | 0.3596 |     0.9057 |  0.5331 |     0.5834 | 0.0109 |
| Dixon-Coles |     8038 | 0.3597 |     0.9065 |  0.5332 |     0.5816 | 0.012  |
| GBM-Poisson |     8038 | 0.341  |     0.8735 |  0.5136 |     0.6039 | 0.0088 |

**Mejor modelo por RPS: GBM-Poisson**

Nota: GBM-Poisson usa HistGradientBoostingRegressor(loss=poisson) de sklearn
(sustituto de LightGBM/XGBoost por Python 3.14). Predice desde Elo, no identidad
del equipo, por lo que generaliza a equipos no vistos en entrenamiento.