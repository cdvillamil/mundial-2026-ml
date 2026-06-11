# Validacion historica (Mundiales 2010, 2014, 2018, 2022)

Simulaciones por torneo: 20,000. Entrenamiento con corte temporal
estricto (solo datos anteriores al inicio de cada Mundial).

## Nivel partido (GBM-Poisson vs baseline Elo vs ensamble GBM+Dixon-Coles)

|   year |   n_matches |   gbm_rps |   ens_rps |   base_rps |   gbm_logloss |   gbm_acc |
|-------:|------------:|----------:|----------:|-----------:|--------------:|----------:|
|   2010 |          64 |    0.3779 |    0.3864 |     0.4059 |        0.9497 |    0.5312 |
|   2014 |          64 |    0.3967 |    0.4195 |     0.4004 |        0.9476 |    0.5938 |
|   2018 |          64 |    0.4195 |    0.43   |     0.4462 |        0.9911 |    0.5625 |
|   2022 |          64 |    0.4418 |    0.4231 |     0.4391 |        1.0533 |    0.5312 |

GBM gana en RPS al baseline en **3/4** mundiales.
El ensamble GBM+Dixon-Coles mejora al GBM en **1/4** mundiales.

## Nivel torneo (ranking de P(campeon))

|   year | predicted_champion   |   predicted_prob | real_champion   |   real_champ_rank |   real_champ_prob |   n_groups |
|-------:|:---------------------|-----------------:|:----------------|------------------:|------------------:|-----------:|
|   2010 | Brazil               |             30.3 | Spain           |                 2 |              26   |          8 |
|   2014 | Brazil               |             28.8 | Germany         |                 3 |              13.3 |          8 |
|   2018 | Brazil               |             26.5 | France          |                 5 |               5.9 |          8 |
|   2022 | Brazil               |             35.5 | Argentina       |                 2 |              25.1 |          8 |

Campeon real en **top-5** de P(campeon) en **4/4** mundiales.

## Gate F6
- (a) GBM supera baseline en RPS en >=3/4: PASS (3/4)
- (b) Campeon real en top-5 en >=3/4: PASS (4/4)

Nota: bracket por siembra y Elo congelado al inicio (ver plan). El campo y
grupos son los REALES de cada torneo (reconstruidos desde la base de datos).