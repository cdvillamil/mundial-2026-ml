# Validacion historica (Mundiales 2010, 2014, 2018, 2022)

Simulaciones por torneo: 20,000. Entrenamiento con corte temporal
estricto (solo datos anteriores al inicio de cada Mundial).

## Nivel partido (GBM-Poisson vs baseline Elo)

|   year |   n_matches |   gbm_rps |   base_rps |   gbm_logloss |   gbm_acc |
|-------:|------------:|----------:|-----------:|--------------:|----------:|
|   2010 |          64 |    0.3796 |     0.4059 |        0.9544 |    0.5312 |
|   2014 |          64 |    0.3861 |     0.4004 |        0.9314 |    0.5938 |
|   2018 |          64 |    0.4182 |     0.4462 |        0.9858 |    0.5469 |
|   2022 |          64 |    0.4325 |     0.4391 |        1.0353 |    0.5312 |

GBM gana en RPS en **4/4** mundiales.

## Nivel torneo (ranking de P(campeon))

|   year | predicted_champion   |   predicted_prob | real_champion   |   real_champ_rank |   real_champ_prob |   n_groups |
|-------:|:---------------------|-----------------:|:----------------|------------------:|------------------:|-----------:|
|   2010 | Brazil               |             28.3 | Spain           |                 2 |              26   |          8 |
|   2014 | Brazil               |             30.9 | Germany         |                 3 |              13   |          8 |
|   2018 | Brazil               |             29.9 | France          |                 5 |               5.7 |          8 |
|   2022 | Brazil               |             29.7 | Argentina       |                 2 |              21.2 |          8 |

Campeon real en **top-5** de P(campeon) en **4/4** mundiales.

## Gate F6
- (a) GBM supera baseline en RPS en >=3/4: PASS (4/4)
- (b) Campeon real en top-5 en >=3/4: PASS (4/4)

Nota: bracket por siembra y Elo congelado al inicio (ver plan). El campo y
grupos son los REALES de cada torneo (reconstruidos desde la base de datos).