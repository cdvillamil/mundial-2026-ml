# Mundial 2026 ML — Predicción del Mundial FIFA 2026

Proyecto personal y educativo de Machine Learning y analítica deportiva: predicción de marcadores exactos, fase de grupos, avance por rondas y probabilidad de campeón del Mundial FIFA 2026, usando exclusivamente fuentes de datos gratuitas y públicas.

> ⚠️ Proyecto educativo. No está diseñado ni destinado a apuestas deportivas.

## 📄 Documento de diseño

Todo el diseño del sistema está en **[docs/DOCUMENTO_TECNICO.md](docs/DOCUMENTO_TECNICO.md)**:

1. Plan de ejecución (con adaptación al calendario: el torneo inicia el 11-jun-2026)
2. Arquitectura (Python, Pandas, Scikit-Learn, XGBoost, LightGBM, Streamlit)
3. Fuentes de datos públicas (Kaggle, GitHub, APIs gratuitas) con plan B por fuente
4. Diseño de base de datos (SQLite + Parquet)
5. Pipeline de ML (modelo híbrido GBM-Poisson + Dixon-Coles)
6. Simulación Monte Carlo del torneo (100k / 500k / 1M corridas)
7. Validación histórica (backtesting mundiales 2010, 2014, 2018, 2022)
8. Roadmap en 7 fases con gates de validación
9. Riesgos técnicos y mitigaciones
10. Recomendaciones de mejora

## Cómo correr el pipeline

```
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m src.etl.run_all              # descarga, limpia, carga, Elo, rankings
.venv\Scripts\python -m src.etl.validate             # reporte del gate F1
.venv\Scripts\python -m src.simulation.build_field   # campo de 48 equipos (ejemplo)
.venv\Scripts\python -m src.simulation.monte_carlo --n 100000   # ¿quién será campeón?
.venv\Scripts\python -m src.inference.predict_2026   # marcadores de la fase de grupos
.venv\Scripts\python -m pytest                       # suite de tests (64)
```

## Cómo ver las predicciones

- **Reporte HTML interactivo** (no requiere servidor): abre `outputs/reports/f7_dashboard.html`.
  Se regenera con `python -m src.dashboard.report`.
- **Dashboard Streamlit**: `streamlit run src/dashboard/app.py` (probabilidades por equipo/fase
  y matriz de marcador interactiva por partido).
- **Predicciones de grupos** (marcador más probable + 1X2): `outputs/predictions/group_stage_2026_latest.csv`.

## Cómo evaluar contra los resultados reales

Conforme se juegue el Mundial, rellena `data/external/wc2026_results.csv`
(`home,away,home_goals,away_goals`) y ejecuta:

```
.venv\Scripts\python -m src.evaluation.evaluate_predictions
```

Devuelve RPS, log-loss, accuracy 1X2 y acierto de marcador exacto de las predicciones
pre-registradas vs lo que realmente ocurrió.

## Validación del modelo (backtesting histórico)

El sistema se validó entrenando solo con datos previos a cada Mundial y prediciéndolo:
el GBM-Poisson **supera al baseline Elo en RPS en los 4 mundiales** (2010–2022) y el
**campeón real quedó en el top-5 de P(campeón) en los 4**. Reproducible con
`python -m src.evaluation.backtest_tournament --n 20000`.

## Estado

- [x] Documento técnico de diseño (v1.0 — 2026-06-10)
- [x] Fase 1 — Setup e ingesta de datos (gate PASS: 49 398 partidos, Elo r=0.972 vs eloratings.net)
- [x] Fase 2 — EDA y feature engineering (gate PASS: 19/19 tests, matriz de features 49k×60, anti-leakage verificado)
- [x] Fase 3 — Modelos estadísticos (Poisson, Dixon-Coles, baseline Elo, métricas RPS/log-loss/Brier, backtest temporal 2018; 34/34 tests — ver `outputs/reports/f3_model_comparison.md`)
- [x] Fase 4 — Modelos ML y ensamble (gate PASS: GBM-Poisson con Elo point-in-time mejora RPS 0.360→0.341 y accuracy 58%→60%, ECE 0.009; 43/43 tests — ver `outputs/reports/f4_model_comparison.md`)
- [x] Fase 5 — Simulador Monte Carlo (gate PASS: 48 equipos/12 grupos/8 terceros, desempates FIFA testeados, 100k corridas; campeón más probable España 23% en campo de ejemplo; 57/57 tests — ver `outputs/reports/f5_simulation.md`)
- [x] Fase 6 — Validación histórica (gate PASS 4/4: GBM supera al baseline Elo en RPS en los 4 mundiales; campeón real en top-5 de P(campeón) en los 4 — España #2, Alemania #3, Francia #5, Argentina #2; grupos reales reconstruidos; 58/58 tests — ver `outputs/reports/f6_historical_validation.md`)
- [x] Fase 7 — Predicción 2026, dashboard y evaluación final (gate PASS: 72 predicciones de grupos pre-registradas, dashboard HTML+Streamlit, marco de evaluación predicho-vs-real; 64/64 tests — ver `outputs/reports/f7_dashboard.html`)

**🎉 Proyecto completo: las 7 fases superaron su gate de validación.**
