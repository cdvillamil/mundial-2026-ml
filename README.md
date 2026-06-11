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
.venv\Scripts\python -m src.etl.run_all    # descarga, limpia, carga, Elo, rankings
.venv\Scripts\python -m src.etl.validate   # reporte del gate F1
.venv\Scripts\python -m pytest             # suite de tests
```

## Estado

- [x] Documento técnico de diseño (v1.0 — 2026-06-10)
- [x] Fase 1 — Setup e ingesta de datos (gate PASS: 49 398 partidos, Elo r=0.972 vs eloratings.net)
- [x] Fase 2 — EDA y feature engineering (gate PASS: 19/19 tests, matriz de features 49k×60, anti-leakage verificado)
- [x] Fase 3 — Modelos estadísticos (Poisson, Dixon-Coles, baseline Elo, métricas RPS/log-loss/Brier, backtest temporal 2018; 34/34 tests — ver `outputs/reports/f3_model_comparison.md`)
- [ ] Fase 4 — Modelos ML y ensamble
- [ ] Fase 5 — Simulador Monte Carlo
- [ ] Fase 6 — Validación histórica
- [ ] Fase 7 — Predicción 2026, dashboard y evaluación final
