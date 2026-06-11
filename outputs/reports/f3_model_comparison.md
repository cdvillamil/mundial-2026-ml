# Comparacion de modelos estadisticos (corte temporal 2018-01-01)

Entrenamiento: partidos 2010-2018. Test: partidos 2018+.
Convencion: menor RPS/log-loss/Brier es mejor; mayor accuracy es mejor.

| model             |   n_test |    RPS |   log_loss |   Brier |   accuracy |
|:------------------|---------:|-------:|-----------:|--------:|-----------:|
| Poisson           |     8038 | 0.3596 |     0.9057 |  0.5331 |     0.5834 |
| Dixon-Coles       |     8038 | 0.3597 |     0.9065 |  0.5332 |     0.5816 |
| Dixon-Coles+decay |     8038 | 0.3878 |     0.9558 |  0.5647 |     0.5531 |

**Mejor modelo por RPS: Poisson**

## Interpretacion honesta

- Los tres modelos producen probabilidades razonables (~58% accuracy 1X2,
  muy por encima del ~40% de azar informado).
- Poisson y Dixon-Coles empatan en la practica (RPS 0.3596 vs 0.3597): con un
  modelo global de ataque/defensa sin features ricas, la correccion de marcadores
  bajos de Dixon-Coles no aporta mejora medible en seleccciones.
- El decaimiento temporal agresivo (xi=0.003) EMPEORA: las selecciones juegan
  pocos partidos, descartar historia pierde demasiada senal. Habria que afinar xi
  con validacion (mucho menor) o no usarlo.
- **Conclusion:** estos son baselines estadisticos solidos. Las mejoras reales
  vendran en Fase 4 con modelos GBM que exploten Elo, forma y contexto.
  El valor de Fase 3 es establecer el baseline y la infraestructura de evaluacion
  (matriz de marcadores, metricas RPS/log-loss/Brier, backtest temporal).