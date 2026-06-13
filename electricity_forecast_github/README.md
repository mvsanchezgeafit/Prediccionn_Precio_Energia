# Pronóstico de precio eléctrico (Colombia)

Repositorio reproducible del trabajo de grado: **EDA** y **comparación de modelos** (SARIMAX, VAR, GARCH, N-HiTS, híbrido).

## Estructura (simple)

```
├── notebooks/
│   ├── 01_eda.ipynb              ← empezar aquí
│   └── 02_modelos_clasicos.ipynb ← modelos y métricas
├── data/
│   ├── xm_daily_max_price_dataset.parquet
│   ├── ENSO_2010_2025_manual.csv
│   └── cache/                    ← resultados guardados (val/test)
└── src/                          ← código Python (no hace falta editarlo)
```

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
jupyter lab notebooks
```

También puede abrir Jupyter desde la **raíz del repo**; los notebooks encuentran `src/` automáticamente.

## Uso

| Notebook | Qué hace |
|----------|----------|
| **01_eda** | Exploración: estacionariedad, distribución, volatilidad, correlaciones |
| **02_modelos** | Pronóstico 60 días: train/val/test, gráficas comparativas |

El notebook 02 carga resultados desde `data/cache/` (`TVT_LOAD_RESULTS_PARQUET=True`). Ejecute *Run All* para regenerar gráficas (vienen sin salidas embebidas).

Para reentrenar todos los modelos: `TVT_LOAD_RESULTS_PARQUET=False` en la celda de configuración.

## Datos

- Panel diario 2015–2025 (XM Colombia + ENSO)
- Objetivo: `precio_prom_ponderado_cop_kwh` (COP/kWh)

## Créditos

Datos de mercado: XM Colombia. Uso académico.
