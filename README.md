# Pronóstico de precio eléctrico — notebooks reproducibles

Repositorio mínimo para ejecutar los notebooks de **EDA** y **modelos clásicos (TVT)** del trabajo de grado.

## Estructura

```
electricity_forecast_github/
├── README.md
├── requirements.txt
├── ENSO_2010_2025_manual.csv      # serie ENSO (merge en dataset)
└── electricity_forecast/          # raíz del proyecto Python
    ├── xm_connection.py           # marcador de raíz + cliente API XM
    ├── *.py                       # módulos de modelado
    ├── notebooks/
    │   ├── 03_eda_exploratorio.ipynb
    │   └── 10_modelos_clasicos_tvt.ipynb
    └── data/processed/
        ├── xm_daily_max_price_dataset.parquet
        └── notebook_models/10_modelos_clasicos_tvt/   # caché resultados TVT
```

## Requisitos

- Python 3.10+ (recomendado 3.11)
- Jupyter Lab o Notebook

```bash
cd electricity_forecast_github
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate     # Linux/macOS
pip install -r requirements.txt
```

## Ejecución

1. Abrir Jupyter con **directorio de trabajo** en `electricity_forecast/` (donde está `xm_connection.py`):

```bash
cd electricity_forecast
jupyter lab
```

2. **Notebook 03 — EDA:** ejecutar todas las celdas en orden. Usa el parquet en `data/processed/`.

3. **Notebook 10 — Modelos clásicos TVT:**
   - Por defecto `TVT_LOAD_RESULTS_PARQUET=True` carga resultados guardados (SARIMAX, VAR, GARCH, híbrido, N-HiTS).
   - `TVT_VAR_FORCE_REFIT=False` — no reentrena VAR (métricas idénticas al parquet).
   - §4.b N-HiTS se omite si las columnas ya vienen del parquet.
   - Los notebooks están **sin salidas embebidas** (repo liviano); ejecute *Run All* para generar gráficas.
   - Para **reentrenar todo** desde cero: `TVT_LOAD_RESULTS_PARQUET=False` en la celda de caché.

Orden recomendado notebook 10: celdas de configuración → caché → §3–§6 (§1–§2 y §5 se saltan con parquet).

## Datos

| Archivo | Descripción |
|---------|-------------|
| `data/processed/xm_daily_max_price_dataset.parquet` | Panel diario 2015–2025 (precio ponderado + covariables) |
| `ENSO_2010_2025_manual.csv` | Índice ENSO (ruta esperada: carpeta padre de `electricity_forecast/`) |
| `data/processed/notebook_models/10_modelos_clasicos_tvt/*.parquet` | Resultados OOS val/test por modelo |

Sin el parquet, los notebooks intentan descargar vía API XM (`servapibi.xm.com.co`); requiere red y es más lento.

## Objetivo de modelado

Columna por defecto: `precio_prom_ponderado_cop_kwh` (promedio diario ponderado por demanda).

## Licencia / uso académico

Material de trabajo de grado. Citar fuente de datos XM Colombia al publicar figuras.
