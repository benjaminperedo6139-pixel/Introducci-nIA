# Predicción del valor de viviendas en California

Proyecto modular de Machine Learning que reorganiza un notebook de marimo en un proyecto de Python, coordinado mediante GitHub.

## Descripción del problema

El objetivo es predecir el **valor mediano de las viviendas** (`median_house_value`) de distritos de California a partir de características como ubicación, antigüedad de las casas, número de habitaciones, población, ingreso mediano y proximidad al océano. Se trata de un problema de **regresión**.

## Conjunto de datos

Se utiliza el dataset *California Housing*, disponible en el repositorio del curso:

```
https://raw.githubusercontent.com/IvTole/Intro_IA_ML_CUGDL/refs/heads/main/data/housing/housing.csv
```

| Característica | Detalle |
|---|---|
| Registros originales | 20,640 |
| Registros eliminados por valores faltantes | 207 (1.00 %), todos en `total_bedrooms` |
| Registros utilizados | 20,433 |
| Variable objetivo | `median_house_value` |
| Variables numéricas | `longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income` |
| Variable categórica | `ocean_proximity` |

## Integrantes

| Nombre | Usuario de GitHub |
|---|---|
| Jorge Alberto Laureano Corona | [GiorgioCrown](https://github.com/GiorgioCrown) |
| Jose Carlos Isai Huitron Barron | [josehuitron6148-svg](https://github.com/josehuitron6148-svg) |
| Benjamín Peredo Alonso | [benjaminperedo6139-pixel](https://github.com/benjaminperedo6139-pixel) |
| Luis Angel Pardave Garcia | [luispardave](https://github.com/luispardave) |

## Organización de los archivos

```
proyecto-housing/
├── notebooks/
│   └── notebook_original.py   # Notebook de marimo original (referencia)
├── src/
│   ├── __init__.py            # Marca src/ como paquete de Python
│   ├── config.py              # URL de datos, variable objetivo, columnas, semilla y proporciones
│   ├── data_loader.py         # Carga, revisión, eliminación de nulos y separación X / y
│   ├── split.py               # División en entrenamiento, validación y prueba con verificaciones
│   ├── features.py            # Creación de variables nuevas y gráfica de exploración
│   ├── preprocessing.py       # ColumnTransformer y Pipeline (escalado + one-hot)
│   ├── models.py              # Modelos a comparar y modelo de referencia
│   └── evaluation.py          # Métricas (MAE, RMSE) y selección del mejor modelo
├── train.py                   # Archivo principal que coordina todo el flujo
├── requirements.txt           # Dependencias
├── .gitignore
└── README.md
```

Al ejecutar el proyecto se crea automáticamente la carpeta `figures/` con la gráfica de exploración. Esta carpeta no se sube al repositorio.

## Instalación

Se requiere **Python 3.10 o superior** y conexión a internet.

1. Clonar el repositorio:

   ```bash
   git clone https://github.com/USUARIO/NOMBRE-REPO.git
   cd NOMBRE-REPO
   ```

2. (Opcional) Crear y activar un entorno virtual:

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

3. Instalar las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

## Obtención de los datos

No es necesario descargar los datos manualmente. La URL del CSV está definida en `src/config.py` y `src/data_loader.py` lo lee directamente con `pandas` al ejecutar el proyecto.

Si se prefiere trabajar sin conexión, se puede guardar el archivo en `data/housing.csv` y cambiar en `src/config.py`:

```python
DATA_URL = "data/housing.csv"
```

## Ejecución

Desde la raíz del proyecto:

```bash
python train.py
```

Para verificar únicamente la división de los datos:

```bash
python -m src.split
```

## Flujo del proyecto

1. **Carga y revisión** (`data_loader.py`): se leen los datos, se revisan tipos y valores faltantes, y se eliminan los registros incompletos.
2. **División** (`split.py`): los datos se separan en entrenamiento (60 %), validación (20 %) y prueba (20 %). El conjunto de prueba se aparta desde el inicio y solo se utiliza al final. Se verifican con `assert` que los conjuntos no se traslapen y que las proporciones sean correctas.
3. **Creación de variables** (`features.py`): se agregan `rooms_per_household` y `bedrooms_per_household`. Como son operaciones fila por fila que no aprenden de los datos, se aplican a los tres conjuntos sin fuga de información. La exploración gráfica se hace solo con entrenamiento.
4. **Preprocesamiento** (`preprocessing.py`): un `ColumnTransformer` estandariza las variables numéricas y codifica `ocean_proximity` con one-hot. Está integrado con el modelo en un `Pipeline`, por lo que se ajusta **únicamente con entrenamiento**.
5. **Entrenamiento y comparación** (`models.py`, `evaluation.py`): se entrenan todos los modelos y se comparan con MAE y RMSE en entrenamiento y validación.
6. **Selección** (`evaluation.py`): se elige automáticamente el modelo con menor MAE en **validación**.
7. **Evaluación final** (`train.py`): el modelo seleccionado se evalúa una sola vez en **prueba**.

## Modelos utilizados

| Modelo | Configuración |
|---|---|
| Referencia (`DummyRegressor`) | Predice siempre la mediana de entrenamiento |
| Regresión lineal | Valores por defecto |
| Árbol de decisión | `max_depth=6`, `random_state=42` |
| Bosque aleatorio | `n_estimators=100`, `random_state=42` |

## Resultados

Comparación en entrenamiento y validación (valores en dólares):

| Modelo | MAE train | RMSE train | MAE valid | RMSE valid |
|---|---:|---:|---:|---:|
| Referencia: mediana | 88,597.14 | 118,610.21 | 86,833.32 | 117,527.45 |
| Regresión lineal | 49,195.14 | 67,474.04 | 50,383.50 | 71,859.24 |
| Árbol de decisión | 45,788.84 | 64,745.58 | 48,081.18 | 68,629.32 |
| **Bosque aleatorio** | 12,479.80 | 19,067.32 | **33,198.46** | **50,191.03** |

**Modelo seleccionado:** Bosque aleatorio (menor MAE en validación).

Evaluación final en prueba:

| Métrica | Valor |
|---|---:|
| MAE | 33,697.54 dólares |
| RMSE | 50,964.58 dólares |

## Interpretación

Los tres modelos superan claramente a la referencia, lo que indica que las variables sí aportan información para predecir el valor de las viviendas. El bosque aleatorio obtuvo el menor error en validación, aunque la gran diferencia entre su error de entrenamiento (MAE de 12,480) y de validación (MAE de 33,198) muestra **sobreajuste**: memoriza parte del conjunto de entrenamiento. Aun así, generaliza mejor que los otros modelos.

El MAE de prueba (33,697.54) es muy cercano al de validación (33,198.46). No tienen que coincidir porque se calculan sobre observaciones diferentes, pero su cercanía sugiere que el desempeño es consistente con datos no vistos. La prueba se utilizó una sola vez: si se cambiara de modelo después de ver ese resultado, la prueba se convertiría en otro conjunto de validación y dejaría de ser una medición independiente.

## Limitaciones y problemas conocidos

- Se requiere **conexión a internet** para descargar los datos. Si el archivo cambia de ubicación en el repositorio del curso, el proyecto dejará de funcionar hasta actualizar `DATA_URL`.
- Se eliminaron las filas con valores faltantes en lugar de imputarlas. Una imputación dentro del pipeline permitiría conservar esos registros.
- El bosque aleatorio presenta sobreajuste. No se realizó ajuste de hiperparámetros (por ejemplo, limitar `max_depth` o `min_samples_leaf`), lo que podría reducirlo.
- La división es aleatoria, por lo que no mide necesariamente el desempeño en regiones geográficas completamente nuevas.
- El dataset tiene un tope en `median_house_value` (500,001), lo que limita la capacidad de los modelos para predecir valores altos.
- La función `root_mean_squared_error` requiere `scikit-learn` 1.4 o superior.

## Tabla de contribuciones

| Integrante | Rama | Archivos | Pull request | Commit | Revisó el PR de |
|---|---|---|---|---|---|
| @usuario1 | `Módulo-de-Dataset` | `config.py`, `data_loader.py`, `__init__.py`, `requirements.txt`, `.gitignore`, notebook original | #1 | `abc1234` | @usuario4 |
| @usuario2 | `Modulo-DataT_T_V` | `split.py`, `features.py` | #2 | `abc1234` | @usuario1 |
| @usuario3 | `Modulo_Preprocessing` | `preprocessing.py`, `models.py` | #3 | `abc1234` | @usuario2 |
| @usuario4 | `train-evaluation` | `evaluation.py`, `train.py` | #4 | `abc1234` | @usuario3 |

La integración se realizó en la rama `staging` y, una vez verificado el funcionamiento completo, se integró a `main`.
