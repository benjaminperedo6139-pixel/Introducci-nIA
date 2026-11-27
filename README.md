# Predicción de Cierre en Menos de 24 Horas ($P_{24h}$) en Solicitudes NYC 311

Proyecto modular de Machine Learning para predecir, al momento de registrar una solicitud de servicio ciudadano en **NYC 311**, si esta será cerrada en menos de 24 horas ($P_{24h}$), utilizando validación temporal y una arquitectura modular coordinada mediante GitHub.

## 1. Objetivo y definición del problema

El objetivo es construir un modelo de **clasificación binaria** capaz de estimar la probabilidad de que un reporte registrado en el sistema NYC 311 se resuelva dentro de las primeras 24 horas posteriores a su creación.

- **Variable objetivo ($P_{24h}$):**
  - `1` (Positivo): La solicitud se cerró en un tiempo estrictamente menor a 24 horas ($\text{Closed Date} - \text{Created Date} < 24\text{ h}$).
  - `0` (Negativo): La solicitud tardó 24 horas o más en cerrarse, o permanecía abierta al momento del corte.

### Prevención de fuga de información (Data Leakage)
Para simular una predicción real al instante del registro (`Created Date`):
1. **Variables disponibles:** Únicamente se utilizan atributos conocidos al momento de levantar el reporte (agencia responsable, tipo de queja, descriptor, ubicación/distrito, canal de ingreso, hora y día de creación). Se excluyen variables posteriores al cierre como `Closed Date`, `Resolution Description` o `Status`.
2. **Integridad temporal de las etiquetas:** En cada corte de entrenamiento, solo se incluyen solicitudes cuya etiqueta ya era observable antes de la fecha de inicio del conjunto de validación (es decir, solicitudes creadas al menos 24 horas antes del corte de validación), evitando conocer desenlaces del futuro.

## 2. Conjunto de datos

Se utiliza el registro histórico de solicitudes de servicio **NYC 311 Service Requests** (NYC Open Data).

| Característica | Detalle |
|---|---|
| Fuente | NYC Open Data — 311 Service Requests |
| Periodo analizado | `[Fecha inicio] a [Fecha fin]` |
| Registros utilizados | `[Total de filas]` |
| Variable objetivo | `P_24h` (binaria: `0` / `1`) |
| Proporción de clases | `[XX.X % clase 1 / XX.X % clase 0]` |

## 3. Integrantes

| Nombre | Usuario de GitHub |
|---|---|
| Jorge Alberto Laureano Corona | [GiorgioCrown](https://github.com/GiorgioCrown) |
| Jose Carlos Isai Huitron Barron | [josehuitron6148-svg](https://github.com/josehuitron6148-svg) |
| Benjamín Peredo Alonso | [benjaminperedo6139-pixel](https://github.com/benjaminperedo6139-pixel) |
| Luis Angel Pardave Garcia | [luispardave](https://github.com/luispardave) |

## 4. Organización del proyecto
Introducci-nIA/
├── notebooks/
│   └── notebook_original.py   # Exploración inicial de referencia
├── src/
│   ├── init.py            # Marca src/ como paquete de Python
│   ├── config.py              # Rutas, columnas, parámetros de cortes y semilla
│   ├── data_loader.py         # Carga, limpieza y construcción del target P_24h
│   ├── split.py               # División temporal (cortes deslizantes + prueba final)
│   ├── features.py            # Ingeniería de variables temporales y de contexto
│   ├── preprocessing.py       # ColumnTransformer y Pipeline (ajustado solo en train)
│   ├── models.py              # Modelo de referencia y modelos con hiperparámetros
│   └── evaluation.py          # Evaluación por cortes, selección por F1 y matriz de confusión
├── figures/                   # Gráficas generadas automáticamente (exploración y matriz)
├── train.py                   # Script principal que ejecuta todo el flujo de extremo a extremo
├── requirements.txt           # Dependencias del proyecto
└── README.md                  # Informe técnico final

## 5. Instalación y ejecución

Requiere **Python 3.10 o superior**.

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/benjaminperedo6139-pixel/Introducci-nIA.git](https://github.com/benjaminperedo6139-pixel/Introducci-nIA.git)
   cd Introducci-nIA
   python -m venv .venv
# Windows: .venv\Scripts\activate | macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
Obtención de datos y ejecución completa:
Los datos se descargan/cargan automáticamente mediante src/data_loader.py siguiendo la ruta configurada en src/config.py. Para ejecutar todo el experimento:

python train.py

## 6. Diseño de la validación temporal
Dado que las solicitudes 311 presentan estacionalidad diaria/semanal y cambios operativos a lo largo del tiempo, una división aleatoria provocaría fuga de información del futuro hacia el pasado. Por ello se empleó una validación temporal mediante ventana deslizante (TimeSeriesSplit), reservando el último bloque cronológico como conjunto de prueba final intocable durante el desarrollo.

## 7. Modelos e hiperparámetros evaluados 
DummyClassifier: Predice la clase mayoritaria (strategy="most_frequent"). Sirve como línea base mínima para verificar que los modelos aprenden patrones reales.

Regresión Logística: Modelo lineal interpretable y eficiente con codificación One-Hot. Limitación: no captura interacciones no lineales complejas entre agencia, hora y ubicación. Hiperparámetros probados.

Árbol de Decisión / Bosque Aleatorio (RandomForestClassifier): Captura reglas no lineales e interacciones entre tipo de reporte, agencia y horario sin requerir escalado estricto. Limitación: tiende a sobreajustar si no se limita la profundidad (max_depth). Hiperparámetros probados.


## 8. Métrica principal y resultadosJustificación de la métrica principal: F1-scoreSe eligió el F1-score promedio entre cortes de validación como métrica principal de selección porque equilibra Precision y Recall sobre la clase positiva ($P_{24h} = 1$). A diferencia del Accuracy, el F1-score evita que un modelo parezca exitoso simplemente prediciendo la clase mayoritaria cuando existe desbalance entre cierres rápidos y lentos.

## Significado de cada tipo de error en el contexto de NYC 311.
Falso Positivo (FP — Predice cierre < 24h, pero tarda más): El sistema estima erróneamente una resolución rápida para una solicitud compleja. En la operación de 311, esto genera expectativas incumplidas en el ciudadano y puede provocar que el caso no reciba escalamiento oportuno.

Falso Negativo (FN — Predice cierre >= 24h, pero se resuelve en < 24h): El modelo subestima la capacidad de respuesta de la agencia para un reporte que en realidad se atiende el mismo día. Operativamente tiene menor gravedad para el usuario final, pero distorsiona la planeación de carga de trabajo y asignación de cuadrillas de emergencia.