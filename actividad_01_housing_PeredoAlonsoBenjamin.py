import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    # Scikit - Learn
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.dummy import DummyRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, root_mean_squared_error

    return (
        ColumnTransformer,
        DecisionTreeRegressor,
        DummyRegressor,
        LinearRegression,
        OneHotEncoder,
        Pipeline,
        RandomForestRegressor,
        StandardScaler,
        mean_absolute_error,
        mo,
        pd,
        plt,
        root_mean_squared_error,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Actividad 1 · Flujo de trabajo de Machine Learning

    **Machine Learning y procesamiento de datos · Trabajo individual**

    **Nombre:**

    Usaremos el conjunto de datos de viviendas trabajado en clase para explicar el flujo de un análisis sencillo, separar los datos correctamente y comparar modelos. Cada fila de los datos describe una zona censal, no una vivienda individual; el objetivo es predecir la variable `median_house_value`, el valor mediano de las viviendas de la zona.

    **Instrucciones**

    1. Completa las cuatro tareas de código marcadas `TODO`.
    2. Escribe tus respuestas en los espacios de las celdas Markdown y sustituye los textos **COMPLETAR** donde aparezcan. Guarda el archivo: las respuestas deben formar parte del notebook, no quedar solo en pantalla.
    3. Usa la misma partición y semilla para todos los modelos. No uses prueba para elegir variables, modelos o hiperparámetros.
    4. Entrega este `.py` con tu nombre y respuestas, incluida la tabla de resultados.
       Antes de entregar, vuelve a abrirlo y comprueba que funciona.

    El código proporcionado también debe poder explicarse con tus palabras.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Cargar y revisar los datos

    Primero inspeccionamos la estructura y los faltantes, sin buscar todavía
    relaciones que nos ayuden a elegir el modelo.

    **Responde:** ¿por qué es un problema de regresión? Escribe dos variables
    predictoras y la variable objetivo. ¿Qué significaría una predicción para
    una fila? ¿Sería correcto interpretarla como el precio de una casa específica?

    **Respuesta:**

    Se considera un problema de regresión porque la variable que queremos estimar es numérica y continua: el valor mediano de las viviendas de cada zona. No estamos intentando asignar una etiqueta o categoría, sino calcular una cantidad.

    Dos variables predictoras pueden ser `total_rooms`, que indica el número total de habitaciones, y `population`, que representa la población de la zona. La variable objetivo es `median_house_value`.

    Para una fila, la predicción sería una estimación del valor mediano de las viviendas que se encuentran en esa zona censal. No debe entenderse como el precio exacto de una casa determinada, ya que los datos están agrupados por zona y no describen propiedades individuales. Por eso, el resultado debe interpretarse como un valor aproximado del mercado de esa área.
    """)
    return


@app.cell
def _(pd):
    # Requiere internet la carga de datos
    url= "https://raw.githubusercontent.com/IvTole/Intro_IA_ML_CUGDL/refs/heads/main/data/housing/housing.csv"
    datos_originales = pd.read_csv(url)
    datos_originales.head()
    return (datos_originales,)


@app.cell
def _(datos_originales, mo):
    mo.vstack([
        mo.md(f"**Filas:** {len(datos_originales):,} · **Columnas:** {datos_originales.shape[1]} (incluyen el objetivo)"),
        datos_originales.dtypes.rename("tipo").to_frame().join(
            datos_originales.isna().sum().rename("faltantes")
        ),
    ])
    return


@app.cell
def _(datos_originales, mo):
    # Para esta primera actividad restringimos el análisis a registros completos.
    # Quitamos filas con datos nulos

    datos = datos_originales.dropna().copy()
    filas_eliminadas = len(datos_originales) - len(datos)
    mo.md(f"Se eliminaron **{filas_eliminadas} filas** ({filas_eliminadas / len(datos_originales):.2%}).")
    return (datos,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Responde:** ¿qué columna tiene faltantes? ¿Qué porcentaje de filas eliminamos?
    ¿Por qué eliminar filas puede cambiar la población que representa el análisis?
    Propón una alternativa y explica con qué conjunto calcularías sus parámetros.

    **Respuesta:**

    Los valores ausentes se encuentran en la columna **`total_bedrooms`**. De los 20,640 registros iniciales, se quitaron 207 filas incompletas, es decir, cerca del 1% del total.

    Aunque parece una cantidad pequeña, eliminar observaciones puede modificar el grupo que estamos estudiando. Si los faltantes se concentran en determinadas zonas, niveles de ingreso o características de las viviendas, el conjunto restante podría dejar de representar correctamente a toda la población original.

    Una alternativa sería completar los valores faltantes mediante la media o, preferentemente, la mediana. Estos valores deben calcularse solamente con el conjunto de entrenamiento y después aplicarse a validación y prueba. También podría utilizarse un modelo para estimar los dormitorios faltantes a partir de otras variables, siempre evitando calcular sus parámetros con información de validación o prueba.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Separar antes de aprender de los datos

    **Tarea de código:** completa la segunda llamada a `train_test_split` para obtener aproximadamente **60% entrenamiento, 20% validación y 20% prueba** del conjunto de registros completos.

    Ya reservamos el 20% para prueba. ¿Qué fracción del 80% restante debes asignar a validación? Sustituye los cuatro `None` por una llamada que use `X_desarrollo`, `y_desarrollo`, `test_size=...` y `random_state=42`.

    **Antes de ejecutar, responde:**

    - ¿Qué se aprende con entrenamiento y qué decisiones se toman con validación?
    - ¿Para qué se reserva prueba y por qué no debemos consultarla repetidamente?
    - ¿Por qué ajustar `StandardScaler` con todos los datos antes de dividirlos
      permite que validación influya en el análisis, aunque no usemos sus etiquetas?

    **Respuesta:**

    - Con el conjunto de entrenamiento (60% de los datos) el modelo realiza el aprendizaje. En esta fase, se ajustan los parámetros del modelo, simultáneamente, durante el entrenamiento se estiman los parámetros del preprocesador, el modelo "ve" e "interpreta patrones" en estas 60% de filas durante el ajuste. Con el conjunto de validación (20% de los datos) se toman decisiones estratégicas sobre qué modelo es mejor. El modelo ya ha sido entrenado y ahora se evalúa en datos que nunca vio durante el aprendizaje. Validación permite comparar diferentes arquitecturas (regresión lineal versus árbol versus bosque), evaluar diferentes valores de hiperparámetros, y elegir el modelo que mejor generaliza. Además, validación nos permite detectar si existe overfitting comparando el error en entrenamiento con el error en validación: si son muy diferentes, el modelo aprendió ruido específico de entrenamiento en lugar de patrones generales

    - El conjunto de prueba (20% restante) se reserva exclusivamente para evaluar el desempeño final del modelo ya elegido en datos completamente nuevos que el modelo nunca ha visto; funciona como simulador del mundo real: mide cómo se comportará el modelo en producción cuando encuentre datos verdaderamente nuevos y desconocidos. Es la evaluación honesta y definitiva del sistema.
    No debemos consultar prueba repetidamente porque cada vez que la miramos para tomar decisiones, la convertimos implícitamente en datos de entrenamiento. Si cambias de modelo porque "se ve mejor en prueba", en realidad estás seleccionando el modelo que overfitea mejor a esos datos específicos de prueba. Este problema se conoce como fuga de información: prueba deja de ser verdaderamente desconocida. Eventualmente, si iteras buscando el mejor desempeño en prueba, lograrás resultados artificialmente buenos que no reflejan la verdadera capacidad del modelo. Por esta razón, prueba debe permanecer intacto hasta el reporte final, y cualquier cambio iterativo debe realizarse únicamente usando validación.

    - Si ajustamos (es decir, ejecutamos .fit()) el StandardScaler con todos los datos antes de dividirlos en entrenamiento, validación y prueba, ocurre una fuga sutil pero importante de información. El scaler calcula la media y la desviación estándar de cada variable numérica usando todas las 20,433 filas completas, incluyendo filas que deberían pertenecer a validación y prueba. Estos estadísticos (media y desv. estándar) reflejan propiedades de la distribución completa, incluyendo información que "se filtra" desde validación y prueba hacia el preprocesador.  Cuando luego aplicamos ese scaler ajustado globalmente a validación, la transformación está calibrada con parámetros que incorporan información de validación misma. Técnicamente, validación influye en cómo se normalizan sus propios datos, violando el principio de independencia. El procedimiento correcto es: primero dividir los datos, luego ajustar (.fit()) StandardScaler únicamente con entrenamiento, y finalmente aplicar (.transform()) esos parámetros aprendidos a validación y prueba. De esta forma, validación y prueba son verdaderamente independientes: sus transformaciones se basan en estadísticos que solo conocen entrenamiento.
    """)
    return


@app.cell
def _(datos, train_test_split):
    X = datos.drop(columns="median_house_value")
    y = datos["median_house_value"]
    X_desarrollo, X_test, y_desarrollo, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    return X, X_desarrollo, X_test, y_desarrollo, y_test


@app.cell
def _(X_desarrollo, train_test_split, y_desarrollo):
    # TODO 1: sustituye esta asignación por train_test_split(...).
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_desarrollo, y_desarrollo, test_size=0.25, random_state=42
    )
    return X_train, X_valid, y_train, y_valid


@app.cell
def _(X, X_test, X_train, X_valid, mo, pd, y_train, y_valid):
    mo.stop(X_train is None, mo.md("⏸ Completa la tarea 1 para continuar."))
    assert X_train.index.equals(y_train.index)
    assert X_valid.index.equals(y_valid.index)
    assert set(X_train.index).isdisjoint(X_valid.index)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert set(X_valid.index).isdisjoint(X_test.index)
    assert len(X_train) + len(X_valid) + len(X_test) == len(X)
    assert abs(len(X_train) / len(X) - 0.60) < 0.01, "Revisa la proporción de entrenamiento."
    particion_lista = True
    pd.DataFrame({
        "Conjunto": ["Entrenamiento", "Validación", "Prueba"],
        "Filas": [len(X_train), len(X_valid), len(X_test)],
        "Fracción": [len(X_train)/len(X), len(X_valid)/len(X), len(X_test)/len(X)],
    })
    return (particion_lista,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Explorar entrenamiento y preparar las variables

    **Tarea de código:** completa la función para crear `rooms_per_household` dividiendo `total_rooms` entre `households`. Conserva las columnas originales. Se utiliza `.replace(0, 1)` en el denominador como regla explícita para evitar una división entre cero.

    Una función permite aplicar la misma operación a cada conjunto. Estas divisiones por fila no estiman parámetros a partir de otras observaciones.
    """)
    return


@app.function
def crear_variables(tabla):
    resultado = tabla.copy()
    resultado["bedrooms_per_household"] = (
        resultado["total_bedrooms"] / resultado["households"].replace(0, 1)
    )
    resultado["rooms_per_household"] = (
        resultado["total_rooms"] / resultado["households"].replace(0, 1)
    )
    
    return resultado


@app.cell
def _(X_train, X_valid, mo, particion_lista):
    # Espera a que la celda de comprobación valide la partición.
    mo.stop(not particion_lista)
    X_train_feat = crear_variables(X_train)
    X_valid_feat = crear_variables(X_valid)
    mo.stop("rooms_per_household" not in X_train_feat,
            mo.md("⏸ Completa la tarea 2 para continuar."))
    X_train_feat.head()
    return X_train_feat, X_valid_feat


@app.cell
def _(X_train_feat, plt, y_train):
    _fig, _ax = plt.subplots(figsize=(7, 4))
    _ax.scatter(X_train_feat["rooms_per_household"], y_train, alpha=0.15, s=8)
    _ax.set(xlabel="Habitaciones por hogar", ylabel="Valor mediano (dólares)",
            title="Exploración del conjunto de entrenamiento")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Responde:**

    - ¿Qué aporta una razón por hogar frente a un total?
    - Identifica valores extremos en la gráfica y propón una posible explicación. Inspecciona las filas correspondientes del conjunto de entrenamiento: ¿qué observas en `total_rooms` y `households`?
    - ¿Hay valores cero en `households` en entrenamiento? ¿Qué implicaría sustituir cero por uno al calcular las razones? Explica una limitación de esta regla, aunque no encuentres ceros.
    - ¿Por qué aplicamos exactamente la misma función a entrenamiento y validación?


    **Respuesta:**

    - Una razón como `rooms_per_household` permite comparar zonas grandes y pequeñas de una forma más equilibrada. Un total elevado de habitaciones puede explicarse simplemente porque hay muchos hogares; al dividirlo entre `households`, obtenemos una medida promedio que describe mejor la situación de cada zona.

    - En la gráfica pueden observarse puntos extremos con cantidades poco comunes de habitaciones por hogar. Una posible causa es que existan pocas viviendas registradas junto con un número alto de habitaciones, o que haya algún registro inusual. Estos casos conviene inspeccionarlos porque pueden influir demasiado en el aprendizaje. Además, la gráfica muestra que una mayor cantidad de habitaciones no garantiza por sí sola un valor de vivienda más alto.

    - Se utiliza `.replace(0, 1)` para impedir una división entre cero. Sin embargo, la solución tiene una desventaja: puede producir una razón artificial. Por ejemplo, 80 habitaciones divididas entre un hogar reemplazado por uno darían 80 habitaciones por hogar, aunque originalmente no hubiera hogares registrados. Sería mejor identificar esos casos y tratarlos de manera especial.

    - La misma función debe aplicarse a entrenamiento y validación para que las variables tengan una definición consistente. Así, el modelo aprende con una transformación y se evalúa con exactamente la misma lógica. En este caso la función realiza cálculos por fila y no estima parámetros globales, por lo que no introduce fuga de información.

    El preprocesamiento siguiente estandariza las variables numéricas y codifica `ocean_proximity`. `Pipeline` une estas operaciones con el modelo: al llamar a `fit(X_train_feat, y_train)`, todo se ajusta solo con entrenamiento. Al llamar a `predict`, se reutilizan esas transformaciones.
    """)
    return


@app.cell
def _(
    ColumnTransformer,
    OneHotEncoder,
    Pipeline,
    StandardScaler,
    X_train_feat,
):
    columnas_numericas = X_train_feat.select_dtypes(include="number").columns.tolist()

    def construir_pipeline(estimador):
        # Cada entrenamiento recibe un preprocesador nuevo.
        preprocesador = ColumnTransformer([
            ("numericas", StandardScaler(), columnas_numericas),
            ("categoricas", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             ["ocean_proximity"]),
        ])
        return Pipeline([("preprocesamiento", preprocesador), ("modelo", estimador)])

    return (construir_pipeline,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Elegir y entrenar con un botón

    **Tarea de código:** agrega un tercer modelo al diccionario siguiendo el patrón proporcionado: un `DecisionTreeRegressor(max_depth=6, random_state=42)` con la etiqueta `Árbol de decisión`.

    El selector está envuelto en `.form()`: cambiar la selección no envía todavía su valor a Python. El botón **Entrenar** confirma la elección. El entrenamiento lee únicamente `formulario_modelo.value` y espera mientras sea `None`.

    Prueba la automatización: entrena regresión lineal, cambia el selector a bosque y observa que los resultados siguen mostrando regresión lineal. Pulsa **Entrenar** y comprueba que ahora muestran bosque. Si editas código o datos de los que depende el entrenamiento, marimo sí puede volver a ejecutar las celdas: el formulario controla los cambios del selector.
    """)
    return


@app.cell
def _(DecisionTreeRegressor, LinearRegression, RandomForestRegressor):
    modelos = {
        "Regresión lineal": LinearRegression(),
        "Bosque aleatorio": RandomForestRegressor(n_estimators=100, random_state=42),
        "Árbol de decisión": DecisionTreeRegressor(max_depth=6, random_state=42),
    }
    return (modelos,)


@app.cell
def _(mo, modelos):
    formulario_modelo = mo.ui.dropdown(
        options=list(modelos), value="Regresión lineal", label="Modelo",
    ).form(submit_button_label="Entrenar", clear_on_submit=False)
    formulario_modelo
    return (formulario_modelo,)


@app.cell
def _(
    X_train_feat,
    X_valid_feat,
    construir_pipeline,
    formulario_modelo,
    mo,
    modelos,
    y_train,
):
    mo.stop(formulario_modelo.value is None,
            mo.md("Selecciona un modelo y pulsa **Entrenar**."))
    nombre_entrenado = formulario_modelo.value
    modelo_entrenado = construir_pipeline(modelos[nombre_entrenado])
    modelo_entrenado.fit(X_train_feat, y_train)
    pred_train = modelo_entrenado.predict(X_train_feat)
    pred_valid = modelo_entrenado.predict(X_valid_feat)
    return nombre_entrenado, pred_train, pred_valid


@app.cell
def _(
    DummyRegressor,
    X_train_feat,
    X_valid_feat,
    mean_absolute_error,
    mo,
    nombre_entrenado,
    pd,
    pred_train,
    pred_valid,
    root_mean_squared_error,
    y_train,
    y_valid,
):
    _base = DummyRegressor(strategy="median")
    _base.fit(X_train_feat, y_train)
    _pred_base = _base.predict(X_valid_feat)
    mo.vstack([
        mo.md(f"**Último modelo entrenado: {nombre_entrenado}**"),
        pd.DataFrame({
            "Evaluación": ["Referencia: mediana · validación", "Modelo · entrenamiento", "Modelo · validación"],
            "MAE": [mean_absolute_error(y_valid, _pred_base),
                    mean_absolute_error(y_train, pred_train), mean_absolute_error(y_valid, pred_valid)],
            "RMSE": [root_mean_squared_error(y_valid, _pred_base),
                     root_mean_squared_error(y_train, pred_train), root_mean_squared_error(y_valid, pred_valid)],
        }),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Comparar e interpretar

    Ejecuta los tres modelos y transcribe sus resultados. La salida del widget muestra la última ejecución; esta tabla conserva tu comparación en el archivo. Usaremos **MAE de validación** como criterio de selección.

    | Modelo | MAE entrenamiento | MAE validación | RMSE validación |
    |---|---|---|---|
    | Referencia: mediana | No requerido | 86,833.33 | 117,527.45 |
    | Regresión lineal | 49,195.14 | 50,383.50 | 71,859.24 |
    | Árbol de decisión | 45,788.84 | 48,081.18 | 68,629.32 |
    | Bosque aleatorio | 12,479.80 | 33,198.46 | 50,191.03 |

    **Responde:**

    1. Interpreta el MAE de un modelo en dólares y en el contexto de estas zonas.

    -R: Un MAE de **$33,198.46** indica que, en promedio, el valor estimado para una zona se separa del valor real en aproximadamente 33 mil dólares. El error puede ser mayor o menor en casos particulares. Aunque todavía representa una diferencia importante, es inferior al error obtenido por los otros modelos y por la predicción de referencia.

    2. ¿Cuánto reduce el MAE frente a predecir siempre la mediana de entrenamiento?

    -R: La diferencia entre ambos errores es de **$53,634.87**. En términos relativos, esto equivale aproximadamente a una reducción del **61.8%** frente a utilizar siempre la mediana como predicción. Por lo tanto, el bosque aleatorio disminuye considerablemente el error de referencia.

    3. ¿Qué sugiere un error de entrenamiento mucho menor que el de validación?

    -R: Una diferencia muy grande entre el error de entrenamiento y el de validación puede indicar **sobreajuste**. En este caso, el bosque obtiene un MAE de entrenamiento de $12,479.80, pero aumenta a $33,198.46 en validación. Esto sugiere que aprendió detalles específicos del conjunto de entrenamiento que no se mantienen de la misma manera en datos nuevos.

    4. Elige un modelo usando la tabla y justifica tu elección antes de abrir prueba.

    -R: Seleccionaría el **Bosque aleatorio**, porque presentó el MAE de validación más bajo: $33,198.46, frente a $48,081.18 del árbol y $50,383.50 de la regresión lineal. También obtuvo el RMSE de validación menor. Aunque existe una diferencia notable entre entrenamiento y validación, sus resultados en datos no utilizados para ajustar fueron los más favorables.

    5. Explica qué hacen `fit` y `predict`, y por qué cambiar el selector no entrena
       hasta pulsar el botón. Describe lo que observaste al comprobarlo.

    -R: `fit` es el método que utiliza los datos de entrenamiento para ajustar los parámetros del modelo. `predict`, por su parte, toma lo aprendido y genera estimaciones para nuevas observaciones sin volver a entrenar.

    -R: El selector está dentro de un formulario, por lo que cambiar la opción solamente modifica la selección visible. El valor no se envía al programa hasta presionar **Entrenar**. Al hacer la prueba, cambiar de regresión lineal a bosque sin pulsar el botón mantuvo los resultados anteriores; después de presionar el botón, sí se ejecutó el modelo seleccionado.

    **Respuestas:**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Evaluación final en prueba

    **Tarea de código:** después de escribir tu justificación, sustituye `None` por la etiqueta exacta del modelo elegido. Esta elección es independiente del selector anterior. Después pulsa **Evaluar en prueba**.

    Para mantener sencillo el ejercicio, ajustaremos un pipeline nuevo únicamente con entrenamiento. No incorporaremos validación al ajuste final en esta tarea. No cambies la elección después de ver prueba: sus resultados se reportan, no se utilizan para seguir buscando el mejor modelo.
    """)
    return


@app.cell
def _():
    # TODO 4: escribe la etiqueta del modelo que justificaste con validación.
    eleccion_final = "Bosque aleatorio"
    return (eleccion_final,)


@app.cell
def _(eleccion_final, mo, modelos):
    mo.stop(eleccion_final is None, mo.md("⏸ Completa la comparación y la tarea 4."))
    assert eleccion_final in modelos, "Bosque aleatorio"
    confirmar_prueba = mo.ui.checkbox(
        label=f"Ya justifiqué mi elección: {eleccion_final}",
    ).form(submit_button_label="Evaluar en prueba")
    confirmar_prueba
    return (confirmar_prueba,)


@app.cell
def _(
    X_test,
    X_train_feat,
    confirmar_prueba,
    construir_pipeline,
    eleccion_final,
    mean_absolute_error,
    mo,
    modelos,
    root_mean_squared_error,
    y_test,
    y_train,
):
    mo.stop(confirmar_prueba.value is not True,
            mo.md("Prueba permanece reservada. Confirma tu elección para evaluarla."))
    _final = construir_pipeline(modelos[eleccion_final])
    _final.fit(X_train_feat, y_train)
    _pred_test = _final.predict(crear_variables(X_test))
    mo.md(f"""
    **Evaluación final: {eleccion_final}**

    - MAE de prueba: **{mean_absolute_error(y_test, _pred_test):,.2f} dólares**.
    - RMSE de prueba: **{root_mean_squared_error(y_test, _pred_test):,.2f} dólares**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Reporte final:** modelo elegido: Bosque aleatorio · MAE prueba: 33,697.54 · RMSE prueba: 50,964.58.

    **Responde:** ¿cómo se compara el error de prueba con el de validación? ¿Por qué no tienen que coincidir? Si el error de prueba es mayor, ¿por qué cambiar repetidamente de modelo mirando prueba dejaría de ser una evaluación final?

    **Respuesta:**

    R: El MAE de prueba fue de **$33,697.54**, mientras que en validación fue de **$33,198.46**. Ambos valores son bastante cercanos, aunque no idénticos, porque se calcularon sobre grupos diferentes de observaciones. Es normal que existan pequeñas variaciones entre los resultados.

    Si después de consultar prueba cambiáramos repetidamente de modelo para mejorar ese número, estaríamos utilizando la prueba como otro conjunto de validación. Dejaría de ser una medición independiente y el resultado final podría parecer mejor de lo que realmente sería al enfrentarse a datos nuevos.

    **Explica el análisis completo** en 6–8 oraciones: carga y revisión, separación, exploración, creación de variables, preprocesamiento, entrenamiento, selección y evaluación final. En cada paso indica para qué sirve y qué conjunto utiliza.

    **Explicación:**

    -R: Primero se cargaron los datos y se revisó su estructura para detectar valores faltantes. Después se eliminaron los registros incompletos y se dividieron los datos en entrenamiento, validación y prueba, manteniendo la prueba apartada desde el principio. Con entrenamiento se exploraron las variables y se generaron nuevas características, como las habitaciones por hogar, aplicando la misma transformación a los conjuntos correspondientes. Posteriormente, el pipeline estandarizó las variables numéricas y convirtió la variable categórica en valores utilizables por los modelos, ajustándose únicamente con entrenamiento. Se entrenaron una regresión lineal, un árbol de decisión y un bosque aleatorio, y sus resultados se compararon mediante MAE y RMSE en validación. El bosque aleatorio fue seleccionado por presentar el menor error de validación, aunque mostró indicios de sobreajuste. Finalmente, se evaluó el modelo elegido con el conjunto de prueba y se obtuvo un error parecido al de validación, lo que sugiere un comportamiento relativamente consistente con datos no vistos.

    ## Para una clase posterior · Ubicación

    No es requisito de esta entrega. Propón una variable derivada de `latitude` y `longitude`: por ejemplo, distancia a un punto de referencia fijo. ¿Qué hipótesis representa? ¿Cómo compararías con y sin ella usando validación? Si defines el punto o agrupas zonas aprendiendo de los datos, ¿con qué conjunto debes hacerlo? ¿Una división aleatoria mide necesariamente el desempeño en
    regiones geográficas completamente nuevas?

    ## Criterios de evaluación

    | Criterio | Puntos |
    |---|---:|
    | Separación correcta y explicación de entrenamiento, validación y prueba | 25 |
    | Variable nueva, revisión de datos y explicación del preprocesamiento | 20 |
    | Tercer modelo y comprobación del selector con botón | 20 |
    | Comparación, interpretación y reporte final de métricas | 25 |
    | Explicación del flujo completo y notebook guardado con respuestas | 10 |

    **Antes de entregar:** comprueba los cuatro TODO, las respuestas, la tabla y
    el reporte final. Los errores pequeños no dan más puntos por sí solos:
    se evalúan el procedimiento y la interpretación.

    **Consulta:** [formularios de marimo](https://docs.marimo.io/api/inputs/form/)
    · [fuga de información y buenas prácticas](https://scikit-learn.org/stable/common_pitfalls.html).
    """)
    return


if __name__ == "__main__":
    app.run()