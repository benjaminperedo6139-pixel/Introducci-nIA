import marimo

__generated_with = "0.25.0"
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
