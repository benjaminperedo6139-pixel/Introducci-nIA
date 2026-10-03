"""Flujo principal: carga -> división -> variables -> entrenamiento -> selección -> prueba."""
import pandas as pd

from src.config import FIGURES_DIR
from src.data_loader import cargar_datos, separar_X_y
from src.evaluation import calcular_metricas, evaluar_modelo, seleccionar_mejor
from src.features import crear_variables, graficar_exploracion
from src.models import obtener_modelos, obtener_referencia
from src.preprocessing import construir_pipeline, obtener_columnas_numericas
from src.split import dividir_datos


def main():
    # 1. Carga y separación
    datos = cargar_datos()
    X, y = separar_X_y(datos)
    X_train, X_valid, X_test, y_train, y_valid, y_test = dividir_datos(X, y)

    # 2. Variables nuevas y exploración (solo train)
    X_train_feat = crear_variables(X_train)
    X_valid_feat = crear_variables(X_valid)
    graficar_exploracion(X_train_feat, y_train, FIGURES_DIR)
    columnas_numericas = obtener_columnas_numericas(X_train_feat)

    # 3. Referencia y entrenamiento de todos los modelos
    resultados, entrenados = {}, {}
    base = obtener_referencia().fit(X_train_feat, y_train)
    resultados["Referencia: mediana"] = evaluar_modelo(base, X_train_feat, y_train, X_valid_feat, y_valid)

    for nombre, estimador in obtener_modelos().items():
        pipe = construir_pipeline(estimador, columnas_numericas)
        pipe.fit(X_train_feat, y_train)
        entrenados[nombre] = pipe
        resultados[nombre] = evaluar_modelo(pipe, X_train_feat, y_train, X_valid_feat, y_valid)

    tabla = pd.DataFrame(resultados).T
    print("Comparación de modelos:")
    print(tabla.round(2), "\n")

    # 4. Selección con validación (se excluye la referencia)
    eleccion_final = seleccionar_mejor(tabla.drop(index="Referencia: mediana"))
    print(f"Modelo seleccionado por validación: {eleccion_final}\n")

    # 5. Evaluación final en prueba (una sola vez)
    pred_test = entrenados[eleccion_final].predict(crear_variables(X_test))
    m_test = calcular_metricas(y_test, pred_test)
    print(f"Evaluación final: {eleccion_final}")
    print(f"- MAE de prueba:  {m_test['MAE']:,.2f} dólares")
    print(f"- RMSE de prueba: {m_test['RMSE']:,.2f} dólares")


if __name__ == "__main__":
    main()
