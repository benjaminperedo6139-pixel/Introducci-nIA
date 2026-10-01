"""Orquestador de entrenamiento.

Sustituye la parte interactiva de marimo (dropdown, formularios y mo.stop):
entrena todos los modelos, los compara en validación, elige automáticamente
el de menor error y solo entonces evalúa ese modelo en prueba.
"""
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from src.data_loader import cargar_datos, dividir_datos, separar_X_y
from src.models import obtener_modelos, obtener_referencia
from src.preprocessing import construir_pipeline, crear_variables, obtener_columnas_numericas


def validar_particion(X, X_train, X_valid, X_test, y_train, y_valid):
    assert X_train.index.equals(y_train.index)
    assert X_valid.index.equals(y_valid.index)
    assert set(X_train.index).isdisjoint(X_valid.index)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert set(X_valid.index).isdisjoint(X_test.index)
    assert len(X_train) + len(X_valid) + len(X_test) == len(X)
    assert abs(len(X_train) / len(X) - 0.60) < 0.01, "Revisa la proporción de entrenamiento."


def evaluar_modelos(modelos, X_train_feat, X_valid_feat, y_train, y_valid):
    columnas_numericas = obtener_columnas_numericas(X_train_feat)
    resultados = []
    modelos_entrenados = {}
    for nombre, estimador in modelos.items():
        pipeline = construir_pipeline(estimador, columnas_numericas)
        pipeline.fit(X_train_feat, y_train)
        pred_train = pipeline.predict(X_train_feat)
        pred_valid = pipeline.predict(X_valid_feat)
        resultados.append({
            "Modelo": nombre,
            "MAE_train": mean_absolute_error(y_train, pred_train),
            "MAE_valid": mean_absolute_error(y_valid, pred_valid),
            "RMSE_train": root_mean_squared_error(y_train, pred_train),
            "RMSE_valid": root_mean_squared_error(y_valid, pred_valid),
        })
        modelos_entrenados[nombre] = pipeline
    return pd.DataFrame(resultados), modelos_entrenados


def main():
    datos = cargar_datos()
    X, y = separar_X_y(datos)
    X_train, X_valid, X_test, y_train, y_valid, y_test = dividir_datos(X, y)
    validar_particion(X, X_train, X_valid, X_test, y_train, y_valid)

    X_train_feat = crear_variables(X_train)
    X_valid_feat = crear_variables(X_valid)

    referencia = obtener_referencia()
    referencia.fit(X_train_feat, y_train)
    pred_referencia = referencia.predict(X_valid_feat)
    print(f"Referencia (mediana) · MAE validación: {mean_absolute_error(y_valid, pred_referencia):,.2f}\n")

    modelos = obtener_modelos()
    tabla_resultados, modelos_entrenados = evaluar_modelos(
        modelos, X_train_feat, X_valid_feat, y_train, y_valid
    )
    print(tabla_resultados.to_string(index=False))

    mejor_nombre = tabla_resultados.loc[tabla_resultados["MAE_valid"].idxmin(), "Modelo"]
    print(f"\nModelo seleccionado automáticamente (menor MAE de validación): {mejor_nombre}")

    modelo_final = modelos_entrenados[mejor_nombre]
    X_test_feat = crear_variables(X_test)
    pred_test = modelo_final.predict(X_test_feat)
    print(f"\nEvaluación final: {mejor_nombre}")
    print(f"MAE de prueba: {mean_absolute_error(y_test, pred_test):,.2f} dólares")
    print(f"RMSE de prueba: {root_mean_squared_error(y_test, pred_test):,.2f} dólares")


if __name__ == "__main__":
    main()
