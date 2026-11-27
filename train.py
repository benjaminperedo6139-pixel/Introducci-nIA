"""Flujo principal: carga -> división temporal -> variables -> cortes -> selección -> prueba final."""
from sklearn.base import clone

from src.config import FIGURES_DIR
from src.data_loader import cargar_datos, separar_X_y
from src.evaluation import (
    evaluar_cortes,
    evaluar_prueba_final,
    seleccionar_mejor,
)
from src.features import crear_variables, graficar_exploracion
from src.models import obtener_modelos, obtener_referencia
from src.preprocessing import construir_pipeline, obtener_columnas_numericas
from src.split import dividir_datos_temporales


def main():
    # 1. Carga y separación de características y target (P_24h)
    datos = cargar_datos()
    X, y = separar_X_y(datos)

    # 2. División temporal: cortes de validación deslizante y bloque de prueba final
    cortes, X_desarrollo, y_desarrollo, X_test, y_test = dividir_datos_temporales(X, y)

    # 3. Creación de variables por corte y exploración solo con el bloque de desarrollo
    cortes_feat = []
    for X_tr, y_tr, X_val, y_val in cortes:
        cortes_feat.append((crear_variables(X_tr), y_tr, crear_variables(X_val), y_val))

    X_des_feat = crear_variables(X_desarrollo)
    X_test_feat = crear_variables(X_test)
    graficar_exploracion(X_des_feat, y_desarrollo, FIGURES_DIR)

    # Función auxiliar para construir el pipeline dentro de cada corte sin fuga de datos
    def armar_pipeline_corte(estimador, X_tr_corte):
        cols_num = obtener_columnas_numericas(X_tr_corte)
        return construir_pipeline(estimador, cols_num)

    # 4. Referencia y evaluación de todos los modelos/configuraciones en los mismos cortes
    nombre_referencia = "Referencia (Dummy)"
    modelos_a_evaluar = {
        nombre_referencia: obtener_referencia(),
        **obtener_modelos(),
    }

    df_cortes, df_resumen = evaluar_cortes(
        modelos_dict=modelos_a_evaluar,
        cortes=cortes_feat,
        construir_pipeline_fn=armar_pipeline_corte,
    )

    print("=== MÉTRICAS POR CORTE TEMPORAL ===")
    print(df_cortes.round(4).to_string(index=False), "\n")

    print("=== RESUMEN PROMEDIO ENTRE CORTES (ORDENADO POR F1 VALIDACIÓN) ===")
    print(df_resumen.round(4).to_string(), "\n")

    # 5. Selección del mejor modelo por F1 promedio entre cortes (excluyendo la referencia)
    eleccion_final = seleccionar_mejor(
        df_resumen,
        metrica="F1_valid_prom",
        excluir=[nombre_referencia],
    )
    print(f"Modelo seleccionado por mayor F1 promedio en validación: {eleccion_final}\n")

    # 6. Evaluación final en el bloque de prueba reservado (UNA SOLA VEZ)
    estimador_ganador = clone(modelos_a_evaluar[eleccion_final])
    modelo_final = armar_pipeline_corte(estimador_ganador, X_des_feat)

    resultado_test = evaluar_prueba_final(
        modelo_final=modelo_final,
        X_desarrollo=X_des_feat,
        y_desarrollo=y_desarrollo,
        X_test=X_test_feat,
        y_test=y_test,
        figures_dir=FIGURES_DIR,
    )

    m_test = resultado_test["metricas"]
    print(f"=== EVALUACIÓN FINAL EN BLOQUE DE PRUEBA: {eleccion_final} ===")
    print(f"- F1-score:  {m_test['F1']:.4f}")
    print(f"- Precision: {m_test['Precision']:.4f}")
    print(f"- Recall:    {m_test['Recall']:.4f}")
    print(f"- Accuracy:  {m_test['Accuracy']:.4f}\n")

    print("Matriz de confusión (filas = real [0, 1], columnas = predicho [0, 1]):")
    print(resultado_test["matriz_confusion"], "\n")

    print("Reporte detallado por clase:")
    print(resultado_test["reporte"])
    print(f"Gráfica guardada en: {resultado_test['ruta_figura']}")


if __name__ == "__main__":
    main()