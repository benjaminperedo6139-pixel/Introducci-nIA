"""Métricas de clasificación, evaluación por cortes temporales, selección y matriz de confusión."""
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def calcular_metricas(y_real, y_pred):
    """Calcula métricas clave para clasificación binaria (target P_24h)."""
    return {
        "F1": f1_score(y_real, y_pred, zero_division=0),
        "Precision": precision_score(y_real, y_pred, zero_division=0),
        "Recall": recall_score(y_real, y_pred, zero_division=0),
        "Accuracy": accuracy_score(y_real, y_pred),
    }


def evaluar_modelo(modelo, X_train, y_train, X_valid, y_valid):
    """Evalúa un modelo ya entrenado en un corte de entrenamiento y validación."""
    m_train = calcular_metricas(y_train, modelo.predict(X_train))
    m_valid = calcular_metricas(y_valid, modelo.predict(X_valid))
    return {
        "F1 train": m_train["F1"],
        "F1 valid": m_valid["F1"],
        "Precision valid": m_valid["Precision"],
        "Recall valid": m_valid["Recall"],
        "Accuracy valid": m_valid["Accuracy"],
    }


def evaluar_cortes(modelos_dict, cortes, construir_pipeline_fn=None):
    """
    Evalúa todos los modelos y configuraciones en los mismos cortes temporales.

    Parámetros:
    - modelos_dict: dict {nombre_config: estimador}
    - cortes: lista de tuplas [(X_train, y_train, X_valid, y_valid), ...]
    - construir_pipeline_fn: función opcional construir_pipeline(estimador, X_train)
    para ajustar el preprocesamiento únicamente con entrenamiento en cada corte.
    """
    registros = []

    for nombre, estimador_base in modelos_dict.items():
        for idx_corte, (X_tr, y_tr, X_val, y_val) in enumerate(cortes, start=1):
            estimador = clone(estimador_base)
            modelo = (
                construir_pipeline_fn(estimador, X_tr)
                if construir_pipeline_fn is not None
                else estimador
            )
            modelo.fit(X_tr, y_tr)

            metricas_corte = evaluar_modelo(modelo, X_tr, y_tr, X_val, y_val)
            registros.append({
                "Modelo": nombre,
                "Corte": idx_corte,
                **metricas_corte,
            })

    df_cortes = pd.DataFrame(registros)

    df_resumen = (
        df_cortes.groupby("Modelo", sort=False)
        .agg(
            F1_valid_prom=("F1 valid", "mean"),
            F1_valid_std=("F1 valid", "std"),
            Precision_valid_prom=("Precision valid", "mean"),
            Recall_valid_prom=("Recall valid", "mean"),
            F1_train_prom=("F1 train", "mean"),
        )
        .sort_values(by="F1_valid_prom", ascending=False)
    )

    return df_cortes, df_resumen


def seleccionar_mejor(df_resumen, metrica="F1_valid_prom", excluir=None):
    """Selecciona el mejor modelo según el F1 promedio entre cortes (excluyendo referencia)."""
    candidatos = df_resumen.drop(index=excluir, errors="ignore") if excluir else df_resumen
    return candidatos[metrica].idxmax()


def evaluar_prueba_final(modelo_final, X_desarrollo, y_desarrollo, X_test, y_test, figures_dir="figures"):
    """
    Entrena el modelo ganador con todo el conjunto de desarrollo y lo evalúa
    UNA SOLA VEZ en el bloque de prueba final. Genera y guarda la matriz de confusión.
    """
    modelo_final.fit(X_desarrollo, y_desarrollo)
    y_pred = modelo_final.predict(X_test)

    metricas = calcular_metricas(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    reporte = classification_report(
        y_test,
        y_pred,
        target_names=["No cierra <24h (0)", "Cierra <24h (1)"],
        zero_division=0,
    )

    os.makedirs(figures_dir, exist_ok=True)
    ruta_fig = os.path.join(figures_dir, "matriz_confusion.png")
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No <24h (0)", "Sí <24h (1)"],
    )
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Matriz de Confusión - Bloque de Prueba Final")
    plt.tight_layout()
    plt.savefig(ruta_fig, dpi=150)
    plt.close(fig)

    return {
        "metricas": metricas,
        "matriz_confusion": cm,
        "reporte": reporte,
        "ruta_figura": ruta_fig,
    }