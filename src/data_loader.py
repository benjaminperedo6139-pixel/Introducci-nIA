"""Carga de datos, revisión y separación de predictores y objetivo."""
import pandas as pd

from src.config import DATA_URL, TARGET


def cargar_datos(url=DATA_URL):
    # Requiere internet la carga de datos
    datos_originales = pd.read_csv(url)
    print(f"Filas: {len(datos_originales):,} · Columnas: {datos_originales.shape[1]} (incluyen el objetivo)")
    print(
        datos_originales.dtypes.rename("tipo").to_frame().join(
            datos_originales.isna().sum().rename("faltantes")
        )
    )

    # Restringimos el análisis a registros completos
    datos = datos_originales.dropna().copy()
    filas_eliminadas = len(datos_originales) - len(datos)
    print(f"Se eliminaron {filas_eliminadas} filas ({filas_eliminadas / len(datos_originales):.2%}).\n")
    return datos


def separar_X_y(datos):
    X = datos.drop(columns=TARGET)
    y = datos[TARGET]
    return X, y
