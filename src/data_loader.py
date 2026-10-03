"""Carga de datos, revisión y separación de predictores y objetivo."""
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_URL, RANDOM_STATE, TARGET, TEST_SIZE, VALID_SIZE


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


def dividir_datos(X, y):
    X_desarrollo, X_test, y_desarrollo, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_desarrollo, y_desarrollo, test_size=VALID_SIZE, random_state=RANDOM_STATE
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test
