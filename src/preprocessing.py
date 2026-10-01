"""Preprocesamiento integrado con el modelo mediante un Pipeline."""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_COLS


def crear_variables(tabla):
    resultado = tabla.copy()
    resultado["bedrooms_per_household"] = (
        resultado["total_bedrooms"] / resultado["households"].replace(0, 1)
    )
    resultado["rooms_per_household"] = (
        resultado["total_rooms"] / resultado["households"].replace(0, 1)
    )
    return resultado


def obtener_columnas_numericas(X_train_feat):
    return X_train_feat.select_dtypes(include="number").columns.tolist()


def construir_pipeline(estimador, columnas_numericas):
    # Cada entrenamiento recibe un preprocesador nuevo; se ajusta solo con train
    preprocesador = ColumnTransformer([
        ("numericas", StandardScaler(), columnas_numericas),
        ("categoricas", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL_COLS),
    ])
    return Pipeline([("preprocesamiento", preprocesador), ("modelo", estimador)])
