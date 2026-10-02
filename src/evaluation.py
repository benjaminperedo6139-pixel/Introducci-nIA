"""Métricas, comparación y selección de modelos."""
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def calcular_metricas(y_real, y_pred):
    return {
        "MAE": mean_absolute_error(y_real, y_pred),
        "RMSE": root_mean_squared_error(y_real, y_pred),
    }


def evaluar_modelo(modelo, X_train, y_train, X_valid, y_valid):
    m_train = calcular_metricas(y_train, modelo.predict(X_train))
    m_valid = calcular_metricas(y_valid, modelo.predict(X_valid))
    return {
        "MAE train": m_train["MAE"], "RMSE train": m_train["RMSE"],
        "MAE valid": m_valid["MAE"], "RMSE valid": m_valid["RMSE"],
    }


def seleccionar_mejor(tabla, metrica="MAE valid"):
    # La selección se hace SOLO con validación
    return tabla[metrica].idxmin()
