"""División en entrenamiento, validación y prueba."""
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TEST_SIZE, VALID_SIZE


def dividir_datos(X, y):
    # Primero se aparta prueba; se reserva hasta la evaluación final
    X_desarrollo, X_test, y_desarrollo, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    # El desarrollo se divide en entrenamiento y validación
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_desarrollo, y_desarrollo, test_size=VALID_SIZE, random_state=RANDOM_STATE
    )
    verificar_particion(X, X_train, X_valid, X_test, y_train, y_valid)
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def verificar_particion(X, X_train, X_valid, X_test, y_train, y_valid):
    assert X_train.index.equals(y_train.index)
    assert X_valid.index.equals(y_valid.index)
    assert set(X_train.index).isdisjoint(X_valid.index)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert set(X_valid.index).isdisjoint(X_test.index)
    assert len(X_train) + len(X_valid) + len(X_test) == len(X)
    assert abs(len(X_train) / len(X) - 0.60) < 0.01, "Revisa la proporción de entrenamiento."

    print(pd.DataFrame({
        "Conjunto": ["Entrenamiento", "Validación", "Prueba"],
        "Filas": [len(X_train), len(X_valid), len(X_test)],
        "Fracción": [len(X_train) / len(X), len(X_valid) / len(X), len(X_test) / len(X)],
    }).to_string(index=False), "\n")


if __name__ == "__main__":
    from src.data_loader import cargar_datos, separar_X_y
    X, y = separar_X_y(cargar_datos())
    dividir_datos(X, y)
    print("División correcta")
