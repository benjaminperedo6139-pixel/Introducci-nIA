"""Creación de variables y exploración (solo con entrenamiento)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def crear_variables(tabla):
    # Transformación fila por fila: no aprende de los datos, no hay fuga
    resultado = tabla.copy()
    resultado["bedrooms_per_household"] = (
        resultado["total_bedrooms"] / resultado["households"].replace(0, 1)
    )
    resultado["rooms_per_household"] = (
        resultado["total_rooms"] / resultado["households"].replace(0, 1)
    )
    return resultado


def graficar_exploracion(X_train_feat, y_train, carpeta="figures"):
    Path(carpeta).mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(X_train_feat["rooms_per_household"], y_train, alpha=0.15, s=8)
    ax.set(xlabel="Habitaciones por hogar", ylabel="Valor mediano (dólares)",
           title="Exploración del conjunto de entrenamiento")
    ruta = Path(carpeta) / "exploracion_train.png"
    fig.savefig(ruta, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfica guardada en {ruta}\n")
