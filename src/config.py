"""Configuración del proyecto: rutas, variables y semillas."""

DATA_URL = (
    "https://raw.githubusercontent.com/IvTole/Intro_IA_ML_CUGDL/"
    "refs/heads/main/data/housing/housing.csv"
)
TARGET = "median_house_value"
CATEGORICAL_COLS = ["ocean_proximity"]

RANDOM_STATE = 42
TEST_SIZE = 0.20   # 20 % del total para prueba
VALID_SIZE = 0.25  # 25 % del desarrollo -> 20 % del total para validación

FIGURES_DIR = "figures"
