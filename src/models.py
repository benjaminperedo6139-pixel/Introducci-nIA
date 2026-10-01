"""Modelos a comparar y modelo de referencia."""
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from src.config import RANDOM_STATE


def obtener_modelos():
    return {
        "Regresión lineal": LinearRegression(),
        "Bosque aleatorio": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
        "Árbol de decisión": DecisionTreeRegressor(max_depth=6, random_state=RANDOM_STATE),
    }


def obtener_referencia():
    return DummyRegressor(strategy="median")
