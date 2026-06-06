import pandas as pd
import pytest
from cat_regression import get_features_cat_regression, plot_features_cat_regression
import math

def test_get_features_cat_regression_ok():

    df = pd.DataFrame({
        "color": ["rojo"] * 4 + ["azul"] * 4,
        "ventas": [100, 101, 102, 103, 1, 2, 3, 4]
    })

    resultado = get_features_cat_regression(
        df=df,
        target_col="ventas",
        pvalue=0.05
    )

    assert isinstance(resultado, list)
    assert "color" in resultado
    

def test_get_features_cat_regression_sin_significativas():

    df = pd.DataFrame({
        "color": ["rojo", "azul", "verde", "amarillo"],
        "ventas": [10, 11, 10, 11]
    })

    resultado = get_features_cat_regression(
        df=df,
        target_col="ventas"
    )

    assert resultado == []
    

def test_get_features_cat_regression_target_inexistente():

    df = pd.DataFrame({
        "color": ["rojo", "azul"]
    })

    resultado = get_features_cat_regression(
        df=df,
        target_col="ventas"
    )

    assert resultado is None
    

def test_plot_features_cat_regression_ok():

    df = pd.DataFrame({
        "color": ["rojo"] * 4 + ["azul"] * 4,
        "ventas": [100, 101, 102, 103, 1, 2, 3, 4]
    })

    resultado = plot_features_cat_regression(
        df=df,
        target_col="ventas"
    )

    assert isinstance(resultado, list)
    assert "color" in resultado
    

def test_plot_features_cat_regression_sin_significativas():

    df = pd.DataFrame({
        "color": ["rojo", "azul", "verde", "amarillo"],
        "ventas": [10, 11, 10, 11]
    })

    resultado = plot_features_cat_regression(
        df=df,
        target_col="ventas"
    )

    assert resultado == []
    

def test_plot_features_cat_regression_target_inexistente():

    df = pd.DataFrame({
        "color": ["rojo", "azul"]
    })

    resultado = plot_features_cat_regression(
        df=df,
        target_col="ventas"
    )

    assert resultado is None