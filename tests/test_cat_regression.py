import pandas as pd
import pytest
from toolbox_ml.eda.cat_regression import get_features_cat_regression, plot_features_cat_regression
import math
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

def test_get_features_cat_regression_correcto():
    df = pd.DataFrame({
        "color": ["rojo"] * 4 + ["azul"] * 4,
        "ventas": [100, 101, 102, 103, 1, 2, 3, 4]})
    resultado = get_features_cat_regression(
        df=df,
        target_col="ventas",
        pvalue=0.05)
    assert isinstance(resultado, list)
    assert "color" in resultado
    

def test_get_features_cat_regression_limite():
    df = pd.DataFrame({
        "color": ["rojo", "azul", "verde", "amarillo"],
        "ventas": [10, 11, 10, 11]})
    resultado = get_features_cat_regression(
        df=df,
        target_col="ventas")
    assert resultado == []
    

def test_get_features_cat_regression_error():
    df = pd.DataFrame({
        "color": ["rojo", "azul"]})
    resultado = get_features_cat_regression(
        df=df,
        target_col="ventas")
    assert resultado is None
    

def test_plot_features_cat_regression_correcto():
    df = pd.DataFrame({
        "color": ["rojo"] * 4 + ["azul"] * 4,
        "ventas": [100, 101, 102, 103, 1, 2, 3, 4]})
    resultado = plot_features_cat_regression(
        df=df,
        target_col="ventas")
    assert isinstance(resultado, list)
    assert "color" in resultado
    

def test_plot_features_cat_regression_limite():
    df = pd.DataFrame({
        "color": ["rojo", "azul", "verde", "amarillo"],
        "ventas": [10, 11, 10, 11]})
    resultado = plot_features_cat_regression(
        df=df,
        target_col="ventas")
    assert resultado == []
    

def test_plot_features_cat_regression_error():
    df = pd.DataFrame({
        "color": ["rojo", "azul"]})
    resultado = plot_features_cat_regression(
        df=df,
        target_col="ventas")
    assert resultado is None