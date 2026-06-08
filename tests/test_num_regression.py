import pandas as pd
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from toolbox_ml.eda.num_regression import get_features_num_regression, plot_features_num_regression

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

def test_get_features_num_regression_correcto():
    x = np.arange(50, dtype=float)
    df = pd.DataFrame({
        "feature_alta": x + np.random.normal(0, 0.5, 50),
        "feature_baja": np.random.normal(0, 10, 50),
        "target": x,
    })
    resultado = get_features_num_regression(
        df=df,
        target_col="target",
        umbral_corr=0.5,
        pvalue=0.05,
    )
    assert isinstance(resultado, list)
    assert "feature_alta" in resultado
    assert "feature_baja" not in resultado


def test_get_features_num_regression_limite():
    np.random.seed(0)
    df = pd.DataFrame({
        "ruido": np.random.normal(0, 1, 30),
        "target": np.random.normal(0, 1, 30),
    })
    resultado = get_features_num_regression(
        df=df,
        target_col="target",
        umbral_corr=0.9,
    )
    assert resultado == []


def test_get_features_num_regression_error():
    df = pd.DataFrame({"feature": [1, 2, 3]})
    resultado = get_features_num_regression(
        df=df,
        target_col="no_existe",
        umbral_corr=0.5,
    )
    assert resultado is None


def test_plot_features_num_regression_correcto():
    x = np.arange(50, dtype=float)
    df = pd.DataFrame({
        "feature_alta": x + np.random.normal(0, 0.5, 50),
        "feature_baja": np.random.normal(0, 10, 50),
        "target": x,
    })
    resultado = plot_features_num_regression(
        df=df,
        target_col="target",
        umbral_corr=0.5,
        pvalue=0.05,
    )
    assert isinstance(resultado, list)
    assert "feature_alta" in resultado
    assert "feature_baja" not in resultado


def test_plot_features_num_regression_limite():
    np.random.seed(0)
    df = pd.DataFrame({
        "ruido": np.random.normal(0, 1, 30),
        "target": np.random.normal(0, 1, 30),
    })
    resultado = plot_features_num_regression(
        df=df,
        target_col="target",
        umbral_corr=0.9,
    )
    assert resultado == []


def test_plot_features_num_regression_error():
    df = pd.DataFrame({"feature": [1, 2, 3]})
    resultado = plot_features_num_regression(
        df=df,
        target_col="no_existe",
    )
    assert resultado is None