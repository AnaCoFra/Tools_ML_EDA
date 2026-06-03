import pytest
import pandas as pd
import numpy as np
from toolbox_ml.eda.core import detect_outliers, get_features_num_classification, get_features_cat_classification



# Tests para detect_outliers


def test_detect_outliers_devuelve_dict():
    """Caso correcto: input válido → retorna dict."""
    df = pd.DataFrame({'a': [1, 2, 3, 4, 100]})
    resultado = detect_outliers(df)
    assert isinstance(resultado, dict)


def test_detect_outliers_detecta_outlier_obvio():
    """Un valor muy alejado debe ser detectado por ambos métodos."""
    df = pd.DataFrame({'a': [10, 11, 10, 12, 11, 10, 500]})
    resultado = detect_outliers(df, method="both")
    assert resultado['a']['iqr']['count'] > 0
    assert resultado['a']['zscore']['count'] > 0


def test_detect_outliers_sin_outliers():
    """Datos uniformes no deben producir outliers."""
    df = pd.DataFrame({'a': [10, 10, 10, 10, 10]})
    resultado = detect_outliers(df, method="both")
    assert resultado['a']['iqr']['count'] == 0
    assert resultado['a']['zscore']['count'] == 0


def test_detect_outliers_method_iqr_solo():
    """Con method='iqr' solo debe haber clave 'iqr', no 'zscore'."""
    df = pd.DataFrame({'a': [1, 2, 3, 4, 100]})
    resultado = detect_outliers(df, method="iqr")
    assert 'iqr' in resultado['a']
    assert 'zscore' not in resultado['a']


def test_detect_outliers_columnas_vacias_usa_todas_numericas():
    """Si columns=[] usa todas las columnas numéricas."""
    df = pd.DataFrame({'a': [1, 2, 100], 'b': [5, 5, 5], 'c': ['x', 'y', 'z']})
    resultado = detect_outliers(df, columns=[])
    assert 'a' in resultado
    assert 'b' in resultado
    assert 'c' not in resultado  # categórica → excluida


def test_detect_outliers_retorna_none_input_invalido():
    """Caso de error: input no es DataFrame → retorna None."""
    assert detect_outliers("no soy un dataframe") is None
    assert detect_outliers([1, 2, 3]) is None


def test_detect_outliers_method_invalido():
    """Method no reconocido → retorna None."""
    df = pd.DataFrame({'a': [1, 2, 3]})
    assert detect_outliers(df, method="inventado") is None



# Tests para get_features_num_classification


def test_get_features_num_classification_devuelve_lista():
    """Caso correcto: input válido → retorna lista."""
    df = pd.DataFrame({
        'edad': [20, 30, 40, 50, 25, 35],
        'target': ['A', 'A', 'A', 'B', 'B', 'B']
    })
    resultado = get_features_num_classification(df, target_col='target')
    assert isinstance(resultado, list)


def test_get_features_num_classification_detecta_feature_relevante():
    """Una columna claramente separada por clases debe ser seleccionada."""
    df = pd.DataFrame({
        'valor': [1, 2, 1, 2, 100, 200, 100, 200],
        'clase': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']
    })
    resultado = get_features_num_classification(df, target_col='clase', pvalue=0.05)
    assert 'valor' in resultado


def test_get_features_num_classification_excluye_feature_irrelevante():
    """Una columna sin diferencia entre clases no debe ser seleccionada."""
    np.random.seed(42)
    df = pd.DataFrame({
        'ruido': np.random.normal(0, 1, 100),
        'clase': ['A'] * 50 + ['B'] * 50
    })
    resultado = get_features_num_classification(df, target_col='clase', pvalue=0.05)
    assert 'ruido' not in resultado


def test_get_features_num_classification_retorna_none_input_invalido():
    """Caso de error: input no es DataFrame → retorna None."""
    assert get_features_num_classification("no soy df", target_col='x') is None


def test_get_features_num_classification_retorna_none_target_no_existe():
    """Target que no existe en el DataFrame → retorna None."""
    df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
    assert get_features_num_classification(df, target_col='inventada') is None


def test_get_features_num_classification_multiclase():
    """Funciona con targets de más de 2 clases (usa ANOVA)."""
    df = pd.DataFrame({
        'valor': [1, 2, 1, 50, 51, 50, 200, 201, 200],
        'clase': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']
    })
    resultado = get_features_num_classification(df, target_col='clase', pvalue=0.05)
    assert 'valor' in resultado



# Tests para get_features_cat_classification


def test_get_features_cat_classification_devuelve_lista():
    """Caso correcto: input válido → retorna lista."""
    df = pd.DataFrame({
        'color': ['rojo', 'azul', 'rojo', 'azul'],
        'target': ['A', 'A', 'B', 'B']
    })
    resultado = get_features_cat_classification(df, target_col='target')
    assert isinstance(resultado, list)


def test_get_features_cat_classification_detecta_feature_relevante():
    """Una categórica perfectamente alineada con el target debe ser seleccionada."""
    df = pd.DataFrame({
        'grupo': ['X'] * 50 + ['Y'] * 50,
        'target': ['A'] * 50 + ['B'] * 50
    })
    resultado = get_features_cat_classification(df, target_col='target', pvalue=0.05)
    assert 'grupo' in resultado


def test_get_features_cat_classification_excluye_feature_irrelevante():
    """Una categórica sin relación con el target no debe ser seleccionada."""
    np.random.seed(0)
    df = pd.DataFrame({
        'aleatoria': np.random.choice(['X', 'Y'], 100),
        'target': np.random.choice(['A', 'B'], 100)
    })
    resultado = get_features_cat_classification(df, target_col='target', pvalue=0.05)
    assert 'aleatoria' not in resultado


def test_get_features_cat_classification_retorna_none_input_invalido():
    """Caso de error: input no es DataFrame → retorna None."""
    assert get_features_cat_classification("no soy df", target_col='x') is None


    