import pytest
import numpy as np
import pandas as pd
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from toolbox_ml.eda.core import describe_df, tipifica_variables
from toolbox_ml.eda.core import get_features_num_regression, plot_features_num_regression
from toolbox_ml.eda.core import get_features_cat_regression, plot_features_cat_regression
from toolbox_ml.eda.core import detect_outliers, get_features_num_classification, get_features_cat_classification

pytestmark = [
    pytest.mark.filterwarnings("ignore::FutureWarning"),
    pytest.mark.filterwarnings("ignore::UserWarning"),
]


##### tests_describe_tipification

# Tests para describe_df

def test_describe_df_devuelve_dataframe():
    """Caso correcto: input válido → retorna DataFrame."""
    df = pd.DataFrame({'a': [1, 2, None], 'b': ['x', 'y', 'z']})
    resultado = describe_df(df)
    assert isinstance(resultado, pd.DataFrame)


def test_describe_df_columnas_correctas():
    """El DataFrame resultado tiene exactamente las columnas esperadas."""
    df = pd.DataFrame({'a': [1, 2, 3]})
    resultado = describe_df(df)
    assert set(resultado.columns) == {
        'tipo', 'porcentaje_nulos', 'valores_unicos', 'porcentaje_cardinalidad'
    }


def test_describe_df_porcentaje_nulos_correcto():
    """Calcula correctamente el porcentaje de nulos."""
    df = pd.DataFrame({'a': [1, None, None, None]})
    resultado = describe_df(df)
    assert resultado.loc['a', 'porcentaje_nulos'] == 75.0
    #assert resultado.loc['a', 'porcentaje_nulos'] == pytest.approx(33.33, abs=0.01)


def test_describe_df_retorna_none_con_input_invalido():
    """Caso de error: input no es DataFrame → retorna None."""
    assert describe_df("esto no es un dataframe") is None
    assert describe_df([1, 2, 3]) is None

""" 
**Nota:** al comparar floats en los tests, usar `pytest.approx()` para evitar fallos por precisión numérica:

"""
# Tests para tipifica_variables


def test_tipifica_variables_devuelve_dataframe():
    """Caso correcto: input válido → retorna DataFrame."""
    df = pd.DataFrame({'a': [1, 2, None], 'b': ['x', 'y', 'z']})
    resultado = tipifica_variables(df,umbral_categorica=3, umbral_continua=80.0)
    assert isinstance(resultado, pd.DataFrame)
    
def test_tipifica_variables_columnas_correctas():
    """Caso correcto: input válido → retorna DataFrame."""
    df = pd.DataFrame({'a': [1, 2, None], 'b': ['x', 'y', 'z']})
    resultado = tipifica_variables(df, umbral_categorica=3, umbral_continua=80.0)
    assert set(resultado.columns) == {'Nombre variable', 'Tipo sugerido'}

    
def test_tipifica_variables_clasificacion_correcta():
    df = pd.DataFrame({
        # 3 valores únicos → Categorica
        'Nombre': ['Ana', 'Luis', 'Carlos', 'Ana', 'Luis', 'Carlos'],

        # 4 valores únicos → 4/6 = 66.7% < 80 → Numérica Discreta
        'Edad': [10, 20, 30, 40, 5, 90],

        # 6 valores únicos → 100% ≥ 80 → Numérica Continua
        'Estatura': [1.60, 1.70, 1.80, 1.90, 2.00, 2.10],

        # 2 valores únicos → Binaria
        'Sexo': ['F', 'M', 'F', 'M', 'F', 'M']
    })

    resultado = tipifica_variables(df, umbral_categorica=4, umbral_continua=80.0)

    tipos = dict(zip(resultado['Nombre variable'], resultado['Tipo sugerido']))

    assert tipos == {
        'Nombre': 'Categorica',
        'Edad': 'Numérica Continua',
        'Estatura': 'Numérica Continua',
        'Sexo': 'Binaria'
    }
    
def test_tipifica_variables_retorna_none_con_input_invalido():
    """Caso de error: input no es DataFrame → retorna None."""
    assert tipifica_variables("esto no es un dataframe", umbral_categorica=3, umbral_continua=80.0) is None
    assert tipifica_variables([1, 2, 3], umbral_categorica=3, umbral_continua=80.0) is None


##### tests_num_regression

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



###test_cat_regression

def test_get_features_cat_regression_correcto():
    """Caso correcto: input válido → devuelve lista con variables significativas."""
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
    """Caso límite: input válido → devuelve lista vacía al no encnotrar variables significativas."""
    df = pd.DataFrame({
        "color": ["rojo", "azul", "verde", "amarillo"],
        "ventas": [10, 11, 10, 11]})
    resultado = get_features_cat_regression(
        df=df,
        target_col="ventas")
    assert resultado == []
    

def test_get_features_cat_regression_error():
    """Caso erróneo: input no válido → retorna None."""
    df = pd.DataFrame({
        "color": ["rojo", "azul"]})
    resultado = get_features_cat_regression(
        df=df,
        target_col="ventas")
    assert resultado is None
    

def test_plot_features_cat_regression_correcto():
    """Caso correcto: input válido → pinta los gráficos y devuelve lista con variables significativas."""
    df = pd.DataFrame({
        "color": ["rojo"] * 4 + ["azul"] * 4,
        "ventas": [100, 101, 102, 103, 1, 2, 3, 4]})
    resultado = plot_features_cat_regression(
        df=df,
        target_col="ventas")
    assert isinstance(resultado, list)
    assert "color" in resultado
    

def test_plot_features_cat_regression_limite():
    """Caso límite: input válido → no pinta gráficos y devuelve lista vacía al no encontrar variables significativas."""
    df = pd.DataFrame({
        "color": ["rojo", "azul", "verde", "amarillo"],
        "ventas": [10, 11, 10, 11]})
    resultado = plot_features_cat_regression(
        df=df,
        target_col="ventas")
    assert resultado == []
    

def test_plot_features_cat_regression_error():
    """Caso erróneo: input no válido → retorna None."""
    df = pd.DataFrame({
        "color": ["rojo", "azul"]})
    resultado = plot_features_cat_regression(
        df=df,
        target_col="ventas")
    assert resultado is None


### BONUS

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
    resultado = detect_outliers(df, method="both", zscore_threshold=2.0)
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


    