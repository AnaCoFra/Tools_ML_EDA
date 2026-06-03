import pytest
import pandas as pd
from toolbox_ml.eda.core import describe_df, tipifica_variables

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
        'Edad': 'Numérica Discreta',
        'Estatura': 'Numérica Discreta',
        'Sexo': 'Binaria'
    }
    
def test_tipifica_variables_retorna_none_con_input_invalido():
    """Caso de error: input no es DataFrame → retorna None."""
    assert tipifica_variables("esto no es un dataframe", umbral_categorica=3, umbral_continua=80.0) is None
    assert tipifica_variables([1, 2, 3], umbral_categorica=3, umbral_continua=80.0) is None