import pandas as pd

def describe_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera un resumen estadístico descriptivo de un DataFrame.
    Argumentos:
        df (pd.DataFrame): DataFrame a analizar.

    Retorna:
        pd.DataFrame: DataFrame con una fila por columna del input y las
        siguientes columnas: 'tipo', 'porcentaje_nulos', 'valores_unicos',
        'porcentaje_cardinalidad'.
        Retorna None si el input no es un DataFrame válido.

    """ 
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    else:
        df_transformed = pd.DataFrame({
            'tipo': df.dtypes.astype(str),
            'porcentaje_nulos': df.isnull().mean() * 100,
            'valores_unicos': df.nunique(), 
            'porcentaje_cardinalidad': df.nunique() / len(df) * 100
        })
        return  df_transformed


def tipifica_variables(df: pd.DataFrame, 
    umbral_categorica: int,
    umbral_continua:float) -> pd.DataFrame:
    """
    Clasifica variables en:
    - Binaria: exactamente 2 valores únicos
    - Categorica: valores únicos <= umbral_categorica
    - Numérica Continua: valores únicos > umbral_categorica y %_card <= umbral_continua
    - Numérica Discreta: valores únicos > umbral_categorica y %_card > umbral_continua
    """

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    if not isinstance(umbral_categorica, int):
        return None
    if not isinstance(umbral_continua, float):
        return None

    datos = []

    for col in df.columns:
        num_unique = df[col].nunique()
        perc_card = num_unique / len(df) * 100

        if num_unique == 2:
            tipo = "Binaria"
        elif num_unique <= umbral_categorica:
            tipo = "Categorica"
        elif num_unique > umbral_categorica and perc_card <= umbral_continua:
            tipo = "Numérica Continua"
        elif num_unique > umbral_categorica and perc_card > umbral_continua:
            tipo = "Numérica Discreta"
        else:
            tipo = "Desconocida"

        datos.append({
            "Nombre variable": col,
            "Tipo sugerido": tipo
        })

    return pd.DataFrame(datos)
