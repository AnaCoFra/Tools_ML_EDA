import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import math
import scipy.stats
from scipy import stats
from scipy.stats import mannwhitneyu, f_oneway
from pandas.api.types import is_numeric_dtype
import seaborn as sns



# función describe

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

# función tipifica

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
        elif num_unique > umbral_categorica and perc_card >= umbral_continua:
            tipo = "Numérica Continua"
        elif num_unique > umbral_categorica and perc_card < umbral_continua:
            tipo = "Numérica Discreta"
        else:
            tipo = "Desconocida"

        datos.append({
            "Nombre variable": col,
            "Tipo sugerido": tipo
        })

    return pd.DataFrame(datos)


# ─────────────────────────────────────────────
# get_features_num_regression
# ─────────────────────────────────────────────

def get_features_num_regression(
    df: pd.DataFrame,
    target_col: str,
    umbral_corr: float,
    pvalue: float = None,
) -> list:
   
    # ── Comprobaciones de entrada ──────────────────────────────────────
    if not isinstance(df, pd.DataFrame):
        print("Error: El parámetro 'df' debe ser un DataFrame de pandas.")
        return None

    if target_col not in df.columns:
        print(
            f"Error: La columna 'target_col' ('{target_col}') no existe en el DataFrame."
        )
        return None

    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print(
            f"Error: La columna 'target_col' ('{target_col}') debe ser de tipo numérico."
        )
        return None

    if (
        not isinstance(umbral_corr, (int, float))
        or not (0 <= umbral_corr <= 1)
    ):
        print("Error: 'umbral_corr' debe ser un número entre 0 y 1.")
        return None

    if pvalue is not None:
        if not isinstance(pvalue, (int, float)) or not (0 <= pvalue <= 1):
            print("Error: 'pvalue' debe ser un número entre 0 y 1 o None.")
            return None

    # ── Identificar columnas categóricas ─────────────────────────────
    cat_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in cat_cols if c != target_col]

    if not cat_cols:
        print("No se encontraron columnas categóricas en el DataFrame.")
        return []

    # ── Aplicar el test adecuado a cada columna ───────────────────────
    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in columnas_numericas:
        columnas_numericas.remove(target_col)

    features_seleccionadas = []

    for col in columnas_numericas:
        # Eliminamos filas con nulos en estas columnas para evitar errores en pearsonr
        df_clean = df[[col, target_col]].dropna()

        # Si tras limpiar no hay suficientes datos, saltamos la columna
        if len(df_clean) < 2:
            continue

        # Calcular coeficiente de correlación y p-valor
        corr, p_val = stats.pearsonr(df_clean[col], df_clean[target_col])

        # Filtrar por valor absoluto de la correlación
        if abs(corr) > umbral_corr:
            # Si se solicita filtro de p-valor, comprobar la significancia
            if pvalue is not None:
                if p_val < pvalue:
                    features_seleccionadas.append(col)
            else:
                # Si pvalue es None, basta con superar el umbral de correlación
                features_seleccionadas.append(col)

    return features_seleccionadas



# ─────────────────────────────────────────────
# plot_features_num_regression
# ─────────────────────────────────────────────

def plot_features_num_regression(
    df: pd.DataFrame,
    target_col: str = "",
    columns: list = [],
    umbral_corr: float = 0,
    pvalue: float = None
) -> list:

    # ── Comprobaciones de entrada ──────────────────────────────────────
    if not isinstance(df, pd.DataFrame):
        print("Error: 'df' debe ser un pd.DataFrame.")
        return None

    if not isinstance(target_col, str) or target_col == "":
        print("Error: 'target_col' debe ser un string no vacío.")
        return None

    if target_col not in df.columns:
        print(f"Error: '{target_col}' no existe en el DataFrame.")
        return None

    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print(f"Error: '{target_col}' debe ser una columna numérica.")
        return None

    if not isinstance(umbral_corr, (int, float)) or not (0 <= umbral_corr <= 1):
        print("Error: 'umbral_corr' debe ser un float entre 0 y 1.")
        return None

    if pvalue is not None:
        if not isinstance(pvalue, (int, float)) or not (0 < pvalue <= 1):
            print("Error: 'pvalue' debe ser un float entre 0 y 1 (o None).")
            return None

    # ── Determinar columnas candidatas ────────────────────────────────
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target_col]

    if len(columns) == 0:
        candidates = numeric_cols
    else:
        invalid = [c for c in columns if c not in df.columns]
        if invalid:
            print(f"Error: las siguientes columnas no existen en el DataFrame: {invalid}")
            return None
        non_numeric = [c for c in columns if c not in numeric_cols]
        if non_numeric:
            print(f"Error: las siguientes columnas no son numéricas: {non_numeric}")
            return None
        candidates = [c for c in columns if c != target_col]

    # ── Filtrar por correlación ──────────────────
    selected = []
    for col in candidates:
        df_clean = df[[target_col, col]].dropna()
        corr_val, p_val = stats.pearsonr(df_clean[target_col], df_clean[col])
        if abs(corr_val) >= umbral_corr:
            if pvalue is None or p_val < pvalue:
                selected.append(col)

    if not selected:
        print("No se encontraron columnas que cumplan los criterios especificados.")
        return []

    # ── Pintar pairplots (grupos de máx. 5 columnas + target) ─────────
    chunk_size = 4  # 4 features + target_col = 5 columnas por plot
    for i in range(0, len(selected), chunk_size):
        group = selected[i: i + chunk_size]
        cols_to_plot = [target_col] + group
        sns.pairplot(df[cols_to_plot].dropna(), diag_kind="kde")
        plt.suptitle(
            f"Pairplot (grupo {i // chunk_size + 1}): {group}",
            y=1.02, fontsize=11
        )
        plt.tight_layout()
        plt.show()

    return selected



# función get cat regression

def get_features_cat_regression(
    df: pd.DataFrame,
    target_col: str,
    pvalue: float = 0.05,
    umbral_card_rel: float = 0.3,
    umbral_n_cat: int = 20
) -> list:

    # Comprobaciones de entrada
    if not isinstance(df, pd.DataFrame):
        print("Error: df debe ser un pd.DataFrame")
        return None

    if target_col not in df.columns:
        print(f"Error: '{target_col}' no existe en el DataFrame")
        return None

    if not is_numeric_dtype(df[target_col]):
        print(f"Error: '{target_col}' debe ser numérica")
        return None

    if not isinstance(pvalue, (int, float)) or not (0 < pvalue < 1):
        print("Error: pvalue debe ser un número entre 0 y 1")
        return None

    if (
        not isinstance(umbral_card_rel, float)
        or not (0 < umbral_card_rel < 1)
    ):
        print(
            "Error: umbral_card_rel debe ser un número entre 0 y 1"
        )
        return None

    columnas_significativas = []

    # Recorrer columnas
    for col in df.columns:

        # Saltar target
        if col == target_col:
            continue

        temp_df = df[[col, target_col]].dropna()

        n_categorias = temp_df[col].nunique()

        cardinalidad_relativa = (
            n_categorias / len(temp_df)
        )

        # Consideramos categórica si:
        # - tiene baja cardinalidad relativa
        # - y además pocas categorías absolutas
        es_categorica = (
            cardinalidad_relativa < umbral_card_rel
            and n_categorias <= umbral_n_cat
        )

        if es_categorica:

            categorias = temp_df[col].unique()

            # EXACTAMENTE 2 categorías
            if len(categorias) == 2:

                grupo1 = temp_df[
                    temp_df[col] == categorias[0]
                ][target_col]

                grupo2 = temp_df[
                    temp_df[col] == categorias[1]
                ][target_col]

                _, p_valor = mannwhitneyu(
                    grupo1,
                    grupo2
                )

                if p_valor < pvalue:
                    columnas_significativas.append(col)

            # MÁS DE 2 categorías
            elif len(categorias) > 2:

                grupos = [
                    temp_df[
                        temp_df[col] == categoria
                    ][target_col]
                    for categoria in categorias
                ]

                _, p_valor = f_oneway(*grupos)

                if p_valor < pvalue:
                    columnas_significativas.append(col)

    return columnas_significativas

#función plot cat regression 

def plot_features_cat_regression(
    df: pd.DataFrame,
    target_col: str = "",
    columns: list = None,
    pvalue: float = 0.05,
    with_individual_plot: bool = False,
    umbral_card_rel: float = 0.3,
    umbral_n_cat: int = 15
) -> list:

    columnas_significativas = get_features_cat_regression(
        df,
        target_col,
        pvalue,
        umbral_card_rel,
        umbral_n_cat
    )

    if columnas_significativas is None:
        return None

    if columns is None:
        columns = columnas_significativas

    columns = [
        col for col in columns
        if col in columnas_significativas
    ]

    if len(columns) == 0:
        return []

    if with_individual_plot:

        for columna in columns:

            plt.figure(figsize=(8, 5))

            categorias = (
                df[columna]
                .value_counts()
                .head(10)
                .index
            )

            for categoria in categorias:

                datos = df[df[columna] == categoria][target_col]

                plt.hist(
                    datos,
                    alpha=0.5,
                    bins=len(df[target_col].unique()),
                    label=str(categoria)
                )

            plt.title(f"{target_col} según {columna}")
            plt.xlabel(target_col)
            plt.ylabel("Frecuencia")
            plt.legend()
            plt.tight_layout()
            plt.show()

    else:

        n_cols = 2
        n_rows = math.ceil(len(columns) / n_cols)

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(12, 5 * n_rows)
        )

        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        axes = axes.flatten()

        for i, columna in enumerate(columns):

            categorias = (
                df[columna]
                .value_counts()
                .head(10)
                .index
            )

            for categoria in categorias:

                datos = df[df[columna] == categoria][target_col]

                axes[i].hist(
                    datos,
                    alpha=0.5,
                    bins=len(df[target_col].unique()),
                    label=str(categoria)
                )

            axes[i].set_title(f"{target_col} según {columna}")
            axes[i].set_xlabel(target_col)
            axes[i].set_ylabel("Frecuencia")
            axes[i].legend()

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    return columns


def detect_outliers(
    df: pd.DataFrame,
    columns: list = [],
    method: str = "both",
    iqr_threshold: float = 1.5,
    zscore_threshold: float = 3.0
) -> dict:
    """
    Detecta outliers en las columnas numéricas de un DataFrame usando IQR y/o Z-score.

    Argumentos:
        df (pd.DataFrame): DataFrame a analizar.
        columns (list): Lista de columnas a analizar. Si está vacía, usa todas las numéricas.
        method (str): Método a usar: 'iqr', 'zscore' o 'both'.
        iqr_threshold (float): Multiplicador del IQR para definir el límite (por defecto 1.5).
        zscore_threshold (float): Umbral del Z-score en desviaciones típicas (por defecto 3.0).

    Retorna:
        dict: Diccionario con una clave por columna analizada. Cada valor es un dict con:
              - 'iqr'     → {count, percentage, indices} (si method es 'iqr' o 'both')
              - 'zscore'  → {count, percentage, indices} (si method es 'zscore' o 'both')
        Retorna None si alguna comprobación de entrada falla.
    """
    # Comprobaciones iniciales

    if not isinstance(df, pd.DataFrame):
        print("Error: 'df' debe ser un pd.DataFrame.")
        return None

    if not isinstance(columns, list):
        print("Error: 'columns' debe ser una lista.")
        return None

    valid_methods = {"iqr", "zscore", "both"}
    if method not in valid_methods:
        print(f"Error: 'method' debe ser uno de {valid_methods}.")
        return None

    if not isinstance(iqr_threshold, (int, float)) or iqr_threshold <= 0:
        print("Error: 'iqr_threshold' debe ser un número positivo.")
        return None

    if not isinstance(zscore_threshold, (int, float)) or zscore_threshold <= 0:
        print("Error: 'zscore_threshold' debe ser un número positivo.")
        return None

    # Seleccionar columnas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(columns) == 0:
        # Si no se especifican columnas, usamos todas las numéricas
        target_cols = numeric_cols
    else:
        # Filtrar solo las que existen y son numéricas
        target_cols = [c for c in columns if c in numeric_cols]
        invalid = [c for c in columns if c not in df.columns]
        if invalid:
            print(f"Advertencia: columnas no encontradas en el DataFrame: {invalid}")

    if len(target_cols) == 0:
        print("Error: no hay columnas numéricas válidas para analizar.")
        return None

    result = {}

    for col in target_cols:
        serie = df[col].dropna()
        col_result = {}

        # --- Método IQR ---
        if method in ("iqr", "both"):
            Q1 = serie.quantile(0.25)
            Q3 = serie.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - iqr_threshold * IQR
            upper_bound = Q3 + iqr_threshold * IQR

            # Índices del DataFrame original (no de la serie sin NaN)
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_indices = df.index[outlier_mask & df[col].notna()].tolist()

            col_result["iqr"] = {
                "count": len(outlier_indices),
                "percentage": round(len(outlier_indices) / len(df) * 100, 2),
                "indices": outlier_indices,
                "lower_bound": round(lower_bound, 4),
                "upper_bound": round(upper_bound, 4)
            }

        # Z-score como método
        if method in ("zscore", "both"):
            mean = serie.mean()
            std = serie.std()

            if std == 0:
                # Desviación típica cero: todos los valores son iguales, no hay outliers
                col_result["zscore"] = {
                    "count": 0,
                    "percentage": 0.0,
                    "indices": []
                }
            else:
                zscores = (df[col] - mean) / std
                outlier_mask = zscores.abs() > zscore_threshold
                outlier_indices = df.index[outlier_mask & df[col].notna()].tolist()

                col_result["zscore"] = {
                    "count": len(outlier_indices),
                    "percentage": round(len(outlier_indices) / len(df) * 100, 2),
                    "indices": outlier_indices
                }

        result[col] = col_result

    return result




def get_features_num_classification(
    df: pd.DataFrame,
    target_col: str,
    pvalue: float = 0.05
) -> list:
    """
    Devuelve las columnas numéricas con relación estadística significativa
    con una variable target categórica (clasificación).

    Usa ANOVA (f_oneway) si el target tiene más de 2 clases, o
    Mann-Whitney U si tiene exactamente 2 clases.

    Argumentos:
        df (pd.DataFrame): DataFrame a analizar.
        target_col (str): Nombre de la columna target (debe ser categórica o binaria).
        pvalue (float): Nivel de significación estadística (entre 0 y 1).

    Retorna:
        list: Lista de nombres de columnas numéricas significativas.
        Retorna None si alguna comprobación de entrada falla.
    """
    # Comprobaciones iniciales

    if not isinstance(df, pd.DataFrame):
        print("Error: 'df' debe ser un pd.DataFrame.")
        return None

    if target_col not in df.columns:
        print(f"Error: '{target_col}' no existe en el DataFrame.")
        return None

    if df[target_col].dtype.kind in ("f", "i") and df[target_col].nunique() > 10:
        print(f"Error: '{target_col}' parece numérica continua; debe ser categórica o discreta.")
        return None

    if not isinstance(pvalue, float) or not (0 < pvalue < 1):
        print("Error: 'pvalue' debe ser un float entre 0 y 1.")
        return None

    # Columnas numéricas (sin el target)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target_col]

    clases = df[target_col].dropna().unique()
    n_clases = len(clases)

    selected = []

    for col in numeric_cols:
        # Eliminar filas con NaN en esta columna o en el target
        datos = df[[col, target_col]].dropna()

        # Crear grupos: lista de arrays, uno por clase
        grupos = [datos.loc[datos[target_col] == clase, col].values for clase in clases]

        # Evitar grupos vacíos o con un solo elemento
        grupos = [g for g in grupos if len(g) > 1]
        if len(grupos) < 2:
            continue

        # Elegir el test según número de clases
        if n_clases == 2:
            _, p = scipy.stats.mannwhitneyu(grupos[0], grupos[1], alternative="two-sided")
        else:
            _, p = scipy.stats.f_oneway(*grupos)

        if p < pvalue:
            selected.append(col)

    return selected






def get_features_cat_classification(
    df: pd.DataFrame,
    target_col: str,
    pvalue: float = 0.05
) -> list:
    """
    Devuelve las columnas categóricas con relación estadística significativa
    con una variable target categórica (clasificación), usando el test Chi-cuadrado.

    Argumentos:
        df (pd.DataFrame): DataFrame a analizar.
        target_col (str): Nombre de la columna target (debe ser categórica).
        pvalue (float): Nivel de significación estadística (entre 0 y 1).

    Retorna:
        list: Lista de nombres de columnas categóricas significativas.
        Retorna None si alguna comprobación de entrada falla.
    """
    # --- Comprobaciones de entrada ---
    if not isinstance(df, pd.DataFrame):
        print("Error: 'df' debe ser un pd.DataFrame.")
        return None

    if target_col not in df.columns:
        print(f"Error: '{target_col}' no existe en el DataFrame.")
        return None

    if not isinstance(pvalue, float) or not (0 < pvalue < 1):
        print("Error: 'pvalue' debe ser un float entre 0 y 1.")
        return None

    # Columnas categóricas (sin el target)

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c != target_col]

    selected = []

    for col in cat_cols:
        # Tabla de contingencia entre la variable categórica y el target
        datos = df[[col, target_col]].dropna()
        tabla = pd.crosstab(datos[col], datos[target_col])

        # Necesitamos al menos 2 filas y 2 columnas para el test
        if tabla.shape[0] < 2 or tabla.shape[1] < 2:
            continue

        # Test Chi-cuadrado de independencia
        _, p, _, _ = scipy.stats.chi2_contingency(tabla)

        if p < pvalue:
            selected.append(col)

    return selected