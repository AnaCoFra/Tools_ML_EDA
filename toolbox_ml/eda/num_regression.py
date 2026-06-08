import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ─────────────────────────────────────────────
# FUNCIÓN 1: get_features_num_regression
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
# FUNCIÓN 2: plot_features_num_regression
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