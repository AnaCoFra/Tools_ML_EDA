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