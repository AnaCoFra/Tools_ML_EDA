import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, f_oneway
from pandas.api.types import is_numeric_dtype
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import math


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