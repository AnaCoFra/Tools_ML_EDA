# toolbox_ml

`toolbox_ml` es un paquete de Python para análisis exploratorio de datos y tareas básicas de regresión con un enfoque práctico, modular y reutilizable. El objetivo del proyecto es construir una herramienta de trabajo real para Data Science, con funciones bien documentadas, tests automáticos y una estructura de paquete instalable.

## Descripción

Este paquete reúne funciones para:

- resumir rápidamente un DataFrame;
- sugerir el tipo de cada variable;
- seleccionar variables numéricas relacionadas con un target de regresión;
- visualizar relaciones entre variables numéricas y el target;
- seleccionar variables categóricas significativas en regresión;
- visualizar distribuciones del target por grupos categóricos.

El proyecto se ha desarrollado siguiendo una estructura de paquete profesional, con pruebas unitarias en `pytest`, documentación en docstrings y control de versiones con Git y GitHub.

## Estructura del repositorio

```text
.github/
└── workflows/
    └── tests.yml

notebooks/
├── demo.ipynb
├── demo_cat_regression.ipynb
├── demo_describe_tipifica.ipynb
└── demo_num_regression.ipynb

data/
└── titanic_cat_reg.csv

tests/
├── __init__.py
├── test_bonus.py
├── test_cat_regression.py
├── test_core.py
├── test_describe_tipifica.py
└── test_num_regression.py

toolbox_ml/
└── eda/
    ├── __init__.py
    ├── cat_regression.py
    ├── core.py
    ├── core_bonus.py
    ├── core_describe_tipifica.py
    └── num_regression.py

.gitignore
README.md
requirements.txt
setup.py
```

## Requisitos

- Python 3.10 o superior
- pandas
- numpy
- scipy
- matplotlib
- seaborn
- pytest
- scikit-learn, solo si se implementa la función bonus

Todas las dependencias están recogidas en `requirements.txt`.

## Instalación

Clona el repositorio e instala el proyecto en un entorno virtual:

```bash
git clone https://github.com/User/Tools_ML_EDA.git
cd Tools_ML_EDA
python -m venv venv
source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
pip install -e .
```

En Windows:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```
## Uso

Una vez instalado el paquete, las funciones pueden importarse así:

```python
from toolbox_ml.eda.core import (
    describe_df,
    tipifica_variables,
    get_features_num_regression,
    plot_features_num_regression,
    get_features_cat_regression,
    plot_features_cat_regression
)
```

### 1. `describe_df`

```python
import seaborn as sns
from toolbox_ml.eda.core import describe_df

df = sns.load_dataset("titanic")
resumen = describe_df(df)
print(resumen)
```

Esta función devuelve un DataFrame con información básica de cada columna: tipo, porcentaje de nulos, número de valores únicos y cardinalidad.

### 2. `tipifica_variables`

```python
from toolbox_ml.eda.core import tipifica_variables

tipos = tipifica_variables(df, umbral_categoria=10, umbral_continua=70.0)
print(tipos)
```

La función propone un tipo sugerido para cada variable según su cardinalidad y porcentaje de cardinalidad.

### 3. `get_features_num_regression`

```python
from toolbox_ml.eda.core import get_features_num_regression

numericas = get_features_num_regression(
    df=df,
    target_col="fare",
    umbral_corr=0.2,
    pvalue=0.05
)
print(numericas)
```

Devuelve las variables numéricas con correlación de Pearson relevante frente al target.

### 4. `plot_features_num_regression`

```python
from toolbox_ml.eda.core import plot_features_num_regression

seleccionadas = plot_features_num_regression(
    df=df,
    target_col="fare",
    columns=[],
    umbral_corr=0.2,
    pvalue=0.05
)
print(seleccionadas)
```

Genera gráficos de relación entre el target y las variables numéricas seleccionadas.

### 5. `get_features_cat_regression`

```python
from toolbox_ml.eda.core import get_features_cat_regression

categoricas = get_features_cat_regression(
    df=df,
    target_col="fare",
    pvalue=0.05
)
print(categoricas)
```

Selecciona variables categóricas con relación estadísticamente significativa con el target.

### 6. `plot_features_cat_regression`

```python
from toolbox_ml.eda.core import plot_features_cat_regression

representadas = plot_features_cat_regression(
    df=df,
    target_col="fare",
    columns=[],
    pvalue=0.05,
    with_individual_plot=False
)
print(representadas)
```

Muestra histogramas agrupados del target por categorías.

## Ejecución de tests

Para ejecutar todos los tests unitarios:

```bash
pytest tests/ -v
```

Los tests cubren casos correctos, casos límite y casos de error para todas las funciones implementadas.

## Notebook de demostración

El archivo `notebooks/demo.ipynb` contiene una demostración completa del uso del paquete con un dataset real, mostrando:

- carga de datos;
- uso de cada función;
- salidas visibles;
- comentarios explicativos.

## Equipo y reparto de tareas

Este proyecto ha sido desarrollado por:

- Ana Corrochano
- Maria Rodriguez
- Melania Fondevilla
- Paula Comas
- William Walker

## Flujo de trabajo Git

El repositorio sigue un flujo basado en ramas de feature y Pull Requests:

1. Crear una rama por funcionalidad.
2. Trabajar en la rama con commits pequeños y claros.
3. Abrir Pull Request hacia `main`.
4. Revisar el código antes de hacer merge.
5. Hacer squash and merge.
6. Sincronizar `main` después de cada integración.

Ejemplos de ramas:

```bash
feature/describe-df
feature/tipifica-variables
feature/num-regression
feature/cat-regression
feature/bonus
```

Se recomienda usar commits con formato Conventional Commits

```bash
feat: add describe_df function
fix: handle empty dataframe in tipifica_variables
docs: update README examples
test: add unit tests for num regression
```

## Notas de implementación

Todas las funciones incluyen:

- validación de entradas;
- type hints;
- docstrings con descripción, argumentos y retornos;
- comentarios en el cuerpo de la función para explicar la lógica principal.

## Licencia

Proyecto académico desarrollado para el Team Challenge de Machine Learning.
