# Tools_ML_EDA

`Tools_ML_EDA` es el paquete de Python que hemos desarrollado para el Team Challenge de EDA y Machine Learning. El proyecto reúne funciones útiles para explorar datos, tipificar variables, analizar relaciones con un target de regresión y generar visualizaciones de apoyo al análisis.

## Qué incluye

El paquete permite:

- obtener un resumen de un DataFrame;
- tipificar variables según su cardinalidad;
- seleccionar variables numéricas relacionadas con un target;
- representar gráficamente relaciones numéricas con el target;
- seleccionar variables categóricas significativas;
- representar distribuciones del target por grupos categóricos.

Todo el proyecto está organizado como un paquete instalable, con funciones documentadas, validación de entradas y tests automáticos con `pytest`.

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
- scikit-learn, solo para la función bonus

Todas las dependencias están en `requirements.txt`.

## Instalación

Clona el repositorio e instala el proyecto en un entorno virtual:

```bash
git clone https://github.com/user/Tools_ML_EDA.git
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

Una vez instalado el paquete, las funciones se importan así:

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

### `describe_df`

```python
import seaborn as sns
from toolbox_ml.eda.core import describe_df

df = sns.load_dataset("titanic")
resumen = describe_df(df)
print(resumen)
```

Devuelve un resumen por columna con tipo de dato, porcentaje de nulos, número de valores únicos y cardinalidad.

### `tipifica_variables`

```python
from toolbox_ml.eda.core import tipifica_variables

tipos = tipifica_variables(df, umbral_categoria=10, umbral_continua=70.0)
print(tipos)
```

### `get_features_num_regression`

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

Devuelve las variables numéricas con correlación relevante con el target.

### `plot_features_num_regression`

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

Genera gráficos de las variables numéricas seleccionadas junto al target.

### `get_features_cat_regression`

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

### `plot_features_cat_regression`

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

## Tests

Para ejecutar los tests:

```bash
pytest tests/ -v
```

Los tests cubren casos válidos, casos límite y casos de error para las funciones del paquete.

## Demo

El notebook `notebooks/demo.ipynb` muestra el uso del paquete con un dataset real y sirve como demostración del proyecto.

## Equipo

El proyecto ha sido realizado por:

- Ana Corrochano
- Paula Comas
- Melania Fondevilla
- Maria Rodriguez
- William Walker


## Flujo de trabajo

Hemos trabajado con ramas de feature y Pull Requests hacia `main`, usando commits con formato Conventional Commits.

Ejemplos:

```bash
feat: add describe_df function
fix: handle empty dataframe in tipifica_variables
docs: update README examples
test: add unit tests for num regression
```
