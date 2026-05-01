# ML_proyecto_final
Estudio completo de un conjunto de datos  de rendimiento académico en educación superior. Se abordan tres tareas complementarias:  clasificación de la situación final del estudiante,  regresión para predecir la calificación media del segundo semestre, y análisis no supervisado mediante clustering para identificar perfiles de alumnado. 

**Adriana Lorenzo Mingorance**
**Asignatura:** Machine Learning, 2ºB iMAT. Universidad Pontifica Comillas.


## Requisitos

Python 3.9 o superior:
pip install pandas numpy matplotlib seaborn scikit-learn scipy

## Orden de ejecución

**Es imprescindible ejecutqar los notebooks en el orden numerado**, ya que el notebook 02 generan los CSVs de los notebooks: 03, 04 y 05.

A lo largo de los notebooks se irán guardando las gráficas realizadas en la carpeta `figuras/`.

### 1. `01_EDA.ipynb` - Análisis Exploratorio de Datos
Carga el dataset orginal (`rendimiento_estudiantes.csv`) y realia un análisis descriptivo completo: distribución de la variable objetivo, análisis de variables numéricas y categóricas, correlaciones y visualizaciones clave.


### 2. `02_preprocesado.ipynb` - Preprocesado y división train/test
Codifica variables categóricas con One-Hot Encoding, excluye variables con riesgo de data leakage en la tarea de regresión, realiza la división 80/20 y aplica StandardScaler.
Guarda los conjuntos resultantes en `data/`, personalizados para cada tarea.

Se crean aquí todos los CSVs de `data/`, siendo el único que escribe en esta carpeta. Es IMPRESCINDIBLE ejecutar este notebook antes que el 3 y el 4, que solo la leen.

### 2. `03_clasificacion.ipynb` - Modelos de clasificación supervisada
Entrena y evalúa cuatro modelos (Reg. Logística, KNN, RF y GB) para predecir la situación final del estudiante. Incluye análisis de desbalanceo, matrices de confusión, curvas ROC e importancia de las variables.

### 2. `04_regresion.ipynb` - Modelos de regresión supervisada
Entrena y evalúa cinco modelos (Reg. Lineal, Ridge, Lasso, RF y GB) para predecir la situación final del estudiante. Incluye validación cruzada, análisis de residuos importancia de las variables.

### 2. `05_clustering.ipynb` - Aprendizaje No supervisado

Aplica tres métodos (K-Means, AGNES y GMM) sobre un subconjunto de 10 variables seleccionadas. Incluye reducción de dimensionalidad (PCA), elección nçumero de clusters y análisis de los perfiles obtenidos.

## Reproducibilidad

Todos los notebooks fijan la semilla aleatoria con `semilla=42` y `np.random.seed(42)`. LA división train/test se realiza con `random_state=True` y `stratify=True`. ejecuanto los notebooks en el orden indicando se obtienen exactamente los mismos resultados que los reportados en el informe.