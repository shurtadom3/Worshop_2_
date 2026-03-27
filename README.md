# Workshop 2 – Machine Learning & Deep Learning Aplicado

**Universidad EAFIT – Introducción a la Inteligencia Artificial (2026-01)**

---

## Descripción General

Este workshop integra dos problemas supervisados independientes: uno de **clasificación** y uno de **regresión**, aplicando el ciclo completo de un proyecto de Machine Learning y Deep Learning: análisis del problema, exploración de datos, preprocesamiento, entrenamiento, evaluación y análisis crítico de resultados.

---

## Problema 1 – Clasificación: Detección de Fatiga Muscular en Ciclismo

### Dataset

| Campo | Detalle |
|-------|---------|
| **Nombre** | Muscle Fatigue Cycling |
| **Fuente** | [HuggingFace – YominE/Muscle_Fatigue_Cycling](https://huggingface.co/datasets/YominE/Muscle_Fatigue_Cycling) |
| **Descripción** | Señales de electromiografía (EMG) registradas en 8 músculos de la pierna dominante de sujetos realizando sprints en bicicleta. El target indica el estado muscular: condición normal (0) vs. desgaste muscular (1). |

### Desarrollo

El notebook [`Clasificación/clasificacion.ipynb`](Clasificación/clasificacion.ipynb) contiene el desarrollo completo:

1. **Análisis Preliminar del Problema** — Preprocesamiento del target a 2 clases (0: normal, 1: desgaste) y clasificación de las variables del dataset (numéricas continuas para señales EMG, categórica binaria para el target).

2. **Extracción de Características (Feature Engineering)** — Ventanas de 1 segundo (1000 muestras a 1000 Hz) sobre los 8 canales EMG. Se extraen características en el dominio del tiempo (RMS, varianza, cruce por cero, pendiente media) y de la frecuencia (frecuencia mediana, frecuencia media, potencia espectral).

3. **Análisis Exploratorio de Datos (EDA)** — Estadísticos descriptivos, distribuciones de características, matriz de correlación, boxplots por clase, análisis de balance de clases y visualización de señales en el tiempo.

4. **Procesamiento de Datos** — Imputación de valores nulos, estandarización de características mediante pipeline de scikit-learn y división en conjuntos de entrenamiento, validación y prueba.

5. **Entrenamiento y Comparación de Modelos** — Se entrenan y comparan cinco clasificadores con ajuste de hiperparámetros (Grid Search para modelos clásicos):
   - k-Nearest Neighbors (kNN)
   - Decision Tree
   - Random Forest
   - Gradient Boosting
   - Deep Neural Network (DNN) con 3+ capas ocultas, Dropout y regularización

   Se presenta una tabla comparativa con Accuracy, Precision, Recall y F1-Score, junto con curvas de entrenamiento/validación para detectar overfitting o underfitting.

6. **Evaluación Final del Mejor Modelo** — Reentrenamiento con datos de entrenamiento + validación, métricas finales sobre test, boxplots de características representativas, matriz de confusión y análisis crítico.

7. **Prueba con Muestra Artificial** — Generación de una muestra sintética con valores cercanos al promedio del dataset para verificar la coherencia de las predicciones del modelo.

---

## Problema 2 – Regresión: Estimación de Edad a partir de Imágenes Faciales

### Dataset

| Campo | Detalle |
|-------|---------|
| **Nombre** | Faces: Age Detection from Images (UTKFace) |
| **Fuente** | [Kaggle – arashnic/faces-age-detection-dataset](https://www.kaggle.com/datasets/arashnic/faces-age-detection-dataset) |
| **Descripción** | Conjunto de imágenes faciales etiquetadas con la edad del sujeto. El objetivo es entrenar un modelo que estime la edad a partir de los píxeles de la imagen. |

### Desarrollo

El notebook [`Regresión/regresion.ipynb`](Regresión/regresion_lightning.ipynb) contiene el desarrollo completo:

1. **Análisis Preliminar del Problema** — Justificación del problema como regresión (variable objetivo continua: edad), descripción de las imágenes de entrada (dimensiones, espacio de color, distribución de edades) y protocolo de adquisición de datos.

2. **Análisis Exploratorio de Datos (EDA)** — Distribución de edades (histograma y estadísticos descriptivos), análisis de balance y sesgos, visualización de muestras representativas y análisis de la calidad y variabilidad de las imágenes.

3. **Procesamiento de Datos** — Redimensionamiento a 128×128 píxeles, normalización con estadísticas de ImageNet, data augmentation en entrenamiento y división 70/15/15 en train, validación y test. Pipeline reproducible con PyTorch `Dataset` y `DataLoader`.

4. **Modelo CNN para Regresión** — Red neuronal convolucional con:
   - 3 bloques convolucionales con MaxPooling
   - Capas densas con Dropout y Batch Normalization
   - Función de pérdida: MSE Loss
   - Optimizador: Adam (lr=1e-3)

   Métricas reportadas: **MAE**, **RMSE** y **R²** sobre train, validación y test. Curvas de pérdida a lo largo de las épocas con análisis de overfitting/underfitting.

5. **Prueba con Muestra Artificial** — Predicción sobre una imagen individual de prueba para verificar la coherencia del modelo y análisis del impacto de variaciones visuales (iluminación, escala, orientación).

---

## Estructura del Repositorio

```
Workshop_2_/
├── README.md                              ← Este archivo
├── Clasificación/
│   └── clasificacion.ipynb                ← Notebook del problema de clasificación
└── Regresión/
    └── regresion.ipynb           ← Notebook del problema de regresión
```

---

## Tecnologías Utilizadas

| Área | Herramientas |
|------|-------------|
| Lenguaje | Python |
| ML Clásico | scikit-learn (KNN, Decision Tree, Random Forest, Gradient Boosting, GridSearchCV, Pipelines) |
| Deep Learning | TensorFlow/Keras (DNN clasificación), PyTorch (CNN regresión) |
| Procesamiento de señales | NumPy, SciPy |
| EDA y visualización | Pandas, Matplotlib, Seaborn |
| Datos | HuggingFace Datasets, Kaggle |
| Entorno | Lightning.ai, Google Colab / Jupyter Notebook |

---

## Cómo Ejecutar

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/shurtadom3/Worshop_2_.git
   ```

2. **Problema de Clasificación:** Abrir `Clasificación/clasificacion.ipynb` en Jupyter Notebook o Google Colab y ejecutar todas las celdas secuencialmente. El dataset se descarga automáticamente desde HuggingFace.

3. **Problema de Regresión:** Abrir `Regresión/regresion.ipynb` en Lightning.ai o Google Colab. Es necesario subir el archivo `UTKFace.zip` (descargable desde Kaggle) al entorno antes de ejecutar.

---

## Autores

- Alyson Dahiana Henao
- Gabriel Atehortua
- Sara Hurtado
