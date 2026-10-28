# Clasificación de Sentimiento de Reseñas de Yelp con Trainer

Fine-tuning de **DistilBERT** para clasificar reseñas de Yelp en tres clases de
sentimiento (`negative`, `neutral`, `positive`) usando la API **Trainer** de
Hugging Face Transformers. El proyecto cubre el flujo completo: exploración de
datos, preprocesamiento, tokenización, entrenamiento, evaluación y análisis de
errores.

## Integrantes

- Samuel Herrera Galvis
- Emmanuel Álvarez Castrillón
- Cristian Cabarcas López

## Descripción del proyecto

El taller construye un clasificador de sentimiento a partir del dataset
[Yelp Review Full](https://huggingface.co/datasets/Yelp/yelp_review_full).
Las 5 estrellas originales se colapsan en 3 clases de sentimiento:

| Estrellas | Clase      | Label |
|-----------|------------|-------|
| 1 - 2     | `negative` | 0     |
| 3         | `neutral`  | 1     |
| 4 - 5     | `positive` | 2     |

**Modelo base:** `distilbert-base-uncased` (~66M parámetros).

### Etapas

1. **Preparación de datos** — carga, exploración, mapeo de etiquetas y tokenización.
2. **Fine-tuning y evaluación** — modelo, métricas, `TrainingArguments`, `Trainer`,
   entrenamiento, evaluación en test, análisis de errores y guardado del modelo.

## Estructura del repositorio

```
.
├── part04_workshop_yelp_trainer.ipynb   # Notebook principal del taller
├── workshop_functions.py                # Funciones puras (limpieza, tokenización, métricas)
├── tests/                               # Pruebas basadas en propiedades (Hypothesis)
└── README.md
```

## Requisitos

- Python 3.10 o superior
- GPU recomendada para el entrenamiento (opcional para ejecutar las pruebas)

Instalación de dependencias:

```bash
pip install "transformers>=5.0" datasets evaluate accelerate scikit-learn seaborn
pip install pytest hypothesis
```

## Cómo ejecutar el notebook

1. Abre `part04_workshop_yelp_trainer.ipynb` en Jupyter o VS Code.
2. Ejecuta la celda de instalación de dependencias y reinicia el kernel.
3. Ejecuta las celdas de arriba hacia abajo. La primera celda de datos descarga
   el dataset de Yelp desde Hugging Face.

> El entrenamiento completo requiere GPU y la descarga del dataset y del modelo
> base.

## Pruebas

El proyecto incluye pruebas basadas en propiedades (con
[Hypothesis](https://hypothesis.readthedocs.io/)) sobre las funciones puras del
taller: limpieza de texto, conversión a `Dataset`, tokenización y cálculo de
métricas.

```bash
python -m pytest tests/ -v
```

## Resultados esperados

- `classification_report` con `accuracy` y `f1_macro` sobre el conjunto de test.
- Matriz de confusión normalizada por clase.
- Análisis de los errores más frecuentes del modelo.
