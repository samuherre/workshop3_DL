"""Funciones puras compartidas del Workshop 3 (clasificador de sentimiento de Yelp).

Este módulo extrae las funciones puras definidas en el notebook
`part04_workshop_yelp_trainer.ipynb` para poder importarlas desde las pruebas
automatizadas (las pruebas no pueden importar funciones directamente de un
notebook).

La implementación de `clean_review` es idéntica a la del notebook (celda id
ba5829cd, TODO 5).
"""

import re


def clean_review(text: str) -> str:
    # Aseguramos trabajar siempre con una cadena, por si el valor no es str.
    text = str(text)
    # 1) Eliminamos URLs (http://, https://, www.) reemplazándolas por un espacio.
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # 2) Normalizamos los espacios en blanco a uno solo y recortamos los extremos.
    #    Hacerlo después de quitar URLs garantiza que la función sea idempotente.
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------------------------------------------------------------------------
# Tokenización (TODO 8 del notebook)
# ---------------------------------------------------------------------------
# Longitud máxima de secuencia (en tokens) usada al tokenizar las reseñas.
# Se define a nivel de módulo para que tanto el notebook como las pruebas
# compartan exactamente el mismo valor.
MAX_LEN = 128


def make_tokenize_function(tokenizer, max_len: int = MAX_LEN):
    """Crea la `tokenize_function` del notebook (TODO 8), parametrizada por tokenizer.

    La `tokenize_function` del notebook depende de un `tokenizer` concreto y de
    la constante `MAX_LEN`, lo que impide importarla directamente. Esta fábrica
    reconstruye esa función manteniendo su lógica EXACTA (truncamiento a
    `max_len`, sin padding) pero recibiendo el `tokenizer` como argumento, de
    modo que sea importable y testeable.

    Args:
        tokenizer: El tokenizer de HuggingFace (p. ej. `distilbert-base-uncased`).
        max_len: Longitud máxima de secuencia; por defecto `MAX_LEN` (128).

    Returns:
        La función `tokenize_function(batch)` lista para usar con `Dataset.map`.
    """

    def tokenize_function(batch):
        # truncation=True + max_length=max_len garantiza que ninguna secuencia
        # exceda max_len tokens; no se aplica padding (lo gestiona el collator).
        return tokenizer(batch["text"], truncation=True, max_length=max_len)

    return tokenize_function

# ---------------------------------------------------------------------------
# Métricas de evaluación (TODO 11 del notebook)
# ---------------------------------------------------------------------------
# La `compute_metrics` del notebook usa `evaluate.load("accuracy")` y
# `evaluate.load("f1")`, que descargan los scripts de métricas de forma remota.
# Eso hace la función lenta y dependiente de la red, lo que impide probarla
# de manera rápida y offline. Aquí reconstruimos la MISMA lógica usando
# `sklearn`, cuyos valores numéricos coinciden exactamente: `accuracy` y
# `f1` con `average="macro"`.
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(eval_pred):
    """Equivalente a la compute_metrics del notebook (TODO 11).

    Usa sklearn en lugar de `evaluate.load` para que sea testeable offline;
    los valores numéricos coinciden: accuracy y f1 con average='macro'.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = float(accuracy_score(labels, predictions))
    f1_macro = float(f1_score(labels, predictions, average="macro"))
    return {"accuracy": accuracy, "f1_macro": f1_macro}
