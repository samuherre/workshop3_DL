"""Prueba basada en propiedades (Property 5) para `compute_metrics`.

Verifica que la función de métricas produce valores válidos y correctos.

# Feature: yelp-sentiment-classifier-workshop, Property 5: compute_metrics produce metricas validas y correctas
"""

import os
import sys

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

# Añadimos el directorio raíz del proyecto al sys.path para poder importar
# `workshop_functions` cuando pytest se ejecuta desde cualquier ubicación.
RAIZ_PROYECTO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RAIZ_PROYECTO not in sys.path:
    sys.path.insert(0, RAIZ_PROYECTO)

from workshop_functions import compute_metrics


# Número de clases del clasificador de sentimiento (negative/neutral/positive).
NUM_CLASES = 3

# Floats finitos y acotados para simular logits realistas (sin NaN ni inf).
logits_float = st.floats(
    min_value=-10, max_value=10, allow_nan=False, allow_infinity=False
)


@settings(max_examples=100, deadline=None)
@given(data=st.data())
def test_compute_metrics_produce_metricas_validas(data):
    """Validates: Requirements 11.2, 11.3, 11.4, 11.5.

    Para cualquier par (logits, labels) bien formado, `compute_metrics`:
    - devuelve un dict con las claves `accuracy` y `f1_macro`;
    - produce métricas en el rango [0.0, 1.0];
    - en el caso perfecto (argmax de logits == labels) ambas valen 1.0.
    """
    # Generamos el número de ejemplos del lote: entre 1 y 30 filas.
    n = data.draw(st.integers(min_value=1, max_value=30))

    # logits de forma (N, 3) con floats finitos acotados.
    logits_lista = data.draw(
        st.lists(
            st.lists(logits_float, min_size=NUM_CLASES, max_size=NUM_CLASES),
            min_size=n,
            max_size=n,
        )
    )
    logits = np.array(logits_lista, dtype=np.float64)

    # labels de forma (N,) con enteros en {0, 1, 2}.
    labels_lista = data.draw(
        st.lists(
            st.integers(min_value=0, max_value=NUM_CLASES - 1),
            min_size=n,
            max_size=n,
        )
    )
    labels = np.array(labels_lista, dtype=np.int64)

    resultado = compute_metrics((logits, labels))

    # El resultado es un dict con exactamente las claves esperadas.
    assert isinstance(resultado, dict)
    assert "accuracy" in resultado
    assert "f1_macro" in resultado

    accuracy = resultado["accuracy"]
    f1_macro = resultado["f1_macro"]

    # Ambas métricas deben estar en el rango válido [0.0, 1.0].
    assert 0.0 <= accuracy <= 1.0, f"accuracy fuera de rango: {accuracy}"
    assert 0.0 <= f1_macro <= 1.0, f"f1_macro fuera de rango: {f1_macro}"

    # Caso perfecto: construimos logits cuyo argmax coincide EXACTAMENTE con
    # los labels (valor alto en la columna = label, bajo en el resto). En ese
    # escenario accuracy y f1_macro deben ser exactamente 1.0.
    logits_perfectos = np.full((n, NUM_CLASES), -1.0, dtype=np.float64)
    logits_perfectos[np.arange(n), labels] = 1.0

    resultado_perfecto = compute_metrics((logits_perfectos, labels))

    assert resultado_perfecto["accuracy"] == 1.0, (
        f"accuracy debería ser 1.0 en el caso perfecto: {resultado_perfecto}"
    )
    assert resultado_perfecto["f1_macro"] == 1.0, (
        f"f1_macro debería ser 1.0 en el caso perfecto: {resultado_perfecto}"
    )
