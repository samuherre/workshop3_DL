"""Prueba basada en propiedades (Property 2) para `clean_review`.

Verifica que la función de limpieza normaliza los espacios en blanco y que es
idempotente (aplicarla dos veces produce el mismo resultado que aplicarla una).

# Feature: yelp-sentiment-classifier-workshop, Property 2: clean_review normaliza espacios y es idempotente
"""

import os
import sys

from hypothesis import given, settings
from hypothesis import strategies as st

# Añadimos el directorio raíz del proyecto al sys.path para poder importar
# `workshop_functions` cuando pytest se ejecuta desde cualquier ubicación.
RAIZ_PROYECTO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RAIZ_PROYECTO not in sys.path:
    sys.path.insert(0, RAIZ_PROYECTO)

from workshop_functions import clean_review


@settings(max_examples=100)
@given(texto=st.text())
def test_clean_review_normaliza_espacios_y_es_idempotente(texto):
    """Para todo texto, la salida normaliza espacios y es idempotente.

    Validates: Requirements 5.1, 5.3.

    Comprobamos cuatro propiedades sobre la salida de `clean_review`:
      1. La salida es siempre de tipo `str`.
      2. No contiene secuencias de 2 o más espacios consecutivos.
      3. No tiene espacios al inicio ni al final (está "recortada").
      4. Es idempotente: volver a limpiarla no cambia el resultado.
    """
    salida = clean_review(texto)

    # 1) La salida debe ser una cadena.
    assert isinstance(salida, str), (
        f"La salida no es str: {type(salida)!r} (entrada: {texto!r})"
    )

    # 2) No deben quedar secuencias de 2+ espacios consecutivos.
    assert "  " not in salida, (
        f"La salida contiene espacios dobles: {salida!r} (entrada: {texto!r})"
    )

    # 3) No debe haber espacios en los extremos.
    assert salida == salida.strip(), (
        f"La salida no está recortada: {salida!r} (entrada: {texto!r})"
    )

    # 4) Idempotencia: aplicar la función de nuevo no debe cambiar la salida.
    assert clean_review(salida) == salida, (
        f"La función no es idempotente para la entrada: {texto!r} "
        f"(salida: {salida!r}, salida doble: {clean_review(salida)!r})"
    )
