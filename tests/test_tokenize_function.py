"""Prueba basada en propiedades (Property 4) para `tokenize_function`.

Verifica que la función de tokenización respeta el truncamiento a `MAX_LEN`
(128 tokens), tal como exigen los requisitos 8.1 y 8.2.

# Feature: yelp-sentiment-classifier-workshop, Property 4: tokenize_function respeta el truncamiento a MAX_LEN
"""

import os
import sys

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# Añadimos el directorio raíz del proyecto al sys.path para poder importar
# `workshop_functions` cuando pytest se ejecuta desde cualquier ubicación.
RAIZ_PROYECTO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RAIZ_PROYECTO not in sys.path:
    sys.path.insert(0, RAIZ_PROYECTO)

from workshop_functions import MAX_LEN, make_tokenize_function


# Cargamos el tokenizer real UNA sola vez a nivel de módulo. Cargarlo dentro de
# cada iteración de Hypothesis sería prohibitivamente lento; además, el valor
# no cambia entre ejecuciones del test.
#
# Si el entorno no tiene instalado `transformers` o no puede descargar los
# pesos del tokenizer (p. ej. sin acceso a red), omitimos la prueba con un
# `skip` explicativo en lugar de fallar: la propiedad no es verificable sin el
# tokenizer real, que es parte de la lógica bajo prueba.
try:
    from transformers import AutoTokenizer

    _tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    _tokenize_function = make_tokenize_function(_tokenizer, MAX_LEN)
    _ERROR_CARGA = None
except Exception as exc:  # pragma: no cover - depende del entorno
    _tokenizer = None
    _tokenize_function = None
    _ERROR_CARGA = exc


# Estrategia de texto: mezclamos texto arbitrario de Hypothesis con textos
# "largos" construidos repitiendo una palabra, para forzar explícitamente el
# camino de truncamiento (muchos más de 128 tokens).
_textos_largos = st.builds(
    lambda palabra, repeticiones: (palabra + " ") * repeticiones,
    st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=10
    ),
    st.integers(min_value=130, max_value=400),
)

_estrategia_texto = st.one_of(st.text(), _textos_largos)


@pytest.mark.skipif(
    _tokenize_function is None,
    reason=f"No se pudo cargar el tokenizer real: {_ERROR_CARGA!r}",
)
@settings(max_examples=100, deadline=None)
@given(texto=_estrategia_texto)
def test_tokenize_function_respeta_max_len(texto):
    """Para todo texto, la secuencia tokenizada no excede `MAX_LEN` tokens.

    La función se invoca en modo "batched": recibe `{"text": [texto]}` y
    devuelve `input_ids` como lista de listas (una por ejemplo del lote). Con
    `truncation=True` y `max_length=MAX_LEN`, la longitud de cada secuencia
    debe ser menor o igual a `MAX_LEN` (128), incluso contando los tokens
    especiales [CLS]/[SEP] que añade DistilBERT.
    """
    resultado = _tokenize_function({"text": [texto]})

    input_ids = resultado["input_ids"][0]

    assert len(input_ids) <= MAX_LEN, (
        f"La secuencia tokenizada tiene {len(input_ids)} tokens, "
        f"excede MAX_LEN={MAX_LEN} (entrada de {len(texto)} caracteres)"
    )
