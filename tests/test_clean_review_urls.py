"""Prueba basada en propiedades (Property 1) para `clean_review`.

Verifica que la función de limpieza elimina las URLs del texto.

# Feature: yelp-sentiment-classifier-workshop, Property 1: clean_review elimina URLs
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


# Caracteres válidos para el "cuerpo" de una URL (dominio, ruta, etc.).
# Excluimos deliberadamente cualquier espacio en blanco: el patrón `\S+` de
# `clean_review` se detiene en el primer espacio, así que para que la URL se
# elimine por completo no debe contener espacios.
CARACTERES_URL = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "-._~:/?#[]@!$&'()*+,;=%"
)

# Esquemas de URL que `clean_review` reconoce explícitamente.
esquemas = st.sampled_from(["http://", "https://", "www."])

# Cuerpo de la URL: al menos un carácter, sin espacios en blanco.
cuerpos_url = st.text(alphabet=CARACTERES_URL, min_size=1, max_size=40)


def _sin_marcas_url(s: str) -> bool:
    """El texto base no debe traer sus propias marcas de URL.

    Si el prefijo o el sufijo contuvieran `http` o `www.` por su cuenta, la
    aserción de "no quedan URLs" podría fallar aunque `clean_review` haya
    eliminado correctamente la URL que insertamos nosotros. Restringimos el
    espacio de entrada para que la propiedad bajo prueba sea exactamente
    "la URL insertada se elimina".
    """
    return "http" not in s and "www." not in s


# Texto base sin marcas de URL propias, para aislar la propiedad bajo prueba.
texto_base = st.text().filter(_sin_marcas_url)


@settings(max_examples=200)
@given(
    prefijo=texto_base,
    sufijo=texto_base,
    esquema=esquemas,
    cuerpo=cuerpos_url,
)
def test_clean_review_elimina_urls(prefijo, sufijo, esquema, cuerpo):
    """Para todo texto base y toda URL insertada, la salida no contiene URLs.

    Insertamos una URL (sin espacios) entre un prefijo y un sufijo arbitrarios
    y comprobamos que la salida de `clean_review` no contiene ni `http` ni
    `www.`, las marcas que delatan una URL no eliminada.
    """
    url = esquema + cuerpo

    # Rodeamos la URL con espacios para garantizar que queda delimitada como un
    # único "token" `\S+`; así `clean_review` puede eliminarla por completo sin
    # arrastrar caracteres del prefijo/sufijo.
    texto = f"{prefijo} {url} {sufijo}"

    salida = clean_review(texto)

    assert "http" not in salida, (
        f"La salida aún contiene 'http': {salida!r} (entrada: {texto!r})"
    )
    assert "www." not in salida, (
        f"La salida aún contiene 'www.': {salida!r} (entrada: {texto!r})"
    )
