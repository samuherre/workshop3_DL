# Feature: yelp-sentiment-classifier-workshop, Property 3: la conversion a DatasetDict preserva filas y columnas
"""
Prueba basada en propiedades (Property 3) para la conversion de un DataFrame
de pandas a un ``Dataset`` de HuggingFace.

Esta prueba valida que, para cualquier DataFrame con columnas ``text`` y
``label``, el ``Dataset`` construido mediante
``Dataset.from_pandas(df[["text", "label"]], preserve_index=False)`` conserva
exactamente el mismo numero de filas que el DataFrame y expone unicamente las
columnas ``{"text", "label"}``.

Validates: Requirements 6.1, 6.2
"""

import pandas as pd
from datasets import Dataset
from hypothesis import given, settings, strategies as st


@settings(max_examples=100, deadline=None)  # deadline=None: construir Datasets puede ser lento
@given(data=st.data())
def test_conversion_preserva_filas_y_columnas(data):
    """La conversion a Dataset preserva el numero de filas y las columnas {text, label}."""
    # Numero de filas a generar (incluye el caso vacio con 0 filas).
    n_filas = data.draw(st.integers(min_value=0, max_value=20))

    # Generadores emparejados: ambas columnas tienen exactamente la misma longitud.
    textos = data.draw(
        st.lists(st.text(), min_size=n_filas, max_size=n_filas)
    )
    etiquetas = data.draw(
        st.lists(st.integers(min_value=0, max_value=2), min_size=n_filas, max_size=n_filas)
    )

    # Construimos el DataFrame con las columnas esperadas por el workshop.
    df = pd.DataFrame({"text": textos, "label": etiquetas})

    # Conversion directa a Dataset de HuggingFace (no dependemos del notebook).
    dataset = Dataset.from_pandas(df[["text", "label"]], preserve_index=False)

    # Aserciones de la propiedad.
    assert len(dataset) == len(df)
    assert set(dataset.column_names) == {"text", "label"}
