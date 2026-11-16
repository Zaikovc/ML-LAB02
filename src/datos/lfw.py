"""Carga de LFW: hasta 50 identidades con al menos 20 fotografías cada una."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import fetch_lfw_people

from src.config import DIR_CACHE
from src.datos.base import CargadorRostros, validar_conjunto
from src.modelos import ConjuntoRostros

MIN_FOTOS_POR_PERSONA = 20
MAX_IDENTIDADES = 50
RESIZE = 0.5


class CargadorLFW(CargadorRostros):
    """Fotografías de LFW con mayor diversidad de pose, luz y fondo.

    1. ``fetch_lfw_people(min_faces_per_person=20, resize=0.5)``.
    2. Conserva las ``max_identidades`` de mayor frecuencia. Los empates se
       resuelven por nombre para que la selección no dependa del orden de carga.
    3. Reindexa ``y`` a 0..C−1 y guarda el nombre de cada identidad.
    4. Lee alto y ancho desde ``images.shape`` y valida el contrato.
    """

    nombre = "lfw"

    def __init__(
        self,
        min_fotos: int = MIN_FOTOS_POR_PERSONA,
        max_identidades: int = MAX_IDENTIDADES,
        resize: float = RESIZE,
    ) -> None:
        if min_fotos < 2 or max_identidades < 2:
            raise ValueError("LFW necesita al menos 2 fotos y 2 identidades.")
        self.min_fotos = int(min_fotos)
        self.max_identidades = int(max_identidades)
        self.resize = float(resize)

    def cargar(self) -> ConjuntoRostros:
        lfw = fetch_lfw_people(
            min_faces_per_person=self.min_fotos,
            resize=self.resize,
            data_home=str(DIR_CACHE),
            download_if_missing=True,
        )
        images = np.asarray(lfw.images)
        y_original = np.asarray(lfw.target, dtype=int)
        nombres_originales = np.asarray(lfw.target_names)

        seleccion = self._identidades_mas_frecuentes(y_original, nombres_originales)
        mascara = np.isin(y_original, seleccion)

        images = images[mascara]
        ids_conservados, y = np.unique(y_original[mascara], return_inverse=True)
        y = y.astype(int)
        nombres = [str(nombres_originales[i]) for i in ids_conservados]

        alto, ancho = int(images.shape[1]), int(images.shape[2])
        X = images.reshape(images.shape[0], alto * ancho).astype(float)

        conjunto = ConjuntoRostros(
            nombre=self.nombre,
            images=images,
            X=X,
            y=y,
            nombres_identidades=nombres,
            alto=alto,
            ancho=ancho,
        )
        validar_conjunto(conjunto)
        self._exigir_minimo(conjunto)
        return conjunto

    def _identidades_mas_frecuentes(
        self,
        y: np.ndarray,
        nombres: np.ndarray,
    ) -> np.ndarray:
        ids, conteos = np.unique(y, return_counts=True)
        orden = sorted(range(ids.size), key=lambda i: (-conteos[i], str(nombres[ids[i]])))
        return ids[orden[: self.max_identidades]]

    def _exigir_minimo(self, conjunto: ConjuntoRostros) -> None:
        _, conteos = np.unique(conjunto.y, return_counts=True)
        if conteos.min() < self.min_fotos:
            raise ValueError(
                f"Hay identidades con {conteos.min()} fotos; se exigían {self.min_fotos}."
            )
        if conjunto.n_clases > self.max_identidades:
            raise ValueError(
                f"LFW quedó con {conjunto.n_clases} identidades; el máximo es "
                f"{self.max_identidades}."
            )
