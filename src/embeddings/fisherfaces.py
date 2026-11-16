"""Fisherfaces: PCA comprime y LDA discrimina, ambos ajustados solo con TRAIN."""

from __future__ import annotations

import warnings

import numpy as np

from src.embeddings.base import ExtractorEmbedding
from src.embeddings.lda import ExtractorLDA
from src.embeddings.pca import ExtractorPCA


class ExtractorFisherfaces(ExtractorEmbedding):
    """Proyección z = W_LDAᵀ W_PCAᵀ (x − μ), ajustada solo con TRAIN.

    Orden:

    1. PCA con ``d_pca`` sobre ``X_train``. Ahí se estima μ y W_PCA.
    2. LDA con ``d_lda`` sobre el PCA de TRAIN, usando ``y_train``.
    3. ``transform`` aplica las dos proyecciones ya aprendidas, sin reajustar.

    Contrato: ``d_lda <= min(C - 1, d_pca)`` y ``d_pca <= min(N_train, p)``.
    """

    nombre = "fisherfaces"

    def __init__(self, d_pca: int, d_lda: int) -> None:
        if d_pca < 1 or d_lda < 1:
            raise ValueError("Fisherfaces requiere d_pca y d_lda positivos.")
        if d_lda > d_pca:
            raise ValueError(
                f"d_lda={d_lda} supera d_pca={d_pca}. "
                "Debe cumplirse d_lda <= min(C-1, d_pca)."
            )
        self.d_pca = int(d_pca)
        self.d_lda = int(d_lda)
        self._pca: ExtractorPCA | None = None
        self._lda: ExtractorLDA | None = None

    @property
    def dimension(self) -> int:
        return self.d_lda

    @property
    def ajustado(self) -> bool:
        return self._pca is not None and self._lda is not None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> ExtractorFisherfaces:
        if y is None:
            raise ValueError("Fisherfaces necesita y_train: la etapa LDA usa las identidades.")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2 or X.shape[0] != y.shape[0]:
            raise ValueError(f"X {X.shape} e y {y.shape} no están alineados.")

        n_train, p = X.shape
        n_clases = int(np.unique(y).size)
        if n_clases < 2:
            raise ValueError("Fisherfaces necesita al menos dos identidades en TRAIN.")
        maximo_lda = min(n_clases - 1, self.d_pca)
        if self.d_lda > maximo_lda:
            raise ValueError(
                f"d_lda={self.d_lda} supera min(C-1, d_pca)={maximo_lda} "
                f"(C={n_clases}, d_pca={self.d_pca})."
            )
        if self.d_pca > min(n_train, p):
            raise ValueError(
                f"d_pca={self.d_pca} supera el rango posible min(N_train, p)={min(n_train, p)}."
            )
        if self.d_pca > n_train - n_clases:
            # Con d_pca > N − C la dispersión intra-clase S_W queda singular.
            warnings.warn(
                f"d_pca={self.d_pca} > N_train - C={n_train - n_clases}: "
                "S_W puede ser singular en el subespacio PCA.",
                RuntimeWarning,
                stacklevel=2,
            )

        pca = ExtractorPCA(n_components=self.d_pca).fit(X)
        Z_pca = pca.transform(X)
        lda = ExtractorLDA(n_components=self.d_lda).fit(Z_pca, y)

        self._pca, self._lda = pca, lda
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.ajustado:
            raise RuntimeError("Fisherfaces: llame a fit con TRAIN antes de transform.")
        X = np.asarray(X, dtype=float)
        Z = self._lda.transform(self._pca.transform(X))
        return self.comprobar(Z, X.shape[0])

    def componentes_imagen(self, alto: int, ancho: int) -> np.ndarray:
        """Cada fisherface en el espacio de píxeles: W_PCA · W_LDA reordenado a H×W.

        ``scalings_`` de LDA lleva coordenadas PCA a coordenadas discriminantes,
        y ``components_`` de PCA lleva píxeles centrados a coordenadas PCA.
        """
        if not self.ajustado:
            raise RuntimeError("Fisherfaces: ajuste el extractor antes de visualizarlo.")
        w_pca = self._pca._modelo.components_  
        w_lda = self._lda._modelo.scalings_[:, : self.d_lda]  
        pesos = (w_pca.T @ w_lda).T  
        if pesos.shape[1] != alto * ancho:
            raise ValueError("La fisherface no coincide con alto × ancho.")
        return pesos.reshape(-1, alto, ancho)

    def rostro_medio(self, alto: int, ancho: int) -> np.ndarray:
        if not self.ajustado:
            raise RuntimeError("Fisherfaces: ajuste el extractor antes de visualizarlo.")
        return self._pca.rostro_medio(alto, ancho)
