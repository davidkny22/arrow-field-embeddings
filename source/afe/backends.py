"""Spatial layout backends for Arrow Field Embeddings."""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Union


class SpatialBackend(ABC):
    """Abstract base class for spatial layout backends."""

    @abstractmethod
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Produce a (n_samples, 3) spatial layout from X."""
        ...

    @abstractmethod
    def transform(self, X_new: np.ndarray) -> np.ndarray:
        """Project new data into existing spatial layout."""
        ...


class PaCMAPBackend(SpatialBackend):
    """PaCMAP-based spatial layout.

    Requires: pip install pacmap
    """

    def __init__(self, random_state=None, verbose=False, **kwargs):
        self.random_state = random_state
        self.verbose = verbose
        self.pacmap_kwargs = kwargs
        self._reducer = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        try:
            import pacmap
        except ImportError:
            raise ImportError(
                "PaCMAP is required but not installed. "
                "Install with: pip install pacmap"
            )

        self._reducer = pacmap.PaCMAP(
            n_components=3,
            random_state=self.random_state,
            verbose=self.verbose,
            **self.pacmap_kwargs,
        )
        return self._reducer.fit_transform(X).astype(np.float32)

    def transform(self, X_new: np.ndarray) -> np.ndarray:
        if self._reducer is None:
            raise RuntimeError("Backend must be fit before calling transform.")
        return self._reducer.transform(X_new).astype(np.float32)


class DREAMSBackend(SpatialBackend):
    """DREAMS-based spatial layout.

    DREAMS (Dimensionality Reduction Enhanced Across Multiple Scales)
    regularizes t-SNE toward PCA positions, balancing local and global
    structure preservation.

    Requires the DREAMS fork of openTSNE:
        git clone --branch tp --single-branch https://github.com/berenslab/DREAMS.git
        cd DREAMS && python setup.py install

    Parameters
    ----------
    reg_lambda : float, default=0.15
        Regularization strength. Higher = more global (PCA-like),
        lower = more local (t-SNE-like).
    perplexity : float, default=30.0
        t-SNE perplexity parameter.
    n_iter : int, default=750
        Number of optimization iterations.
    random_state : int or None, default=None
        Random seed.
    verbose : bool, default=False
        Print progress.
    """

    def __init__(
        self,
        reg_lambda=0.15,
        perplexity=30.0,
        n_iter=750,
        random_state=None,
        verbose=False,
        **kwargs,
    ):
        self.reg_lambda = reg_lambda
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.random_state = random_state
        self.verbose = verbose
        self.extra_kwargs = kwargs
        self._embedding = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        try:
            from openTSNE import TSNE
        except ImportError:
            raise ImportError(
                "DREAMS is experimental and requires the openTSNE fork from berenslab. "
                "Install with:\n"
                "  git clone --branch tp --single-branch "
                "https://github.com/berenslab/DREAMS.git\n"
                "  cd DREAMS && python setup.py install"
            )
        import inspect

        dreams_params = set(inspect.signature(TSNE.__init__).parameters)
        required_params = {"regularization", "reg_lambda", "reg_embedding"}
        if not required_params.issubset(dreams_params):
            raise ImportError(
                "DREAMSBackend requires the experimental berenslab DREAMS "
                "openTSNE fork with regularization/reg_embedding support. "
                "A standard openTSNE install is not compatible."
            )
        try:
            from openTSNE import tsne as opentsne_tsne
            optimizer_params = set(
                inspect.signature(opentsne_tsne.gradient_descent.__call__).parameters
            )
        except (ImportError, AttributeError, ValueError):
            optimizer_params = set()
        if "X" not in optimizer_params:
            raise ImportError(
                "DREAMSBackend found openTSNE regularization parameters, but "
                "the optimizer is not the compatible DREAMS fork expected by "
                "this backend. Install the berenslab DREAMS openTSNE fork."
            )

        from sklearn.decomposition import PCA

        # Compute PCA reference embedding (3D)
        pca = PCA(n_components=3, random_state=self.random_state)
        pca_embedding = pca.fit_transform(X).astype(np.float64)

        # Scale reference embedding to small initial scale
        pca_ref = pca_embedding / pca_embedding[:, 0].std() * 0.0001

        embedder = TSNE(
            n_components=3,
            perplexity=self.perplexity,
            n_iter=self.n_iter,
            initialization=pca_ref,
            regularization=True,
            reg_lambda=self.reg_lambda,
            reg_embedding=pca_ref,
            random_state=self.random_state or 42,
            verbose=self.verbose,
            **self.extra_kwargs,
        )

        self._embedding = np.asarray(embedder.fit(X), dtype=np.float32)
        return self._embedding.copy()

    def transform(self, X_new: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "DREAMSBackend does not currently support transform on new data."
        )


class UMAPBackend(SpatialBackend):
    """UMAP-based spatial layout.

    Requires: pip install umap-learn
    """

    def __init__(self, random_state=None, verbose=False, **kwargs):
        self.random_state = random_state
        self.verbose = verbose
        self.umap_kwargs = kwargs
        self._reducer = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        try:
            import umap
        except ImportError:
            raise ImportError(
                "UMAP is required but not installed. "
                "Install with: pip install umap-learn"
            )

        self._reducer = umap.UMAP(
            n_components=3,
            random_state=self.random_state,
            verbose=self.verbose,
            **self.umap_kwargs,
        )
        return self._reducer.fit_transform(X).astype(np.float32)

    def transform(self, X_new: np.ndarray) -> np.ndarray:
        if self._reducer is None:
            raise RuntimeError("Backend must be fit before calling transform.")
        return self._reducer.transform(X_new).astype(np.float32)


class TSNEBackend(SpatialBackend):
    """t-SNE spatial layout using sklearn.

    This is a standalone t-SNE backend, separate from DREAMSBackend
    (which uses a regularized openTSNE fork).

    sklearn's TSNE is already a core dependency — no extra install needed.
    """

    def __init__(self, random_state=None, verbose=False, **kwargs):
        self.random_state = random_state
        self.verbose = verbose
        self.tsne_kwargs = kwargs
        self._embedding = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        from sklearn.manifold import TSNE

        tsne = TSNE(
            n_components=3,
            random_state=self.random_state or 42,
            verbose=1 if self.verbose else 0,
            **self.tsne_kwargs,
        )
        self._embedding = tsne.fit_transform(X).astype(np.float32)
        return self._embedding.copy()

    def transform(self, X_new: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "TSNEBackend does not support transform on new data. "
            "t-SNE is a non-parametric method without an "
            "out-of-sample extension."
        )


class TriMAPBackend(SpatialBackend):
    """TriMAP-based spatial layout.

    Requires: pip install trimap
    """

    def __init__(self, random_state=None, verbose=False, **kwargs):
        self.random_state = random_state
        self.verbose = verbose
        self.trimap_kwargs = kwargs
        self._reducer = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        try:
            import trimap
        except ImportError:
            raise ImportError(
                "TriMAP is required but not installed. "
                "Install with: pip install trimap"
            )

        # Pass random_state directly to TRIMAP instead of mutating global state
        trimap_kwargs = dict(self.trimap_kwargs)
        if self.random_state is not None and "random_state" not in trimap_kwargs:
            trimap_kwargs["random_state"] = self.random_state

        self._reducer = trimap.TRIMAP(
            n_dims=3,
            verbose=self.verbose,
            **trimap_kwargs,
        )
        result = self._reducer.fit_transform(X).astype(np.float32)

        return result

    def transform(self, X_new: np.ndarray) -> np.ndarray:
        if self._reducer is None:
            raise RuntimeError("Backend must be fit before calling transform.")
        return self._reducer.transform(X_new).astype(np.float32)


class ManualBackend(SpatialBackend):
    """Use a pre-computed (n, 3) spatial embedding as the backend."""

    def __init__(self, embedding: np.ndarray):
        embedding = np.asarray(embedding, dtype=np.float32)
        if embedding.ndim != 2 or embedding.shape[1] != 3:
            raise ValueError(
                f"ManualBackend expects (n, 3) array, got {embedding.shape}"
            )
        self._embedding = embedding

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        if len(X) != len(self._embedding):
            raise ValueError(
                f"Data has {len(X)} points but embedding has "
                f"{len(self._embedding)} points."
            )
        return self._embedding.copy()

    def transform(self, X_new: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "ManualBackend does not support transform on new data."
        )


def get_backend(
    backend: Union[str, np.ndarray, SpatialBackend],
    backend_kwargs: Optional[dict] = None,
    random_state: Optional[int] = None,
    verbose: bool = False,
) -> SpatialBackend:
    """Factory: string name or array -> SpatialBackend instance."""
    backend_kwargs = backend_kwargs or {}

    if isinstance(backend, SpatialBackend):
        return backend

    if isinstance(backend, np.ndarray):
        return ManualBackend(backend)

    if isinstance(backend, str):
        backend = backend.lower()
        if backend == "pacmap":
            return PaCMAPBackend(
                random_state=random_state, verbose=verbose, **backend_kwargs
            )
        elif backend == "dreams":
            return DREAMSBackend(
                random_state=random_state, verbose=verbose, **backend_kwargs
            )
        elif backend == "umap":
            return UMAPBackend(
                random_state=random_state, verbose=verbose, **backend_kwargs
            )
        elif backend == "tsne":
            return TSNEBackend(
                random_state=random_state, verbose=verbose, **backend_kwargs
            )
        elif backend == "trimap":
            return TriMAPBackend(
                random_state=random_state, verbose=verbose, **backend_kwargs
            )
        elif backend == "manual":
            raise ValueError(
                "For manual backend, pass a numpy array directly "
                "instead of the string 'manual'."
            )
        else:
            raise ValueError(
                f"Unknown backend '{backend}'. "
                f"Choose from: 'pacmap', 'umap', 'tsne', 'trimap', "
                f"'dreams', or pass an ndarray."
            )

    raise TypeError(f"Unsupported backend type: {type(backend)}")


__all__ = [
    "SpatialBackend",
    "PaCMAPBackend",
    "DREAMSBackend",
    "UMAPBackend",
    "TSNEBackend",
    "TriMAPBackend",
    "ManualBackend",
    "get_backend",
]
