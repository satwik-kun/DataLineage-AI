"""FAISS index utilities for vector storage and search."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


class FaissVectorStore:
    """Small wrapper around FAISS index with metadata tracking."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.metadata: List[Dict] = []
        self._numpy_vectors = np.zeros((0, dim), dtype=np.float32)
        self._faiss = None
        self.index = None
        self._init_index()

    def _init_index(self) -> None:
        try:
            import faiss

            self._faiss = faiss
            self.index = faiss.IndexFlatIP(self.dim)
        except Exception:
            self._faiss = None
            self.index = None

    def add(self, vectors: np.ndarray, metadata: Sequence[Dict]) -> None:
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        vectors = vectors.astype(np.float32)

        if vectors.shape[1] != self.dim:
            raise ValueError(f"Expected vectors with dim={self.dim}, got dim={vectors.shape[1]}")
        if len(metadata) != vectors.shape[0]:
            raise ValueError("metadata length must match number of vectors")

        if self.index is not None:
            self.index.add(vectors)
        else:
            self._numpy_vectors = np.vstack([self._numpy_vectors, vectors])

        self.metadata.extend(list(metadata))

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict]:
        if len(self.metadata) == 0:
            return []

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        query_vector = query_vector.astype(np.float32)

        k = min(top_k, len(self.metadata))

        if self.index is not None:
            scores, ids = self.index.search(query_vector, k)
            result_ids = ids[0]
            result_scores = scores[0]
        else:
            sims = query_vector @ self._numpy_vectors.T
            order = np.argsort(-sims[0])[:k]
            result_ids = order
            result_scores = sims[0][order]

        results: List[Dict] = []
        for idx, score in zip(result_ids, result_scores):
            if idx < 0:
                continue
            meta = dict(self.metadata[int(idx)])
            meta["score"] = float(score)
            results.append(meta)
        return results


if __name__ == "__main__":
    store = FaissVectorStore(dim=4)
    vectors = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    store.add(vectors, [{"id": "a"}, {"id": "b"}])
    print(store.search(np.array([1, 0, 0, 0], dtype=np.float32), top_k=1))
