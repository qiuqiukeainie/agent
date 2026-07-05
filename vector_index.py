from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SearchHit:
    index: int
    score: float


class VectorIndex:
    def __init__(self) -> None:
        self.backend = "numpy"
        self.ids: list[int] = []
        self.matrix = np.empty((0, 0), dtype=np.float32)
        self.faiss_index = None

    def build(self, vectors: list[np.ndarray], ids: list[int]) -> None:
        self.ids = ids
        if not vectors:
            self.matrix = np.empty((0, 0), dtype=np.float32)
            self.faiss_index = None
            return

        self.matrix = np.vstack([normalize(vector) for vector in vectors]).astype(np.float32)
        self.faiss_index = build_faiss_index(self.matrix)
        self.backend = "faiss-flatip" if self.faiss_index is not None else "numpy"

    def search(self, query: np.ndarray, top_k: int) -> list[SearchHit]:
        if self.matrix.size == 0:
            return []
        query = normalize(query.astype(np.float32)).reshape(1, -1)
        top_k = min(top_k, len(self.ids))
        if self.faiss_index is not None:
            scores, indexes = self.faiss_index.search(query, top_k)
            return [
                SearchHit(index=self.ids[int(index)], score=float(score))
                for score, index in zip(scores[0], indexes[0])
                if index >= 0
            ]

        scores = self.matrix @ query[0]
        ranked = np.argsort(scores)[::-1][:top_k]
        return [SearchHit(index=self.ids[int(index)], score=float(scores[index])) for index in ranked]


def build_faiss_index(matrix: np.ndarray):
    try:
        import faiss
    except Exception:
        return None

    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return index


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm
