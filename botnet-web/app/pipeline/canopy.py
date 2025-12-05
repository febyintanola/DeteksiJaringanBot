"""Canopy clustering utilities that build TF-IDF embeddings and optional ANN indexes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from loguru import logger
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

_CANOPY_DIM = 256
_CANOPY_MAX_FEATURES = 6000


@dataclass
class CanopyArtifacts:
    """Container for canopy outputs returned to the pipeline."""

    assignments: Dict[str, str]
    canopies: Dict[str, List[str]]
    embeddings: Dict[str, np.ndarray]
    vectorizer: Optional[TfidfVectorizer] = None
    reducer: Optional[TruncatedSVD] = None


class _ANNIndex:
    """Thin wrapper around hnswlib index with graceful fallback when unavailable."""

    def __init__(self, dim: int) -> None:
        self._index = None
        self.dim = dim
        try:
            import hnswlib  # type: ignore

            index = hnswlib.Index(space="cosine", dim=dim)
            self._index = index
        except Exception as exc:  # pragma: no cover - hnswlib missing or unsupported
            logger.warning("hnswlib unavailable, falling back to brute-force canopy search: {}", exc)

    def init(self, total: int, m: int = 32, ef_construction: int = 200) -> None:
        if not self.is_available:
            return
        self._index.init_index(max_elements=total, ef_construction=ef_construction, M=m)
        self._index.set_ef(64)

    def add(self, vectors: np.ndarray) -> None:
        if not self.is_available:
            return
        ids = np.arange(vectors.shape[0])
        self._index.add_items(vectors, ids)

    def query(self, vector: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_available:
            raise RuntimeError("ANN index is not available")
        labels, distances = self._index.knn_query(vector, k=k)
        return labels[0], distances[0]

    @property
    def is_available(self) -> bool:
        return self._index is not None


def _fit_vectorizer(user_texts: Dict[str, str], max_features: int = _CANOPY_MAX_FEATURES) -> Tuple[TfidfVectorizer, np.ndarray, List[str]]:
    user_ids = []
    corpus = []
    for uid, text in user_texts.items():
        cleaned = (text or "").strip()
        if not cleaned:
            continue
        user_ids.append(uid)
        corpus.append(cleaned)
    if not user_ids:
        raise ValueError("No user texts available for TF-IDF canopy computation")

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        strip_accents="unicode",
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        norm="l2",
    )
    try:
        matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        # Fallback for tiny datasets where min_df=2 removes everything
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            strip_accents="unicode",
            lowercase=True,
            ngram_range=(1, 1),
            min_df=1,
            norm="l2",
        )
        matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix, user_ids


def _reduce(matrix, target_dim: int = _CANOPY_DIM) -> Tuple[np.ndarray, Optional[TruncatedSVD]]:
    n_samples, n_features = matrix.shape
    if n_features <= target_dim:
        dense = matrix.toarray()
        return normalize(dense), None

    dim = min(target_dim, n_features - 1, n_samples - 1 if n_samples > 1 else target_dim)
    if dim < 2:
        dense = matrix.toarray()
        return normalize(dense), None

    reducer = TruncatedSVD(n_components=dim, random_state=42, algorithm="randomized")
    reduced = reducer.fit_transform(matrix)
    return normalize(reduced), reducer


def _canopy_from_vectors(
    user_ids: Sequence[str],
    vectors: np.ndarray,
    t1: float,
    t2: float,
    use_ann: bool = True,
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    if t1 <= 0 or t2 <= 0:
        raise ValueError("Canopy thresholds must be > 0")
    if t1 < t2:
        logger.warning("Swapping canopy thresholds to enforce T1 >= T2 (received T1=%s, T2=%s)", t1, t2)
        t1, t2 = t2, t1

    n = len(user_ids)
    ann = _ANNIndex(vectors.shape[1]) if use_ann and n > 10 else None
    if ann and ann.is_available:
        ann.init(n)
        ann.add(vectors)

    assignments: Dict[str, str] = {}
    canopies: Dict[str, List[str]] = {}
    remaining = set(range(n))
    canopy_idx = 0

    while remaining:
        pivot = remaining.pop()
        pivot_vec = vectors[pivot]
        members = []
        to_remove = set()

        if ann and ann.is_available:
            labels, distances = ann.query(pivot_vec, k=n)
            for idx, dist in zip(labels, distances):
                if idx >= n:
                    continue
                sim = 1.0 - float(dist)
                if math.isnan(sim):
                    continue
                if sim >= t1:
                    members.append(idx)
                if sim >= t2:
                    to_remove.add(idx)
        else:
            sims = vectors @ pivot_vec
            for idx, sim in enumerate(sims):
                if sim >= t1:
                    members.append(idx)
                if sim >= t2:
                    to_remove.add(idx)

        if pivot not in members:
            members.append(pivot)
        if pivot not in to_remove:
            to_remove.add(pivot)

        canopy_label = f"canopy_{canopy_idx:03d}"
        assigned_users = []
        for idx in members:
            uid = user_ids[idx]
            assignments.setdefault(uid, canopy_label)
            assigned_users.append(uid)
        canopies[canopy_label] = assigned_users

        remaining -= to_remove
        canopy_idx += 1

    return assignments, canopies


def build_canopies(
    user_texts: Dict[str, str],
    t1: float,
    t2: float,
    use_ann: bool = True,
) -> CanopyArtifacts:
    if not user_texts:
        logger.info("Skipping canopy clustering: empty user_texts")
        return CanopyArtifacts(assignments={}, canopies={}, embeddings={})

    vectorizer, matrix, user_ids = _fit_vectorizer(user_texts)
    reduced, reducer = _reduce(matrix)

    assignments, canopies = _canopy_from_vectors(user_ids, reduced, t1, t2, use_ann=use_ann)
    embeddings = {uid: reduced[idx] for idx, uid in enumerate(user_ids)}

    return CanopyArtifacts(
        assignments=assignments,
        canopies=canopies,
        embeddings=embeddings,
        vectorizer=vectorizer,
        reducer=reducer,
    )