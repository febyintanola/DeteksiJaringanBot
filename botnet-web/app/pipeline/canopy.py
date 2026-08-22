"""Canopy clustering utilities that build TF-IDF embeddings and optional ANN indexes."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from loguru import logger
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from .text_features import fit_tuned_tfidf

_CANOPY_DIM = 256
_CANOPY_MAX_FEATURES = 20000
_THRESHOLD_TUNING_SAMPLE_LIMIT = 1500
_THRESHOLD_GRID = tuple(round(0.60 + (0.05 * idx), 2) for idx in range(8))


def _normalize_similarity_thresholds(t1: float, t2: float) -> Tuple[float, float]:
    """Treat T1 as the loose outer threshold and T2 as the strict inner threshold."""
    if t1 <= 0 or t2 <= 0:
        raise ValueError("Ambang canopy harus lebih besar dari 0.")
    if t1 > t2:
        logger.warning(
            "Menukar ambang canopy agar T1/S_outer <= T2/S_inner untuk cosine similarity "
            "(nilai diterima T1={}, T2={})",
            t1,
            t2,
        )
        return t2, t1
    return t1, t2


@dataclass
class CanopyTiming:
    """Timing breakdown for canopy stages used by benchmarks and diagnostics."""

    vectorize_sec: float = 0.0
    reduce_sec: float = 0.0
    ann_index_sec: float = 0.0
    ann_query_sec: float = 0.0
    brute_force_sec: float = 0.0
    assignment_sec: float = 0.0
    threshold_grid_sec: float = 0.0
    total_sec: float = 0.0
    used_ann: bool = False
    ann_fallback: bool = False


@dataclass
class CanopyArtifacts:
    """Container for canopy outputs returned to the pipeline."""

    assignments: Dict[str, str]
    canopies: Dict[str, List[str]]
    embeddings: Dict[str, np.ndarray]
    vectorizer: Optional[TfidfVectorizer] = None
    reducer: Optional[TruncatedSVD] = None
    timing: CanopyTiming = field(default_factory=CanopyTiming)
    threshold_t1: Optional[float] = None
    threshold_t2: Optional[float] = None
    threshold_silhouette: Optional[float] = None
    threshold_candidates: int = 0


class _ANNIndex:
    """Thin wrapper around hnswlib index with graceful fallback when unavailable."""

    def __init__(self, dim: int) -> None:
        self._index = None
        self._count = 0
        self.dim = dim
        try:
            import hnswlib  # type: ignore

            index = hnswlib.Index(space="cosine", dim=dim)
            self._index = index
        except Exception as exc:  # pragma: no cover - hnswlib missing or unsupported
            logger.warning("hnswlib tidak tersedia, memakai pencarian canopy brute-force: {}", exc)

    def init(self, total: int, m: int = 32, ef_construction: int = 200) -> None:
        if not self.is_available:
            return
        self._count = total
        self._index.init_index(max_elements=total, ef_construction=ef_construction, M=m)
        self._index.set_ef(max(64, total))

    def add(self, vectors: np.ndarray) -> None:
        if not self.is_available:
            return
        ids = np.arange(vectors.shape[0])
        self._index.add_items(vectors, ids)

    def query(self, vector: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_available:
            raise RuntimeError("Indeks ANN tidak tersedia.")
        k = max(1, min(k, self._count))
        self._index.set_ef(max(64, k))
        labels, distances = self._index.knn_query(vector, k=k)
        return labels[0], distances[0]

    @property
    def is_available(self) -> bool:
        return self._index is not None


def _fit_vectorizer(user_texts: Dict[str, str], max_features: int = _CANOPY_MAX_FEATURES) -> Tuple[TfidfVectorizer, np.ndarray, List[str]]:
    try:
        user_ids, vectorizer, matrix = fit_tuned_tfidf(user_texts, max_features=max_features, min_users=1)
    except ValueError as exc:
        raise ValueError("Tidak ada teks pengguna yang bisa dipakai untuk komputasi TF-IDF canopy.") from exc
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
) -> Tuple[Dict[str, str], Dict[str, List[str]], CanopyTiming]:
    t1, t2 = _normalize_similarity_thresholds(t1, t2)

    n = len(user_ids)
    ann = _ANNIndex(vectors.shape[1]) if use_ann and n > 10 else None
    timing = CanopyTiming(used_ann=bool(ann and ann.is_available))
    if ann and ann.is_available:
        t_ann_index_start = perf_counter()
        ann.init(n)
        ann.add(vectors)
        timing.ann_index_sec = perf_counter() - t_ann_index_start

    assignments: Dict[str, str] = {}
    canopies: Dict[str, List[str]] = {}
    remaining = set(range(n))
    canopy_idx = 0
    ann_failed = False

    while remaining:
        pivot = remaining.pop()
        pivot_vec = vectors[pivot]
        members = []
        to_remove = set()

        if ann and ann.is_available and not ann_failed:
            t_ann_query_start = perf_counter()
            try:
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
            except RuntimeError as exc:
                ann_failed = True
                timing.ann_fallback = True
                logger.warning("Query ANN canopy gagal, memakai pencarian brute-force: {}", exc)
            finally:
                timing.ann_query_sec += perf_counter() - t_ann_query_start

        if not members and not to_remove:
            t_bruteforce_start = perf_counter()
            sims = vectors @ pivot_vec
            for idx, sim in enumerate(sims):
                if sim >= t1:
                    members.append(idx)
                if sim >= t2:
                    to_remove.add(idx)
            timing.brute_force_sec += perf_counter() - t_bruteforce_start

        t_assignment_start = perf_counter()
        if pivot not in members:
            members.append(pivot)
        if pivot not in to_remove:
            to_remove.add(pivot)

        canopy_label = f"canopy_{canopy_idx:03d}"
        assigned_indexes = sorted(set(members) | set(to_remove))
        assigned_users = []
        for idx in assigned_indexes:
            uid = user_ids[idx]
            assignments.setdefault(uid, canopy_label)
            assigned_users.append(uid)
        canopies[canopy_label] = assigned_users

        remaining -= to_remove
        canopy_idx += 1
        timing.assignment_sec += perf_counter() - t_assignment_start

    return assignments, canopies, timing


def _canopy_from_similarity_matrix(
    user_ids: Sequence[str],
    similarities: np.ndarray,
    t1: float,
    t2: float,
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    t1, t2 = _normalize_similarity_thresholds(t1, t2)

    n = len(user_ids)
    assignments: Dict[str, str] = {}
    canopies: Dict[str, List[str]] = {}
    remaining = set(range(n))
    canopy_idx = 0

    while remaining:
        pivot = remaining.pop()
        row = similarities[pivot]
        members = set(int(idx) for idx in np.flatnonzero(row >= t1))
        to_remove = set(int(idx) for idx in np.flatnonzero(row >= t2))
        members.add(pivot)
        to_remove.add(pivot)

        canopy_label = f"canopy_{canopy_idx:03d}"
        assigned_users = []
        for idx in sorted(members | to_remove):
            uid = user_ids[idx]
            assignments.setdefault(uid, canopy_label)
            assigned_users.append(uid)
        canopies[canopy_label] = assigned_users

        remaining -= to_remove
        canopy_idx += 1

    return assignments, canopies


def _silhouette_for_assignments(
    user_ids: Sequence[str],
    vectors: np.ndarray,
    assignments: Dict[str, str],
) -> Optional[float]:
    labels = [assignments.get(uid) for uid in user_ids]
    if any(label is None for label in labels):
        return None

    sample_count = len(labels)
    label_count = len(set(labels))
    if sample_count < 3 or label_count < 2 or label_count >= sample_count:
        return None

    try:
        return float(silhouette_score(vectors, labels, metric="cosine"))
    except Exception as exc:
        logger.debug("Kandidat silhouette ambang canopy dilewati: {}", exc)
        return None


def _tune_thresholds(
    user_ids: Sequence[str],
    vectors: np.ndarray,
    default_t1: float,
    default_t2: float,
) -> Tuple[float, float, Optional[float], int, float]:
    if len(user_ids) < 3:
        return default_t1, default_t2, None, 0, 0.0

    t0 = perf_counter()
    if len(user_ids) > _THRESHOLD_TUNING_SAMPLE_LIMIT:
        rng = np.random.default_rng(42)
        sample_indices = np.sort(rng.choice(len(user_ids), size=_THRESHOLD_TUNING_SAMPLE_LIMIT, replace=False))
        tuning_user_ids = [user_ids[int(idx)] for idx in sample_indices]
        tuning_vectors = vectors[sample_indices]
    else:
        tuning_user_ids = user_ids
        tuning_vectors = vectors

    grid_values = sorted({*_THRESHOLD_GRID, round(float(default_t1), 2), round(float(default_t2), 2)})
    similarities = np.clip(tuning_vectors @ tuning_vectors.T, -1.0, 1.0)
    best: Optional[Tuple[float, float, float, int]] = None
    candidates = 0

    for candidate_t1 in grid_values:
        for candidate_t2 in grid_values:
            if candidate_t1 > candidate_t2:
                continue
            candidates += 1
            assignments, canopies = _canopy_from_similarity_matrix(tuning_user_ids, similarities, candidate_t1, candidate_t2)
            score = _silhouette_for_assignments(tuning_user_ids, tuning_vectors, assignments)
            if score is None:
                continue
            cluster_count = len(canopies)
            if best is None:
                best = (score, candidate_t1, candidate_t2, cluster_count)
                continue

            _, best_t1, best_t2, best_cluster_count = best
            candidate_distance = abs(candidate_t1 - default_t1) + abs(candidate_t2 - default_t2)
            best_distance = abs(best_t1 - default_t1) + abs(best_t2 - default_t2)
            target_cluster_count = math.sqrt(len(tuning_user_ids))
            candidate_key = (score, -candidate_distance, -abs(cluster_count - target_cluster_count))
            best_key = (best[0], -best_distance, -abs(best_cluster_count - target_cluster_count))
            if candidate_key > best_key:
                best = (score, candidate_t1, candidate_t2, cluster_count)

    elapsed = perf_counter() - t0
    if best is None:
        return default_t1, default_t2, None, candidates, elapsed
    return best[1], best[2], best[0], candidates, elapsed


def build_canopies(
    user_texts: Dict[str, str],
    t1: float,
    t2: float,
    use_ann: bool = True,
    auto_tune: bool = True,
) -> CanopyArtifacts:
    t_total_start = perf_counter()
    if not user_texts:
        logger.info("Canopy clustering dilewati: user_texts kosong")
        return CanopyArtifacts(assignments={}, canopies={}, embeddings={})

    t_vectorize_start = perf_counter()
    vectorizer, matrix, user_ids = _fit_vectorizer(user_texts)
    vectorize_sec = perf_counter() - t_vectorize_start

    t_reduce_start = perf_counter()
    reduced, reducer = _reduce(matrix)
    reduce_sec = perf_counter() - t_reduce_start

    selected_t1, selected_t2 = _normalize_similarity_thresholds(t1, t2)
    threshold_silhouette: Optional[float] = None
    threshold_candidates = 0
    threshold_grid_sec = 0.0
    if auto_tune:
        selected_t1, selected_t2, threshold_silhouette, threshold_candidates, threshold_grid_sec = _tune_thresholds(
            user_ids,
            reduced,
            selected_t1,
            selected_t2,
        )

    assignments, canopies, timing = _canopy_from_vectors(user_ids, reduced, selected_t1, selected_t2, use_ann=use_ann)
    timing.vectorize_sec = vectorize_sec
    timing.reduce_sec = reduce_sec
    timing.threshold_grid_sec = threshold_grid_sec
    timing.total_sec = perf_counter() - t_total_start
    embeddings = {uid: reduced[idx] for idx, uid in enumerate(user_ids)}

    return CanopyArtifacts(
        assignments=assignments,
        canopies=canopies,
        embeddings=embeddings,
        vectorizer=vectorizer,
        reducer=reducer,
        timing=timing,
        threshold_t1=selected_t1,
        threshold_t2=selected_t2,
        threshold_silhouette=threshold_silhouette,
        threshold_candidates=threshold_candidates,
    )
