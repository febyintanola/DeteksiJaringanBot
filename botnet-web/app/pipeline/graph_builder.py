from typing import Any, Dict, List, Tuple, Optional, Union
import math
from collections import defaultdict

import numpy as np

from .parser import parse_comments
from .text_features import fit_tuned_tfidf

_CONTENT_MAX_FEATURES = 20000


def _normalize_layer(weights: Dict[Tuple[str, str], float]) -> Dict[Tuple[str, str], float]:
    if not weights:
        return {}
    degree = defaultdict(float)
    for (a, b), value in weights.items():
        degree[a] += value
        degree[b] += value
    normalized: Dict[Tuple[str, str], float] = {}
    for (a, b), value in weights.items():
        denom = max(1.0, degree[a] * degree[b])
        normalized[(a, b)] = value / math.log(2.0 + denom)
    return normalized


def _fit_tfidf(user_texts: Dict[str, str]) -> Tuple[List[str], Any]:
    try:
        user_ids, _vectorizer, matrix = fit_tuned_tfidf(user_texts, max_features=_CONTENT_MAX_FEATURES, min_users=2)
    except ValueError:
        return [], None

    return user_ids, matrix


def _content_similarity_edges(user_texts: Dict[str, str], k_neighbors: int) -> Dict[Tuple[str, str], float]:
    if k_neighbors <= 0:
        return {}

    user_ids, matrix = _fit_tfidf(user_texts)
    if matrix is None or len(user_ids) < 2:
        return {}

    sim_matrix = (matrix @ matrix.T).tocsr()
    sim_matrix.setdiag(0.0)
    sim_matrix.eliminate_zeros()

    weights: Dict[Tuple[str, str], float] = {}
    for row_idx, src in enumerate(user_ids):
        row = sim_matrix.getrow(row_idx)
        if row.nnz == 0:
            continue

        data = row.data
        indices = row.indices
        limit = min(k_neighbors, len(data))
        if len(data) > limit:
            top_positions = np.argpartition(data, -limit)[-limit:]
            data = data[top_positions]
            indices = indices[top_positions]

        order = np.argsort(-data)
        for pos in order:
            sim = float(data[pos])
            if sim <= 0:
                continue
            dst = user_ids[int(indices[pos])]
            pair = (src, dst) if src < dst else (dst, src)
            weights[pair] = max(weights.get(pair, 0.0), sim)

    return weights


def _temporal_pair_score(left: Dict[str, float], right: Dict[str, float], time_scale: float) -> float:
    center_gap = abs(left["center"] - right["center"])
    center_proximity = max(0.0, 1.0 - (center_gap / max(1.0, time_scale)))

    left_span = left["last"] - left["first"]
    right_span = right["last"] - right["first"]
    if left_span <= 0 and right_span <= 0:
        overlap_ratio = center_proximity
    else:
        overlap = max(0.0, min(left["last"], right["last"]) - max(left["first"], right["first"]))
        union = max(1.0, max(left["last"], right["last"]) - min(left["first"], right["first"]))
        overlap_ratio = overlap / union if overlap > 0 else max(0.0, 1.0 - (center_gap / max(1.0, time_scale * 1.5)))

    rate_gap = abs(math.log1p(left["rate"]) - math.log1p(right["rate"]))
    rate_similarity = max(0.0, 1.0 - (rate_gap / math.log1p(5.0)))

    return max(0.0, min(1.0, (0.55 * center_proximity) + (0.25 * overlap_ratio) + (0.20 * rate_similarity)))


def _temporal_similarity_edges(user_summary: Dict[str, Dict[str, Any]], k_neighbors: int) -> Dict[Tuple[str, str], float]:
    if k_neighbors <= 0:
        return {}

    activity_windows: List[Dict[str, float]] = []
    for uid, summary in user_summary.items():
        raw_timestamps = summary.get("timestamps", [])
        timestamps = sorted(float(ts) for ts in raw_timestamps if ts is not None) if isinstance(raw_timestamps, list) else []
        if not timestamps:
            first_ts = summary.get("first_timestamp")
            last_ts = summary.get("last_timestamp")
            if first_ts is not None:
                timestamps.append(float(first_ts))
            if last_ts is not None and last_ts != first_ts:
                timestamps.append(float(last_ts))
        if not timestamps:
            continue

        first = timestamps[0]
        last = timestamps[-1]
        span = max(0.0, last - first)
        count = max(1.0, float(summary.get("comment_count", len(timestamps)) or len(timestamps) or 1.0))
        activity_windows.append(
            {
                "uid": uid,
                "first": first,
                "last": last,
                "center": (first + last) / 2.0,
                "rate": count / max(1.0, span),
            }
        )

    if len(activity_windows) < 2:
        return {}

    activity_windows.sort(key=lambda item: item["center"])
    global_span = max(1.0, activity_windows[-1]["center"] - activity_windows[0]["center"])
    time_scale = max(120.0, min(global_span, max(180.0, global_span * 0.12)))
    search_window = max(6, k_neighbors * 3)

    weights: Dict[Tuple[str, str], float] = {}
    for index, src in enumerate(activity_windows):
        candidates: List[Tuple[float, str]] = []
        left_bound = max(0, index - search_window)
        right_bound = min(len(activity_windows), index + search_window + 1)
        for cursor in range(left_bound, right_bound):
            if cursor == index:
                continue
            dst = activity_windows[cursor]
            score = _temporal_pair_score(src, dst, time_scale)
            if score <= 0.18:
                continue
            candidates.append((score, str(dst["uid"])))

        candidates.sort(key=lambda item: item[0], reverse=True)
        for score, dst_uid in candidates[:k_neighbors]:
            pair = (str(src["uid"]), dst_uid)
            if pair[0] > pair[1]:
                pair = (pair[1], pair[0])
            weights[pair] = max(weights.get(pair, 0.0), score)

    return weights


def build_graph(
    comments: List[Dict],
    alpha: float = 1.0,
    beta: float = 0.8,
    gamma: float = 0.5,
    delta: float = 0.5,
    k_neighbors: int = 15,
    parsed: Optional[Dict[str, Any]] = None,
    canopy_assignments: Optional[Dict[str, str]] = None,
    cross_canopy_penalty: float = 0.6,
    return_context: bool = False,
) -> Union[Tuple[List[Dict], List[Dict]], Tuple[List[Dict], List[Dict], Dict[str, Any]]]:
    """Build weighted graph from parsed comment layers.

    When ``parsed`` is provided, skips re-parsing comments and reuses the aggregates.
    If canopy assignments are given, cross-canopy edges are down-weighted.
    """
    if parsed is None:
        parsed = parse_comments(comments)
    user_summary: Dict[str, Dict[str, Any]] = parsed["user_summary"]
    mention_norm = _normalize_layer(parsed["mention_edges"])
    reply_norm = _normalize_layer(parsed["reply_edges"])
    thread_norm = _normalize_layer(parsed["co_thread_edges"])
    temporal_norm = _normalize_layer(_temporal_similarity_edges(user_summary, k_neighbors))
    user_texts = parsed["user_texts"]

    combined = defaultdict(float)
    # `delta` now represents the structural conversation layer plus temporal synchronization.
    for layer, weight in (
        (mention_norm, alpha),
        (reply_norm, beta),
        (thread_norm, delta * 0.7),
        (temporal_norm, delta * 0.3),
    ):
        for key, value in layer.items():
            combined[key] += weight * value

    user_ids = sorted(user_summary.keys())
    for key, value in _content_similarity_edges(user_texts, k_neighbors).items():
        combined[key] += gamma * value

    adjacency: Dict[str, Dict[str, float]] = defaultdict(dict)
    for (a, b), value in combined.items():
        if value <= 0:
            continue
        if canopy_assignments and cross_canopy_penalty < 1.0:
            ca = canopy_assignments.get(a)
            cb = canopy_assignments.get(b)
            if ca and cb and ca != cb:
                value *= max(0.0, cross_canopy_penalty)
        adjacency[a][b] = max(adjacency[a].get(b, 0.0), value)
        adjacency[b][a] = max(adjacency[b].get(a, 0.0), value)

    candidate_pairs = set()
    for uid, neighbors in adjacency.items():
        top_neighbors = sorted(neighbors.items(), key=lambda item: item[1], reverse=True)[:k_neighbors]
        for vid, _ in top_neighbors:
            candidate_pairs.add(tuple(sorted((uid, vid))))

    edges: List[Dict] = []
    for a, b in sorted(candidate_pairs):
        weight = adjacency[a].get(b)
        if weight is None:
            continue
        edges.append({"source": a, "target": b, "weight": float(weight)})

    nodes: List[Dict] = []
    for uid in user_ids:
        summary = user_summary.get(uid, {})
        canopy_id = canopy_assignments.get(uid) if canopy_assignments else None
        nodes.append(
            {
                "id": uid,
                "label": summary.get("username") or uid,
                "canopy": canopy_id,
                "comment_count": summary.get("comment_count", 0),
                "reply_count": summary.get("reply_count", 0),
                "mention_count": summary.get("mention_count", 0),
                "thread_count": summary.get("thread_count", 0),
                "total_likes": summary.get("total_likes", 0),
                "avg_likes": summary.get("avg_likes", 0.0),
                "first_timestamp": summary.get("first_timestamp"),
                "last_timestamp": summary.get("last_timestamp"),
                "threads": summary.get("threads", []),
                "text_sample": (user_texts.get(uid, "")[:280] if user_texts.get(uid) else ""),
            }
        )

    if return_context:
        return nodes, edges, parsed
    return nodes, edges
