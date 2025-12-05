from typing import Any, Dict, List, Tuple, Optional, Union
import math
from collections import defaultdict

from .parser import parse_comments


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


def _tokenize(text: str) -> List[str]:
    return [token for token in text.lower().split() if token]


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    if intersection == 0:
        return 0.0
    return intersection / float(len(a | b))


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
    user_texts = parsed["user_texts"]

    combined = defaultdict(float)
    for layer, weight in (
        (mention_norm, alpha),
        (reply_norm, beta),
        (thread_norm, delta),
    ):
        for key, value in layer.items():
            combined[key] += weight * value

    # Content similarity placeholder (simple token Jaccard)
    token_sets = {uid: set(_tokenize(text)) for uid, text in user_texts.items() if text}
    user_ids = sorted(user_summary.keys())
    for index, src in enumerate(user_ids):
        tokens_src = token_sets.get(src)
        if not tokens_src:
            continue
        for dst in user_ids[index + 1 : index + 1 + k_neighbors]:
            tokens_dst = token_sets.get(dst)
            if not tokens_dst:
                continue
            sim = _jaccard(tokens_src, tokens_dst)
            if sim <= 0:
                continue
            pair = (src, dst) if src < dst else (dst, src)
            combined[pair] += gamma * sim

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
