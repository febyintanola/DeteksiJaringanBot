from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from .text_features import fit_tuned_tfidf

_COMPONENT_WEIGHTS = {
    "cluster_density": 0.25,
    "content_repetition": 0.25,
    "temporal_burst": 0.20,
    "account_age": 0.15,
    "high_frequency": 0.15,
}
_CONTENT_MAX_FEATURES = 20000


def _safe_mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _quantile(values: List[float], q: float, fallback: float = 0.0) -> float:
    if not values:
        return fallback
    return float(np.quantile(np.asarray(values, dtype=float), q))


def _build_text_matrix(nodes: List[Dict[str, Any]], parsed: Optional[Dict[str, Any]]) -> Tuple[Dict[str, int], Any]:
    user_texts = (parsed or {}).get("user_texts") or {}
    if not user_texts:
        user_texts = {node["id"]: node.get("text_sample", "") for node in nodes if node.get("text_sample")}

    try:
        user_ids, _vectorizer, matrix = fit_tuned_tfidf(user_texts, max_features=_CONTENT_MAX_FEATURES, min_users=2)
    except ValueError:
        return {}, None
    return {uid: idx for idx, uid in enumerate(user_ids)}, matrix


def _mean_pairwise_similarity(members: List[str], text_index: Dict[str, int], text_matrix: Any) -> float:
    if text_matrix is None:
        return 0.0
    indices = [text_index[uid] for uid in members if uid in text_index]
    n = len(indices)
    if n < 2:
        return 0.0
    subset = text_matrix[indices]
    sim_matrix = (subset @ subset.T).toarray()
    total = float(sim_matrix.sum() - np.trace(sim_matrix))
    return _clamp01(total / (n * (n - 1)))


def _weighted_component_average(components: Dict[str, Optional[float]]) -> float:
    weighted_sum = 0.0
    total_weight = 0.0
    for name, weight in _COMPONENT_WEIGHTS.items():
        value = components.get(name)
        if value is None:
            continue
        weighted_sum += weight * _clamp01(value)
        total_weight += weight
    if total_weight <= 0:
        return 0.0
    return weighted_sum / total_weight


def _relative_excess_score(value: float, baseline: float, stretch: float = 4.0) -> float:
    if baseline <= 0 or value <= baseline:
        return 0.0
    ratio = value / baseline
    return _clamp01((ratio - 1.0) / max(1.0, stretch))


def _inverse_span_score(value: float, reference: float) -> float:
    if reference <= 0:
        return 0.0
    return _clamp01(1.0 - _clamp01(value / reference))


def _cluster_density_score(structural_density: float, avg_weight: float, edge_weight_p90: float) -> Tuple[float, float]:
    weighted_density = _clamp01(avg_weight / edge_weight_p90) if edge_weight_p90 > 0 else 0.0
    return _clamp01((0.55 * structural_density) + (0.45 * weighted_density)), weighted_density


def _extract_activity_points(node: Dict[str, Any], summary: Dict[str, Any]) -> Tuple[List[float], Optional[float]]:
    raw_timestamps = summary.get("timestamps", [])
    timestamps: List[float] = []
    if isinstance(raw_timestamps, list) and raw_timestamps:
        timestamps.extend(float(ts) for ts in raw_timestamps if ts is not None)
    else:
        first_ts = node.get("first_timestamp")
        last_ts = node.get("last_timestamp")
        if first_ts is not None:
            timestamps.append(float(first_ts))
        if last_ts is not None and last_ts != first_ts:
            timestamps.append(float(last_ts))

    if not timestamps:
        return [], None

    timestamps.sort()
    return timestamps, (timestamps[0] + timestamps[-1]) / 2.0


def _temporal_burst_score(
    cluster_comment_total: float,
    cluster_timestamps: List[float],
    member_centers: List[float],
    global_comment_rate: float,
    global_span: float,
) -> Tuple[float, float, float, float, float]:
    if not cluster_timestamps:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    cluster_span = max(1.0, max(cluster_timestamps) - min(cluster_timestamps)) if len(cluster_timestamps) >= 2 else 1.0
    cluster_comment_rate = cluster_comment_total / cluster_span if cluster_comment_total > 0 else 0.0
    rate_pressure = _relative_excess_score(cluster_comment_rate, global_comment_rate, stretch=3.0)
    compact_window = _inverse_span_score(cluster_span, global_span)

    if len(member_centers) >= 2:
        center_alignment = _inverse_span_score(max(member_centers) - min(member_centers), global_span)
    elif member_centers:
        center_alignment = compact_window
    else:
        center_alignment = 0.0

    score = _clamp01((0.45 * rate_pressure) + (0.30 * compact_window) + (0.25 * center_alignment))
    return score, rate_pressure, compact_window, center_alignment, cluster_span


def _high_frequency_score(
    cluster_comment_counts: List[float],
    median_comment_count: float,
    p90_excess_comments: float,
) -> Tuple[float, float, float]:
    if not cluster_comment_counts:
        return 0.0, 0.0, 0.0

    repeat_threshold = max(1.0, median_comment_count)
    repeat_ratio = _safe_mean([1.0 if count > repeat_threshold else 0.0 for count in cluster_comment_counts])
    mean_excess = _safe_mean([max(0.0, count - repeat_threshold) for count in cluster_comment_counts])
    excess_pressure = _clamp01(mean_excess / max(1.0, p90_excess_comments))
    return _clamp01((0.5 * repeat_ratio) + (0.5 * excess_pressure)), repeat_ratio, excess_pressure


def _top_reasons(stats: Dict[str, Any]) -> List[str]:
    candidates = [
        ("kepadatan cluster tinggi", stats.get("cluster_density", 0.0)),
        ("repetisi konten tinggi", stats.get("content_repetition", 0.0)),
        ("aktivitas temporal sinkron", stats.get("temporal_burst", 0.0)),
        ("frekuensi komentar tinggi", stats.get("high_frequency", 0.0)),
    ]
    account_age = stats.get("account_age")
    if account_age is not None:
        candidates.append(("proporsi akun baru tinggi", account_age))

    top = [label for label, value in sorted(candidates, key=lambda item: item[1], reverse=True) if value >= 0.45]
    return top[:3]


def score_clusters(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    clusters: Dict[str, List[str]],
    canopy_assignments: Optional[Dict[str, str]] = None,
    parsed: Optional[Dict[str, Any]] = None,
):
    """
    Score suspicious clusters using a composite unsupervised signal:
    cluster density, content repetition, temporal burst, optional new-account ratio,
    and high comment frequency.

    Every score-producing component is normalized to the 0..1 range where
    larger values always mean "more suspicious".

    Returns tuple of (suspicious_user_ids, metrics_dict, suspicious_details, cluster_stats, node_scores).
    """
    G = nx.Graph()
    label_map: Dict[str, str] = {}
    node_map = {node["id"]: node for node in nodes}
    for node in nodes:
        G.add_node(node["id"], label=node.get("label"))
        label_map[node["id"]] = node.get("label", node["id"])

    total_edge_weight = 0.0
    intra_canopy_weight = 0.0
    edge_weights: List[float] = []
    for edge in edges:
        weight = float(edge.get("weight", 0.0))
        u = edge["source"]
        v = edge["target"]
        G.add_edge(u, v, weight=weight)
        total_edge_weight += weight
        edge_weights.append(weight)
        if canopy_assignments:
            cu = canopy_assignments.get(u)
            cv = canopy_assignments.get(v)
            if cu and cv and cu == cv:
                intra_canopy_weight += weight

    summary_lookup = (parsed or {}).get("user_summary") or {}
    text_index, text_matrix = _build_text_matrix(nodes, parsed)

    global_comment_counts = [float(node.get("comment_count", 0) or 0.0) for node in nodes]
    global_excess_comments = [max(0.0, count - 1.0) for count in global_comment_counts]
    median_comment_count = _quantile(global_comment_counts, 0.5, fallback=1.0)
    p90_excess_comments = max(1.0, _quantile(global_excess_comments, 0.9, fallback=1.0))

    all_timestamps: List[float] = []
    for node in nodes:
        uid = node["id"]
        summary = summary_lookup.get(uid, {})
        timestamps, _ = _extract_activity_points(node, summary)
        all_timestamps.extend(timestamps)

    global_span = max(1.0, (max(all_timestamps) - min(all_timestamps)) if len(all_timestamps) >= 2 else 1.0)
    global_comment_total = max(1.0, float(sum(global_comment_counts)))
    global_comment_rate = global_comment_total / global_span
    edge_weight_p90 = max(1.0, _quantile(edge_weights, 0.9, fallback=1.0))

    cluster_scores: Dict[str, float] = {}
    cluster_stats: Dict[str, Dict[str, Any]] = {}
    node_score = {node["id"]: 0.0 for node in nodes}
    node_prominence = {node["id"]: 0.0 for node in nodes}
    node_cluster: Dict[str, str] = {}

    for cid, members in clusters.items():
        for uid in members:
            node_cluster[uid] = cid

        sub = G.subgraph(members)
        if sub.number_of_nodes() <= 1:
            cluster_stats[cid] = {
                "cluster_density": 0.0,
                "structural_density": 0.0,
                "weighted_density": 0.0,
                "content_repetition": 0.0,
                "temporal_burst": 0.0,
                "temporal_rate_pressure": 0.0,
                "temporal_compactness": 0.0,
                "temporal_alignment": 0.0,
                "cluster_span_sec": 0.0,
                "account_age": None,
                "account_age_coverage": 0.0,
                "high_frequency": 0.0,
                "high_frequency_repeat_ratio": 0.0,
                "high_frequency_excess": 0.0,
                "avg_weight": 0.0,
                "deg_conc": 0.0,
                "deg_conc_raw": 0.0,
                "cluster_size": len(members),
                "score": 0.0,
                "reasons": [],
                "is_suspicious": False,
            }
            cluster_scores[cid] = 0.0
            continue

        weights = [float(data.get("weight", 0.0)) for _, _, data in sub.edges(data=True)]
        avg_weight = _safe_mean(weights)
        structural_density = _clamp01(nx.density(sub))
        cluster_density, weighted_density = _cluster_density_score(structural_density, avg_weight, edge_weight_p90)

        degrees = [deg for _, deg in sub.degree()]
        avg_degree = _safe_mean([float(deg) for deg in degrees])
        deg_conc_raw = (max(degrees) / avg_degree) if avg_degree > 0 else 0.0
        deg_conc = _clamp01((deg_conc_raw - 1.0) / 3.0) if deg_conc_raw > 0 else 0.0

        content_repetition = _mean_pairwise_similarity(members, text_index, text_matrix)

        cluster_comment_counts: List[float] = []
        cluster_comment_count_map: Dict[str, float] = {}
        cluster_timestamps: List[float] = []
        member_centers: List[float] = []
        account_age_values: List[float] = []
        for uid in members:
            node = node_map.get(uid, {})
            summary = summary_lookup.get(uid, {})
            comment_count = float(node.get("comment_count", 0) or summary.get("comment_count", 0) or 0.0)
            cluster_comment_counts.append(comment_count)
            cluster_comment_count_map[uid] = comment_count

            timestamps, center = _extract_activity_points(node, summary)
            cluster_timestamps.extend(timestamps)
            if center is not None:
                member_centers.append(center)

            new_account_signal = summary.get("new_account_signal")
            if new_account_signal is not None:
                account_age_values.append(_clamp01(float(new_account_signal)))

        cluster_comment_total = sum(cluster_comment_counts)
        temporal_burst, rate_pressure, compact_window, center_alignment, cluster_span = _temporal_burst_score(
            cluster_comment_total,
            cluster_timestamps,
            member_centers,
            global_comment_rate,
            global_span,
        )

        high_frequency, repeat_ratio, excess_pressure = _high_frequency_score(
            cluster_comment_counts,
            median_comment_count,
            p90_excess_comments,
        )

        account_age = _safe_mean(account_age_values) if account_age_values else None
        account_age_coverage = (len(account_age_values) / len(members)) if members else 0.0

        components = {
            "cluster_density": cluster_density,
            "content_repetition": content_repetition,
            "temporal_burst": temporal_burst,
            "account_age": account_age,
            "high_frequency": high_frequency,
        }
        score = _weighted_component_average(components)

        stats = {
            "cluster_density": cluster_density,
            "structural_density": structural_density,
            "weighted_density": weighted_density,
            "content_repetition": content_repetition,
            "temporal_burst": temporal_burst,
            "temporal_rate_pressure": rate_pressure,
            "temporal_compactness": compact_window,
            "temporal_alignment": center_alignment,
            "cluster_span_sec": cluster_span,
            "account_age": account_age,
            "account_age_coverage": account_age_coverage,
            "high_frequency": high_frequency,
            "high_frequency_repeat_ratio": repeat_ratio,
            "high_frequency_excess": excess_pressure,
            "avg_weight": avg_weight,
            "deg_conc": deg_conc,
            "deg_conc_raw": deg_conc_raw,
            "cluster_size": len(members),
            "score": score,
            "is_suspicious": False,
        }
        stats["reasons"] = _top_reasons(stats)

        cluster_scores[cid] = score
        cluster_stats[cid] = stats
        node_strengths = {uid: float(sub.degree(uid, weight="weight")) for uid in members}
        max_strength = max(node_strengths.values()) if node_strengths else 1.0
        max_comments = max(cluster_comment_count_map.values()) if cluster_comment_count_map else 1.0
        for uid in members:
            prominence = _clamp01(
                (0.6 * _clamp01(node_strengths.get(uid, 0.0) / max(1.0, max_strength)))
                + (0.4 * _clamp01(cluster_comment_count_map.get(uid, 0.0) / max(1.0, max_comments)))
            )
            node_prominence[uid] = prominence
            node_score[uid] = _clamp01((0.7 * score) + (0.3 * prominence))

    positive_scores = [score for score in cluster_scores.values() if score > 0]
    suspicious_clusters: List[str] = []
    cutoff = 0.0
    if positive_scores:
        percentile_cutoff = _quantile(positive_scores, 0.8, fallback=max(positive_scores))
        cutoff = max(0.5, percentile_cutoff)
        suspicious_clusters = [cid for cid, score in cluster_scores.items() if score >= cutoff and score > 0]
        if not suspicious_clusters:
            top_cluster = max(cluster_scores.items(), key=lambda item: item[1])
            if top_cluster[1] >= 0.35:
                suspicious_clusters = [top_cluster[0]]
                cutoff = top_cluster[1]

    suspicious_cluster_set = set(suspicious_clusters)
    for cid, stats in cluster_stats.items():
        stats["is_suspicious"] = cid in suspicious_cluster_set

    suspicious = [uid for uid, cid in node_cluster.items() if cid in suspicious_cluster_set]

    try:
        graph_weight = G.size(weight="weight")
        if graph_weight == 0:
            modularity = 0.0
        else:
            modularity = 0.0
            for members in clusters.values():
                sub = G.subgraph(members)
                lc = sub.size(weight="weight")
                dc = sum(dict(G.degree(members, weight="weight")).values())
                modularity += (lc / graph_weight) - (dc / (2 * graph_weight)) ** 2
    except Exception:
        modularity = 0.0

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    avg_degree = (2.0 * n_edges / n_nodes) if n_nodes > 0 else 0.0
    density = (2.0 * n_edges / (n_nodes * (n_nodes - 1))) if n_nodes > 1 else 0.0
    try:
        num_components = nx.number_connected_components(G)
    except Exception:
        num_components = 0

    conductances = []
    for members in clusters.values():
        if len(members) < 2 or len(members) >= len(G) - 1:
            continue
        cut_weight = 0.0
        vol_s = 0.0
        vol_not_s = 0.0
        member_set = set(members)
        for u, v, data in G.edges(data=True):
            weight = float(data.get("weight", 0.0))
            if u in member_set and v in member_set:
                vol_s += weight
            elif u not in member_set and v not in member_set:
                vol_not_s += weight
            else:
                cut_weight += weight
        denom = min(vol_s + cut_weight, vol_not_s + cut_weight)
        if denom > 0:
            conductances.append(cut_weight / denom)
    conductance_mean = _safe_mean(conductances)

    cluster_density_values = [stats["cluster_density"] for stats in cluster_stats.values()]
    content_values = [stats["content_repetition"] for stats in cluster_stats.values()]
    temporal_values = [stats["temporal_burst"] for stats in cluster_stats.values()]
    high_frequency_values = [stats["high_frequency"] for stats in cluster_stats.values()]
    account_age_values = [float(stats["account_age"]) for stats in cluster_stats.values() if stats.get("account_age") is not None]
    account_age_coverage_values = [float(stats["account_age_coverage"]) for stats in cluster_stats.values()]

    suspicious_details = []
    for uid in suspicious:
        cid = node_cluster.get(uid)
        stats = cluster_stats.get(cid, {})
        degree_unweighted = G.degree(uid)
        degree_weighted = G.degree(uid, weight="weight")
        top_neighbors = []
        if uid in G:
            for vid, attr in sorted(G[uid].items(), key=lambda item: item[1].get("weight", 0.0), reverse=True)[:5]:
                top_neighbors.append(
                    {
                        "peer_id": vid,
                        "peer_label": label_map.get(vid, vid),
                        "weight": float(attr.get("weight", 0.0)),
                    }
                )

        suspicious_details.append(
            {
                "user_id": uid,
                "username": label_map.get(uid, uid),
                "cluster_id": cid,
                "cluster_size": int(stats.get("cluster_size", 0) or 0),
                "cluster_is_suspicious": bool(stats.get("is_suspicious", False)),
                "canopy_id": canopy_assignments.get(uid) if canopy_assignments else None,
                "score": node_score.get(uid, 0.0),
                "node_prominence": float(node_prominence.get(uid, 0.0) or 0.0),
                "cluster_avg_weight": float(stats.get("avg_weight", 0.0) or 0.0),
                "cluster_degree_concentration": float(stats.get("deg_conc", 0.0) or 0.0),
                "cluster_degree_concentration_raw": float(stats.get("deg_conc_raw", 0.0) or 0.0),
                "cluster_density": float(stats.get("cluster_density", 0.0) or 0.0),
                "content_repetition": float(stats.get("content_repetition", 0.0) or 0.0),
                "temporal_burst": float(stats.get("temporal_burst", 0.0) or 0.0),
                "temporal_alignment": float(stats.get("temporal_alignment", 0.0) or 0.0),
                "temporal_compactness": float(stats.get("temporal_compactness", 0.0) or 0.0),
                "temporal_rate_pressure": float(stats.get("temporal_rate_pressure", 0.0) or 0.0),
                "account_age_ratio": stats.get("account_age"),
                "account_age_coverage": float(stats.get("account_age_coverage", 0.0) or 0.0),
                "high_frequency": float(stats.get("high_frequency", 0.0) or 0.0),
                "high_frequency_repeat_ratio": float(stats.get("high_frequency_repeat_ratio", 0.0) or 0.0),
                "high_frequency_excess": float(stats.get("high_frequency_excess", 0.0) or 0.0),
                "reasons": stats.get("reasons", []),
                "degree": degree_unweighted,
                "strength": degree_weighted,
                "top_edges": top_neighbors,
            }
        )

    metrics = {
        "modularity": modularity,
        "conductance_mean": conductance_mean,
        "avg_degree": avg_degree,
        "density": density,
        "num_components": float(num_components),
        "num_nodes": float(n_nodes),
        "num_edges": float(n_edges),
        "num_clusters": float(len(clusters)),
        "num_suspicious_clusters": float(len(suspicious_cluster_set)),
        "num_suspicious_users": float(len(suspicious)),
        "suspicious_score_cutoff": float(cutoff),
        "avg_cluster_density": _safe_mean(cluster_density_values),
        "avg_content_repetition": _safe_mean(content_values),
        "avg_temporal_burst": _safe_mean(temporal_values),
        "avg_high_frequency": _safe_mean(high_frequency_values),
        "account_age_coverage": _safe_mean(account_age_coverage_values),
    }
    if account_age_values:
        metrics["avg_account_age_ratio"] = _safe_mean(account_age_values)

    if canopy_assignments and total_edge_weight > 0:
        metrics["intra_canopy_edge_ratio"] = intra_canopy_weight / total_edge_weight

    return suspicious, metrics, suspicious_details, cluster_stats, node_score
