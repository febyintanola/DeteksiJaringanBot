from typing import List, Dict, Tuple, Optional
import networkx as nx


def score_clusters(
    nodes: List[Dict],
    edges: List[Dict],
    clusters: Dict[str, List[str]],
    canopy_assignments: Optional[Dict[str, str]] = None,
):
    """
    Simple unsupervised scoring: nodes in clusters with higher internal average
    edge weights and degree concentration are more suspicious.
    Returns tuple of (suspicious_user_ids, metrics_dict, suspicious_details)
    where suspicious_details berisi penjelasan singkat mengapa user dicurigai.
    """
    G = nx.Graph()
    label_map = {}
    for n in nodes:
        G.add_node(n["id"], label=n.get("label"))
        label_map[n["id"]] = n.get("label", n["id"])
    total_edge_weight = 0.0
    intra_canopy_weight = 0.0
    for e in edges:
        w = float(e.get("weight", 0.0))
        u = e["source"]
        v = e["target"]
        G.add_edge(u, v, weight=w)
        total_edge_weight += w
        if canopy_assignments:
            cu = canopy_assignments.get(u)
            cv = canopy_assignments.get(v)
            if cu and cv and cu == cv:
                intra_canopy_weight += w

    cluster_scores: Dict[str, float] = {}
    cluster_stats: Dict[str, Dict[str, float]] = {}
    node_score = {n["id"]: 0.0 for n in nodes}
    node_cluster: Dict[str, str] = {}

    for cid, members in clusters.items():
        sub = G.subgraph(members)
        if sub.number_of_nodes() <= 1 or sub.number_of_edges() == 0:
            cluster_scores[cid] = 0.0
            cluster_stats[cid] = {"avg_weight": 0.0, "deg_conc": 0.0}
            for u in members:
                node_cluster[u] = cid
            continue

        avg_w = sum(d.get("weight", 0.0) for _, _, d in sub.edges(data=True)) / sub.number_of_edges()
        degrees = [deg for _, deg in sub.degree()]
        avg_deg = sum(degrees) / len(degrees) if degrees else 0.0
        deg_conc = (max(degrees) / avg_deg) if avg_deg > 0 else 0.0

        score = 0.7 * avg_w + 0.3 * deg_conc
        cluster_scores[cid] = score
        cluster_stats[cid] = {"avg_weight": avg_w, "deg_conc": deg_conc}
        for u in members:
            node_score[u] = score
            node_cluster[u] = cid

    vals = sorted(node_score.values(), reverse=True)
    if not vals:
        return [], {"modularity": 0.0, "conductance_mean": 0.0}, []

    cutoff_index = max(0, int(0.2 * (len(vals) - 1)))
    cutoff = vals[cutoff_index]
    suspicious = [u for u, s in node_score.items() if s >= cutoff and s > 0]

    try:
        m = G.size(weight="weight")
        if m == 0:
            modularity = 0.0
        else:
            modularity = 0.0
            for cid, members in clusters.items():
                sub = G.subgraph(members)
                lc = sub.size(weight="weight")
                dc = sum(dict(G.degree(members, weight="weight")).values())
                modularity += (lc / m) - (dc / (2 * m)) ** 2
    except Exception:
        modularity = 0.0

    # Global readability metrics
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    avg_degree = (2.0 * n_edges / n_nodes) if n_nodes > 0 else 0.0
    density = (2.0 * n_edges / (n_nodes * (n_nodes - 1))) if n_nodes > 1 else 0.0
    try:
        num_components = nx.number_connected_components(G)
    except Exception:
        num_components = 0

    conductances = []
    for cid, members in clusters.items():
        if len(members) < 2 or len(members) >= len(G) - 1:
            continue
        cut_w = 0.0
        vol_S = 0.0
        vol_notS = 0.0
        S = set(members)
        for u, v, d in G.edges(data=True):
            w = d.get("weight", 0.0)
            if u in S and v in S:
                vol_S += w
            elif u not in S and v not in S:
                vol_notS += w
            else:
                cut_w += w
        denom = min(vol_S + cut_w, vol_notS + cut_w)
        if denom > 0:
            conductances.append(cut_w / denom)
    conductance_mean = sum(conductances) / len(conductances) if conductances else 0.0

    suspicious_details = []
    for uid in suspicious:
        cid = node_cluster.get(uid)
        stats = cluster_stats.get(cid, {"avg_weight": 0.0, "deg_conc": 0.0})
        degree_unweighted = G.degree(uid)
        degree_weighted = G.degree(uid, weight="weight")
        neigh = []
        if uid in G:
            for vid, attr in sorted(G[uid].items(), key=lambda item: item[1].get("weight", 0.0), reverse=True)[:5]:
                neigh.append({
                    "peer_id": vid,
                    "peer_label": label_map.get(vid, vid),
                    "weight": float(attr.get("weight", 0.0)),
                })
        suspicious_details.append({
            "user_id": uid,
            "username": label_map.get(uid, uid),
            "cluster_id": cid,
            "canopy_id": canopy_assignments.get(uid) if canopy_assignments else None,
            "score": node_score.get(uid, 0.0),
            "cluster_avg_weight": stats.get("avg_weight", 0.0),
            "cluster_degree_concentration": stats.get("deg_conc", 0.0),
            "degree": degree_unweighted,
            "strength": degree_weighted,
            "top_edges": neigh,
        })

    metrics = {
        "modularity": modularity,
        "conductance_mean": conductance_mean,
        "avg_degree": avg_degree,
        "density": density,
        "num_components": float(num_components),
        "num_nodes": float(n_nodes),
        "num_edges": float(n_edges),
        "num_clusters": float(len(clusters)),
    }

    if canopy_assignments and total_edge_weight > 0:
        metrics["intra_canopy_edge_ratio"] = intra_canopy_weight / total_edge_weight

    return suspicious, metrics, suspicious_details

