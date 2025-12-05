from typing import List, Dict, Tuple
import networkx as nx


def mst_cluster(nodes: List[Dict], edges: List[Dict], use_overlay: bool = True) -> Dict[str, List[str]]:
    """
    Build MST per connected component and cut weak links to form clusters.
    For v1, we cut edges below median MST weight per component.

    Returns mapping cluster_id -> list of user_ids.
    """
    G = nx.Graph()
    for n in nodes:
        G.add_node(n["id"], label=n.get("label"))
    for e in edges:
        # Convert similarity to distance for MST: higher weight -> smaller distance
        w = float(e.get("weight", 0.0))
        d = 1.0 / (w + 1e-9)
        G.add_edge(e["source"], e["target"], weight=w, distance=d)

    clusters: Dict[str, List[str]] = {}
    cid = 0

    for comp_nodes in nx.connected_components(G):
        sub = G.subgraph(comp_nodes).copy()
        if sub.number_of_edges() == 0:
            clusters[str(cid)] = list(sub.nodes())
            cid += 1
            continue
        # Minimum spanning tree using distance
        T = nx.minimum_spanning_tree(sub, weight="distance")
        # derive cut threshold: median of similarity weights on MST
        weights = [sub[u][v]["weight"] for u, v in T.edges()]
        if not weights:
            clusters[str(cid)] = list(sub.nodes())
            cid += 1
            continue
        thresh = sorted(weights)[len(weights) // 2]
        # remove weak links on MST and take connected components as clusters
        T_cut = T.copy()
        for u, v, d in list(T_cut.edges(data=True)):
            if d.get("weight", 0.0) < thresh:
                T_cut.remove_edge(u, v)
        for c in nx.connected_components(T_cut):
            clusters[str(cid)] = list(c)
            cid += 1

    return clusters
