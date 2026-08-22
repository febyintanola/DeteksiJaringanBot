import asyncio
import os
from app.pipeline.tiktok_client import fetch_comments
from app.pipeline.parser import parse_comments
from app.pipeline.canopy import build_canopies, CanopyArtifacts
from app.pipeline.graph_builder import build_graph
from app.pipeline.mst_cluster import mst_cluster
from app.pipeline.scoring import score_clusters

async def test_pipeline():
    # Use a known working URL
    url = "https://www.tiktok.com/@seventeen17_official/video/7425561216552062226?is_from_webapp=1&sender_device=pc"
    ms_token = os.getenv("ms_token")
    if not ms_token:
        print("ms_token not set, skipping fetch")
        return

    print("Fetching comments...")
    comments = await fetch_comments(url, 50, ms_token)
    print(f"Fetched {len(comments)} comments")

    if not comments:
        print("No comments, skipping")
        return

    print("Parsing comments...")
    parsed = parse_comments(comments)

    print("Building canopy...")
    try:
        canopy = build_canopies(parsed.get("user_texts", {}), t1=0.6, t2=0.8, use_ann=False)
    except ValueError:
        canopy = CanopyArtifacts(assignments={}, canopies={}, embeddings={})
    print(f"Canopies: {len(canopy.canopies)} (avg size ~{(sum(len(v) for v in canopy.canopies.values()) / len(canopy.canopies)) if canopy.canopies else 0:.1f})")

    print("Building graph...")
    nodes, edges = build_graph(
        comments,
        alpha=1.0,
        beta=0.8,
        gamma=0.5,
        delta=0.5,
        k_neighbors=10,
        parsed=parsed,
        canopy_assignments=canopy.assignments,
    )
    print(f"Graph: {len(nodes)} nodes, {len(edges)} edges")

    print("Clustering...")
    clusters = mst_cluster(nodes, edges, use_overlay=True)
    print(f"Clusters: {len(clusters)}")

    print("Scoring...")
    suspicious, metrics, details, _cluster_stats, _node_scores = score_clusters(
        nodes,
        edges,
        clusters,
        canopy_assignments=canopy.assignments,
        parsed=parsed,
    )
    print(f"Suspicious users: {len(suspicious)}")
    print(f"Metrics: {metrics}")
    if details:
        print("Top suspicious explanation:")
        for item in details[:3]:
            top_peers = ", ".join(f"{e['peer_label']}({e['weight']:.2f})" for e in item.get("top_edges", [])[:3])
            print(f"  - {item['username']} | skor {item['score']:.3f} | cluster {item['cluster_id']} | koneksi: {top_peers}")

    # Print sample
    print("Sample clusters:")
    for cid, members in list(clusters.items())[:3]:
        print(f"  {cid}: {len(members)} users")

if __name__ == "__main__":
    asyncio.run(test_pipeline())
