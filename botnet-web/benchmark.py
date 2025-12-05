import asyncio
import os
import statistics
from time import perf_counter
from app.pipeline.tiktok_client import fetch_comments
from app.pipeline.parser import parse_comments
from app.pipeline.canopy import build_canopies, CanopyArtifacts
from app.pipeline.graph_builder import build_graph
from app.pipeline.mst_cluster import mst_cluster
from app.pipeline.scoring import score_clusters

VIDEO_URL = os.getenv("BENCH_VIDEO_URL", "https://www.tiktok.com/@seventeen17_official/video/7425561216552062226?is_from_webapp=1&sender_device=pc")
MS_TOKEN = os.getenv("ms_token") or os.getenv("MS_TOKEN")
SIZES = [50, 100, 150]
REPEATS = 2

async def run_once(n):
    comments = await fetch_comments(VIDEO_URL, n, MS_TOKEN)
    parsed = parse_comments(comments)
    try:
        canopy = build_canopies(parsed.get("user_texts", {}), t1=0.8, t2=0.6, use_ann=True)
    except ValueError:
        canopy = CanopyArtifacts(assignments={}, canopies={}, embeddings={})
    t0 = perf_counter()
    nodes, edges = build_graph(
        comments,
        alpha=1.0,
        beta=0.8,
        gamma=0.5,
        delta=0.5,
        k_neighbors=15,
        parsed=parsed,
        canopy_assignments=canopy.assignments,
    )
    t_build = perf_counter() - t0
    t1 = perf_counter()
    clusters = mst_cluster(nodes, edges, use_overlay=True)
    t_cluster = perf_counter() - t1
    t2 = perf_counter()
    suspicious, metrics, details = score_clusters(nodes, edges, clusters, canopy_assignments=canopy.assignments)
    t_score = perf_counter() - t2
    return {
        "n_comments": len(comments),
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "n_canopies": len(canopy.canopies),
        "build_sec": t_build,
        "cluster_sec": t_cluster,
        "score_sec": t_score,
    }

async def main():
    if not MS_TOKEN:
        print("ms_token not set; aborting benchmark.")
        return
    rows = []
    for size in SIZES:
        for r in range(REPEATS):
            print(f"Running size={size} repeat={r+1}")
            rows.append(await run_once(size))
    # Aggregate
    print("\nRESULTS:")
    for size in SIZES:
        subset = [row for row in rows if row["n_comments"] >= size]
        if not subset:
            continue
        def avg(key):
            return statistics.mean(row[key] for row in subset)
        print(f"size>={size}: build={avg('build_sec'):.3f}s cluster={avg('cluster_sec'):.3f}s score={avg('score_sec'):.3f}s nodes~{statistics.mean(row['n_nodes'] for row in subset):.1f} edges~{statistics.mean(row['n_edges'] for row in subset):.1f}")

    # Persist detailed results for reporting
    try:
        out_dir = os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(out_dir, exist_ok=True)
        json_path = os.path.join(out_dir, "bench_results.json")
        csv_path = os.path.join(out_dir, "bench_results.csv")

        import json, csv
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

        fieldnames = [
            "n_comments","n_nodes","n_edges","n_canopies",
            "build_sec","cluster_sec","score_sec",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k) for k in fieldnames})
        print(f"Saved detailed results to {json_path} and {csv_path}")
    except Exception as e:
        print(f"Warning: failed to save benchmark outputs: {e}")

if __name__ == "__main__":
    asyncio.run(main())
