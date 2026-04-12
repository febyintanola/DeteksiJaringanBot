import asyncio
import csv
import json
import os
from pathlib import Path
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
SIZES = [50, 100, 150, 200, 300, 500, 750, 1000]
REPEATS = 2
MODES = ["pipeline", "canopy_only", "kruskal_only"]
USE_LIVE_FETCH = os.getenv("BENCH_USE_LIVE", "0").strip().lower() in {"1", "true", "yes"}
DATA_DIR = Path(__file__).resolve().parent / "data"


def _build_canopy(parsed):
    try:
        return build_canopies(parsed.get("user_texts", {}), t1=0.8, t2=0.6, use_ann=True)
    except ValueError:
        return CanopyArtifacts(assignments={}, canopies={}, embeddings={})


def _load_local_comments_fixture():
    candidates = [DATA_DIR / "comments.json"]
    candidates.extend(sorted(DATA_DIR.glob("comments_*.json"), reverse=True))
    for path in candidates:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        comments = payload.get("comments") if isinstance(payload, dict) else None
        if isinstance(comments, list) and comments:
            return comments, str(path)
    return [], None


async def _get_comments(limit: int):
    if USE_LIVE_FETCH:
        comments = await fetch_comments(VIDEO_URL, limit, MS_TOKEN)
        return comments, "live"

    comments, source = _load_local_comments_fixture()
    if comments:
        return comments[:limit], source

    if MS_TOKEN:
        comments = await fetch_comments(VIDEO_URL, limit, MS_TOKEN)
        return comments, "live"

    raise RuntimeError("No local comments snapshot found and BENCH_USE_LIVE is disabled.")

async def run_once(n, mode: str, comments):
    comments = comments[:n]
    parsed = parse_comments(comments)

    if mode == "canopy_only":
        t0 = perf_counter()
        canopy = _build_canopy(parsed)
        t_canopy = perf_counter() - t0
        return {
            "mode": mode,
            "n_comments": len(comments),
            "n_nodes": len(parsed.get("user_summary", {})),
            "n_edges": 0,
            "n_canopies": len(canopy.canopies),
            "n_clusters": len(canopy.canopies),
            "canopy_sec": t_canopy,
            "build_sec": 0.0,
            "cluster_sec": 0.0,
            "score_sec": 0.0,
            "total_sec": t_canopy,
        }

    t_canopy_start = perf_counter()
    canopy = _build_canopy(parsed)
    t_canopy = perf_counter() - t_canopy_start

    t_build_start = perf_counter()
    nodes, edges = build_graph(
        comments,
        alpha=1.0,
        beta=0.8,
        gamma=0.5,
        delta=0.5,
        k_neighbors=15,
        parsed=parsed,
        canopy_assignments=None if mode == "kruskal_only" else canopy.assignments,
    )
    t_build = perf_counter() - t_build_start

    t_cluster_start = perf_counter()
    clusters = mst_cluster(nodes, edges, use_overlay=True)
    t_cluster = perf_counter() - t_cluster_start

    t_score_start = perf_counter()
    suspicious, metrics, details = score_clusters(
        nodes,
        edges,
        clusters,
        canopy_assignments=None if mode == "kruskal_only" else canopy.assignments,
        parsed=parsed,
    )
    t_score = perf_counter() - t_score_start

    return {
        "mode": mode,
        "n_comments": len(comments),
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "n_canopies": len(canopy.canopies),
        "n_clusters": len(clusters),
        "canopy_sec": t_canopy,
        "build_sec": t_build,
        "cluster_sec": t_cluster,
        "score_sec": t_score,
        "total_sec": t_canopy + t_build + t_cluster + t_score,
    }

async def main():
    comments, source = await _get_comments(max(SIZES))
    print(f"Using comments source: {source} ({len(comments)} comments available)")
    rows = []
    for size in SIZES:
        for mode in MODES:
            for r in range(REPEATS):
                print(f"Running mode={mode} size={size} repeat={r+1}")
                rows.append(await run_once(size, mode, comments))
    # Aggregate
    print("\nRESULTS:")
    for mode in MODES:
        for size in SIZES:
            subset = [row for row in rows if row["mode"] == mode and row["n_comments"] >= size]
            if not subset:
                continue

            def avg(key):
                return statistics.mean(row[key] for row in subset)

            print(
                f"{mode} size>={size}: total={avg('total_sec'):.3f}s canopy={avg('canopy_sec'):.3f}s "
                f"build={avg('build_sec'):.3f}s cluster={avg('cluster_sec'):.3f}s score={avg('score_sec'):.3f}s "
                f"nodes~{statistics.mean(row['n_nodes'] for row in subset):.1f} edges~{statistics.mean(row['n_edges'] for row in subset):.1f}"
            )

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
            "mode",
            "n_comments",
            "n_nodes",
            "n_edges",
            "n_canopies",
            "n_clusters",
            "canopy_sec",
            "build_sec",
            "cluster_sec",
            "score_sec",
            "total_sec",
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
