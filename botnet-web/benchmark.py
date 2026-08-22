import argparse
import asyncio
import csv
import json
import os
import statistics
from datetime import datetime, timezone
from time import perf_counter
import numpy as np
from sklearn.metrics import silhouette_score
from app.pipeline.tiktok_client import TikTokFetchError, fetch_comments
from app.pipeline.parser import parse_comments
from app.pipeline.canopy import build_canopies, CanopyArtifacts
from app.pipeline.graph_builder import build_graph
from app.pipeline.mst_cluster import mst_cluster
from app.pipeline.scoring import score_clusters

DEFAULT_VIDEO_URL = "https://vt.tiktok.com/ZSHoa9M6H/"
MS_TOKEN = os.getenv("ms_token") or os.getenv("MS_TOKEN")
SIZES = [100, 200, 500, 1000]
REPEATS = 2
MODES = ["pipeline", "canopy_only", "kruskal_only"]
PERFORMANCE_FIELDS = [
    "fetch_sec",
    "parse_sec",
    "feature_prep_sec",
    "preprocess_tfidf_sec",
    "tfidf_sec",
    "reduce_sec",
    "ann_index_sec",
    "ann_query_sec",
    "ann_prefilter_sec",
    "canopy_threshold_grid_sec",
    "canopy_bruteforce_sec",
    "canopy_assign_sec",
    "canopy_cluster_sec",
    "canopy_sec",
    "build_sec",
    "cluster_sec",
    "score_sec",
    "total_sec",
    "total_with_fetch_sec",
]
QUALITY_FIELDS = [
    "modularity",
    "conductance_mean",
    "silhouette_score",
    "silhouette_samples",
    "silhouette_clusters",
    "canopy_t1",
    "canopy_t2",
    "canopy_threshold_silhouette",
    "canopy_threshold_candidates",
    "avg_degree",
    "density",
    "num_components",
    "num_suspicious_clusters",
    "num_suspicious_users",
    "suspicious_cluster_ratio",
    "suspicious_user_ratio",
    "avg_cluster_density",
    "avg_content_repetition",
    "avg_temporal_burst",
    "avg_high_frequency",
    "avg_account_age_ratio",
    "account_age_coverage",
    "intra_canopy_edge_ratio",
    "mean_suspicious_score",
    "max_suspicious_score",
]
QUALITY_TEXT_FIELDS = ["quality_interpretation"]


def _build_canopy(parsed):
    try:
        return build_canopies(parsed.get("user_texts", {}), t1=0.6, t2=0.8, use_ann=True)
    except ValueError:
        return CanopyArtifacts(assignments={}, canopies={}, embeddings={})


def _parse_args():
    parser = argparse.ArgumentParser(description="Benchmark botnet detection pipeline from a TikTok video URL.")
    parser.add_argument(
        "--video-url",
        default=os.getenv("BENCH_VIDEO_URL", DEFAULT_VIDEO_URL),
        help="TikTok video URL to fetch comments from.",
    )
    return parser.parse_args()


async def _get_comments(video_url: str, limit: int):
    if not MS_TOKEN:
        raise RuntimeError("Atur ms_token atau MS_TOKEN sebelum menjalankan benchmark langsung.")
    comments = await fetch_comments(video_url, limit, MS_TOKEN)
    return comments, video_url


async def _fetch_comments_for_size(video_url: str, limit: int):
    t_fetch_start = perf_counter()
    comments, source = await _get_comments(video_url, limit)
    return comments[:limit], source, perf_counter() - t_fetch_start


def _quality_stub():
    values = {field: None for field in QUALITY_FIELDS}
    values.update({field: "" for field in QUALITY_TEXT_FIELDS})
    return values


def _performance_stub(fetch_sec: float = 0.0):
    values = {field: 0.0 for field in PERFORMANCE_FIELDS}
    values["fetch_sec"] = fetch_sec
    values["total_with_fetch_sec"] = fetch_sec
    return values


def _extract_canopy_metrics(canopy, parse_sec: float):
    timing = getattr(canopy, "timing", None)
    if timing is None:
        metrics = _performance_stub()
        metrics["parse_sec"] = parse_sec
        metrics["preprocess_tfidf_sec"] = parse_sec
        metrics["total_sec"] = parse_sec
        metrics["total_with_fetch_sec"] = parse_sec
        return metrics

    feature_prep_sec = float(timing.vectorize_sec or 0.0) + float(timing.reduce_sec or 0.0)
    ann_prefilter_sec = float(timing.ann_index_sec or 0.0) + float(timing.ann_query_sec or 0.0)
    canopy_cluster_sec = float(timing.brute_force_sec or 0.0) + float(timing.assignment_sec or 0.0)
    canopy_sec = float(timing.total_sec or 0.0)
    return {
        "fetch_sec": 0.0,
        "parse_sec": parse_sec,
        "feature_prep_sec": feature_prep_sec,
        "preprocess_tfidf_sec": parse_sec + feature_prep_sec,
        "tfidf_sec": float(timing.vectorize_sec or 0.0),
        "reduce_sec": float(timing.reduce_sec or 0.0),
        "ann_index_sec": float(timing.ann_index_sec or 0.0),
        "ann_query_sec": float(timing.ann_query_sec or 0.0),
        "ann_prefilter_sec": ann_prefilter_sec,
        "canopy_threshold_grid_sec": float(timing.threshold_grid_sec or 0.0),
        "canopy_bruteforce_sec": float(timing.brute_force_sec or 0.0),
        "canopy_assign_sec": float(timing.assignment_sec or 0.0),
        "canopy_cluster_sec": canopy_cluster_sec,
        "canopy_sec": canopy_sec,
        "build_sec": 0.0,
        "cluster_sec": 0.0,
        "score_sec": 0.0,
        "total_sec": parse_sec + canopy_sec,
        "total_with_fetch_sec": parse_sec + canopy_sec,
    }


def _compute_silhouette_metrics(embeddings, clusters):
    if not embeddings or not clusters:
        return {}

    cluster_lookup = {}
    for cluster_id, members in clusters.items():
        for uid in members:
            cluster_lookup[uid] = cluster_id

    vectors = []
    labels = []
    for uid, vector in embeddings.items():
        cluster_id = cluster_lookup.get(uid)
        if not cluster_id:
            continue
        vectors.append(np.asarray(vector, dtype=float))
        labels.append(cluster_id)

    sample_count = len(vectors)
    label_count = len(set(labels))
    metrics = {
        "silhouette_samples": float(sample_count),
        "silhouette_clusters": float(label_count),
    }
    if sample_count < 3 or label_count < 2 or label_count >= sample_count:
        return metrics

    try:
        metrics["silhouette_score"] = float(silhouette_score(np.vstack(vectors), labels, metric="cosine"))
    except Exception:
        pass
    return metrics


def _interpret_quality_metrics(modularity, silhouette, conductance, n_clusters, mode: str):
    if not n_clusters:
        return "Tidak ada cluster yang terbentuk."

    notes = [f"Terbentuk sekitar {int(round(n_clusters))} cluster"]
    if modularity is None and conductance is None and silhouette is None:
        if mode == "canopy_only":
            notes.append("mode canopy-only belum punya metrik graf penuh")
        else:
            notes.append("metrik kualitas belum cukup untuk dievaluasi")
        return "; ".join(notes) + "."

    if modularity is not None:
        if modularity >= 0.30:
            notes.append("struktur komunitas kuat")
        elif modularity >= 0.10:
            notes.append("struktur komunitas sedang")
        else:
            notes.append("struktur komunitas lemah")

    if silhouette is not None:
        if silhouette >= 0.50:
            notes.append("pemisahan cluster sangat baik")
        elif silhouette >= 0.25:
            notes.append("pemisahan cluster cukup jelas")
        elif silhouette >= 0.00:
            notes.append("pemisahan cluster masih lemah")
        else:
            notes.append("cluster saling tumpang tindih")
    else:
        notes.append("silhouette belum dapat dihitung")

    if conductance is not None:
        if conductance <= 0.35:
            notes.append("isolasi cluster baik")
        elif conductance <= 0.60:
            notes.append("isolasi cluster sedang")
        else:
            notes.append("isolasi cluster lemah")

    return "; ".join(notes) + "."


def _extract_quality_metrics(metrics, suspicious, details, n_nodes, n_clusters, mode: str):
    detail_scores = [float(item.get("score", 0.0)) for item in (details or [])]
    suspicious_count = len(suspicious or [])
    suspicious_clusters = float(metrics.get("num_suspicious_clusters", 0.0) or 0.0)
    result = {
        "modularity": metrics.get("modularity"),
        "conductance_mean": metrics.get("conductance_mean"),
        "silhouette_score": metrics.get("silhouette_score"),
        "silhouette_samples": metrics.get("silhouette_samples"),
        "silhouette_clusters": metrics.get("silhouette_clusters"),
        "avg_degree": metrics.get("avg_degree"),
        "density": metrics.get("density"),
        "num_components": metrics.get("num_components"),
        "num_suspicious_clusters": suspicious_clusters,
        "num_suspicious_users": float(metrics.get("num_suspicious_users", suspicious_count) or suspicious_count),
        "suspicious_cluster_ratio": (suspicious_clusters / n_clusters) if n_clusters else 0.0,
        "suspicious_user_ratio": (suspicious_count / n_nodes) if n_nodes else 0.0,
        "avg_cluster_density": metrics.get("avg_cluster_density"),
        "avg_content_repetition": metrics.get("avg_content_repetition"),
        "avg_temporal_burst": metrics.get("avg_temporal_burst"),
        "avg_high_frequency": metrics.get("avg_high_frequency"),
        "avg_account_age_ratio": metrics.get("avg_account_age_ratio"),
        "account_age_coverage": metrics.get("account_age_coverage"),
        "intra_canopy_edge_ratio": metrics.get("intra_canopy_edge_ratio"),
        "mean_suspicious_score": (sum(detail_scores) / len(detail_scores)) if detail_scores else 0.0,
        "max_suspicious_score": max(detail_scores) if detail_scores else 0.0,
    }
    result["quality_interpretation"] = _interpret_quality_metrics(
        result.get("modularity"),
        result.get("silhouette_score"),
        result.get("conductance_mean"),
        n_clusters,
        mode,
    )
    return result


def _mean_or_none(rows, key):
    values = [row.get(key) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return statistics.mean(values)


def _fmt_metric(value, digits=3):
    if value is None:
        return "-"
    return f"{value:.{digits}f}"

async def run_once(requested_n, mode: str, comments, fetch_sec: float = 0.0):
    comments = comments[:requested_n]
    t_parse_start = perf_counter()
    parsed = parse_comments(comments)
    t_parse = perf_counter() - t_parse_start

    canopy = _build_canopy(parsed)
    performance = _extract_canopy_metrics(canopy, t_parse)
    performance["fetch_sec"] = fetch_sec

    base_row = {
        "mode": mode,
        "requested_comments": requested_n,
        "n_comments": len(comments),
        "n_canopies": len(canopy.canopies),
        "canopy_t1": canopy.threshold_t1,
        "canopy_t2": canopy.threshold_t2,
        "canopy_threshold_silhouette": canopy.threshold_silhouette,
        "canopy_threshold_candidates": canopy.threshold_candidates,
        **performance,
    }

    if mode == "canopy_only":
        base_row["n_nodes"] = len(parsed.get("user_summary", {}))
        base_row["n_edges"] = 0
        base_row["n_clusters"] = len(canopy.canopies)
        quality_metrics = _compute_silhouette_metrics(canopy.embeddings, canopy.canopies)
        base_row["total_with_fetch_sec"] = fetch_sec + base_row["total_sec"]
        return {
            **base_row,
            **_extract_quality_metrics(quality_metrics, [], [], base_row["n_nodes"], base_row["n_clusters"], mode),
        }

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
    suspicious, metrics, details, _cluster_stats, _node_scores = score_clusters(
        nodes,
        edges,
        clusters,
        canopy_assignments=None if mode == "kruskal_only" else canopy.assignments,
        parsed=parsed,
    )
    t_score = perf_counter() - t_score_start
    metrics.update(_compute_silhouette_metrics(canopy.embeddings, clusters))

    base_row["n_nodes"] = len(nodes)
    base_row["n_edges"] = len(edges)
    base_row["n_clusters"] = len(clusters)
    base_row["build_sec"] = t_build
    base_row["cluster_sec"] = t_cluster
    base_row["score_sec"] = t_score
    base_row["total_sec"] = base_row["parse_sec"] + base_row["canopy_sec"] + t_build + t_cluster + t_score
    base_row["total_with_fetch_sec"] = fetch_sec + base_row["total_sec"]

    return {
        **base_row,
        **_extract_quality_metrics(metrics, suspicious, details, len(nodes), len(clusters), mode),
    }

async def main():
    args = _parse_args()
    rows = []
    for size in SIZES:
        try:
            comments, source, fetch_sec = await _fetch_comments_for_size(args.video_url, size)
        except TikTokFetchError as exc:
            print(f"Melewati size={size}: pengambilan komentar gagal - {exc}")
            continue

        print(f"Berhasil mengambil size={size} dari {source}: {len(comments)} komentar dalam {fetch_sec:.3f}s")
        for mode in MODES:
            for r in range(REPEATS):
                print(f"Menjalankan mode={mode} size={size} repeat={r+1}")
                try:
                    rows.append(await run_once(size, mode, comments, fetch_sec=fetch_sec))
                except Exception as exc:
                    print(f"Melewati mode={mode} size={size} repeat={r+1}: {exc}")
                    continue
    # Aggregate
    print("\nPERFORMANCE RESULTS:")
    for mode in MODES:
        for size in SIZES:
            subset = [row for row in rows if row["mode"] == mode and row["requested_comments"] == size]
            if not subset:
                continue

            def avg(key):
                return statistics.mean(row[key] for row in subset)

            print(
                f"{mode} size={size}: fetch={avg('fetch_sec'):.3f}s parse={avg('parse_sec'):.3f}s "
                f"prep+tfidf={avg('preprocess_tfidf_sec'):.3f}s ann={avg('ann_prefilter_sec'):.3f}s "
                f"canopy={avg('canopy_cluster_sec'):.3f}s build={avg('build_sec'):.3f}s "
                f"mst={avg('cluster_sec'):.3f}s score={avg('score_sec'):.3f}s "
                f"total={avg('total_sec'):.3f}s e2e={avg('total_with_fetch_sec'):.3f}s "
                f"clusters~{statistics.mean(row['n_clusters'] for row in subset):.1f} "
                f"nodes~{statistics.mean(row['n_nodes'] for row in subset):.1f} edges~{statistics.mean(row['n_edges'] for row in subset):.1f}"
            )

    print("\nQUALITY RESULTS:")
    for mode in MODES:
        for size in SIZES:
            subset = [row for row in rows if row["mode"] == mode and row["requested_comments"] == size]
            if not subset:
                continue
            clusters_mean = statistics.mean(row["n_clusters"] for row in subset)
            interpretation = _interpret_quality_metrics(
                _mean_or_none(subset, "modularity"),
                _mean_or_none(subset, "silhouette_score"),
                _mean_or_none(subset, "conductance_mean"),
                clusters_mean,
                mode,
            )
            print(
                f"{mode} size={size}: clusters~{clusters_mean:.1f} "
                f"modularity={_fmt_metric(_mean_or_none(subset, 'modularity'))} "
                f"silhouette={_fmt_metric(_mean_or_none(subset, 'silhouette_score'))} "
                f"conductance={_fmt_metric(_mean_or_none(subset, 'conductance_mean'))} "
                f"density={_fmt_metric(_mean_or_none(subset, 'density'), 4)} "
                f"suspicious_clusters~{_fmt_metric(_mean_or_none(subset, 'num_suspicious_clusters'), 1)} "
                f"suspicious_users~{_fmt_metric(_mean_or_none(subset, 'num_suspicious_users'), 1)} "
                f"suspicious_ratio={_fmt_metric(_mean_or_none(subset, 'suspicious_user_ratio'))} "
                f"cluster_density={_fmt_metric(_mean_or_none(subset, 'avg_cluster_density'))} "
                f"content_rep={_fmt_metric(_mean_or_none(subset, 'avg_content_repetition'))} "
                f"burst={_fmt_metric(_mean_or_none(subset, 'avg_temporal_burst'))} "
                f"high_freq={_fmt_metric(_mean_or_none(subset, 'avg_high_frequency'))} "
                f"interpretation={interpretation}"
            )

    # Persist detailed results for reporting
    try:
        out_dir = os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(out_dir, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        json_path = os.path.join(out_dir, "bench_results.json")
        csv_path = os.path.join(out_dir, "bench_results.csv")
        json_fallback_path = os.path.join(out_dir, f"bench_results_{timestamp}.json")
        csv_fallback_path = os.path.join(out_dir, f"bench_results_{timestamp}.csv")

        import json, csv
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, indent=2)
        except PermissionError:
            with open(json_fallback_path, "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, indent=2)
            json_path = json_fallback_path

        fieldnames = [
            "mode",
            "requested_comments",
            "n_comments",
            "n_nodes",
            "n_edges",
            "n_canopies",
            "n_clusters",
        ] + PERFORMANCE_FIELDS + QUALITY_FIELDS + QUALITY_TEXT_FIELDS
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow({k: row.get(k) for k in fieldnames})
        except PermissionError:
            with open(csv_fallback_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow({k: row.get(k) for k in fieldnames})
            csv_path = csv_fallback_path
        print(f"Hasil detail disimpan ke {json_path} dan {csv_path}")
    except Exception as e:
        print(f"Peringatan: gagal menyimpan output benchmark: {e}")

if __name__ == "__main__":
    asyncio.run(main())
