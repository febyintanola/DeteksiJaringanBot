import argparse
import asyncio
import csv
import json
import os
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from sklearn.metrics import silhouette_score

from app.pipeline.canopy import CanopyArtifacts, build_canopies
from app.pipeline.graph_builder import build_graph
from app.pipeline.mst_cluster import mst_cluster
from app.pipeline.parser import parse_comments
from app.pipeline.scoring import score_clusters
from app.pipeline.tiktok_client import TikTokFetchError, fetch_comments

DEFAULT_INPUT_JSON = Path(__file__).resolve().parent / "data" / "comments_20260420T090153Z.json"
DEFAULT_OUTPUT_PREFIX = Path(__file__).resolve().parent / "data" / "baseline_louvain_comparison"
DEFAULT_SIZES = [100, 200, 500, 1000]
DEFAULT_VIDEO_URL = "https://vt.tiktok.com/ZSHoa9M6H/"
MS_TOKEN = os.getenv("ms_token") or os.getenv("MS_TOKEN")

METHOD_PROPOSED = "canopy_mst"
METHOD_LOUVAIN = "louvain"

QUALITY_FIELDS = [
    "modularity",
    "conductance_mean",
    "silhouette_score",
    "silhouette_samples",
    "silhouette_clusters",
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
TEXT_FIELDS = ["quality_interpretation"]


@dataclass
class MethodProfile:
    method_id: str
    method_label: str
    comparison_role: str
    platform: str
    clustering_method: str
    graph_reduction_method: str
    system_output: str


METHOD_PROFILES = {
    METHOD_PROPOSED: MethodProfile(
        method_id=METHOD_PROPOSED,
        method_label="Canopy + Kruskal MST (Usulan)",
        comparison_role="Metode usulan",
        platform="TikTok, berbasis komentar publik dan interaksi antar pengguna",
        clustering_method="Canopy Clustering sebagai pre-clustering, dilanjutkan Kruskal MST clustering",
        graph_reduction_method="Kruskal MST membentuk backbone graf, lalu edge lemah pada MST dipotong",
        system_output="Graf pengguna, cluster terdeteksi, skor kecurigaan, alasan cluster, dan daftar akun mencurigakan",
    ),
    METHOD_LOUVAIN: MethodProfile(
        method_id=METHOD_LOUVAIN,
        method_label="Louvain Community Detection (Baseline)",
        comparison_role="Baseline penelitian terdahulu",
        platform="Graf sosial umum; pada benchmark ini diterapkan ke komentar TikTok yang sama",
        clustering_method="Louvain Community Detection berbasis optimasi modularity",
        graph_reduction_method="Tidak memakai reduksi MST; community detection dijalankan pada graf berbobot penuh",
        system_output="Komunitas/cluster pengguna dan skor kecurigaan menggunakan modul scoring yang sama untuk pembanding",
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark metode usulan Canopy + Kruskal MST terhadap baseline "
            "Louvain Community Detection pada data komentar TikTok yang sama."
        )
    )
    parser.add_argument(
        "--input-json",
        default=str(DEFAULT_INPUT_JSON),
        help="File JSON komentar lokal. Mendukung format list komentar atau payload {'comments': [...]}",
    )
    parser.add_argument(
        "--video-url",
        default=None,
        help="Jika diisi, komentar diambil live dari TikTok dan input-json diabaikan.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=max(DEFAULT_SIZES),
        help="Batas komentar saat fetch live dari TikTok.",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=DEFAULT_SIZES,
        help="Ukuran dataset yang diuji, misalnya --sizes 100 200 500 1000.",
    )
    parser.add_argument("--repeats", type=int, default=1, help="Jumlah pengulangan tiap ukuran.")
    parser.add_argument(
        "--output-prefix",
        default=str(DEFAULT_OUTPUT_PREFIX),
        help="Prefix output tanpa ekstensi. Akan dibuat .json, .csv, dan .md.",
    )
    parser.add_argument("--k-neighbors", type=int, default=15, help="Jumlah tetangga maksimum saat build graph.")
    parser.add_argument("--canopy-t1", type=float, default=0.6, help="Threshold luar Canopy.")
    parser.add_argument("--canopy-t2", type=float, default=0.8, help="Threshold dalam Canopy.")
    parser.add_argument("--no-ann", action="store_true", help="Matikan ANN untuk Canopy.")
    parser.add_argument("--no-auto-tune", action="store_true", help="Matikan auto tuning threshold Canopy.")
    parser.add_argument("--louvain-resolution", type=float, default=1.0, help="Resolusi Louvain.")
    parser.add_argument("--seed", type=int, default=42, help="Seed Louvain.")
    return parser.parse_args()


def _load_comments(path: Path) -> Tuple[List[Dict[str, Any]], str]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    if isinstance(payload, list):
        return payload, str(path)
    if isinstance(payload, dict) and isinstance(payload.get("comments"), list):
        source = payload.get("video_url") or str(path)
        return payload["comments"], str(source)
    raise ValueError(f"Format komentar tidak dikenali: {path}")


async def _fetch_live_comments(video_url: str, limit: int) -> Tuple[List[Dict[str, Any]], str, float]:
    if not MS_TOKEN:
        raise RuntimeError("Atur ms_token atau MS_TOKEN sebelum menjalankan benchmark live.")
    t0 = perf_counter()
    comments = await fetch_comments(video_url, limit, MS_TOKEN)
    return comments[:limit], video_url, perf_counter() - t0


def _build_canopy(parsed: Dict[str, Any], args: argparse.Namespace) -> CanopyArtifacts:
    try:
        return build_canopies(
            parsed.get("user_texts", {}),
            t1=args.canopy_t1,
            t2=args.canopy_t2,
            use_ann=not args.no_ann,
            auto_tune=not args.no_auto_tune,
        )
    except ValueError:
        return CanopyArtifacts(assignments={}, canopies={}, embeddings={})


def _as_graph(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> nx.Graph:
    graph = nx.Graph()
    for node in nodes:
        graph.add_node(node["id"], label=node.get("label"))
    for edge in edges:
        weight = float(edge.get("weight", 0.0) or 0.0)
        distance = 1.0 / (weight + 1e-9)
        graph.add_edge(edge["source"], edge["target"], weight=weight, distance=distance)
    return graph


def _mst_reduction_summary(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> Dict[str, float]:
    graph = _as_graph(nodes, edges)
    mst_edges = 0
    kept_edges = 0

    for component_nodes in nx.connected_components(graph):
        subgraph = graph.subgraph(component_nodes).copy()
        if subgraph.number_of_edges() == 0:
            continue

        tree = nx.minimum_spanning_tree(subgraph, weight="distance", algorithm="kruskal")
        mst_edges += tree.number_of_edges()
        weights = [float(data.get("weight", 0.0) or 0.0) for _, _, data in tree.edges(data=True)]
        if not weights:
            continue

        threshold = sorted(weights)[len(weights) // 2]
        kept_edges += sum(1 for value in weights if value >= threshold)

    original_edges = len(edges)
    reduction_ratio = 0.0
    if original_edges:
        reduction_ratio = max(0.0, 1.0 - (kept_edges / original_edges))

    return {
        "mst_backbone_edges": float(mst_edges),
        "edges_after_reduction": float(kept_edges),
        "edge_reduction_ratio": float(reduction_ratio),
    }


def _louvain_clusters(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    resolution: float,
    seed: int,
) -> Dict[str, List[str]]:
    graph = _as_graph(nodes, edges)
    if graph.number_of_edges() == 0:
        return {str(index): [node] for index, node in enumerate(graph.nodes())}

    communities = nx.algorithms.community.louvain_communities(
        graph,
        weight="weight",
        resolution=resolution,
        seed=seed,
    )
    return {str(index): sorted(list(members)) for index, members in enumerate(communities)}


def _compute_silhouette_metrics(
    embeddings: Dict[str, Any],
    clusters: Dict[str, List[str]],
) -> Dict[str, float]:
    if not embeddings or not clusters:
        return {}

    cluster_lookup: Dict[str, str] = {}
    for cluster_id, members in clusters.items():
        for uid in members:
            cluster_lookup[uid] = cluster_id

    vectors: List[np.ndarray] = []
    labels: List[str] = []
    for uid, vector in embeddings.items():
        cluster_id = cluster_lookup.get(uid)
        if cluster_id is None:
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


def _interpret_quality_metrics(
    modularity: Optional[float],
    silhouette: Optional[float],
    conductance: Optional[float],
    n_clusters: int,
) -> str:
    if not n_clusters:
        return "Tidak ada cluster yang terbentuk."

    notes = [f"Terbentuk {n_clusters} cluster"]
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


def _extract_quality_metrics(
    metrics: Dict[str, Any],
    suspicious: Sequence[str],
    details: Sequence[Dict[str, Any]],
    n_nodes: int,
    n_clusters: int,
) -> Dict[str, Any]:
    detail_scores = [float(item.get("score", 0.0) or 0.0) for item in details]
    suspicious_count = len(suspicious)
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
    )
    return result


def _profile_fields(method_id: str) -> Dict[str, str]:
    profile = METHOD_PROFILES[method_id]
    return {
        "method": profile.method_id,
        "method_label": profile.method_label,
        "comparison_role": profile.comparison_role,
        "platform": profile.platform,
        "clustering_method": profile.clustering_method,
        "graph_reduction_method": profile.graph_reduction_method,
        "system_output": profile.system_output,
    }


def _run_proposed(
    comments: List[Dict[str, Any]],
    parsed: Dict[str, Any],
    parse_sec: float,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], CanopyArtifacts]:
    t_canopy = perf_counter()
    canopy = _build_canopy(parsed, args)
    canopy_sec = perf_counter() - t_canopy

    t_build = perf_counter()
    nodes, edges = build_graph(
        comments,
        alpha=1.0,
        beta=0.8,
        gamma=0.5,
        delta=0.5,
        k_neighbors=args.k_neighbors,
        parsed=parsed,
        canopy_assignments=canopy.assignments,
    )
    build_sec = perf_counter() - t_build

    t_cluster = perf_counter()
    clusters = mst_cluster(nodes, edges, use_overlay=True)
    cluster_sec = perf_counter() - t_cluster

    reduction = _mst_reduction_summary(nodes, edges)

    t_score = perf_counter()
    suspicious, metrics, details, _cluster_stats, _node_scores = score_clusters(
        nodes,
        edges,
        clusters,
        canopy_assignments=canopy.assignments,
        parsed=parsed,
    )
    metrics.update(_compute_silhouette_metrics(canopy.embeddings, clusters))
    score_sec = perf_counter() - t_score

    total_sec = parse_sec + canopy_sec + build_sec + cluster_sec + score_sec
    row = {
        **_profile_fields(METHOD_PROPOSED),
        "requested_comments": len(comments),
        "n_comments": len(comments),
        "n_nodes": len(nodes),
        "n_edges_before_reduction": len(edges),
        "n_edges_after_reduction": reduction["edges_after_reduction"],
        "edge_reduction_ratio": reduction["edge_reduction_ratio"],
        "mst_backbone_edges": reduction["mst_backbone_edges"],
        "n_canopies": len(canopy.canopies),
        "n_clusters": len(clusters),
        "canopy_t1": canopy.threshold_t1,
        "canopy_t2": canopy.threshold_t2,
        "canopy_threshold_silhouette": canopy.threshold_silhouette,
        "canopy_threshold_candidates": canopy.threshold_candidates,
        "parse_sec": parse_sec,
        "canopy_sec": canopy_sec,
        "build_sec": build_sec,
        "reduction_sec": cluster_sec,
        "cluster_sec": cluster_sec,
        "score_sec": score_sec,
        "total_sec": total_sec,
    }
    row.update(_extract_quality_metrics(metrics, suspicious, details, len(nodes), len(clusters)))
    return row, canopy


def _run_louvain(
    comments: List[Dict[str, Any]],
    parsed: Dict[str, Any],
    parse_sec: float,
    embeddings: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    t_build = perf_counter()
    nodes, edges = build_graph(
        comments,
        alpha=1.0,
        beta=0.8,
        gamma=0.5,
        delta=0.5,
        k_neighbors=args.k_neighbors,
        parsed=parsed,
        canopy_assignments=None,
    )
    build_sec = perf_counter() - t_build

    t_cluster = perf_counter()
    clusters = _louvain_clusters(nodes, edges, resolution=args.louvain_resolution, seed=args.seed)
    cluster_sec = perf_counter() - t_cluster

    t_score = perf_counter()
    suspicious, metrics, details, _cluster_stats, _node_scores = score_clusters(
        nodes,
        edges,
        clusters,
        canopy_assignments=None,
        parsed=parsed,
    )
    metrics.update(_compute_silhouette_metrics(embeddings, clusters))
    score_sec = perf_counter() - t_score

    total_sec = parse_sec + build_sec + cluster_sec + score_sec
    row = {
        **_profile_fields(METHOD_LOUVAIN),
        "requested_comments": len(comments),
        "n_comments": len(comments),
        "n_nodes": len(nodes),
        "n_edges_before_reduction": len(edges),
        "n_edges_after_reduction": len(edges),
        "edge_reduction_ratio": 0.0,
        "mst_backbone_edges": 0.0,
        "n_canopies": 0,
        "n_clusters": len(clusters),
        "canopy_t1": None,
        "canopy_t2": None,
        "canopy_threshold_silhouette": None,
        "canopy_threshold_candidates": 0,
        "parse_sec": parse_sec,
        "canopy_sec": 0.0,
        "build_sec": build_sec,
        "reduction_sec": 0.0,
        "cluster_sec": cluster_sec,
        "score_sec": score_sec,
        "total_sec": total_sec,
    }
    row.update(_extract_quality_metrics(metrics, suspicious, details, len(nodes), len(clusters)))
    return row


def _mean_or_none(rows: Sequence[Dict[str, Any]], key: str) -> Optional[float]:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return float(statistics.mean(float(value) for value in values))


def _aggregate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["method"], int(row["requested_comments"])), []).append(row)

    aggregated: List[Dict[str, Any]] = []
    numeric_fields = [
        "n_comments",
        "n_nodes",
        "n_edges_before_reduction",
        "n_edges_after_reduction",
        "edge_reduction_ratio",
        "mst_backbone_edges",
        "n_canopies",
        "n_clusters",
        "canopy_t1",
        "canopy_t2",
        "canopy_threshold_silhouette",
        "canopy_threshold_candidates",
        "parse_sec",
        "canopy_sec",
        "build_sec",
        "reduction_sec",
        "cluster_sec",
        "score_sec",
        "total_sec",
        *QUALITY_FIELDS,
    ]
    for (method_id, requested_comments), subset in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        base = _profile_fields(method_id)
        base["requested_comments"] = requested_comments
        base["repeats"] = len(subset)
        for field in numeric_fields:
            base[field] = _mean_or_none(subset, field)
        base["quality_interpretation"] = _interpret_quality_metrics(
            base.get("modularity"),
            base.get("silhouette_score"),
            base.get("conductance_mean"),
            int(round(float(base.get("n_clusters") or 0))),
        )
        aggregated.append(base)
    return aggregated


def _fmt(value: Any, digits: int = 3, suffix: str = "") -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}f}{suffix}"


def _fmt_int(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return str(int(round(float(value))))
    except (TypeError, ValueError):
        return str(value)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    table = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        table.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(table)


def _build_markdown_report(aggregated: List[Dict[str, Any]], source: str) -> str:
    concept_rows = [
        [
            "Platform",
            METHOD_PROFILES[METHOD_LOUVAIN].platform,
            METHOD_PROFILES[METHOD_PROPOSED].platform,
        ],
        [
            "Metode clustering",
            METHOD_PROFILES[METHOD_LOUVAIN].clustering_method,
            METHOD_PROFILES[METHOD_PROPOSED].clustering_method,
        ],
        [
            "Metode graph reduction",
            METHOD_PROFILES[METHOD_LOUVAIN].graph_reduction_method,
            METHOD_PROFILES[METHOD_PROPOSED].graph_reduction_method,
        ],
        [
            "Output sistem",
            METHOD_PROFILES[METHOD_LOUVAIN].system_output,
            METHOD_PROFILES[METHOD_PROPOSED].system_output,
        ],
        [
            "Kebaruan yang ditonjolkan",
            "Berfokus pada pembentukan komunitas graf berdasarkan modularity.",
            "Menggabungkan pre-clustering berbasis konten, reduksi backbone MST, dan scoring multi-sinyal untuk deteksi jaringan bot.",
        ],
    ]

    result_rows = []
    for row in sorted(aggregated, key=lambda item: (int(item["requested_comments"]), item["method"])):
        result_rows.append(
            [
                _fmt_int(row.get("requested_comments")),
                row.get("method_label"),
                _fmt_int(row.get("n_edges_before_reduction")),
                _fmt_int(row.get("n_edges_after_reduction")),
                _fmt((row.get("edge_reduction_ratio") or 0.0) * 100, 1, "%"),
                _fmt(row.get("total_sec"), 3),
                _fmt_int(row.get("n_clusters")),
                _fmt(row.get("modularity"), 3),
                _fmt(row.get("silhouette_score"), 3),
                _fmt(row.get("conductance_mean"), 3),
                _fmt_int(row.get("num_suspicious_users")),
            ]
        )

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    sections = [
        "# Perbandingan Canopy + Kruskal MST dengan Louvain",
        "",
        f"Sumber data: `{source}`",
        f"Dibuat: {generated_at}",
        "",
        "## Tabel Perbandingan dengan Penelitian Terdahulu / Baseline",
        "",
        _markdown_table(["Aspek", "Baseline Louvain", "Penelitian Ini"], concept_rows),
        "",
        "## Tabel Benchmark Empiris",
        "",
        _markdown_table(
            [
                "Ukuran",
                "Metode",
                "Edge awal",
                "Edge setelah reduksi",
                "Reduksi",
                "Total detik",
                "Cluster",
                "Modularity",
                "Silhouette",
                "Conductance",
                "Akun mencurigakan",
            ],
            result_rows,
        ),
        "",
        "Catatan: Louvain dipakai sebagai baseline karena sama-sama bekerja pada graf berbobot, "
        "sehingga perbandingannya lebih langsung dibanding baseline non-graf. Nilai silhouette "
        "dihitung pada ruang embedding teks yang sama agar kualitas pemisahan cluster dapat dibandingkan.",
        "",
    ]
    return "\n".join(sections)


def _write_outputs(rows: List[Dict[str, Any]], aggregated: List[Dict[str, Any]], markdown: str, output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")
    md_path = output_prefix.with_suffix(".md")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "methods": {key: profile.__dict__ for key, profile in METHOD_PROFILES.items()},
        "rows": rows,
        "aggregated": aggregated,
    }
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    fieldnames = [
        "method",
        "method_label",
        "comparison_role",
        "platform",
        "clustering_method",
        "graph_reduction_method",
        "system_output",
        "requested_comments",
        "repeats",
        "n_comments",
        "n_nodes",
        "n_edges_before_reduction",
        "n_edges_after_reduction",
        "edge_reduction_ratio",
        "mst_backbone_edges",
        "n_canopies",
        "n_clusters",
        "canopy_t1",
        "canopy_t2",
        "canopy_threshold_silhouette",
        "canopy_threshold_candidates",
        "parse_sec",
        "canopy_sec",
        "build_sec",
        "reduction_sec",
        "cluster_sec",
        "score_sec",
        "total_sec",
        *QUALITY_FIELDS,
        *TEXT_FIELDS,
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in aggregated:
            writer.writerow({field: row.get(field) for field in fieldnames})

    with md_path.open("w", encoding="utf-8") as fh:
        fh.write(markdown)

    print(f"Hasil detail disimpan ke {json_path}")
    print(f"Tabel agregat disimpan ke {csv_path}")
    print(f"Tabel Markdown disimpan ke {md_path}")


async def main() -> None:
    args = _parse_args()
    output_prefix = Path(args.output_prefix)

    fetch_sec = 0.0
    if args.video_url:
        try:
            comments, source, fetch_sec = await _fetch_live_comments(args.video_url, args.limit)
        except TikTokFetchError as exc:
            raise RuntimeError(f"Pengambilan komentar gagal: {exc}") from exc
    else:
        comments, source = _load_comments(Path(args.input_json))

    if not comments:
        raise RuntimeError("Data komentar kosong.")

    print(f"Sumber data: {source}")
    print(f"Komentar tersedia: {len(comments)}")
    if fetch_sec:
        print(f"Fetch live selesai dalam {fetch_sec:.3f}s")

    rows: List[Dict[str, Any]] = []
    sizes = [size for size in args.sizes if size <= len(comments)]
    skipped_sizes = [size for size in args.sizes if size > len(comments)]
    for size in skipped_sizes:
        print(f"Melewati size={size}: data hanya berisi {len(comments)} komentar.")

    for size in sizes:
        sample = comments[:size]
        for repeat in range(args.repeats):
            print(f"Menjalankan size={size} repeat={repeat + 1}/{args.repeats}")
            t_parse = perf_counter()
            parsed = parse_comments(sample)
            parse_sec = perf_counter() - t_parse

            proposed_row, canopy = _run_proposed(sample, parsed, parse_sec, args)
            proposed_row["repeat"] = repeat + 1
            rows.append(proposed_row)

            louvain_row = _run_louvain(sample, parsed, parse_sec, canopy.embeddings, args)
            louvain_row["repeat"] = repeat + 1
            rows.append(louvain_row)

            print(
                "  "
                f"usulan: total={proposed_row['total_sec']:.3f}s, "
                f"clusters={proposed_row['n_clusters']}, "
                f"Q={_fmt(proposed_row.get('modularity'))}, "
                f"sil={_fmt(proposed_row.get('silhouette_score'))}"
            )
            print(
                "  "
                f"louvain: total={louvain_row['total_sec']:.3f}s, "
                f"clusters={louvain_row['n_clusters']}, "
                f"Q={_fmt(louvain_row.get('modularity'))}, "
                f"sil={_fmt(louvain_row.get('silhouette_score'))}"
            )

    if not rows:
        raise RuntimeError("Tidak ada ukuran benchmark yang dapat dijalankan.")

    aggregated = _aggregate(rows)
    markdown = _build_markdown_report(aggregated, source)
    _write_outputs(rows, aggregated, markdown, output_prefix)

    print("\nRINGKASAN:")
    for row in aggregated:
        print(
            f"{row['method']} size={int(row['requested_comments'])}: "
            f"total={_fmt(row.get('total_sec'))}s, "
            f"edges={_fmt_int(row.get('n_edges_before_reduction'))}->{_fmt_int(row.get('n_edges_after_reduction'))}, "
            f"clusters={_fmt_int(row.get('n_clusters'))}, "
            f"modularity={_fmt(row.get('modularity'))}, "
            f"silhouette={_fmt(row.get('silhouette_score'))}, "
            f"conductance={_fmt(row.get('conductance_mean'))}"
        )


if __name__ == "__main__":
    asyncio.run(main())
