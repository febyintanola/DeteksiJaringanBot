import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import networkx as nx

from app.pipeline.canopy import CanopyArtifacts, build_canopies
from app.pipeline.graph_builder import build_graph
from app.pipeline.parser import parse_comments


DEFAULT_SIZES = [100, 200, 500, 1000]


def _load_comments(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    if isinstance(payload, dict):
        comments = payload.get("comments", [])
    else:
        comments = payload

    if not isinstance(comments, list):
        raise ValueError(f"File {path} tidak berisi daftar komentar.")
    return comments


def _json_cell(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_cell(row.get(key, "")) for key in fieldnames})


def _build_canopy(parsed: Dict[str, Any], t1: float, t2: float, use_ann: bool, auto_tune: bool) -> CanopyArtifacts:
    try:
        return build_canopies(
            parsed.get("user_texts", {}),
            t1=t1,
            t2=t2,
            use_ann=use_ann,
            auto_tune=auto_tune,
        )
    except ValueError:
        return CanopyArtifacts(assignments={}, canopies={}, embeddings={})


def _connected_components(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    graph = nx.Graph()
    label_lookup = {node["id"]: node.get("label", node["id"]) for node in nodes}

    for node in nodes:
        graph.add_node(node["id"])
    for edge in edges:
        graph.add_edge(edge["source"], edge["target"], weight=float(edge.get("weight", 0.0)))

    components = sorted(nx.connected_components(graph), key=lambda item: (-len(item), sorted(item)[0]))
    rows: List[Dict[str, Any]] = []
    for index, members in enumerate(components, start=1):
        member_list = sorted(members)
        for node_id in member_list:
            rows.append(
                {
                    "component_id": index,
                    "component_size": len(member_list),
                    "node_id": node_id,
                    "label": label_lookup.get(node_id, node_id),
                }
            )
    return rows


def _component_summary(component_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[int, Dict[str, Any]] = {}
    for row in component_rows:
        component_id = int(row["component_id"])
        grouped.setdefault(
            component_id,
            {
                "component_id": component_id,
                "component_size": row["component_size"],
                "node_ids": [],
                "labels": [],
            },
        )
        grouped[component_id]["node_ids"].append(row["node_id"])
        grouped[component_id]["labels"].append(row["label"])
    return [grouped[key] for key in sorted(grouped)]


def _export_size(
    comments: List[Dict[str, Any]],
    requested_size: int,
    out_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    subset = comments[:requested_size]
    parsed = parse_comments(subset)
    canopy = _build_canopy(parsed, args.canopy_t1, args.canopy_t2, args.use_ann, args.auto_tune)
    canopy_assignments = None if args.skip_canopy else canopy.assignments

    nodes, edges = build_graph(
        subset,
        alpha=args.alpha_mention,
        beta=args.beta_reply,
        gamma=args.gamma_content,
        delta=args.delta_thread,
        k_neighbors=args.k_neighbors,
        parsed=parsed,
        canopy_assignments=canopy_assignments,
    )

    component_rows = _connected_components(nodes, edges)
    component_summary_rows = _component_summary(component_rows)
    isolated_nodes = sum(1 for row in component_rows if int(row["component_size"]) == 1)
    largest_component = max((int(row["component_size"]) for row in component_summary_rows), default=0)

    prefix = out_dir / f"graph_{requested_size}"
    node_fields = [
        "id",
        "label",
        "canopy",
        "comment_count",
        "reply_count",
        "mention_count",
        "thread_count",
        "total_likes",
        "avg_likes",
        "first_timestamp",
        "last_timestamp",
        "threads",
        "text_sample",
    ]
    _write_csv(prefix.with_name(f"{prefix.name}_nodes.csv"), nodes, node_fields)
    _write_csv(prefix.with_name(f"{prefix.name}_edges.csv"), edges, ["source", "target", "weight"])
    _write_csv(
        prefix.with_name(f"{prefix.name}_components.csv"),
        component_rows,
        ["component_id", "component_size", "node_id", "label"],
    )
    _write_csv(
        prefix.with_name(f"{prefix.name}_component_summary.csv"),
        component_summary_rows,
        ["component_id", "component_size", "node_ids", "labels"],
    )

    return {
        "requested_comments": requested_size,
        "n_comments": len(subset),
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "n_components": len(component_summary_rows),
        "largest_component_size": largest_component,
        "isolated_nodes": isolated_nodes,
        "n_canopies": len(canopy.canopies),
        "canopy_t1": canopy.threshold_t1 or args.canopy_t1,
        "canopy_t2": canopy.threshold_t2 or args.canopy_t2,
        "used_canopy_overlay": not args.skip_canopy,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export GraphBuilder nodes, edges, and connected components.")
    parser.add_argument("--comments", default="data/comments.json", help="Path ke file komentar JSON.")
    parser.add_argument("--out-dir", default="data/graph_exports", help="Folder output CSV.")
    parser.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES, help="Ukuran dataset yang diekspor.")
    parser.add_argument("--alpha-mention", type=float, default=1.0)
    parser.add_argument("--beta-reply", type=float, default=0.8)
    parser.add_argument("--gamma-content", type=float, default=0.5)
    parser.add_argument("--delta-thread", type=float, default=0.5)
    parser.add_argument("--k-neighbors", type=int, default=15)
    parser.add_argument("--canopy-t1", type=float, default=0.6)
    parser.add_argument("--canopy-t2", type=float, default=0.8)
    parser.add_argument("--use-ann", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--auto-tune", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-canopy", action="store_true", help="Bangun graf tanpa overlay/penalty canopy.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    comments_path = Path(args.comments)
    out_dir = Path(args.out_dir)
    comments = _load_comments(comments_path)

    summary_rows = []
    for size in args.sizes:
        if size > len(comments):
            raise ValueError(f"Ukuran {size} melebihi jumlah komentar tersedia ({len(comments)}).")
        summary = _export_size(comments, size, out_dir, args)
        summary_rows.append(summary)
        print(
            "size={requested_comments}: nodes={n_nodes}, edges={n_edges}, "
            "components={n_components}, largest={largest_component_size}, isolated={isolated_nodes}".format(**summary)
        )

    _write_csv(
        out_dir / "graph_summary.csv",
        summary_rows,
        [
            "requested_comments",
            "n_comments",
            "n_nodes",
            "n_edges",
            "n_components",
            "largest_component_size",
            "isolated_nodes",
            "n_canopies",
            "canopy_t1",
            "canopy_t2",
            "used_canopy_overlay",
        ],
    )
    print(f"Output disimpan di: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
