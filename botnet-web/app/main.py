import os
import asyncio
from time import perf_counter
import platform
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from enum import Enum
from uuid import uuid4
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl
from loguru import logger
from sklearn.metrics import silhouette_score

from .pipeline.tiktok_client import (
    fetch_comments,
    normalize_ms_token,
    TikTokDependencyError,
    TikTokFetchError,
    TikTokTokenError,
)
from .pipeline.parser import parse_comments
from .pipeline.canopy import build_canopies, CanopyArtifacts
from .pipeline.graph_builder import build_graph
from .pipeline.mst_cluster import mst_cluster
from .pipeline.scoring import score_clusters

app = FastAPI(title="Bot Network Detector (TikTok)")

# Windows safe event loop policy for subprocess (Playwright needs this)
if platform.system() == "Windows":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend under /static and expose index at /
static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
static_dir = os.path.abspath(static_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


def _format_validation_error(error: Dict[str, Any]) -> str:
    loc = [str(item) for item in error.get("loc", []) if item not in {"body", "query", "path"}]
    field = ".".join(loc) or "data"
    err_type = str(error.get("type", "")).lower()
    ctx = error.get("ctx") or {}
    min_value = ctx.get("ge", ctx.get("limit_value"))
    max_value = ctx.get("le", ctx.get("limit_value"))

    if "missing" in err_type or "required" in err_type:
        return f"Kolom `{field}` wajib diisi."
    if "url" in err_type or field == "url":
        return f"Kolom `{field}` harus berisi URL yang valid."
    if "greater_than_equal" in err_type or "not_ge" in err_type or err_type.endswith(".ge"):
        return f"Kolom `{field}` minimal {min_value}."
    if "less_than_equal" in err_type or "not_le" in err_type or err_type.endswith(".le"):
        return f"Kolom `{field}` maksimal {max_value}."
    if "integer" in err_type or "int" in err_type:
        return f"Kolom `{field}` harus berupa angka bulat."
    if "float" in err_type or "number" in err_type:
        return f"Kolom `{field}` harus berupa angka."
    if "bool" in err_type:
        return f"Kolom `{field}` harus bernilai benar atau salah."
    return f"Kolom `{field}` tidak valid."


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Request, exc: RequestValidationError):
    details = [_format_validation_error(error) for error in exc.errors()]
    return JSONResponse(
        status_code=422,
        content={"detail": "Permintaan tidak valid: " + " ".join(details)},
    )


@app.get("/")
async def root_index():
    return FileResponse(os.path.join(static_dir, "index.html"))


class RunParams(BaseModel):
    url: HttpUrl
    max_comments: int = Field(200, ge=10, le=10000)
    alpha_mention: float = Field(1.0, ge=0)
    beta_reply: float = Field(0.8, ge=0)
    gamma_content: float = Field(0.5, ge=0)
    delta_thread: float = Field(0.5, ge=0)
    k_neighbors: int = Field(15, ge=1, le=200)
    canopy_t1: float = Field(0.6, ge=0, le=1)
    canopy_t2: float = Field(0.8, ge=0, le=1)
    auto_tune_canopy: bool = True
    use_ann: bool = True
    use_mst_overlay: bool = True


class RunResult(BaseModel):
    nodes: List[Dict]
    edges: List[Dict]
    clusters: Dict[str, List[str]]  # cluster_id -> list of user_ids
    canopies: Dict[str, List[str]]
    suspicious: List[str]  # user_ids tagged suspicious
    suspicious_details: List[Dict]
    cluster_stats: Dict[str, Dict]
    metrics: Dict[str, float]
    timings: Dict[str, float]


AsyncProgressCallback = Callable[[str, float, Optional[str]], Awaitable[None]]


class JobStatusEnum(str, Enum):
    pending = "pending"
    running = "running"
    success = "success"
    error = "error"


@dataclass
class PipelineJob:
    job_id: str
    params: RunParams
    status: JobStatusEnum = JobStatusEnum.pending
    stage: str = "pending"
    progress: float = 0.0
    detail: str = ""
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    def to_response(self) -> "JobStatusResponse":
        result_model: Optional[RunResult] = None
        if self.result:
            result_model = _make_run_result(self.result)
        return JobStatusResponse(
            job_id=self.job_id,
            status=self.status,
            stage=self.stage,
            progress=self.progress,
            detail=self.detail or None,
            error=self.error,
            started_at=self.started_at,
            finished_at=self.finished_at,
            result=result_model,
        )


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatusEnum
    stage: Optional[str] = None
    progress: float = 0.0
    detail: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    result: Optional[RunResult] = None


class RunAsyncResponse(BaseModel):
    job_id: str
    status_url: str


jobs_lock = asyncio.Lock()
jobs: Dict[str, PipelineJob] = {}


def _make_run_result(data: Dict[str, Any]) -> RunResult:
    if hasattr(RunResult, "model_validate"):
        return RunResult.model_validate(data)  # type: ignore[attr-defined]
    return RunResult.parse_obj(data)  # type: ignore[attr-defined]


def _clone_params(params: RunParams) -> RunParams:
    if hasattr(params, "model_dump"):
        data = params.model_dump()
    else:
        data = params.dict()  # type: ignore[attr-defined]
    return RunParams(**data)


def _run_result_to_dict(result: RunResult) -> Dict[str, Any]:
    if hasattr(result, "model_dump"):
        return result.model_dump()
    return result.dict()  # type: ignore[attr-defined]


def _load_ms_token_from_env() -> str:
    errors: List[str] = []
    for key in ("ms_token", "MS_TOKEN"):
        raw_value = os.getenv(key)
        if raw_value is None:
            continue
        try:
            return normalize_ms_token(raw_value)
        except TikTokTokenError as exc:
            errors.append(str(exc))

    if errors:
        raise ValueError(errors[0])

    raise ValueError(
        "Variabel lingkungan `ms_token` belum diatur. Isi `ms_token` atau `MS_TOKEN` dengan cookie `msToken` TikTok yang valid."
    )


def _compute_silhouette_metrics(embeddings: Dict[str, Any], clusters: Dict[str, List[str]]) -> Dict[str, float]:
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
    except Exception as exc:
        logger.warning("Gagal menghitung silhouette score: {}", exc)
    return metrics


def _round_log_value(value: Any, digits: int = 4) -> Any:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value


def _summarize_group_sizes(groups: Dict[str, List[str]]) -> Dict[str, Any]:
    sizes = [len(members) for members in groups.values()]
    if not sizes:
        return {"count": 0, "avg_size": 0.0, "max_size": 0, "singletons": 0}
    return {
        "count": len(sizes),
        "avg_size": round(sum(sizes) / len(sizes), 2),
        "max_size": max(sizes),
        "singletons": sum(1 for size in sizes if size == 1),
    }


def _summarize_graph(nodes: List[Dict], edges: List[Dict]) -> Dict[str, Any]:
    connected_user_ids = set()
    weights: List[float] = []
    for edge in edges:
        connected_user_ids.add(edge.get("source"))
        connected_user_ids.add(edge.get("target"))
        weights.append(float(edge.get("weight", 0.0) or 0.0))

    return {
        "nodes": len(nodes),
        "edges": len(edges),
        "isolated_nodes": max(0, len(nodes) - len(connected_user_ids)),
        "avg_weight": round(sum(weights) / len(weights), 4) if weights else 0.0,
        "max_weight": round(max(weights), 4) if weights else 0.0,
    }


def _summarize_top_clusters(cluster_stats: Dict[str, Dict], limit: int = 5) -> List[Dict[str, Any]]:
    ranked = sorted(
        cluster_stats.items(),
        key=lambda item: float(item[1].get("score", 0.0) or 0.0),
        reverse=True,
    )
    return [
        {
            "cluster_id": cluster_id,
            "size": stats.get("cluster_size", 0),
            "score": _round_log_value(stats.get("score")),
            "suspicious": bool(stats.get("is_suspicious", False)),
            "reasons": stats.get("reasons", []),
        }
        for cluster_id, stats in ranked[:limit]
    ]


def _summarize_top_suspicious_users(details: List[Dict], limit: int = 5) -> List[Dict[str, Any]]:
    ranked = sorted(details, key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
    return [
        {
            "user_id": item.get("user_id"),
            "username": item.get("username"),
            "cluster_id": item.get("cluster_id"),
            "score": _round_log_value(item.get("score")),
            "reasons": item.get("reasons", []),
        }
        for item in ranked[:limit]
    ]


async def _emit_progress(progress_cb: Optional[AsyncProgressCallback], stage: str, progress: float, detail: Optional[str] = None) -> None:
    if not progress_cb:
        return
    try:
        await progress_cb(stage, progress, detail)
    except Exception:
        logger.debug("Callback progres gagal pada tahap {}", stage, exc_info=True)


async def _register_job(params: RunParams) -> PipelineJob:
    job_id = uuid4().hex
    job = PipelineJob(job_id=job_id, params=params)
    async with jobs_lock:
        jobs[job_id] = job
    return job


async def _update_job(job_id: str, **kwargs: Any) -> None:
    async with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return
        for key, value in kwargs.items():
            setattr(job, key, value)


def _persist_raw_comments(video_url: str, comments: List[Dict]) -> Optional[Path]:
    data_dir = Path(__file__).resolve().parents[1] / "data"
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "video_url": video_url,
        "saved_at": timestamp,
        "count": len(comments),
        "comments": comments,
    }
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = data_dir / f"comments_{timestamp}.json"
        with snapshot_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
        latest_path = data_dir / "comments.json"
        with latest_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
        return snapshot_path
    except Exception:
        logger.exception("Gagal menyimpan komentar mentah ke disk")
        return None


def _load_fallback_comments(max_count: int) -> tuple[List[Dict[str, Any]], Path]:
    """Load a known-good comment snapshot when TikTok cannot be reached."""
    configured_path = os.getenv("TIKTOK_FALLBACK_COMMENTS", "").strip()
    fallback_path = (
        Path(configured_path).expanduser()
        if configured_path
        else Path(__file__).resolve().parents[1] / "data" / "comments_20260622T154550Z.json"
    )

    if not fallback_path.is_file():
        raise FileNotFoundError(f"File komentar fallback tidak ditemukan: {fallback_path}")

    with fallback_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    raw_comments = payload.get("comments") if isinstance(payload, dict) else payload
    if not isinstance(raw_comments, list):
        raise ValueError("File komentar fallback tidak memiliki daftar `comments` yang valid.")

    comments = [item for item in raw_comments if isinstance(item, dict)]
    if not comments:
        raise ValueError("File komentar fallback tidak berisi komentar yang dapat dianalisis.")

    return comments[:max_count], fallback_path.resolve()


async def _execute_pipeline(
    params: RunParams,
    progress_cb: Optional[AsyncProgressCallback] = None,
    pipeline_id: Optional[str] = None,
) -> RunResult:
    pipeline_id = pipeline_id or uuid4().hex[:8]
    video_url = str(params.url)
    logger.info(
        (
            "[pipeline:{}] Mulai pipeline: url='{}', max_comments={}, "
            "weights=(mention={}, reply={}, content={}, thread={}), k_neighbors={}, "
            "canopy=(t1={}, t2={}, auto_tune={}, use_ann={}), mst_overlay={}"
        ),
        pipeline_id,
        video_url,
        params.max_comments,
        params.alpha_mention,
        params.beta_reply,
        params.gamma_content,
        params.delta_thread,
        params.k_neighbors,
        params.canopy_t1,
        params.canopy_t2,
        params.auto_tune_canopy,
        params.use_ann,
        params.use_mst_overlay,
    )

    t_total_start = perf_counter()

    await _emit_progress(progress_cb, "fetch", 0.05, "Mengambil komentar dari TikTok")
    logger.info("[pipeline:{}] Tahap fetch dimulai", pipeline_id)
    t_fetch_start = perf_counter()
    fetch_error: Exception | None = None
    try:
        ms_token = _load_ms_token_from_env()
        logger.info("[pipeline:{}] ms_token tersedia: {}", pipeline_id, bool(ms_token))
        comments = await fetch_comments(video_url, params.max_comments, ms_token=ms_token)
    except TikTokTokenError as exc:
        fetch_error = ValueError(str(exc))
        logger.warning("[pipeline:{}] Token TikTok tidak valid; mencoba file fallback", pipeline_id)
    except ValueError as exc:
        fetch_error = exc
        logger.warning("[pipeline:{}] Konfigurasi ms_token tidak valid; mencoba file fallback", pipeline_id)
    except TikTokFetchError as exc:
        fetch_error = exc
        logger.exception("[pipeline:{}] Pengambilan komentar TikTok gagal", pipeline_id)
    except TikTokDependencyError as exc:
        fetch_error = RuntimeError(str(exc))
        logger.exception("[pipeline:{}] Dependensi klien TikTok bermasalah", pipeline_id)
    except Exception as exc:
        fetch_error = RuntimeError(f"Gagal mengambil komentar TikTok: {exc}")
        logger.exception("[pipeline:{}] Pengambilan komentar gagal", pipeline_id)

    if fetch_error is not None:
        try:
            comments, fallback_path = _load_fallback_comments(params.max_comments)
        except Exception as fallback_exc:
            logger.exception("[pipeline:{}] File komentar fallback gagal dimuat", pipeline_id)
            raise RuntimeError(
                f"Pengambilan komentar TikTok gagal ({fetch_error}) dan file fallback "
                f"tidak dapat digunakan: {fallback_exc}"
            ) from fallback_exc
        logger.warning(
            "[pipeline:{}] Menggunakan {} komentar fallback dari '{}' karena: {}",
            pipeline_id,
            len(comments),
            fallback_path,
            fetch_error,
        )

    t_fetch = perf_counter() - t_fetch_start
    logger.info(
        "[pipeline:{}] Tahap fetch selesai: comments={}, elapsed={}s",
        pipeline_id,
        len(comments),
        round(t_fetch, 4),
    )
    await _emit_progress(progress_cb, "fetch", 0.15, f"Berhasil mengambil {len(comments)} komentar")

    if not comments:
        logger.warning("[pipeline:{}] Pipeline dihentikan: komentar publik kosong", pipeline_id)
        raise ValueError(
            "Video ini tidak memiliki komentar publik yang bisa dianalisis. "
            "Coba gunakan video lain yang komentarnya aktif dan dapat dilihat publik."
        )

    snapshot_path = _persist_raw_comments(video_url, comments)
    if snapshot_path:
        logger.info("[pipeline:{}] Komentar mentah disimpan ke {}", pipeline_id, snapshot_path)

    await _emit_progress(progress_cb, "parse", 0.3, "Menyusun layer komentar")
    logger.info("[pipeline:{}] Tahap parse dimulai", pipeline_id)
    t_parse_start = perf_counter()
    parsed = parse_comments(comments)
    t_parse = perf_counter() - t_parse_start
    logger.info(
        (
            "[pipeline:{}] Tahap parse selesai: users={}, user_texts={}, "
            "mention_pairs={}, reply_pairs={}, co_thread_pairs={}, elapsed={}s"
        ),
        pipeline_id,
        len(parsed.get("user_summary", {})),
        len([text for text in parsed.get("user_texts", {}).values() if text]),
        len(parsed.get("mention_edges", {})),
        len(parsed.get("reply_edges", {})),
        len(parsed.get("co_thread_edges", {})),
        round(t_parse, 4),
    )

    await _emit_progress(progress_cb, "canopy", 0.45, "Membentuk assignment canopy")
    logger.info("[pipeline:{}] Tahap canopy dimulai", pipeline_id)
    t_canopy_start = perf_counter()
    try:
        canopy_artifacts = build_canopies(
            parsed.get("user_texts", {}),
            params.canopy_t1,
            params.canopy_t2,
            use_ann=params.use_ann,
            auto_tune=params.auto_tune_canopy,
        )
    except ValueError as exc:
        logger.warning("[pipeline:{}] Canopy clustering dilewati: {}", pipeline_id, exc)
        canopy_artifacts = CanopyArtifacts(assignments={}, canopies={}, embeddings={})
    t_canopy = perf_counter() - t_canopy_start
    canopy_size_summary = _summarize_group_sizes(canopy_artifacts.canopies)
    logger.info(
        (
            "[pipeline:{}] Tahap canopy selesai: {}, assignments={}, "
            "selected_t1={}, selected_t2={}, silhouette={}, candidates={}, "
            "used_ann={}, ann_fallback={}, vectorize={}s, reduce={}s, "
            "threshold_grid={}s, ann_index={}s, ann_query={}s, brute_force={}s, "
            "assignment={}s, elapsed={}s"
        ),
        pipeline_id,
        canopy_size_summary,
        len(canopy_artifacts.assignments),
        _round_log_value(canopy_artifacts.threshold_t1 or params.canopy_t1),
        _round_log_value(canopy_artifacts.threshold_t2 or params.canopy_t2),
        _round_log_value(canopy_artifacts.threshold_silhouette),
        canopy_artifacts.threshold_candidates,
        canopy_artifacts.timing.used_ann,
        canopy_artifacts.timing.ann_fallback,
        round(canopy_artifacts.timing.vectorize_sec, 4),
        round(canopy_artifacts.timing.reduce_sec, 4),
        round(canopy_artifacts.timing.threshold_grid_sec, 4),
        round(canopy_artifacts.timing.ann_index_sec, 4),
        round(canopy_artifacts.timing.ann_query_sec, 4),
        round(canopy_artifacts.timing.brute_force_sec, 4),
        round(canopy_artifacts.timing.assignment_sec, 4),
        round(t_canopy, 4),
    )

    await _emit_progress(progress_cb, "graph", 0.6, "Membangun graf interaksi")
    logger.info("[pipeline:{}] Tahap graph dimulai", pipeline_id)
    t_build_start = perf_counter()
    nodes, edges = build_graph(
        comments,
        alpha=params.alpha_mention,
        beta=params.beta_reply,
        gamma=params.gamma_content,
        delta=params.delta_thread,
        k_neighbors=params.k_neighbors,
        parsed=parsed,
        canopy_assignments=canopy_artifacts.assignments,
    )
    t_build = perf_counter() - t_build_start
    logger.info(
        "[pipeline:{}] Tahap graph selesai: {}, elapsed={}s",
        pipeline_id,
        _summarize_graph(nodes, edges),
        round(t_build, 4),
    )

    await _emit_progress(progress_cb, "cluster", 0.75, "Menjalankan MST clustering")
    logger.info("[pipeline:{}] Tahap cluster dimulai", pipeline_id)
    t_cluster_start = perf_counter()
    clusters = mst_cluster(nodes, edges, use_overlay=params.use_mst_overlay)
    t_cluster = perf_counter() - t_cluster_start
    logger.info(
        "[pipeline:{}] Tahap cluster selesai: {}, elapsed={}s",
        pipeline_id,
        _summarize_group_sizes(clusters),
        round(t_cluster, 4),
    )

    await _emit_progress(progress_cb, "score", 0.9, "Menghitung skor cluster")
    logger.info("[pipeline:{}] Tahap score dimulai", pipeline_id)
    t_score_start = perf_counter()
    suspicious, metrics, suspicious_details, cluster_stats, node_scores = score_clusters(
        nodes,
        edges,
        clusters,
        canopy_assignments=canopy_artifacts.assignments,
        parsed=parsed,
    )
    suspicious_set = set(suspicious)
    for node in nodes:
        uid = node["id"]
        node["suspicious_score"] = float(node_scores.get(uid, 0.0) or 0.0)
        node["suspicious_flag"] = uid in suspicious_set
    metrics.update(_compute_silhouette_metrics(canopy_artifacts.embeddings, clusters))
    t_score = perf_counter() - t_score_start
    logger.info(
        (
            "[pipeline:{}] Tahap score selesai: suspicious_users={}, "
            "suspicious_clusters={}, cutoff={}, top_clusters={}, top_users={}, elapsed={}s"
        ),
        pipeline_id,
        len(suspicious),
        int(metrics.get("num_suspicious_clusters", 0.0) or 0.0),
        _round_log_value(metrics.get("suspicious_score_cutoff")),
        _summarize_top_clusters(cluster_stats),
        _summarize_top_suspicious_users(suspicious_details),
        round(t_score, 4),
    )

    t_total = perf_counter() - t_total_start

    timings = {
        "fetch_sec": round(t_fetch, 4),
        "parse_sec": round(t_parse, 4),
        "canopy_sec": round(t_canopy, 4),
        "build_sec": round(t_build, 4),
        "cluster_sec": round(t_cluster, 4),
        "score_sec": round(t_score, 4),
        "total_sec": round(t_total, 4),
    }
    timings["canopy_threshold_grid_sec"] = round(canopy_artifacts.timing.threshold_grid_sec, 4)

    if canopy_artifacts.canopies:
        canopy_sizes = [len(members) for members in canopy_artifacts.canopies.values()]
        if canopy_sizes:
            metrics.update(
                {
                    "num_canopies": float(len(canopy_sizes)),
                    "avg_canopy_size": float(sum(canopy_sizes) / len(canopy_sizes)),
                    "max_canopy_size": float(max(canopy_sizes)),
                    "canopy_t1": float(canopy_artifacts.threshold_t1 or params.canopy_t1),
                    "canopy_t2": float(canopy_artifacts.threshold_t2 or params.canopy_t2),
                    "canopy_threshold_candidates": float(canopy_artifacts.threshold_candidates),
                }
            )
            if canopy_artifacts.threshold_silhouette is not None:
                metrics["canopy_threshold_silhouette"] = float(canopy_artifacts.threshold_silhouette)

    await _emit_progress(progress_cb, "finalizing", 0.97, "Menyiapkan hasil akhir")
    logger.info(
        "[pipeline:{}] Pipeline selesai: timings={}, metrics_summary={}",
        pipeline_id,
        timings,
        {
            "num_nodes": metrics.get("num_nodes"),
            "num_edges": metrics.get("num_edges"),
            "num_clusters": metrics.get("num_clusters"),
            "num_suspicious_users": metrics.get("num_suspicious_users"),
            "modularity": _round_log_value(metrics.get("modularity")),
            "conductance_mean": _round_log_value(metrics.get("conductance_mean")),
            "silhouette_score": _round_log_value(metrics.get("silhouette_score")),
        },
    )

    return RunResult(
        nodes=nodes,
        edges=edges,
        clusters=clusters,
        canopies=canopy_artifacts.canopies,
        suspicious=suspicious,
        suspicious_details=suspicious_details,
        cluster_stats=cluster_stats,
        metrics=metrics,
        timings=timings,
    )


async def _run_job(job_id: str, params: RunParams) -> None:
    await _update_job(
        job_id,
        status=JobStatusEnum.running,
        stage="starting",
        progress=0.02,
        detail="Menyiapkan pipeline",
        started_at=datetime.utcnow(),
    )

    async def progress_cb(stage: str, progress: float, detail: Optional[str] = None) -> None:
        logger.info(
            "[pipeline:{}] Progress {:.0f}%: stage='{}', detail='{}'",
            job_id,
            progress * 100,
            stage,
            detail or "",
        )
        await _update_job(job_id, stage=stage, progress=progress, detail=(detail or ""))

    try:
        result = await _execute_pipeline(params, progress_cb, pipeline_id=job_id)
    except Exception as exc:
        logger.exception("Proses pipeline {} gagal", job_id)
        await _update_job(
            job_id,
            status=JobStatusEnum.error,
            stage="failed",
            progress=1.0,
            detail=str(exc),
            error=str(exc),
            finished_at=datetime.utcnow(),
        )
    else:
        await _update_job(
            job_id,
            status=JobStatusEnum.success,
            stage="completed",
            progress=1.0,
            detail="Selesai",
            result=_run_result_to_dict(result),
            finished_at=datetime.utcnow(),
        )
@app.post("/api/run", response_model=RunResult)
async def run_pipeline(params: RunParams):
    try:
        return await _execute_pipeline(params)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TikTokFetchError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/run_async", response_model=RunAsyncResponse)
async def run_pipeline_async(params: RunParams):
    params_copy = _clone_params(params)
    job = await _register_job(params_copy)
    asyncio.create_task(_run_job(job.job_id, params_copy))
    return RunAsyncResponse(job_id=job.job_id, status_url=f"/api/status/{job.job_id}")


@app.get("/api/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    async with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Proses analisis tidak ditemukan.")
        response = job.to_response()
    return response


class CanopyRequest(BaseModel):
    # Kirim salah satu: `user_texts` atau `comments` (list berisi dict komentar).
    user_texts: Optional[Dict[str, str]] = None
    comments: Optional[List[Dict[str, Any]]] = None
    t1: float = 0.6
    t2: float = 0.8
    auto_tune: bool = True
    use_ann: bool = True


class CanopyResult(BaseModel):
    assignments: Dict[str, str]
    canopies: Dict[str, List[str]]
    num_canopies: int
    avg_canopy_size: float
    selected_t1: float
    selected_t2: float
    threshold_silhouette: Optional[float] = None
    threshold_candidates: int
    timings: Dict[str, float]


@app.post("/api/canopy", response_model=CanopyResult)
async def canopy_only(req: CanopyRequest):
    """Run only the canopy clustering stage.

    Accepts either `user_texts` mapping or raw `comments` payload. Returns assignments and basic timings.
    """
    t0 = perf_counter()
    if not req.user_texts and not req.comments:
        raise HTTPException(status_code=400, detail="Kirim salah satu: `user_texts` atau `comments` pada isi request.")

    if req.user_texts:
        user_texts = req.user_texts
    else:
        # parse comments to extract user_texts
        try:
            parsed = parse_comments(req.comments or [])
            user_texts = parsed.get("user_texts", {})
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Gagal memproses komentar: {exc}") from exc

    if not user_texts:
        raise HTTPException(status_code=400, detail="Tidak ada teks pengguna yang bisa dipakai untuk komputasi canopy.")

    try:
        canopy = build_canopies(user_texts, req.t1, req.t2, use_ann=bool(req.use_ann), auto_tune=bool(req.auto_tune))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    t1 = perf_counter() - t0
    sizes = [len(m) for m in canopy.canopies.values()] if canopy.canopies else []
    avg = float(sum(sizes) / len(sizes)) if sizes else 0.0

    return CanopyResult(
        assignments=canopy.assignments,
        canopies=canopy.canopies,
        num_canopies=len(sizes),
        avg_canopy_size=avg,
        selected_t1=float(canopy.threshold_t1 or req.t1),
        selected_t2=float(canopy.threshold_t2 or req.t2),
        threshold_silhouette=canopy.threshold_silhouette,
        threshold_candidates=int(canopy.threshold_candidates),
        timings={
            "canopy_sec": round(t1, 4),
            "threshold_grid_sec": round(canopy.timing.threshold_grid_sec, 4),
        },
    )


class MSTRequest(BaseModel):
    # Kirim salah satu: `nodes` + `edges` atau `comments` (graf akan dibangun dari komentar).
    nodes: Optional[List[Dict[str, Any]]] = None
    edges: Optional[List[Dict[str, Any]]] = None
    comments: Optional[List[Dict[str, Any]]] = None
    alpha_mention: float = 1.0
    beta_reply: float = 0.8
    gamma_content: float = 0.5
    delta_thread: float = 0.5
    k_neighbors: int = 15


class MSTResult(BaseModel):
    clusters: Dict[str, List[str]]
    timings: Dict[str, float]


@app.post("/api/mst", response_model=MSTResult)
async def mst_only(req: MSTRequest):
    """Run only MST clustering. Accepts pre-built `nodes`+`edges` or raw `comments` to build graph first."""
    t0 = perf_counter()
    if (not req.nodes or not req.edges) and not req.comments:
        raise HTTPException(status_code=400, detail="Kirim salah satu: `nodes` + `edges` atau `comments` pada isi request.")

    if req.nodes and req.edges:
        nodes = req.nodes
        edges = req.edges
    else:
        try:
            parsed = parse_comments(req.comments or [])
            nodes, edges = build_graph(
                req.comments or [],
                alpha=req.alpha_mention,
                beta=req.beta_reply,
                gamma=req.gamma_content,
                delta=req.delta_thread,
                k_neighbors=req.k_neighbors,
                parsed=parsed,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Gagal membangun graf dari komentar: {exc}") from exc

    t_build = perf_counter() - t0
    t_cluster_start = perf_counter()
    clusters = mst_cluster(nodes, edges, use_overlay=True)
    t_cluster = perf_counter() - t_cluster_start

    return MSTResult(clusters=clusters, timings={"build_sec": round(t_build, 4), "cluster_sec": round(t_cluster, 4)})
