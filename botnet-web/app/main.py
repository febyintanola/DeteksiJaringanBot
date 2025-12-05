import os
import asyncio
from time import perf_counter
import platform
import json
from datetime import datetime
from pathlib import Path
from enum import Enum
from uuid import uuid4
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl
from loguru import logger

from .pipeline.tiktok_client import fetch_comments
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
    canopy_t1: float = Field(0.8, ge=0, le=1)
    canopy_t2: float = Field(0.6, ge=0, le=1)
    use_ann: bool = True
    use_mst_overlay: bool = True


class RunResult(BaseModel):
    nodes: List[Dict]
    edges: List[Dict]
    clusters: Dict[str, List[str]]  # cluster_id -> list of user_ids
    canopies: Dict[str, List[str]]
    suspicious: List[str]  # user_ids tagged suspicious
    suspicious_details: List[Dict]
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


async def _emit_progress(progress_cb: Optional[AsyncProgressCallback], stage: str, progress: float, detail: Optional[str] = None) -> None:
    if not progress_cb:
        return
    try:
        await progress_cb(stage, progress, detail)
    except Exception:
        logger.debug("Progress callback failed for stage {}", stage, exc_info=True)


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
        logger.exception("Failed to persist raw comments to disk")
        return None


async def _execute_pipeline(params: RunParams, progress_cb: Optional[AsyncProgressCallback] = None) -> RunResult:
    ms_token = os.getenv("ms_token") or os.getenv("MS_TOKEN")
    if not ms_token:
        raise ValueError("ms_token env var is required to access TikTok API. Set ms_token in your environment.")

    logger.info("ms_token present: {}", bool(ms_token))

    video_url = str(params.url)
    t_total_start = perf_counter()

    await _emit_progress(progress_cb, "fetch", 0.05, "Fetching comments from TikTok")
    t_fetch_start = perf_counter()
    try:
        comments = await fetch_comments(video_url, params.max_comments, ms_token=ms_token)
    except Exception as exc:
        logger.exception("Fetch comments failed")
        raise RuntimeError(f"Failed to fetch comments: {exc}") from exc
    t_fetch = perf_counter() - t_fetch_start
    await _emit_progress(progress_cb, "fetch", 0.15, f"Fetched {len(comments)} comments")

    if not comments:
        raise ValueError("No comments retrieved.")

    snapshot_path = _persist_raw_comments(video_url, comments)
    if snapshot_path:
        logger.info("Persisted raw comments to {}", snapshot_path)

    await _emit_progress(progress_cb, "parse", 0.3, "Parsing comment layers")
    t_parse_start = perf_counter()
    parsed = parse_comments(comments)
    t_parse = perf_counter() - t_parse_start

    await _emit_progress(progress_cb, "canopy", 0.45, "Building canopy assignments")
    t_canopy_start = perf_counter()
    try:
        canopy_artifacts = build_canopies(
            parsed.get("user_texts", {}),
            params.canopy_t1,
            params.canopy_t2,
            use_ann=params.use_ann,
        )
    except ValueError as exc:
        logger.warning("Canopy clustering skipped: {}", exc)
        canopy_artifacts = CanopyArtifacts(assignments={}, canopies={}, embeddings={})
    t_canopy = perf_counter() - t_canopy_start

    await _emit_progress(progress_cb, "graph", 0.6, "Building interaction graph")
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

    await _emit_progress(progress_cb, "cluster", 0.75, "Running MST clustering")
    t_cluster_start = perf_counter()
    clusters = mst_cluster(nodes, edges, use_overlay=params.use_mst_overlay)
    t_cluster = perf_counter() - t_cluster_start

    await _emit_progress(progress_cb, "score", 0.9, "Scoring clusters")
    t_score_start = perf_counter()
    suspicious, metrics, suspicious_details = score_clusters(
        nodes,
        edges,
        clusters,
        canopy_assignments=canopy_artifacts.assignments,
    )
    t_score = perf_counter() - t_score_start

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

    if canopy_artifacts.canopies:
        canopy_sizes = [len(members) for members in canopy_artifacts.canopies.values()]
        if canopy_sizes:
            metrics.update(
                {
                    "num_canopies": float(len(canopy_sizes)),
                    "avg_canopy_size": float(sum(canopy_sizes) / len(canopy_sizes)),
                    "max_canopy_size": float(max(canopy_sizes)),
                }
            )

    await _emit_progress(progress_cb, "finalizing", 0.97, "Packaging result")

    return RunResult(
        nodes=nodes,
        edges=edges,
        clusters=clusters,
        canopies=canopy_artifacts.canopies,
        suspicious=suspicious,
        suspicious_details=suspicious_details,
        metrics=metrics,
        timings=timings,
    )


async def _run_job(job_id: str, params: RunParams) -> None:
    await _update_job(
        job_id,
        status=JobStatusEnum.running,
        stage="starting",
        progress=0.02,
        detail="Starting pipeline",
        started_at=datetime.utcnow(),
    )

    async def progress_cb(stage: str, progress: float, detail: Optional[str] = None) -> None:
        await _update_job(job_id, stage=stage, progress=progress, detail=(detail or ""))

    try:
        result = await _execute_pipeline(params, progress_cb)
    except Exception as exc:
        logger.exception("Pipeline job {} failed", job_id)
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
            detail="Completed",
            result=_run_result_to_dict(result),
            finished_at=datetime.utcnow(),
        )
@app.post("/api/run", response_model=RunResult)
async def run_pipeline(params: RunParams):
    try:
        return await _execute_pipeline(params)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
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
            raise HTTPException(status_code=404, detail="Job not found")
        response = job.to_response()
    return response


class CanopyRequest(BaseModel):
    # Provide either `user_texts` or `comments` (list of comment dicts)
    user_texts: Optional[Dict[str, str]] = None
    comments: Optional[List[Dict[str, Any]]] = None
    t1: float = 0.8
    t2: float = 0.6
    use_ann: bool = True


class CanopyResult(BaseModel):
    assignments: Dict[str, str]
    canopies: Dict[str, List[str]]
    num_canopies: int
    avg_canopy_size: float
    timings: Dict[str, float]


@app.post("/api/canopy", response_model=CanopyResult)
async def canopy_only(req: CanopyRequest):
    """Run only the canopy clustering stage.

    Accepts either `user_texts` mapping or raw `comments` payload. Returns assignments and basic timings.
    """
    t0 = perf_counter()
    if not req.user_texts and not req.comments:
        raise HTTPException(status_code=400, detail="Provide either `user_texts` or `comments` in the request body")

    if req.user_texts:
        user_texts = req.user_texts
    else:
        # parse comments to extract user_texts
        try:
            parsed = parse_comments(req.comments or [])
            user_texts = parsed.get("user_texts", {})
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to parse comments: {exc}") from exc

    if not user_texts:
        raise HTTPException(status_code=400, detail="No user texts available for canopy computation")

    try:
        canopy = build_canopies(user_texts, req.t1, req.t2, use_ann=bool(req.use_ann))
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
        timings={"canopy_sec": round(t1, 4)},
    )


class MSTRequest(BaseModel):
    # Provide either `nodes`+`edges` or `comments` (then graph will be built)
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
        raise HTTPException(status_code=400, detail="Provide either `nodes`+`edges` or `comments` in the request body")

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
            raise HTTPException(status_code=400, detail=f"Failed to build graph from comments: {exc}") from exc

    t_build = perf_counter() - t0
    t_cluster_start = perf_counter()
    clusters = mst_cluster(nodes, edges, use_overlay=True)
    t_cluster = perf_counter() - t_cluster_start

    return MSTResult(clusters=clusters, timings={"build_sec": round(t_build, 4), "cluster_sec": round(t_cluster, 4)})
