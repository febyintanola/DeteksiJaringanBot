import os
import re
import sys
import asyncio
import platform
from typing import List, Dict, Any, Callable, Any as AnyType
from pathlib import Path
from loguru import logger


class TikTokFetchError(RuntimeError):
    """Raised when TikTok comments cannot be fetched after retry strategies."""


def extract_video_id_from_url(url: str) -> str:
    # Typical tiktok video URL contains '/video/<id>'
    m = re.search(r"/video/(\d+)", url)
    if m:
        return m.group(1)
    # fallback: let TikTokApi handle URL (it can resolve id internally), but we still keep id=None
    return ""


def _ensure_tiktokapi_on_path():
    """Ensure local TikTok-Api repo is importable if not installed."""
    api_path = os.getenv("TIKTOK_API_PATH")
    if not api_path:
        here = Path(__file__).resolve()
        repo_root = here.parents[3]  # .../Prototyping
        candidate = repo_root / "TikTok-Api"
        if candidate.exists():
            api_path = str(candidate)
    if api_path and api_path not in sys.path:
        sys.path.insert(0, api_path)


async def fetch_comments(url: str, max_count: int, ms_token: str) -> List[Dict[str, Any]]:
    """
    Fetch comments but run Playwright/TikTokApi in an isolated selector event loop
    in a background thread to avoid Windows Proactor limitations.

    Returns list of dicts: {comment_id, user_id, username, text, timestamp, likes, raw}
    """

    def run_in_isolated_loop(coro_func: Callable[..., AnyType], *args, **kwargs):
        # Use ProactorEventLoopPolicy for subprocess support on Windows
        if platform.system() == "Windows":
            try:
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            except Exception:
                pass
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro_func(*args, **kwargs))
        finally:
            try:
                loop.run_until_complete(asyncio.sleep(0))
            except Exception:
                pass
            loop.close()

    async def _fetch_inner(url: str, max_count: int, ms_token: str) -> List[Dict[str, Any]]:
        # Lazy import to avoid requiring playwright on app startup
        _ensure_tiktokapi_on_path()
        try:
            from TikTokApi import TikTokApi  # type: ignore
            from TikTokApi.exceptions import EmptyResponseException  # type: ignore
        except Exception as e:
            print(f"Failed to import TikTokApi: {e}")
            raise RuntimeError(
                "TikTokApi (dan dependency Playwright) belum siap. "
                "Pastikan repo TikTok-Api ada atau set TIKTOK_API_PATH, lalu: "
                "pip install playwright; python -m playwright install chromium firefox webkit. "
                "Set juga TIKTOK_BROWSER=webkit bila hanya webkit yang terpasang."
            ) from e

        video_id = extract_video_id_from_url(url)
        print(f"Fetching comments for video_id={video_id}, max_count={max_count}, ms_token={'***' if ms_token else None}")
        preferred_browser = os.getenv("TIKTOK_BROWSER", "chromium")
        executable_path = os.getenv("TIKTOK_EXECUTABLE_PATH") or None
        prefer_headless = os.getenv("TIKTOK_HEADLESS", "true").strip().lower() not in {"0", "false", "no"}

        attempts = [
            (preferred_browser, prefer_headless),
            ("webkit", True),
            ("webkit", False),
            ("chromium", False),
        ]

        # Preserve attempt order while removing duplicates.
        seen = set()
        ordered_attempts = []
        for browser, headless in attempts:
            key = (browser, headless)
            if key in seen:
                continue
            seen.add(key)
            ordered_attempts.append(key)

        last_exc: Exception | None = None
        for browser, headless in ordered_attempts:
            logger.info("TikTok fetch attempt using browser='{}', headless={}", browser, headless)
            comments: List[Dict[str, Any]] = []
            try:
                async with TikTokApi() as api:
                    await api.create_sessions(
                        ms_tokens=[ms_token],
                        num_sessions=1,
                        sleep_after=3,
                        browser=browser,
                        executable_path=executable_path,
                        headless=headless,
                    )
                    video = api.video(id=video_id) if video_id else api.video(url=url)
                    async for c in video.comments(count=max_count):
                        raw = getattr(c, "as_dict", {}) or {}
                        usr = raw.get("user", {})
                        item = {
                            "comment_id": raw.get("cid"),
                            "user_id": usr.get("uid"),
                            "username": usr.get("unique_id"),
                            "text": raw.get("text", ""),
                            "timestamp": raw.get("create_time"),
                            "likes": raw.get("digg_count", 0),
                            "raw": raw,
                        }
                        comments.append(item)
                    if comments:
                        return comments
            except EmptyResponseException as exc:
                last_exc = exc
                logger.warning(
                    "TikTok empty response on browser='{}', headless={}. Trying fallback.",
                    browser,
                    headless,
                )
                continue
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "TikTok fetch attempt failed on browser='{}', headless={}: {}",
                    browser,
                    headless,
                    exc,
                )
                continue

        guidance = (
            "TikTok returned empty/blocked responses after retry attempts. "
            "Try a fresh ms_token, set TIKTOK_BROWSER=webkit, set TIKTOK_HEADLESS=false, "
            "or use a residential proxy."
        )
        raise TikTokFetchError(guidance) from last_exc

    # Run isolated to_thread to avoid current loop policy limitations
    return await asyncio.to_thread(run_in_isolated_loop, _fetch_inner, url, max_count, ms_token)
