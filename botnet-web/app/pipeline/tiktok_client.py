import os
import re
import sys
import asyncio
import platform
import urllib.error
import urllib.parse
import urllib.request
from typing import List, Dict, Any, Callable, Any as AnyType, Optional
from pathlib import Path
from loguru import logger


class TikTokFetchError(RuntimeError):
    """Raised when TikTok comments cannot be fetched after retry strategies."""


class TikTokTokenError(ValueError):
    """Raised when the ms_token configuration is missing or malformed."""


class TikTokDependencyError(RuntimeError):
    """Raised when TikTokApi / Playwright dependencies are not ready."""


_PLACEHOLDER_TOKENS = {
    "your_ms_token_here",
    "<isi_ms_token>",
    "isi_ms_token",
    "<your_ms_token_here>",
    "your_token_here",
}


_VIDEO_ID_QUERY_KEYS = {
    "aweme_id",
    "awemeId",
    "item_id",
    "itemId",
    "share_item_id",
    "shareItemId",
    "video_id",
    "videoId",
}


def extract_video_id_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(str(url or "").strip())
    text = urllib.parse.unquote(str(url or ""))

    for pattern in (
        r"/video/(\d{5,25})",
        r"/photo/(\d{5,25})",
        r"/share/video/(\d{5,25})",
        r"/i18n/share/video/(\d{5,25})",
        r"/embed/(?:v2/)?(\d{5,25})",
    ):
        match = re.search(pattern, text)
        if match:
            return match.group(1)

    query = urllib.parse.parse_qs(parsed.query)
    for key in _VIDEO_ID_QUERY_KEYS:
        for value in query.get(key, []):
            if re.fullmatch(r"\d{5,25}", value):
                return value

    return ""


def _is_tiktok_short_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    host = (parsed.netloc or "").lower().split(":")[0]
    if host in {"vt.tiktok.com", "vm.tiktok.com", "t.tiktok.com"}:
        return True
    return host in {"www.tiktok.com", "m.tiktok.com"} and parsed.path.startswith("/t/")


def _extract_video_id_from_html(html_text: str) -> str:
    normalized = urllib.parse.unquote(html_text or "")
    normalized = normalized.replace("\\u002F", "/").replace("\\/", "/")

    for pattern in (
        r'"(?:aweme_id|awemeId|item_id|itemId|share_item_id|shareItemId|video_id|videoId)"\s*[:=]\s*"?(\d{5,25})"?',
        r"(?:aweme_id|item_id|share_item_id|video_id)=([0-9]{5,25})",
        r'"ItemModule"\s*:\s*\{\s*"(\d{5,25})"\s*:',
        r"https://www\.tiktok\.com/@[^\"']+/(?:video|photo)/(\d{5,25})",
        r"/(?:video|photo)/(\d{5,25})",
    ):
        match = re.search(pattern, normalized)
        if match:
            return match.group(1)

    return ""


def _extract_tiktok_username(url: str) -> str:
    match = re.search(r"/@([^/?#]+)", urllib.parse.unquote(str(url or "")))
    if match:
        return match.group(1).strip("@")
    return "tiktok"


def _canonical_tiktok_video_url(video_id: str, source_url: str) -> str:
    username = _extract_tiktok_username(source_url) or "tiktok"
    return f"https://www.tiktok.com/@{username}/video/{video_id}"


def resolve_tiktok_video_url(url: str) -> str:
    """
    Expand TikTok shortlinks before passing them to TikTokApi.

    TikTokApi expects canonical video URLs such as
    https://www.tiktok.com/@user/video/6829267836783971589. Shortlinks like
    https://vt.tiktok.com/... redirect there, but TikTokApi does not resolve
    that format reliably on its own.
    """
    cleaned = str(url or "").strip()
    if not cleaned or extract_video_id_from_url(cleaned) or not _is_tiktok_short_url(cleaned):
        return cleaned

    request = urllib.request.Request(
        cleaned,
        method="GET",
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            resolved = response.geturl()
            charset = response.headers.get_content_charset() or "utf-8"
            body = response.read(2_000_000).decode(charset, errors="ignore")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise TikTokFetchError(
            "Link TikTok pendek tidak bisa di-resolve otomatis. "
            "Buka link itu di browser, lalu salin URL final yang berbentuk "
            "`https://www.tiktok.com/@username/video/ID`."
        ) from exc

    video_id = extract_video_id_from_url(resolved) or _extract_video_id_from_html(body)
    if not video_id:
        raise TikTokFetchError(
            "Link TikTok pendek berhasil dibuka, tetapi TikTok mengarahkannya ke halaman non-video "
            "atau halaman proteksi/login. Buka link itu di browser, lalu salin URL final yang berbentuk "
            "`https://www.tiktok.com/@username/video/ID`."
        )

    canonical = resolved if extract_video_id_from_url(resolved) else _canonical_tiktok_video_url(video_id, resolved)
    logger.info("Resolved TikTok short URL '{}' to '{}'", cleaned, canonical)
    return canonical


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


def normalize_ms_token(ms_token: Optional[str]) -> str:
    raw_value = (ms_token or "").strip()
    if not raw_value:
        raise TikTokTokenError(
            "Variabel lingkungan `ms_token` belum diisi. Isi dengan cookie `msToken` TikTok yang valid sebelum menjalankan analisis."
        )

    cleaned = raw_value.strip().strip("'").strip('"')
    if not cleaned:
        raise TikTokTokenError(
            "Variabel lingkungan `ms_token` kosong. Isi dengan cookie `msToken` TikTok yang valid sebelum menjalankan analisis."
        )

    if cleaned.lower() in _PLACEHOLDER_TOKENS:
        raise TikTokTokenError(
            "Variabel lingkungan `ms_token` masih placeholder. Ganti dengan nilai cookie `msToken` TikTok dari sesi peramban yang masih aktif."
        )

    return cleaned


def _mask_token(ms_token: str) -> str:
    if not ms_token:
        return "<empty>"
    if len(ms_token) <= 8:
        return "***"
    return f"{ms_token[:4]}...{ms_token[-4:]}"


def _build_fetch_error_message(
    last_exc: Optional[BaseException],
    *,
    blocked_response: bool,
    invalid_response: bool,
) -> str:
    exc_name = last_exc.__class__.__name__ if last_exc else "UnknownError"
    exc_text = str(last_exc or "").strip()
    exc_text_lower = exc_text.lower()

    if blocked_response:
        return (
            "TikTok mengembalikan respons kosong saat mengambil komentar. "
            "Ini biasanya berarti request dari aplikasi terdeteksi atau dibatasi anti-bot TikTok, "
            "bukan berarti komentar videonya tidak ada. Coba perbarui `ms_token`, jalankan dengan "
            "`TIKTOK_HEADLESS=false` atau `TIKTOK_BROWSER=webkit`, ganti jaringan/proxy, atau tunggu beberapa saat."
        )

    if invalid_response or "invalidresponse" in exc_name.lower() or "invalid json" in exc_text_lower:
        return (
            "TikTok mengembalikan respons yang tidak valid untuk video ini. "
            "Kemungkinan komentar video tidak bisa diakses atau sesi `ms_token` tidak punya akses yang cukup. "
            f"Detail teknis: {exc_name}."
        )

    if "executable doesn't exist" in exc_text_lower or "browsertype.launch" in exc_text_lower or "playwright" in exc_text_lower:
        return (
            "Peramban Playwright belum siap untuk TikTokApi. "
            "Jalankan `python -m playwright install chromium firefox webkit`, lalu coba lagi."
        )

    if "invalid browser argument" in exc_text_lower:
        return "Konfigurasi `TIKTOK_BROWSER` tidak valid. Gunakan `chromium`, `firefox`, atau `webkit`."

    if exc_text:
        return f"Gagal mengambil komentar TikTok: {exc_text}"

    return "Gagal mengambil komentar TikTok karena sesi tidak dapat dibuat."


async def fetch_comments(url: str, max_count: int, ms_token: str) -> List[Dict[str, Any]]:
    """
    Fetch comments but run Playwright/TikTokApi in an isolated selector event loop
    in a background thread to avoid Windows Proactor limitations.

    Returns list of dicts: {comment_id, user_id, username, text, timestamp, likes, raw}
    """
    ms_token = normalize_ms_token(ms_token)

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
            from TikTokApi.exceptions import EmptyResponseException, InvalidJSONException, InvalidResponseException  # type: ignore
        except Exception as e:
            logger.exception("Failed to import TikTokApi dependencies")
            raise TikTokDependencyError(
                "TikTokApi (dan dependensi Playwright) belum siap. "
                "Pastikan repo TikTok-Api ada atau atur TIKTOK_API_PATH, lalu: "
                "pip install playwright; python -m playwright install chromium firefox webkit. "
                "Atur juga TIKTOK_BROWSER=webkit bila hanya webkit yang terpasang."
            ) from e

        url = resolve_tiktok_video_url(url)
        video_id = extract_video_id_from_url(url)
        logger.info(
            "Fetching comments for video_id='{}', max_count={}, ms_token={}",
            video_id or "<resolve-from-url>",
            max_count,
            _mask_token(ms_token),
        )
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
        blocked_response = False
        invalid_response = False
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
                blocked_response = True
                logger.warning(
                    "TikTok empty response on browser='{}', headless={}. Trying fallback.",
                    browser,
                    headless,
                )
                continue
            except (InvalidResponseException, InvalidJSONException) as exc:
                last_exc = exc
                invalid_response = True
                logger.warning(
                    "TikTok invalid response on browser='{}', headless={}: {}",
                    browser,
                    headless,
                    exc,
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

        if last_exc is None:
            raise TikTokFetchError(
                "Endpoint komentar TikTok berhasil dibaca, tetapi daftar komentarnya kosong. "
                "Kemungkinan video belum memiliki komentar publik, komentar dinonaktifkan/dibatasi, "
                "atau komentar tidak tersedia untuk sesi TikTok saat ini."
            )

        raise TikTokFetchError(
            _build_fetch_error_message(
                last_exc,
                blocked_response=blocked_response,
                invalid_response=invalid_response,
            )
        ) from last_exc

    # Run isolated to_thread to avoid current loop policy limitations
    return await asyncio.to_thread(run_in_isolated_loop, _fetch_inner, url, max_count, ms_token)
