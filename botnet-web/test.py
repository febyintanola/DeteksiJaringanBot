import asyncio
import os
import platform

# Windows safe event loop policy for subprocess (Playwright needs this)
if platform.system() == "Windows":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

from app.pipeline.tiktok_client import fetch_comments

async def test():
    # Use the video_id from the working example
    video_id = 7248300636498890011
    url = f"https://www.tiktok.com/@tiktok/video/{video_id}"
    ms_token = os.getenv("ms_token")
    comments = await fetch_comments(url, 10, ms_token)
    print(f"Comments fetched: {len(comments)}")
    for c in comments[:3]:
        print(c)

asyncio.run(test())

