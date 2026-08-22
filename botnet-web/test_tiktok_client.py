import unittest
from unittest.mock import patch

from app.pipeline.tiktok_client import (
    _build_fetch_error_message,
    _extract_video_id_from_html,
    _is_tiktok_short_url,
    extract_video_id_from_url,
    resolve_tiktok_video_url,
)


class _FakeHeaders:
    def get_content_charset(self):
        return "utf-8"


class _FakeResponse:
    headers = _FakeHeaders()

    def __init__(self, resolved_url, body):
        self.resolved_url = resolved_url
        self.body = body.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def geturl(self):
        return self.resolved_url

    def read(self, _limit):
        return self.body


class TikTokUrlTests(unittest.TestCase):
    def test_extracts_video_id_from_common_tiktok_urls(self):
        cases = {
            "https://www.tiktok.com/@abc/video/7543347195295190302?is_from_webapp=1": "7543347195295190302",
            "https://www.tiktok.com/@abc/photo/7543347195295190302": "7543347195295190302",
            "https://t.tiktok.com/i18n/share/video/7543347195295190302/?share_item_id=111": "7543347195295190302",
            "https://www.tiktok.com/?share_item_id=7543347195295190302": "7543347195295190302",
        }

        for url, expected in cases.items():
            with self.subTest(url=url):
                self.assertEqual(extract_video_id_from_url(url), expected)

    def test_recognizes_tiktok_short_url_hosts_and_paths(self):
        self.assertTrue(_is_tiktok_short_url("https://vt.tiktok.com/ZSHoa9M6H/"))
        self.assertTrue(_is_tiktok_short_url("https://www.tiktok.com/t/ZSHoa9M6H/"))
        self.assertFalse(_is_tiktok_short_url("https://www.tiktok.com/@abc/video/7543347195295190302"))

    def test_extracts_video_id_from_response_html(self):
        self.assertEqual(
            _extract_video_id_from_html('{"itemId":"7543347195295190302"}'),
            "7543347195295190302",
        )
        self.assertEqual(
            _extract_video_id_from_html("https:\\/\\/www.tiktok.com\\/@abc\\/video\\/7543347195295190302"),
            "7543347195295190302",
        )

    def test_resolve_shortlink_uses_video_id_from_html_when_final_url_is_not_canonical(self):
        response = _FakeResponse(
            "https://www.tiktok.com/foryou?lang=id",
            '{"itemId":"7543347195295190302"}',
        )

        with patch("app.pipeline.tiktok_client.urllib.request.urlopen", return_value=response):
            resolved = resolve_tiktok_video_url("https://vt.tiktok.com/ZSHoa9M6H/")

        self.assertEqual(
            resolved,
            "https://www.tiktok.com/@tiktok/video/7543347195295190302",
        )

    def test_empty_tiktok_response_is_presented_as_antibot_block(self):
        message = _build_fetch_error_message(
            RuntimeError("empty response"),
            blocked_response=True,
            invalid_response=False,
        )

        self.assertIn("respons kosong", message)
        self.assertIn("anti-bot", message)
        self.assertIn("bukan berarti komentar videonya tidak ada", message)


if __name__ == "__main__":
    unittest.main()
