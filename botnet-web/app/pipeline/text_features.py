"""Shared text normalization and TF-IDF helpers for content-based signals."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION_RE = re.compile(r"(?<!\w)@[\w.]+", re.UNICODE)
_HASHTAG_RE = re.compile(r"(?<!\w)#(\w+)", re.UNICODE)
_TOKEN_RE = re.compile(r"[a-z0-9_]+", re.IGNORECASE)
_EMOJI_RE = re.compile(
    "["
    "\U0001f1e6-\U0001f1ff"
    "\U0001f300-\U0001f5ff"
    "\U0001f600-\U0001f64f"
    "\U0001f680-\U0001f6ff"
    "\U0001f700-\U0001f77f"
    "\U0001f780-\U0001f7ff"
    "\U0001f800-\U0001f8ff"
    "\U0001f900-\U0001f9ff"
    "\U0001fa00-\U0001fa6f"
    "\U0001fa70-\U0001faff"
    "\u2600-\u26ff"
    "\u2700-\u27bf"
    "]+",
    flags=re.UNICODE,
)

_TEXT_KEYS = (
    "text",
    "comment",
    "content",
    "desc",
    "description",
    "reply_text",
    "reply_comment",
    "comment_text",
    "title",
)
_PROFILE_TEXT_KEYS = (
    "nickname",
    "signature",
    "bio",
    "bio_description",
)

_INDONESIAN_STOPWORDS = {
    "ada",
    "adalah",
    "agar",
    "akan",
    "aku",
    "anda",
    "apa",
    "apakah",
    "atas",
    "atau",
    "awal",
    "bagai",
    "bagaimana",
    "bagi",
    "bahwa",
    "banyak",
    "baru",
    "begini",
    "begitu",
    "belum",
    "benar",
    "berada",
    "berikut",
    "bersama",
    "bisa",
    "buat",
    "cara",
    "cukup",
    "dalam",
    "dan",
    "dapat",
    "dari",
    "daripada",
    "dekat",
    "demi",
    "dengan",
    "depan",
    "dia",
    "di",
    "diri",
    "dong",
    "guna",
    "hal",
    "hampir",
    "hanya",
    "harus",
    "hingga",
    "ia",
    "ini",
    "itu",
    "jadi",
    "jika",
    "juga",
    "justru",
    "kala",
    "kalian",
    "kami",
    "kamu",
    "kan",
    "karena",
    "kata",
    "ke",
    "kembali",
    "kemudian",
    "kepada",
    "ketika",
    "khusus",
    "kita",
    "kok",
    "lagi",
    "lah",
    "lalu",
    "lewat",
    "luar",
    "maka",
    "makin",
    "mana",
    "masih",
    "masing",
    "mau",
    "melalui",
    "memang",
    "mereka",
    "meski",
    "mungkin",
    "namun",
    "nih",
    "nya",
    "oleh",
    "pada",
    "paling",
    "para",
    "per",
    "pernah",
    "pula",
    "pun",
    "saat",
    "saja",
    "saling",
    "sama",
    "sambil",
    "sampai",
    "sana",
    "sangat",
    "satu",
    "saya",
    "sebab",
    "sebagai",
    "sebelum",
    "sebuah",
    "sedang",
    "sejak",
    "sekali",
    "sekitar",
    "selain",
    "selalu",
    "seluruh",
    "semua",
    "sementara",
    "sempat",
    "sendiri",
    "serta",
    "setelah",
    "setiap",
    "siapa",
    "sih",
    "tanpa",
    "tapi",
    "telah",
    "tentang",
    "tentu",
    "terus",
    "tetap",
    "tiap",
    "toh",
    "untuk",
    "usah",
    "via",
    "yaitu",
    "yakni",
    "yang",
}
_SLANG_LEMMAS = {
    "aq": "aku",
    "ak": "aku",
    "gw": "aku",
    "gue": "aku",
    "gua": "aku",
    "loe": "kamu",
    "lu": "kamu",
    "km": "kamu",
    "gk": "tidak",
    "ga": "tidak",
    "gak": "tidak",
    "ngga": "tidak",
    "nggak": "tidak",
    "tdk": "tidak",
    "yg": "yang",
    "dgn": "dengan",
    "dr": "dari",
    "utk": "untuk",
    "bgt": "banget",
    "banget": "sangat",
}
_STOPWORDS = set(ENGLISH_STOP_WORDS) | _INDONESIAN_STOPWORDS
_PRESERVED_TOKENS = {"url_token", "emoji_token", "mention_token"}
_NGRAM_CANDIDATES: Tuple[Tuple[int, int], ...] = ((1, 3), (1, 2), (1, 1))


def _flatten_strings(value: Any) -> List[str]:
    if value in (None, "", [], {}, ()):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        values: List[str] = []
        for inner in value.values():
            values.extend(_flatten_strings(inner))
        return values
    if isinstance(value, (list, tuple, set)):
        values = []
        for inner in value:
            values.extend(_flatten_strings(inner))
        return values
    return [str(value)]


def collect_comment_text_fragments(comment: Dict[str, Any], raw: Optional[Dict[str, Any]] = None) -> List[str]:
    """Collect comment/profile text fields that can describe a user's content style."""
    raw = raw or {}
    fragments: List[str] = []

    for source in (comment, raw):
        for key in _TEXT_KEYS:
            value = source.get(key) if isinstance(source, dict) else None
            fragments.extend(_flatten_strings(value))

    raw_user = raw.get("user") if isinstance(raw, dict) else None
    if isinstance(raw_user, dict):
        for key in _PROFILE_TEXT_KEYS:
            fragments.extend(_flatten_strings(raw_user.get(key)))

    seen = set()
    unique: List[str] = []
    for fragment in fragments:
        text = str(fragment).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def _strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _lemma_token(token: str) -> str:
    token = _SLANG_LEMMAS.get(token, token)
    if token in _PRESERVED_TOKENS or len(token) <= 3:
        return token

    for suffix in ("lah", "kah", "pun", "nya", "ku", "mu"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            token = token[: -len(suffix)]
            break

    if token.endswith("ies") and len(token) > 5:
        return token[:-3] + "y"
    for suffix in ("ing", "ed"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 4:
        token = token[:-1]

    for prefix in ("meng", "meny", "men", "mem", "ber", "ter", "peng", "peny", "pen", "pem", "per", "di", "ke", "se", "me", "pe"):
        if token.startswith(prefix) and len(token) - len(prefix) >= 4:
            token = token[len(prefix) :]
            break

    for suffix in ("kan", "an", "i"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            token = token[: -len(suffix)]
            break

    return token


def normalize_text(text: str) -> str:
    """Normalize noisy social text before vectorization."""
    text = _strip_accents(str(text or "").lower())
    text = _URL_RE.sub(" url_token ", text)
    text = _MENTION_RE.sub(" mention_token ", text)
    text = _HASHTAG_RE.sub(r" \1 ", text)
    text = _EMOJI_RE.sub(" emoji_token ", text)
    tokens = []
    for token in _TOKEN_RE.findall(text):
        lemma = _lemma_token(token.lower())
        if len(lemma) < 2 and lemma not in _PRESERVED_TOKENS:
            continue
        if lemma in _STOPWORDS and lemma not in _PRESERVED_TOKENS:
            continue
        tokens.append(lemma)
    return " ".join(tokens)


def _adaptive_min_df(n_docs: int) -> List[int]:
    if n_docs < 30:
        return [1]
    if n_docs < 80:
        return [1, 2]
    return [2, 1]


def fit_tuned_tfidf(
    user_texts: Dict[str, str],
    *,
    max_features: int = 20000,
    min_users: int = 1,
    ngram_candidates: Sequence[Tuple[int, int]] = _NGRAM_CANDIDATES,
    min_df: Optional[int] = None,
    max_df: float = 0.95,
) -> Tuple[List[str], TfidfVectorizer, Any]:
    """Fit TF-IDF with higher feature cap, adaptive min_df, and n-gram fallbacks."""
    user_ids: List[str] = []
    corpus: List[str] = []
    for uid, text in user_texts.items():
        cleaned = normalize_text(text)
        if not cleaned:
            continue
        user_ids.append(uid)
        corpus.append(cleaned)

    if len(user_ids) < min_users:
        raise ValueError("Jumlah teks pengguna belum cukup untuk komputasi TF-IDF.")

    min_df_values = [min_df] if min_df is not None else _adaptive_min_df(len(corpus))
    effective_max_df = 1.0 if len(corpus) < 30 else max_df
    best = None
    errors: List[str] = []

    for ngram_range in ngram_candidates:
        for min_df_value in min_df_values:
            try:
                vectorizer = TfidfVectorizer(
                    max_features=max_features,
                    strip_accents="unicode",
                    lowercase=False,
                    ngram_range=ngram_range,
                    min_df=min_df_value,
                    max_df=effective_max_df,
                    norm="l2",
                    token_pattern=r"(?u)\b[a-z0-9_]{2,}\b",
                )
                matrix = vectorizer.fit_transform(corpus)
            except ValueError as exc:
                errors.append(str(exc))
                continue

            score = (matrix.nnz, matrix.shape[1], ngram_range[1], -int(min_df_value))
            if best is None or score > best[0]:
                best = (score, vectorizer, matrix)

    if best is None:
        raise ValueError("Tidak ada kosakata TF-IDF yang bisa dipakai setelah preprocessing.")

    return user_ids, best[1], best[2]


def normalize_user_texts(user_texts: Dict[str, str]) -> Dict[str, str]:
    return {uid: normalized for uid, text in user_texts.items() if (normalized := normalize_text(text))}
