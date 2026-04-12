"""Utilities for parsing TikTok comments into structured graph-friendly data."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple

MENTION_RE = re.compile(r"@([A-Za-z0-9_\.]+)")
_NEW_ACCOUNT_KEYWORDS = ("new", "baru", "recent", "fresh")


def _extract_parent_id(comment: Dict[str, Any], raw: Dict[str, Any]) -> Optional[str]:
    """Best-effort extraction of the parent comment identifier."""
    candidate_keys = (
        "reply_comment_id",
        "reply_to_comment_id",
        "parent_comment_id",
        "reply_id",
        "reply_comment",
        "parent_id",
    )
    for key in candidate_keys:
        value = comment.get(key)
        if value:
            break
    else:
        value = None
    if not value and raw:
        for key in candidate_keys:
            value = raw.get(key)
            if value:
                break
    if isinstance(value, dict):
        value = value.get("cid") or value.get("id") or value.get("comment_id")
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if value in (None, "", 0, "0"):
        return None
    return str(value)


def _ensure_summary(summary: Dict[str, Any], username: str) -> Dict[str, Any]:
    summary.setdefault("username", username)
    summary.setdefault("comment_count", 0)
    summary.setdefault("reply_count", 0)
    summary.setdefault("mention_count", 0)
    summary.setdefault("total_likes", 0)
    summary.setdefault("first_timestamp", None)
    summary.setdefault("last_timestamp", None)
    summary.setdefault("timestamps", [])
    summary.setdefault("threads", set())
    summary.setdefault("_texts", [])
    summary.setdefault("new_account_signal", None)
    summary.setdefault("profile_metadata_available", False)
    return summary


def _flatten_profile_strings(value: Any) -> List[str]:
    if value in (None, "", [], {}, ()):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        flattened: List[str] = []
        for inner in value.values():
            flattened.extend(_flatten_profile_strings(inner))
        return flattened
    if isinstance(value, (list, tuple, set)):
        flattened = []
        for inner in value:
            flattened.extend(_flatten_profile_strings(inner))
        return flattened
    return [str(value)]


def _extract_new_account_signal(raw_user: Dict[str, Any]) -> Tuple[Optional[float], bool]:
    profile_strings: List[str] = []
    for field in ("account_labels", "user_tags", "type_label"):
        profile_strings.extend(_flatten_profile_strings(raw_user.get(field)))
    if not profile_strings:
        return None, False
    joined = " ".join(profile_strings).lower()
    is_new = any(keyword in joined for keyword in _NEW_ACCOUNT_KEYWORDS)
    return (1.0 if is_new else 0.0), True


def parse_comments(comments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Parse raw comment payloads into structured aggregates."""
    user_summary: Dict[str, Dict[str, Any]] = {}
    username_to_id: Dict[str, str] = {}
    comment_records: List[Dict[str, Any]] = []

    for comment in comments:
        user_id = comment.get("user_id")
        username = (comment.get("username") or "").strip()
        if not user_id:
            continue
        username_to_id[username.lower()] = user_id
        summary = _ensure_summary(user_summary.setdefault(user_id, {}), username)
        summary["comment_count"] += 1
        summary["total_likes"] += int(comment.get("likes") or 0)
        text = comment.get("text") or ""
        summary["_texts"].append(text)
        timestamp = comment.get("timestamp")
        if timestamp is not None:
            summary["timestamps"].append(timestamp)
            if summary["first_timestamp"] is None or timestamp < summary["first_timestamp"]:
                summary["first_timestamp"] = timestamp
            if summary["last_timestamp"] is None or timestamp > summary["last_timestamp"]:
                summary["last_timestamp"] = timestamp
        raw = comment.get("raw") or {}
        raw_user = raw.get("user") or {}
        new_account_signal, has_profile_metadata = _extract_new_account_signal(raw_user)
        if has_profile_metadata:
            summary["profile_metadata_available"] = True
        if new_account_signal is not None:
            current_signal = summary.get("new_account_signal")
            if current_signal is None or new_account_signal > float(current_signal):
                summary["new_account_signal"] = new_account_signal
        parent_id = _extract_parent_id(comment, raw)
        comment_id = comment.get("comment_id") or raw.get("cid")
        comment_id = str(comment_id) if comment_id else None
        mentions = [m.lower() for m in MENTION_RE.findall(text)]
        summary["mention_count"] += len(mentions)
        if parent_id:
            summary["reply_count"] += 1
        comment_records.append(
            {
                "comment_id": comment_id,
                "user_id": user_id,
                "parent_id": parent_id,
                "mentions": mentions,
            }
        )

    mention_weights: DefaultDict[Tuple[str, str], float] = defaultdict(float)
    reply_weights: DefaultDict[Tuple[str, str], float] = defaultdict(float)
    threads: DefaultDict[str, Set[str]] = defaultdict(set)
    comment_lookup: Dict[str, Dict[str, Any]] = {}
    for record in comment_records:
        if record["comment_id"]:
            comment_lookup[record["comment_id"]] = record

    def resolve_root(comment_id: Optional[str]) -> Optional[str]:
        if not comment_id:
            return None
        seen: Set[str] = set()
        current = comment_id
        while current:
            if current in seen:
                break
            seen.add(current)
            record = comment_lookup.get(current)
            if not record:
                break
            parent = record.get("parent_id")
            if not parent or parent == current:
                break
            current = parent
        return current

    for record in comment_records:
        src = record["user_id"]
        comment_id = record.get("comment_id")
        parent_id = record.get("parent_id")
        # Mentions
        for mention in record["mentions"]:
            dst = username_to_id.get(mention)
            if not dst or dst == src:
                continue
            a, b = sorted((src, dst))
            mention_weights[(a, b)] += 1.0
        # Replies
        if parent_id:
            parent_record = comment_lookup.get(parent_id)
            if parent_record:
                dst = parent_record.get("user_id")
                if dst and dst != src:
                    a, b = sorted((src, dst))
                    reply_weights[(a, b)] += 1.0
        # Thread membership
        root = resolve_root(parent_id or comment_id)
        if root:
            threads[root].add(src)
            if parent_id:
                threads[root].add(comment_lookup.get(parent_id, {}).get("user_id", ""))

    co_thread_weights: DefaultDict[Tuple[str, str], float] = defaultdict(float)
    for root_id, participants in threads.items():
        participants = {uid for uid in participants if uid}
        if len(participants) < 2:
            continue
        sorted_participants = sorted(participants)
        for i, a in enumerate(sorted_participants):
            for b in sorted_participants[i + 1 :]:
                key = (a, b)
                co_thread_weights[key] += 1.0
        for uid in participants:
            summary = user_summary.get(uid)
            if summary is not None:
                summary.setdefault("threads", set()).add(root_id)

    user_texts: Dict[str, str] = {}
    for uid, summary in user_summary.items():
        corpus = " ".join(summary.pop("_texts", []))
        user_texts[uid] = corpus
        threads_set = summary.get("threads", set())
        if isinstance(threads_set, set):
            normalized_threads = sorted(list(threads_set))
            summary["threads"] = normalized_threads
            summary["thread_count"] = len(normalized_threads)
        else:
            summary["thread_count"] = len(threads_set)
        timestamps = summary.get("timestamps", [])
        if isinstance(timestamps, list):
            timestamps.sort()
        if len(timestamps) >= 2:
            summary["activity_span_sec"] = float(max(0, timestamps[-1] - timestamps[0]))
        else:
            summary["activity_span_sec"] = 0.0
        summary["avg_likes"] = (summary["total_likes"] / summary["comment_count"]) if summary["comment_count"] else 0.0

    return {
        "user_summary": user_summary,
        "mention_edges": dict(mention_weights),
        "reply_edges": dict(reply_weights),
        "co_thread_edges": dict(co_thread_weights),
        "user_texts": user_texts,
    }
