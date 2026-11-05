#!/usr/bin/env python3
"""
pubmed_summarizer.py

Cleans messy PubMed-style JSON and asks a locally hosted Llama 3.3 70B
(vLLM, OpenAI-compatible API) to summarize the clinical outcome of each
study in one sentence.

Usage
    python pubmed_summarizer.py --input abstracts.json --output summaries.json
    python pubmed_summarizer.py --dry-run     # clean + show payload, no network
    (with no --input, the embedded sample data is used)

Exit codes
    0   every usable record was summarized
    1   at least one record failed or was not attempted
    2   fatal: unreadable input, server unreachable, or model not served
    130 interrupted with Ctrl-C (partial results are still written)

Requires Python 3.8+ and `pip install requests`.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Tuple

import requests

LOG = logging.getLogger("pubmed_summarizer")

# --------------------------------------------------------------------------
# Configuration (override with env vars or CLI flags)
# --------------------------------------------------------------------------
DEFAULT_URL = os.getenv("LLM_URL", "http://192.168.1.100:8000/v1/chat/completions")
# Must match vLLM's --served-model-name
DEFAULT_MODEL = os.getenv("LLM_MODEL", "Llama-3.3-70B-Instruct")

CONNECT_TIMEOUT_S = 5.0      # fail fast if the box is off / wrong IP
# A 70B model on a DGX Spark decodes at only a few tokens/sec, so a single
# answer can legitimately take 30-60 s. The read timeout must be well above that.
READ_TIMEOUT_S = 120.0
SLOW_RESPONSE_WARN_S = 30.0  # log a warning (but still succeed) above this
MAX_ATTEMPTS = 3
BACKOFF_BASE_S = 2.0
BACKOFF_CAP_S = 30.0
MAX_CONSECUTIVE_FAILURES = 3  # circuit breaker: stop hammering a dead server

MAX_TOKENS = 120              # one sentence; also bounds worst-case latency
MAX_ABSTRACT_CHARS = 8_000    # ~2k tokens; keeps prompt inside --max-model-len 4096
MIN_ABSTRACT_CHARS = 20

RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}

SYSTEM_PROMPT = (
    "You are a careful clinical research assistant. Use only the information "
    "in the abstract you are given. Do not add facts, numbers, or conclusions "
    "that the text does not state."
)
USER_TEMPLATE = (
    "Summarize the clinical outcome of this study in one sentence.\n"
    "If the abstract does not report a clinical outcome, or its statements "
    "contradict each other, say so in that sentence instead of guessing.\n\n"
    "Title: {title}\n"
    "Abstract: {abstract}"
)

# --------------------------------------------------------------------------
# 1. Parsing and cleaning
# --------------------------------------------------------------------------
_NULLISH = {"", "null", "none", "n/a", "na", "nan", "undefined", "unknown", "-", "--"}
_WS_RE = re.compile(r"\s+")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Only strip tags PubMed/XML exports actually contain, so clinical text such as
# "p<0.05 and HR>1.2" is never mistaken for markup.
_TAG_RE = re.compile(
    r"</?(?:abstracttext|b|i|u|em|strong|sub|sup|p|br|span|mml:[a-z]+)\b[^<>]*>", re.I
)
_PMID_RE = re.compile(r"^(?:PMID\s*[:#_-]?\s*)?(\d{1,9})$", re.I)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
_FAILED = object()

# Field names after lower-casing and stripping non-alphanumerics, so
# "abstract_text", "AbstractText" and "Abstract" all resolve correctly.
ID_ALIASES = ("id", "pmid", "uid")
TITLE_ALIASES = ("articletitle", "title")
ABSTRACT_ALIASES = ("abstracttext", "abstract")
AUTHOR_ALIASES = ("authorlist", "authors")


@dataclass
class Record:
    record_id: str            # "PMID:349281" or "record_3" when the id is unusable
    pmid: Optional[str]
    title: Optional[str]
    abstract: str
    authors: List[str] = field(default_factory=list)
    abstract_truncated: bool = False


def _clean_text(value: Any) -> Optional[str]:
    """Return normalized text, or None for null-ish / non-text values."""
    if isinstance(value, (list, tuple)):  # abstracts split into sections
        value = " ".join(v for v in value if isinstance(v, str))
    if not isinstance(value, str):
        return None
    value = html.unescape(value)
    value = _TAG_RE.sub(" ", value)
    value = _CTRL_RE.sub("", value)
    value = _WS_RE.sub(" ", value).strip()
    return None if value.lower() in _NULLISH else value


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) == 0
    if isinstance(value, str):
        return _clean_text(value) is None
    return False


def _norm_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def _lookup(fields: dict, aliases: Tuple[str, ...]) -> Any:
    for alias in aliases:
        value = fields.get(alias)
        if not _is_missing(value):
            return value
    return None


def _normalize_pmid(raw: Any) -> Optional[str]:
    """'PMID:349281' / 349281 / '349281' -> '349281'; anything else -> None."""
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        text = str(raw)
    elif isinstance(raw, float) and raw.is_integer():
        text = str(int(raw))
    elif isinstance(raw, str):
        text = raw.strip()
    else:
        return None
    match = _PMID_RE.match(text)
    return match.group(1) if match else None


def _clean_authors(value: Any) -> List[str]:
    if value is None:
        return []
    items = value if isinstance(value, (list, tuple)) else [value]
    cleaned = (_clean_text(item) for item in items)
    return [a for a in cleaned if a]


def clean_record(raw: Any, index: int) -> Tuple[Optional[Record], Optional[str]]:
    """Return (record, None) if usable, else (None, reason it was skipped)."""
    if not isinstance(raw, dict):
        return None, f"entry is {type(raw).__name__}, not an object"

    fields = {_norm_key(k): v for k, v in raw.items()}

    abstract = _clean_text(_lookup(fields, ABSTRACT_ALIASES))
    if abstract is None:
        return None, "no usable abstract (missing, null or empty)"
    if len(abstract) < MIN_ABSTRACT_CHARS:
        return None, f"abstract too short to summarize ({len(abstract)} chars)"

    truncated = False
    if len(abstract) > MAX_ABSTRACT_CHARS:
        abstract = abstract[:MAX_ABSTRACT_CHARS].rsplit(" ", 1)[0] + " [...]"
        truncated = True

    pmid = _normalize_pmid(_lookup(fields, ID_ALIASES))
    if pmid is None:
        LOG.warning("Entry %d has no valid PMID; labelling it record_%d", index, index)
    return (
        Record(
            record_id=f"PMID:{pmid}" if pmid else f"record_{index}",
            pmid=pmid,
            title=_clean_text(_lookup(fields, TITLE_ALIASES)),
            abstract=abstract,
            authors=_clean_authors(_lookup(fields, AUTHOR_ALIASES)),
            abstract_truncated=truncated,
        ),
        None,
    )


def _try_parse(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return _FAILED


def load_raw_entries(raw_text: str) -> list:
    """Parse legacy JSON defensively; raises ValueError only if nothing is salvageable."""
    text = raw_text.lstrip("\ufeff").strip()
    if not text:
        raise ValueError("input is empty")

    data = _try_parse(text)
    if data is _FAILED:  # common legacy-export damage: trailing commas
        data = _try_parse(_TRAILING_COMMA_RE.sub(r"\1", text))
    if data is _FAILED:  # last resort: treat as JSON Lines, skip unparseable lines
        data, bad = [], 0
        for line in text.splitlines():
            line = line.strip().rstrip(",")
            if not line or line in {"[", "]"}:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
        if bad:
            LOG.warning("Input was not valid JSON; salvaged %d line(s), dropped %d", len(data), bad)
        if not data:
            raise ValueError("input is not valid JSON and nothing could be salvaged")

    if isinstance(data, dict):  # unwrap {"records": [...]}-style envelopes
        for key in ("records", "articles", "results", "data", "items"):
            if isinstance(data.get(key), list):
                return data[key]
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError(f"unexpected top-level JSON type: {type(data).__name__}")


def clean_all(entries: list) -> Tuple[List[Record], List[dict]]:
    records: List[Record] = []
    skipped: List[dict] = []
    seen = set()
    for index, raw in enumerate(entries, start=1):
        record, reason = clean_record(raw, index)
        if record and record.pmid:
            if record.pmid in seen:
                record, reason = None, "duplicate PMID"
            else:
                seen.add(record.pmid)
        if record:
            records.append(record)
        else:
            raw_id = raw.get("id") if isinstance(raw, dict) else None
            skipped.append({"index": index, "id": None if raw_id is None else str(raw_id), "reason": reason})
            LOG.warning("Skipping entry %d (%s): %s", index, raw_id, reason)
    return records, skipped


# --------------------------------------------------------------------------
# 2. Payload
# --------------------------------------------------------------------------
def build_payload(record: Record, model: str) -> dict:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    title=record.title or "(not provided)", abstract=record.abstract
                ),
            },
        ],
        "temperature": 0.0,   # deterministic; also means retrying can't "fix" a bad answer
        "max_tokens": MAX_TOKENS,
        "stream": False,
    }


# --------------------------------------------------------------------------
# 3. HTTP client with retries
# --------------------------------------------------------------------------
class LLMError(Exception):
    def __init__(self, message: str, retryable: bool = False, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retryable = retryable
        self.retry_after = retry_after
        self.attempts = 0


class LLMClient:
    def __init__(self, url: str, connect_timeout: float, read_timeout: float,
                 max_attempts: int, api_key: Optional[str] = None):
        self.url = url
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.max_attempts = max(1, max_attempts)
        self.session = requests.Session()
        self.session.headers["Content-Type"] = "application/json"
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

    def close(self) -> None:
        self.session.close()

    def preflight(self, model: str) -> None:
        """Fail fast, with a useful message, before we queue any real work."""
        models_url = self.url.split("/chat/completions")[0] + "/models"
        try:
            resp = self.session.get(models_url, timeout=(self.connect_timeout, 10))
        except requests.exceptions.RequestException as exc:
            raise LLMError(
                f"cannot reach {models_url} ({exc.__class__.__name__}). Is the vLLM "
                "container running? A 70B model can take several minutes to load."
            )
        if resp.status_code != 200:
            LOG.warning("Preflight: GET %s -> HTTP %s; continuing anyway", models_url, resp.status_code)
            return
        try:
            served = [m["id"] for m in resp.json()["data"]]
        except (ValueError, KeyError, TypeError):
            LOG.warning("Preflight: unexpected /models response; continuing anyway")
            return
        if model not in served:
            raise LLMError(
                f"model '{model}' is not served here (server offers {served}). "
                "Match --model to vLLM's --served-model-name."
            )
        LOG.info("Preflight OK: server is up and serving %s", model)

    def summarize(self, payload: dict) -> Tuple[str, Optional[str], int]:
        """POST with retry. Returns (text, finish_reason, attempts) or raises LLMError."""
        for attempt in range(1, self.max_attempts + 1):
            started = time.monotonic()
            try:
                resp = self.session.post(
                    self.url, json=payload, timeout=(self.connect_timeout, self.read_timeout)
                )
                elapsed = time.monotonic() - started
                text, finish_reason = self._parse(resp)
                if elapsed > SLOW_RESPONSE_WARN_S:
                    LOG.warning("Slow response: %.1fs (succeeded; read timeout is %.0fs)",
                                elapsed, self.read_timeout)
                return text, finish_reason, attempt
            except requests.exceptions.ConnectTimeout:
                err = LLMError(f"connect timeout after {self.connect_timeout:.0f}s", True)
            except requests.exceptions.ReadTimeout:
                err = LLMError(f"no response within {self.read_timeout:.0f}s", True)
            except requests.exceptions.ConnectionError as exc:
                err = LLMError(f"connection refused/dropped ({exc.__class__.__name__})", True)
            except requests.exceptions.ChunkedEncodingError:
                err = LLMError("connection dropped mid-response", True)
            except requests.exceptions.RequestException as exc:
                err = LLMError(f"request failed: {exc}", False)
            except LLMError as exc:
                err = exc

            err.attempts = attempt
            if not err.retryable or attempt == self.max_attempts:
                raise err
            delay = min(BACKOFF_CAP_S, BACKOFF_BASE_S * 2 ** (attempt - 1))
            delay = random.uniform(delay / 2, delay)  # jitter
            if err.retry_after is not None:
                delay = min(max(delay, err.retry_after), BACKOFF_CAP_S)
            LOG.warning("Attempt %d/%d failed (%s); retrying in %.1fs",
                        attempt, self.max_attempts, err, delay)
            time.sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover

    @staticmethod
    def _parse(resp: requests.Response) -> Tuple[str, Optional[str]]:
        snippet = resp.text[:200].replace("\n", " ")
        if resp.status_code in RETRYABLE_STATUS:
            retry_after = None
            header = resp.headers.get("Retry-After", "")
            if header.isdigit():
                retry_after = float(header)
            raise LLMError(f"HTTP {resp.status_code}: {snippet}", True, retry_after)
        if resp.status_code != 200:  # 400/401/404/422: retrying will not help
            raise LLMError(f"HTTP {resp.status_code}: {snippet}", False)
        try:
            body = resp.json()
        except ValueError:
            raise LLMError("HTTP 200 but body was not valid JSON (truncated?)", True)
        try:
            choice = body["choices"][0]
            text = choice["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise LLMError("unexpected response schema (no choices[0].message.content)", False)
        text = " ".join((text or "").split())
        if not text:
            raise LLMError("model returned an empty completion", False)
        return text, choice.get("finish_reason")


# --------------------------------------------------------------------------
# 4. Orchestration
# --------------------------------------------------------------------------
@dataclass
class Result:
    id: str
    title: Optional[str]
    status: str                      # ok | failed | not_attempted
    summary: Optional[str] = None
    error: Optional[str] = None
    attempts: int = 0
    latency_s: Optional[float] = None
    note: Optional[str] = None


def read_input(path: Optional[str]) -> str:
    if path is None:
        LOG.info("No --input given; using the embedded sample data")
        return SAMPLE_JSON
    if path == "-":
        return sys.stdin.read()
    return Path(path).read_text(encoding="utf-8", errors="replace")


def write_report(report: dict, path: Optional[str]) -> None:
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if not path:
        print(text)
        return
    tmp = f"{path}.tmp"
    Path(tmp).write_text(text + "\n", encoding="utf-8")
    os.replace(tmp, path)  # atomic: readers never see a half-written file
    LOG.info("Wrote %s", path)


def parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", "-i", help="JSON file ('-' for stdin). Default: embedded sample")
    p.add_argument("--output", "-o", help="Write the JSON report here. Default: stdout")
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--connect-timeout", type=float, default=CONNECT_TIMEOUT_S)
    p.add_argument("--read-timeout", type=float, default=READ_TIMEOUT_S)
    p.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)
    p.add_argument("--max-consecutive-failures", type=int, default=MAX_CONSECUTIVE_FAILURES)
    p.add_argument("--skip-preflight", action="store_true", help="Do not GET /v1/models first")
    p.add_argument("--dry-run", action="store_true", help="Clean data and show a payload; no network")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        stream=sys.stderr,
    )

    try:
        entries = load_raw_entries(read_input(args.input))
    except (OSError, ValueError) as exc:
        LOG.error("Cannot load input: %s", exc)
        return 2

    records, skipped = clean_all(entries)
    LOG.info("%d entries read: %d usable, %d skipped", len(entries), len(records), len(skipped))

    if args.dry_run:
        write_report(
            {
                "cleaned": [asdict(r) for r in records],
                "skipped": skipped,
                "example_payload": build_payload(records[0], args.model) if records else None,
            },
            args.output,
        )
        return 0

    stats_base = {"entries_read": len(entries), "usable": len(records), "skipped": len(skipped)}
    results: List[Result] = []
    interrupted = False
    client = LLMClient(
        args.url, args.connect_timeout, args.read_timeout, args.max_attempts,
        api_key=os.getenv("LLM_API_KEY"),
    )

    try:
        if records and not args.skip_preflight:
            try:
                client.preflight(args.model)
            except LLMError as exc:
                LOG.error("Preflight failed: %s", exc)
                return 2

        consecutive_failures = 0
        try:
            for pos, rec in enumerate(records, start=1):
                if consecutive_failures >= args.max_consecutive_failures:
                    remaining = records[pos - 1:]
                    LOG.error("%d consecutive failures - server looks unhealthy; not attempting the "
                              "remaining %d record(s).", consecutive_failures, len(remaining))
                    results.extend(
                        Result(r.record_id, r.title, "not_attempted",
                               error="aborted after repeated server failures")
                        for r in remaining
                    )
                    break

                started = time.monotonic()
                try:
                    text, finish_reason, attempts = client.summarize(build_payload(rec, args.model))
                except LLMError as exc:
                    consecutive_failures += 1
                    LOG.error("[%d/%d] %s FAILED after %d attempt(s): %s",
                              pos, len(records), rec.record_id, exc.attempts, exc)
                    results.append(Result(rec.record_id, rec.title, "failed", error=str(exc),
                                          attempts=exc.attempts,
                                          latency_s=round(time.monotonic() - started, 2)))
                    continue

                consecutive_failures = 0
                notes = []
                if finish_reason == "length":
                    notes.append("summary hit max_tokens and may be cut off")
                if rec.abstract_truncated:
                    notes.append(f"abstract truncated to {MAX_ABSTRACT_CHARS} chars")
                latency = round(time.monotonic() - started, 2)
                LOG.info("[%d/%d] %s ok in %.1fs (attempts=%d)",
                         pos, len(records), rec.record_id, latency, attempts)
                results.append(Result(rec.record_id, rec.title, "ok", summary=text,
                                      attempts=attempts, latency_s=latency,
                                      note="; ".join(notes) or None))
        except KeyboardInterrupt:
            interrupted = True
            LOG.warning("Interrupted - writing partial results")
    finally:
        client.close()

    stats = dict(stats_base)
    for status in ("ok", "failed", "not_attempted"):
        stats[status] = sum(1 for r in results if r.status == status)
    write_report(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "endpoint": args.url,
            "model": args.model,
            "stats": stats,
            "results": [asdict(r) for r in results],
            "skipped": skipped,
        },
        args.output,
    )
    if interrupted:
        return 130
    return 1 if (stats["failed"] or stats["not_attempted"]) else 0


# --------------------------------------------------------------------------
# Embedded sample (the messy legacy data from the brief)
# --------------------------------------------------------------------------
SAMPLE_JSON = """
[
  {
    "id": "PMID:349281",
    "ArticleTitle": "Efficacy of Drug X in severe cardiac failure.",
    "abstract_text": "We tested 500 patients. The drug worked in 40% of cases but caused nausea.",
    "authorList": ["Smith, J", "Doe, A"]
  },
  {
    "id": 8829102,
    "title": null,
    "Abstract": "Null hypothesis rejected. No significant improvement noted in cohort B vs placebo.",
    "authors": "Unknown"
  },
  {
    "id": "ERR_NOT_FOUND",
    "ArticleTitle": "Corrupted Entry",
    "abstract_text": null
  }
]
"""

if __name__ == "__main__":
    sys.exit(main())
