"""Build the nanoChat pretraining corpus.

Single entry point. Streams text from pinned upstream sources, cleans it per a
fixed rule list, writes one-document-per-line JSONL partitioned by domain, and
emits a manifest plus a smoke subset.

See the accompanying spec for the contract. Nothing here tokenizes, shards, or
trains -- it only lands text on disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

SEED = 20260818
MAX_FILE_BYTES = 480 * 1024 * 1024
VAL_FRACTION = 0.005
VAL_MIN, VAL_MAX = 200, 20000
MIN_DOC_BYTES = 200
MIN_DOC_BYTES_TERMINAL = 50
CHECKPOINT_EVERY_BYTES = 256 * 1024 * 1024
CHECKPOINT_EVERY_SECONDS = 120.0
MAX_SHARD_RETRIES = 6


class SourceUnavailable(Exception):
    """Raised when an upstream source cannot be used as specified.

    The domain is marked ``blocked`` and reported; we never silently swap in a
    different dataset.
    """

    def __init__(self, reason: str, action: str):
        super().__init__(reason)
        self.reason = reason
        self.action = action


@dataclass
class HFSource:
    repo: str
    revision: str
    text_field: str
    license: str
    config: Optional[str] = None
    data_dir: Optional[str] = None
    kind: str = "huggingface"

    def manifest(self) -> dict:
        d = {
            "kind": self.kind,
            "repo": self.repo,
            "revision": self.revision,
            "text_field": self.text_field,
            "license": self.license,
        }
        if self.config:
            d["config"] = self.config
        if self.data_dir:
            d["data_dir"] = self.data_dir
        return d


@dataclass
class HTTPJsonlSource:
    url: str
    text_field: str
    license: str
    repo: str
    revision: str
    kind: str = "http_jsonl"

    def manifest(self) -> dict:
        return {
            "kind": self.kind,
            "repo": self.repo,
            "revision": self.revision,
            "url": self.url,
            "text_field": self.text_field,
            "license": self.license,
        }


@dataclass
class LocalSource:
    description: str
    license: str
    kind: str = "local"

    def manifest(self) -> dict:
        return {"kind": self.kind, "description": self.description, "license": self.license}


@dataclass
class Domain:
    name: str
    share: float
    bytes_per_token: float
    slots: list
    min_doc_bytes: int = MIN_DOC_BYTES

    @property
    def all_sources(self) -> list:
        return [c for slot in self.slots for c in slot]


REV_FINEWEB_EDU = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
REV_DCLM = "a3b142c183aebe5af344955ae20836eb34dcf69b"
REV_SMOLLM = "3ba9d605774198c5868892d7a8deda78031a781f"
REV_STACK_EDU = "eeec5caac5cc3758a18f1d3ba4416837a9ba814c"
REV_STARCODER = "9fc30b578cedaec69e47302df72cf00feed7c8c4"
REV_REDPAJAMA = "398f92572e94f4793e41c22ab7ea2a788d9e7de4"

DOMAINS: list[Domain] = [
    Domain("web_edu", 0.20, 4.0, [
        [HFSource("HuggingFaceFW/fineweb-edu", REV_FINEWEB_EDU, "text", "ODC-By-1.0",
                  config="sample-10BT")],
    ]),
    Domain("web_dclm", 0.15, 4.0, [
        [HFSource("mlfoundations/dclm-baseline-1.0", REV_DCLM, "text", "CC-BY-4.0")],
    ]),
    Domain("cosmopedia", 0.12, 4.0, [
        [HFSource("HuggingFaceTB/smollm-corpus", REV_SMOLLM, "text", "ODC-By-1.0",
                  config="cosmopedia-v2")],
    ]),
    Domain("code_python", 0.20, 3.8, [
        [HFSource("HuggingFaceTB/stack-edu", REV_STACK_EDU, "text", "ODC-By-1.0",
                  config="Python"),
         HFSource("bigcode/starcoderdata", REV_STARCODER, "content",
                  "other (BigCode OpenRAIL-M)", data_dir="python")],
    ]),
    Domain("code_shell", 0.05, 3.8, [
        [HFSource("bigcode/starcoderdata", REV_STARCODER, "content",
                  "other (BigCode OpenRAIL-M)", data_dir="shell")],
    ]),
    Domain("code_issues", 0.10, 3.8, [
        [HFSource("bigcode/starcoderdata", REV_STARCODER, "content",
                  "other (BigCode OpenRAIL-M)", data_dir="github-issues-filtered-structured")],
        [HFSource("bigcode/starcoderdata", REV_STARCODER, "content",
                  "other (BigCode OpenRAIL-M)", data_dir="git-commits-cleaned")],
    ]),
    Domain("qa_stackexchange", 0.15, 4.0, [
        [HTTPJsonlSource(
            "https://data.together.xyz/redpajama-data-1T/v1.0.0/stackexchange/stackexchange.jsonl",
            "text", "varied (StackExchange CC-BY-SA)",
            repo="togethercomputer/RedPajama-Data-1T", revision=REV_REDPAJAMA)],
    ]),
    Domain("terminal_docs", 0.03, 4.0, [
        [LocalSource("system man pages (man1/man5/man8) + tldr-pages",
                     "man: varied; tldr: CC-BY-4.0")],
    ], min_doc_bytes=MIN_DOC_BYTES_TERMINAL),
]

DOMAINS_BY_NAME = {d.name: d for d in DOMAINS}


DROP_REASONS = ("utf8", "too_short", "duplicate", "control_char")


def clean_document(raw: Any, min_bytes: int) -> tuple[Optional[str], Optional[str]]:
    """Return (text, None) for a keeper, or (None, reason) for a drop."""
    if isinstance(raw, bytes):
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None, "utf8"
    elif isinstance(raw, str):
        text = raw
        try:
            text.encode("utf-8")
        except UnicodeEncodeError:
            return None, "utf8"
    else:
        return None, "utf8"

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.strip()

    if "\x00" in text:
        return None, "control_char"
    if len(text.encode("utf-8")) < min_bytes:
        return None, "too_short"
    return text, None


def _atomic_write_text(path: Path, data: str) -> None:
    tmp = path.with_suffix(path.suffix + ".swap")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


class RotatingWriter:
    """Append-only JSONL writer with `.tmp` staging and size-based rotation.

    A file being written lives as ``<name>.tmp``; it is ``os.replace``d into
    place only once complete, so a kill never leaves a half file in the corpus.
    """

    def __init__(self, directory: Path, prefix: str, compress: bool,
                 start_index: int = 0, start_bytes: int = 0,
                 finished: Optional[list[str]] = None):
        self.dir = directory
        self.prefix = prefix
        self.compress = compress
        self.index = start_index
        self.cur_bytes = start_bytes
        self.finished: list[str] = list(finished or [])
        self._fh = None
        self._raw = None

    @property
    def ext(self) -> str:
        return ".jsonl.zst" if self.compress else ".jsonl"

    def _name(self, index: int) -> str:
        return f"{self.prefix}_{index:03d}{self.ext}"

    def _tmp_path(self, index: int) -> Path:
        return self.dir / (self._name(index) + ".tmp")

    def final_path(self, index: int) -> Path:
        return self.dir / self._name(index)

    def _open(self) -> None:
        if self._fh is not None:
            return
        path = self._tmp_path(self.index)
        if self.compress:
            import zstandard
            if self.cur_bytes and path.exists():
                raise RuntimeError("cannot resume mid-file with --compress zstd")
            self._raw = open(path, "wb")
            self._fh = zstandard.ZstdCompressor(level=6).stream_writer(self._raw)
        else:
            if path.exists():
                with open(path, "r+b") as fh:
                    fh.truncate(self.cur_bytes)
            else:
                self.cur_bytes = 0
            self._fh = open(path, "ab")

    def write(self, line: bytes) -> None:
        if self._fh is None:
            self._open()
        if self.cur_bytes and self.cur_bytes + len(line) > MAX_FILE_BYTES:
            self.rotate()
            self._open()
        self._fh.write(line)
        self.cur_bytes += len(line)

    def flush(self) -> None:
        if self._fh is None:
            return
        self._fh.flush()
        target = self._raw if self.compress else self._fh
        if not self.compress:
            os.fsync(target.fileno())

    def _close(self) -> None:
        if self._fh is None:
            return
        self._fh.flush()
        if self.compress:
            self._fh.close()
            self._raw.flush()
            os.fsync(self._raw.fileno())
            self._raw.close()
            self._raw = None
        else:
            os.fsync(self._fh.fileno())
            self._fh.close()
        self._fh = None

    def rotate(self) -> None:
        """Finalize the current file and advance to the next index."""
        self._close()
        tmp = self._tmp_path(self.index)
        if tmp.exists() and self.cur_bytes > 0:
            os.replace(tmp, self.final_path(self.index))
            self.finished.append(self._name(self.index))
        elif tmp.exists():
            tmp.unlink()
        self.index += 1
        self.cur_bytes = 0

    def close_final(self) -> list[str]:
        if self.cur_bytes > 0 or self._fh is not None:
            self.rotate()
        return list(self.finished)

    def state(self) -> dict:
        return {"index": self.index, "bytes": self.cur_bytes, "finished": list(self.finished)}


def json_line(text: str, source: str, doc_id: str) -> bytes:
    obj = {"text": text, "source": source, "id": doc_id}
    return (json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def _hf_stream(src: HFSource, cursor: dict, log: Callable[[str], None]) -> Iterator[tuple[Any, dict]]:
    """Yield (raw_text, cursor) from a HuggingFace dataset in streaming mode.

    The cursor is (shard, row) so a resume skips whole completed shards instead
    of re-downloading them.
    """
    from datasets import load_dataset

    kwargs: dict[str, Any] = {
        "path": src.repo,
        "split": "train",
        "streaming": True,
        "revision": src.revision,
    }
    if src.config:
        kwargs["name"] = src.config
    if src.data_dir:
        kwargs["data_dir"] = src.data_dir

    try:
        ds = load_dataset(**kwargs)
    except Exception as exc:
        msg = str(exc)
        if "gated" in msg.lower() or "authenticated" in msg.lower() or "401" in msg or "403" in msg:
            raise SourceUnavailable(
                f"{src.repo} is gated: {msg.splitlines()[0][:300]}",
                f"Accept the terms at https://huggingface.co/datasets/{src.repo} with your "
                f"HuggingFace account, then authenticate (`hf auth login`, or export "
                f"HF_TOKEN=<token>) and re-run with --only <domain>.",
            ) from exc
        if "no longer supported" in msg or "Dataset scripts" in msg:
            raise SourceUnavailable(
                f"{src.repo} still ships a loading script, which datasets>=4 refuses to run: "
                f"{msg.splitlines()[0][:300]}",
                "Pin datasets<4, or point this domain at a parquet-native mirror of the same "
                "dataset after confirming the substitution.",
            ) from exc
        if "404" in msg or "doesn't exist" in msg or "not found" in msg.lower():
            raise SourceUnavailable(
                f"{src.repo} could not be resolved: {msg.splitlines()[0][:300]}",
                f"Check https://huggingface.co/datasets/{src.repo} -- the dataset may have been "
                f"renamed or withdrawn.",
            ) from exc
        raise

    probe = next(iter(ds.take(1)), None)
    if probe is None:
        raise SourceUnavailable(f"{src.repo} yielded no records", "Inspect the dataset on the Hub.")
    if src.text_field not in probe:
        raise SourceUnavailable(
            f"{src.repo}"
            + (f" (config {src.config})" if src.config else "")
            + (f" (data_dir {src.data_dir})" if src.data_dir else "")
            + f" has no {src.text_field!r} column; available columns: {sorted(probe)}",
            "The upstream schema does not match the spec. Pick a source that ships document text "
            "directly, or add a resolver for the referenced blobs, then re-run --only <domain>.",
        )
    if not isinstance(probe[src.text_field], str):
        raise SourceUnavailable(
            f"{src.repo} column {src.text_field!r} is {type(probe[src.text_field]).__name__}, not str",
            "The upstream schema does not match the spec.",
        )

    n_shards = ds.num_shards
    start_shard = int(cursor.get("shard", 0))
    start_row = int(cursor.get("row", 0))
    log(f"    {src.repo}{'/' + src.config if src.config else ''}"
        f"{'/' + src.data_dir if src.data_dir else ''}: {n_shards} shards, "
        f"resuming at shard {start_shard} row {start_row}")

    for shard_idx in range(start_shard, n_shards):
        skip_rows = start_row if shard_idx == start_shard else 0
        attempt = 0
        while True:
            row_idx = 0
            try:
                shard = ds.shard(num_shards=n_shards, index=shard_idx) if n_shards > 1 else ds
                for rec in shard:
                    if row_idx < skip_rows:
                        row_idx += 1
                        continue
                    row_idx += 1
                    yield rec.get(src.text_field), {"shard": shard_idx, "row": row_idx}
                break
            except (KeyboardInterrupt, SourceUnavailable):
                raise
            except Exception as exc:
                attempt += 1
                if attempt > MAX_SHARD_RETRIES:
                    raise
                skip_rows = max(skip_rows, row_idx)
                wait = min(60.0, 2.0 ** attempt)
                log(f"    shard {shard_idx}: {type(exc).__name__}: {str(exc)[:160]} "
                    f"-- retry {attempt}/{MAX_SHARD_RETRIES} from row {skip_rows} in {wait:.0f}s")
                time.sleep(wait)


def _http_jsonl_stream(src: HTTPJsonlSource, cursor: dict,
                       log: Callable[[str], None]) -> Iterator[tuple[Any, dict]]:
    """Yield (raw_text, cursor) from a remote newline-delimited JSON file.

    Resume is a byte offset replayed through a Range request, so an interrupted
    run never refetches what it already consumed.
    """
    import requests

    offset = int(cursor.get("offset", 0))
    log(f"    {src.url}: resuming at byte offset {offset}")
    attempt = 0
    while True:
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            resp = requests.get(src.url, headers=headers, stream=True, timeout=(30, 120))
            if resp.status_code == 416:
                return
            if offset and resp.status_code != 206:
                raise SourceUnavailable(
                    f"{src.url} ignored a Range request (status {resp.status_code}); "
                    f"resume would silently duplicate data",
                    "Delete the domain directory and re-run from scratch, or use a mirror that "
                    "supports byte ranges.")
            if resp.status_code >= 400:
                raise SourceUnavailable(
                    f"{src.url} returned HTTP {resp.status_code}",
                    "Check whether the file has moved; do not substitute another dataset without "
                    "confirming.")
            attempt = 0
            pending = b""
            for chunk in resp.iter_content(chunk_size=8 << 20):
                if not chunk:
                    continue
                pending += chunk
                while True:
                    nl = pending.find(b"\n")
                    if nl < 0:
                        break
                    line, pending = pending[:nl], pending[nl + 1:]
                    offset += nl + 1
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    yield rec.get(src.text_field), {"offset": offset}
            if pending.strip():
                try:
                    rec = json.loads(pending)
                    offset += len(pending)
                    yield rec.get(src.text_field), {"offset": offset}
                except json.JSONDecodeError:
                    pass
            return
        except (KeyboardInterrupt, SourceUnavailable):
            raise
        except Exception as exc:
            attempt += 1
            if attempt > MAX_SHARD_RETRIES:
                raise
            wait = min(60.0, 2.0 ** attempt)
            log(f"    http: {type(exc).__name__}: {str(exc)[:160]} -- retry "
                f"{attempt}/{MAX_SHARD_RETRIES} from offset {offset} in {wait:.0f}s")
            time.sleep(wait)


_OVERSTRIKE = re.compile(r".\x08")


def _render_man_page(path: Path) -> Optional[str]:
    """Render one man page to plain text, dropping backspace-overstrike bolding."""
    section = path.parent.name[-1]
    name = path.name
    for suffix in (".gz", ".bz2", ".xz", ".Z"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    stem = name.rsplit(".", 1)[0]
    env = dict(os.environ, MANWIDTH="80", MANPAGER="cat", PAGER="cat", LC_ALL="en_US.UTF-8")
    try:
        out = subprocess.run(["man", "-P", "cat", section, stem], capture_output=True,
                             timeout=30, env=env).stdout
    except Exception:
        return None
    if not out:
        return None
    return _OVERSTRIKE.sub("", out.decode("utf-8", errors="strict")) if _is_utf8(out) else None


def _is_utf8(b: bytes) -> bool:
    try:
        b.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def _local_terminal_docs_stream(src: LocalSource, cursor: dict,
                                log: Callable[[str], None], work_dir: Path,
                                notes: list[str]) -> Iterator[tuple[Any, dict]]:
    """Yield man pages then tldr pages. Cursor is an index into a stable list."""
    if shutil.which("man") is None:
        raise SourceUnavailable(
            "`man` is not installed",
            "Install man-db/manpages (Debian: apt-get install -y man-db manpages manpages-dev "
            "coreutils) and re-run --only terminal_docs.")

    if shutil.which("apt-get") is None:
        notes.append(
            f"no apt-get on this host ({sys.platform}); the spec's "
            f"`apt-get install man-db manpages manpages-dev coreutils` step was skipped and "
            f"only the man pages already installed were rendered")
    if shutil.which("apt-get") and os.environ.get("CORPUS_APT_INSTALL") == "1":
        log("    apt-get: installing man-db manpages manpages-dev coreutils")
        subprocess.run(
            ["apt-get", "install", "-y", "man-db", "manpages", "manpages-dev", "coreutils"],
            env=dict(os.environ, DEBIAN_FRONTEND="noninteractive"),
            capture_output=True, timeout=1800)

    man_files: list[Path] = []
    for sec in ("man1", "man5", "man8"):
        d = Path("/usr/share/man") / sec
        if d.is_dir():
            man_files.extend(sorted(p for p in d.iterdir() if p.is_file()))
    log(f"    man pages found: {len(man_files)}")
    notes.append(f"{len(man_files)} man pages in /usr/share/man/man{{1,5,8}}")

    tldr_dir = work_dir / "tldr"
    tldr_files: list[Path] = []
    if shutil.which("git"):
        try:
            if not tldr_dir.exists():
                log("    cloning tldr-pages")
                subprocess.run(
                    ["git", "clone", "--depth", "1", "https://github.com/tldr-pages/tldr",
                     str(tldr_dir)],
                    capture_output=True, timeout=900, check=True)
            pages = tldr_dir / "pages"
            if pages.is_dir():
                tldr_files = sorted(pages.rglob("*.md"))
        except Exception as exc:
            log(f"    tldr clone failed: {type(exc).__name__}: {str(exc)[:160]}")
    log(f"    tldr pages found: {len(tldr_files)}")
    notes.append(f"{len(tldr_files)} tldr pages")

    items = [("man", p) for p in man_files] + [("tldr", p) for p in tldr_files]
    start = int(cursor.get("index", 0))
    for i in range(start, len(items)):
        kind, path = items[i]
        pos = {"index": i + 1}
        if kind == "man":
            text = _render_man_page(path)
            if text is None:
                yield None, pos
                continue
            yield text, pos
        else:
            try:
                yield path.read_bytes(), pos
            except Exception:
                yield None, pos


def open_source(src, cursor: dict, log, work_dir: Path,
                notes: list[str]) -> Iterator[tuple[Any, dict]]:
    if isinstance(src, HFSource):
        return _hf_stream(src, cursor, log)
    if isinstance(src, HTTPJsonlSource):
        return _http_jsonl_stream(src, cursor, log)
    if isinstance(src, LocalSource):
        return _local_terminal_docs_stream(src, cursor, log, work_dir, notes)
    raise TypeError(f"unknown source {src!r}")


class DomainBuilder:
    def __init__(self, domain: Domain, out_root: Path, target_bytes: int,
                 compress: bool, work_dir: Path, log: Callable[[str], None]):
        self.domain = domain
        self.dir = out_root / domain.name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.target_bytes = target_bytes
        self.compress = compress
        self.work_dir = work_dir
        self.log = log
        self.progress_path = self.dir / ".progress.json"
        self.reservoir_path = self.dir / ".val_reservoir.jsonl"
        self.blocked: list[dict] = []
        self.notes: list[str] = []

        st = self._load_progress()
        if st.get("pre_finalize"):
            st = self._rollback_finalize(st)
        self.next_id: int = st["next_id"]
        self.kept_docs: int = st["kept_docs"]
        self.kept_bytes: int = st["kept_bytes"]
        self.dropped: dict[str, int] = st["dropped"]
        self.source_index: int = st["source_index"]
        self.cursor: dict = st["cursor"]
        self.seen_for_reservoir: int = st["seen_for_reservoir"]
        self.rng = random.Random(SEED + zlib.crc32(domain.name.encode()))
        if st["rng_state"] is not None:
            self.rng.setstate(_tuple_state(st["rng_state"]))
        self.writer = RotatingWriter(self.dir, "train", compress,
                                     st["writer"]["index"], st["writer"]["bytes"],
                                     st["writer"]["finished"])
        self.reservoir: list[bytes] = self._load_reservoir(st["reservoir_len"])
        self.hashes: set[bytes] = set()
        self._rebuild_hashes()
        self._last_ckpt_time = time.time()
        self._bytes_since_ckpt = 0
        self._reservoir_dirty = True

    def _rollback_finalize(self, st: dict) -> dict:
        """Undo a completed finalize so the domain can keep growing.

        Finalize drains the val reservoir into train and writes val files. If a
        later run has more to fetch -- a blocked source that got unblocked, a
        raised target -- those two steps must be taken back first, otherwise the
        documents that went to val would be dropped on the floor by the next
        finalize.
        """
        pre = st["pre_finalize"]
        ext = ".jsonl.zst" if self.compress else ".jsonl"
        idx = int(pre["writer"]["index"])
        nbytes = int(pre["writer"]["bytes"])
        cur_name = f"train_{idx:03d}{ext}"
        protect = set(pre["writer"]["finished"]) | {cur_name}

        for p in sorted(self.dir.glob("val_*")):
            p.unlink()
        for p in sorted(self.dir.glob("train_*")):
            if p.name not in protect and not p.name.endswith(".tmp"):
                p.unlink()

        final = self.dir / cur_name
        tmp = self.dir / (cur_name + ".tmp")
        if final.exists():
            os.replace(final, tmp)
        if nbytes == 0:
            if tmp.exists():
                tmp.unlink()
        elif tmp.exists() and not self.compress:
            with open(tmp, "r+b") as fh:
                fh.truncate(nbytes)

        self.log(f"    rolled back a previous finalize of {self.domain.name} to "
                 f"{pre['kept_bytes'] / 1e9:.2f} GB / {pre['kept_docs']} docs")
        merged = dict(st)
        merged.update({k: v for k, v in pre.items() if k != "writer"})
        merged["writer"] = pre["writer"]
        merged.pop("pre_finalize", None)
        merged.pop("finalized", None)
        return merged


    def _blank(self) -> dict:
        return {
            "next_id": 0, "kept_docs": 0, "kept_bytes": 0,
            "dropped": {r: 0 for r in DROP_REASONS},
            "source_index": 0, "cursor": {}, "seen_for_reservoir": 0,
            "rng_state": None, "reservoir_len": 0,
            "writer": {"index": 0, "bytes": 0, "finished": []},
        }

    def _load_progress(self) -> dict:
        blank = self._blank()
        if not self.progress_path.exists():
            return blank
        try:
            st = json.loads(self.progress_path.read_text())
        except Exception:
            self.log(f"    progress file unreadable, starting {self.domain.name} from scratch")
            return blank
        for k, v in blank.items():
            st.setdefault(k, v)
        st["dropped"] = {r: int(st["dropped"].get(r, 0)) for r in DROP_REASONS}
        return st

    def _load_reservoir(self, expected_len: int) -> list[bytes]:
        if not self.reservoir_path.exists():
            return []
        lines = self.reservoir_path.read_bytes().splitlines(keepends=True)
        lines = [l for l in lines if l.strip()]
        if expected_len and len(lines) != expected_len:
            self.log(f"    reservoir length {len(lines)} != checkpointed {expected_len}; "
                     f"truncating to checkpoint")
            lines = lines[:expected_len]
        return lines

    def _rebuild_hashes(self) -> None:
        """Rebuild the dedup set from what is actually on disk.

        Doing this from the files rather than trusting a serialized set means a
        resume agrees with reality even if the last checkpoint lagged the data.
        """
        n = 0
        for line in self._iter_written_lines():
            try:
                obj = json.loads(line)
            except Exception:
                continue
            self.hashes.add(hashlib.sha256(obj["text"].encode("utf-8")).digest())
            n += 1
        for line in self.reservoir:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            self.hashes.add(hashlib.sha256(obj["text"].encode("utf-8")).digest())
            n += 1
        if n:
            self.log(f"    resumed {self.domain.name}: {n} documents already on disk, "
                     f"{self.kept_bytes / 1e9:.2f} GB kept")

    def _iter_written_lines(self) -> Iterator[bytes]:
        for name in self.writer.finished:
            p = self.dir / name
            if p.exists():
                yield from _read_jsonl(p, self.compress)
        tmp = self.dir / (f"train_{self.writer.index:03d}"
                          + (".jsonl.zst" if self.compress else ".jsonl") + ".tmp")
        if tmp.exists() and not self.compress:
            with open(tmp, "rb") as fh:
                data = fh.read(self.writer.cur_bytes)
            for line in data.splitlines():
                if line.strip():
                    yield line

    def checkpoint(self, force: bool = False) -> None:
        now = time.time()
        if not force and self._bytes_since_ckpt < CHECKPOINT_EVERY_BYTES \
                and now - self._last_ckpt_time < CHECKPOINT_EVERY_SECONDS:
            return
        if self.compress:
            self.writer.rotate()
        else:
            self.writer.flush()
        if self._reservoir_dirty:
            _atomic_write_bytes(self.reservoir_path, b"".join(self.reservoir))
            self._reservoir_dirty = False
        st = {
            "next_id": self.next_id, "kept_docs": self.kept_docs, "kept_bytes": self.kept_bytes,
            "dropped": self.dropped, "source_index": self.source_index, "cursor": self.cursor,
            "seen_for_reservoir": self.seen_for_reservoir,
            "rng_state": _list_state(self.rng.getstate()),
            "reservoir_len": len(self.reservoir),
            "writer": self.writer.state(),
        }
        _atomic_write_text(self.progress_path, json.dumps(st, indent=1))
        self._last_ckpt_time = now
        self._bytes_since_ckpt = 0


    def run(self) -> dict:
        """Stream until the byte target is met, then split out val and finalize."""
        slots = self.domain.slots
        per_slot = self.target_bytes / max(1, len(slots))
        t0 = time.time()
        status = "complete"

        while self.source_index < len(slots):
            slot = slots[self.source_index]
            slot_target = min(self.target_bytes, int(per_slot * (self.source_index + 1)))
            if self.kept_bytes >= slot_target:
                self.source_index += 1
                self.cursor = {}
                continue

            failures = []
            done = False
            for rank, src in enumerate(slot):
                try:
                    self._pull(src, slot_target, t0)
                    done = True
                    break
                except SourceUnavailable as exc:
                    label = f"{src.repo}" if hasattr(src, "repo") else "local"
                    self.log(f"  unavailable ({self.domain.name} candidate {rank + 1}"
                             f"/{len(slot)}, {label}): {exc.reason}")
                    failures.append({"source": src.manifest(), "reason": exc.reason,
                                     "action": exc.action})
                    self.cursor = {}
                    continue
                except KeyboardInterrupt:
                    self.checkpoint(force=True)
                    raise

            if not done:
                self.blocked.extend(failures)
                status = "blocked"
                break
            if self.kept_bytes < slot_target:
                label = getattr(src, "repo", None) or "local source"
                self.notes.append(
                    f"{label} exhausted at {self.kept_bytes / 1e9:.2f} GB of the "
                    f"{slot_target / 1e9:.2f} GB asked of it")
                if status == "complete":
                    status = "partial"
            self.source_index += 1
            self.cursor = {}
            self.checkpoint(force=True)

        if status != "blocked" and self.kept_bytes < self.target_bytes * 0.95:
            status = "partial"
        if status == "blocked" and self.kept_bytes > 0:
            status = "partial"

        self.checkpoint(force=True)
        return self._finalize(status)

    def _pull(self, src, src_target: int, t0: float) -> None:
        min_bytes = self.domain.min_doc_bytes
        name = self.domain.name
        last_report = time.time()
        stream = open_source(src, self.cursor, self.log, self.work_dir, self.notes)
        for raw, cursor in stream:
            self.cursor = cursor
            if raw is None:
                continue
            text, reason = clean_document(raw, min_bytes)
            if reason:
                self.dropped[reason] += 1
                continue
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            if digest in self.hashes:
                self.dropped["duplicate"] += 1
                continue
            self.hashes.add(digest)

            doc_id = f"{name}/{self.next_id:09d}"
            self.next_id += 1
            line = json_line(text, name, doc_id)
            self.kept_docs += 1
            self.kept_bytes += len(line)
            self._bytes_since_ckpt += len(line)
            self._offer(line)

            if time.time() - last_report > 30:
                el = time.time() - t0
                pct = 100.0 * self.kept_bytes / self.target_bytes
                rate = self.kept_bytes / 1e6 / max(el, 1e-9)
                eta = (self.target_bytes - self.kept_bytes) / 1e6 / max(rate, 1e-9)
                self.log(f"    {name}: {self.kept_bytes / 1e9:.2f}/{self.target_bytes / 1e9:.2f} GB "
                         f"({pct:.1f}%) {self.kept_docs} docs {rate:.1f} MB/s ETA {eta / 60:.0f}m")
                last_report = time.time()
            self.checkpoint()
            if self.kept_bytes >= src_target:
                self.checkpoint(force=True)
                stream.close()
                return

    def _offer(self, line: bytes) -> None:
        """Reservoir-sample (Algorithm R) into the val pool; losers go to train."""
        self.seen_for_reservoir += 1
        if len(self.reservoir) < VAL_MAX:
            self.reservoir.append(line)
            self._reservoir_dirty = True
            return
        j = self.rng.randrange(self.seen_for_reservoir)
        if j < VAL_MAX:
            evicted = self.reservoir[j]
            self.reservoir[j] = line
            self._reservoir_dirty = True
            self.writer.write(evicted)
        else:
            self.writer.write(line)

    def _finalize(self, status: str) -> dict:
        self.checkpoint(force=True)
        pre_finalize = {
            "writer": self.writer.state(),
            "next_id": self.next_id, "kept_docs": self.kept_docs,
            "kept_bytes": self.kept_bytes, "dropped": dict(self.dropped),
            "source_index": self.source_index, "cursor": self.cursor,
            "seen_for_reservoir": self.seen_for_reservoir,
            "rng_state": _list_state(self.rng.getstate()),
            "reservoir_len": len(self.reservoir),
        }
        n = self.kept_docs
        k = int(round(VAL_FRACTION * n))
        k = max(VAL_MIN, min(VAL_MAX, k))
        k = min(k, len(self.reservoir), max(0, n - 1))
        pick_rng = random.Random(SEED + zlib.crc32((self.domain.name + ":val").encode()))
        chosen = set(pick_rng.sample(range(len(self.reservoir)), k)) if k else set()

        for i, line in enumerate(self.reservoir):
            if i not in chosen:
                self.writer.write(line)
        train_files = self.writer.close_final()

        val_writer = RotatingWriter(self.dir, "val", self.compress)
        val_docs = 0
        val_bytes = 0
        for i in sorted(chosen):
            val_writer.write(self.reservoir[i])
            val_docs += 1
            val_bytes += len(self.reservoir[i])
        val_files = val_writer.close_final()
        if not val_files and n > 0:
            empty = self.dir / ("val_000" + (".jsonl.zst" if self.compress else ".jsonl"))
            empty.touch()
            val_files = [empty.name]

        train_bytes = sum((self.dir / f).stat().st_size for f in train_files)
        val_bytes = sum((self.dir / f).stat().st_size for f in val_files)
        train_docs = n - val_docs

        st = {
            "next_id": self.next_id, "kept_docs": self.kept_docs, "kept_bytes": self.kept_bytes,
            "dropped": self.dropped, "source_index": self.source_index, "cursor": self.cursor,
            "seen_for_reservoir": self.seen_for_reservoir,
            "rng_state": _list_state(self.rng.getstate()),
            "reservoir_len": len(self.reservoir),
            "writer": {"index": self.writer.index, "bytes": 0, "finished": train_files},
            "finalized": True, "status": status, "pre_finalize": pre_finalize,
        }
        _atomic_write_text(self.progress_path, json.dumps(st, indent=1))

        return {
            "target_bytes": self.target_bytes,
            "actual_bytes": train_bytes + val_bytes,
            "bytes_per_token": self.domain.bytes_per_token,
            "documents_train": train_docs,
            "documents_val": val_docs,
            "files": train_files + val_files,
            "source": self.domain.all_sources[0].manifest()
                      if len(self.domain.all_sources) == 1
                      else [s.manifest() for s in self.domain.all_sources],
            "dropped": dict(self.dropped),
            "status": status,
            "blocked": self.blocked,
            "notes": self.notes,
        }


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".swap")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _list_state(state) -> list:
    return [state[0], list(state[1]), state[2]]


def _tuple_state(state) -> tuple:
    return (state[0], tuple(state[1]), state[2])


def _read_jsonl(path: Path, compress: bool) -> Iterator[bytes]:
    if str(path).endswith(".zst"):
        import zstandard
        with open(path, "rb") as raw:
            with zstandard.ZstdDecompressor().stream_reader(raw) as fh:
                buf = b""
                while True:
                    chunk = fh.read(8 << 20)
                    if not chunk:
                        break
                    buf += chunk
                    while True:
                        nl = buf.find(b"\n")
                        if nl < 0:
                            break
                        line, buf = buf[:nl], buf[nl + 1:]
                        if line.strip():
                            yield line
                if buf.strip():
                    yield buf
    else:
        with open(path, "rb") as fh:
            for line in fh:
                if line.strip():
                    yield line


def build_smoke(corpus_root: Path, smoke_root: Path, manifest: dict, divisor: int,
                compress: bool, log) -> dict:
    """Mirror the corpus at 1/divisor scale by resampling the produced files."""
    smoke_root.mkdir(parents=True, exist_ok=True)
    domains: dict[str, dict] = {}
    for name, dm in manifest["domains"].items():
        src_dir = corpus_root / name
        dst_dir = smoke_root / name
        if not src_dir.is_dir():
            continue
        train_src = sorted(f for f in dm["files"] if f.startswith("train"))
        if not train_src:
            if dst_dir.is_dir() and not any(dst_dir.iterdir()):
                dst_dir.rmdir()
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        for stale in dst_dir.glob("*.jsonl*"):
            stale.unlink()
        target = dm["target_bytes"] / divisor
        rng = random.Random(SEED + zlib.crc32((name + ":smoke").encode()))

        train_files = train_src
        val_files = sorted(f for f in dm["files"] if f.startswith("val"))

        keep_p = min(1.0, (target / max(1, dm["actual_bytes"])) * 1.15) if dm["actual_bytes"] else 1.0
        writer = RotatingWriter(dst_dir, "train", compress)
        n_train = 0
        acc = 0
        for f in train_files:
            if acc >= target:
                break
            for line in _read_jsonl(src_dir / f, compress):
                if rng.random() < keep_p:
                    writer.write(line if line.endswith(b"\n") else line + b"\n")
                    n_train += 1
                    acc += len(line) + 1
                    if acc >= target:
                        break
        if n_train == 0:
            for f in train_files:
                if acc >= target or n_train >= 50:
                    break
                for line in _read_jsonl(src_dir / f, compress):
                    writer.write(line if line.endswith(b"\n") else line + b"\n")
                    n_train += 1
                    acc += len(line) + 1
                    if acc >= target or n_train >= 50:
                        break
        out_train = writer.close_final()

        val_lines = []
        for f in val_files:
            val_lines.extend(_read_jsonl(src_dir / f, compress))
        k = max(1, min(len(val_lines), max(int(round(VAL_FRACTION * n_train)), 20)))
        k = min(k, len(val_lines))
        vw = RotatingWriter(dst_dir, "val", compress)
        for i in sorted(rng.sample(range(len(val_lines)), k)) if val_lines else []:
            line = val_lines[i]
            vw.write(line if line.endswith(b"\n") else line + b"\n")
        out_val = vw.close_final()
        if not out_val:
            empty = dst_dir / ("val_000" + (".jsonl.zst" if compress else ".jsonl"))
            empty.touch()
            out_val = [empty.name]

        files = out_train + out_val
        actual = sum((dst_dir / f).stat().st_size for f in files)
        domains[name] = {
            "target_bytes": int(target),
            "actual_bytes": actual,
            "bytes_per_token": dm["bytes_per_token"],
            "target_share": dm["target_share"],
            "actual_share": 0.0,
            "documents_train": n_train,
            "documents_val": k if val_lines else 0,
            "files": files,
            "source": dm["source"],
            "dropped": {r: 0 for r in DROP_REASONS},
            "status": dm["status"],
            "note": f"resampled from data/corpus/{name} at 1/{divisor} scale",
        }
        log(f"  smoke {name}: {actual / 1e6:.1f} MB, {n_train} train + "
            f"{domains[name]['documents_val']} val docs")
    return domains


def finish_manifest(domains: dict, seed: int, total_tokens: int) -> dict:
    total_bytes = sum(d["actual_bytes"] for d in domains.values())
    total_docs = sum(d["documents_train"] + d["documents_val"] for d in domains.values())
    est_tokens = 0
    for d in domains.values():
        d["actual_share"] = (d["actual_bytes"] / total_bytes) if total_bytes else 0.0
        d["estimated_tokens"] = int(d["actual_bytes"] / d["bytes_per_token"])
        est_tokens += d["estimated_tokens"]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "target_tokens": total_tokens,
        "total": {"documents": total_docs, "bytes": total_bytes, "estimated_tokens": est_tokens},
        "domains": domains,
    }


def print_report(manifest: dict, smoke_manifest: Optional[dict], corpus_root: Path,
                 smoke_root: Path, elapsed: float) -> None:
    W = 78
    p = print
    p("")
    p("=" * W)
    p("CORPUS BUILD REPORT")
    p("=" * W)
    p(f"{'domain':<18}{'target':>10}{'actual':>10}{'share':>9}{'target':>9}{'delta':>8}  status")
    p("-" * W)
    for name, d in manifest["domains"].items():
        tgt = d["target_bytes"] / 1e9
        act = d["actual_bytes"] / 1e9
        sh = d["actual_share"] * 100
        tsh = d["target_share"] * 100
        p(f"{name:<18}{tgt:>9.2f}G{act:>9.2f}G{sh:>8.1f}%{tsh:>8.1f}%{sh - tsh:>+7.1f}p  {d['status']}")
    p("-" * W)
    t = manifest["total"]
    p(f"{'TOTAL':<18}{'':>10}{t['bytes'] / 1e9:>9.2f}G   "
      f"{t['documents']:,} docs   ~{t['estimated_tokens'] / 1e9:.2f}B tokens")
    p("")
    p("DROPPED DOCUMENTS")
    p(f"{'domain':<18}{'utf8':>10}{'too_short':>12}{'duplicate':>12}{'control':>10}")
    for name, d in manifest["domains"].items():
        dr = d["dropped"]
        p(f"{name:<18}{dr['utf8']:>10,}{dr['too_short']:>12,}{dr['duplicate']:>12,}"
          f"{dr['control_char']:>10,}")
    p("")
    blocked = [(n, d) for n, d in manifest["domains"].items() if d.get("blocked")]
    if blocked:
        p("BLOCKED DOMAINS -- manual action required")
        p("-" * W)
        for name, d in blocked:
            for b in d["blocked"]:
                p(f"* {name}  [{d['status']}]")
                p(f"    why: {b['reason']}")
                p(f"    do : {b['action']}")
        p("")
    else:
        p("BLOCKED DOMAINS: none")
        p("")
    partial = [(n, d) for n, d in manifest["domains"].items()
               if d.get("status") == "partial" and d.get("notes")]
    if partial:
        p("SHORTFALLS -- domains that ran out of source before their target")
        p("-" * W)
        for name, d in partial:
            got = d["actual_bytes"] / 1e9
            want = d["target_bytes"] / 1e9
            p(f"* {name}: {got:.2f} GB of {want:.2f} GB")
            for note in d["notes"]:
                p(f"    - {note}")
        p("")

    du = _dir_bytes(corpus_root) + (_dir_bytes(smoke_root) if smoke_root.exists() else 0)
    p(f"Disk: corpus {_dir_bytes(corpus_root) / 1e9:.2f} GB"
      + (f" + smoke {_dir_bytes(smoke_root) / 1e9:.3f} GB" if smoke_root.exists() else "")
      + f" = {du / 1e9:.2f} GB total")
    if smoke_manifest:
        p(f"Smoke subset: {smoke_manifest['total']['bytes'] / 1e6:.1f} MB, "
          f"{smoke_manifest['total']['documents']:,} docs -> {smoke_root}")
    p(f"Elapsed: {elapsed / 3600:.2f} h ({elapsed:.0f} s)")
    p("=" * W)


def _dir_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build the nanoChat pretraining corpus.")
    ap.add_argument("--out", default="data/corpus", type=Path)
    ap.add_argument("--smoke-out", default="data/corpus_smoke", type=Path)
    ap.add_argument("--target-tokens", type=int, default=6_000_000_000)
    ap.add_argument("--only", action="append", default=None,
                    help="build just this domain (repeatable)")
    ap.add_argument("--compress", choices=["none", "zstd"], default="none")
    ap.add_argument("--smoke-divisor", type=int, default=500)
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--smoke-only", action="store_true",
                    help="rebuild the smoke subset from an existing corpus")
    ap.add_argument("--work-dir", default=None, type=Path)
    ap.add_argument("--checkpoint-mb", type=int, default=256,
                    help="bytes between resume checkpoints, in MB")
    args = ap.parse_args(argv)

    out_root: Path = args.out.resolve()
    smoke_root: Path = args.smoke_out.resolve()
    work_dir: Path = (args.work_dir or (out_root.parent / ".corpus_work")).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)
    compress = args.compress == "zstd"

    global CHECKPOINT_EVERY_BYTES
    CHECKPOINT_EVERY_BYTES = max(1, args.checkpoint_mb) * 1024 * 1024

    selected = [d for d in DOMAINS if (args.only is None or d.name in args.only)]
    if args.only:
        unknown = set(args.only) - set(DOMAINS_BY_NAME)
        if unknown:
            print(f"unknown domain(s): {sorted(unknown)}", file=sys.stderr)
            return 2

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    scale = args.target_tokens
    targets = {d.name: int(round(scale * d.share * d.bytes_per_token)) for d in DOMAINS}

    t_start = time.time()

    if not args.smoke_only:
        need = sum(targets[d.name] for d in selected)
        need = int(need * 1.05) + 2 * 1024 ** 3
        free = shutil.disk_usage(out_root).free
        log(f"disk: {free / 1e9:.1f} GB free, need ~{need / 1e9:.1f} GB")
        if free < need:
            print(f"ERROR: insufficient disk space: {free / 1e9:.1f} GB free, "
                  f"{need / 1e9:.1f} GB required", file=sys.stderr)
            return 1

        try:
            from huggingface_hub import get_token
            token = get_token()
        except Exception:
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token:
            log("note: not authenticated to HuggingFace -- gated datasets will be blocked and "
                "rate limits are lower (`hf auth login`)")

        for d in selected:
            log(f"== {d.name}: target {targets[d.name] / 1e9:.2f} GB "
                f"({scale * d.share / 1e9:.2f}B tokens @ {d.bytes_per_token} B/tok)")
            b = DomainBuilder(d, out_root, targets[d.name], compress, work_dir, log)
            try:
                res = b.run()
            except KeyboardInterrupt:
                log("interrupted -- progress checkpointed, re-run to resume")
                return 130
            except Exception:
                traceback.print_exc()
                b.checkpoint(force=True)
                log(f"   {d.name} failed; progress checkpointed")
                continue
            _write_domain_result(out_root, d, res)
            log(f"   {d.name}: {res['actual_bytes'] / 1e9:.2f} GB, "
                f"{res['documents_train']:,} train + {res['documents_val']:,} val, "
                f"status={res['status']}")

    domains: dict[str, dict] = {}
    for d in DOMAINS:
        res = _read_domain_result(out_root, d)
        if res is None:
            continue
        res["target_share"] = d.share
        res["target_bytes"] = targets[d.name]
        domains[d.name] = res
    manifest = finish_manifest(domains, SEED, scale)
    _atomic_write_text(out_root / "manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))

    smoke_manifest = None
    if not args.skip_smoke and domains:
        log(f"building smoke subset at 1/{args.smoke_divisor} scale")
        sdomains = build_smoke(out_root, smoke_root, manifest, args.smoke_divisor, compress, log)
        smoke_manifest = finish_manifest(sdomains, SEED, scale // args.smoke_divisor)
        smoke_manifest["derived_from"] = str(out_root)
        _atomic_write_text(smoke_root / "manifest.json",
                           json.dumps(smoke_manifest, indent=2, ensure_ascii=False))

    verifier = Path(__file__).resolve().parent / "verify_corpus.py"
    if verifier.exists():
        shutil.copyfile(verifier, out_root / "verify_corpus.py")

    print_report(manifest, smoke_manifest, out_root, smoke_root, time.time() - t_start)
    return 0


def _result_path(out_root: Path, d: Domain) -> Path:
    return out_root / d.name / ".result.json"


def _write_domain_result(out_root: Path, d: Domain, res: dict) -> None:
    _atomic_write_text(_result_path(out_root, d), json.dumps(res, indent=1, ensure_ascii=False))


def _read_domain_result(out_root: Path, d: Domain) -> Optional[dict]:
    p = _result_path(out_root, d)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
