#!/usr/bin/env python3
"""Self-check for a corpus produced by build_corpus.py.

Runs standalone against a corpus directory, prints PASS/FAIL per check, and
exits 0 only if every check passed.

    python3 verify_corpus.py [--root data/corpus] [--smoke data/corpus_smoke]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterator, Optional

MIN_DOC_BYTES = 200
MIN_DOC_BYTES_BY_DOMAIN = {"terminal_docs": 50}
MAX_FILE_BYTES = 512 * 1024 * 1024
SHARE_TOLERANCE_PP = 5.0
SMOKE_MIN_BYTES = 30 * 1000 * 1000
SMOKE_MAX_BYTES = 80 * 1000 * 1000
EXPECTED_KEYS = {"text", "source", "id"}


class Results:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, bool, str]] = []

    def add(self, cid: str, title: str, ok: Optional[bool], detail: str = "") -> None:
        self.rows.append((cid, title, ok, detail))

    def report(self) -> int:
        width = max(len(t) for _, t, _, _ in self.rows) + 2
        failed = 0
        for cid, title, ok, detail in self.rows:
            tag = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
            if ok is False:
                failed += 1
            print(f"{cid:<4} {title:<{width}} {tag}   {detail}")
        print()
        print(f"{len(self.rows)} checks: "
              f"{sum(1 for r in self.rows if r[2] is True)} passed, "
              f"{failed} failed, "
              f"{sum(1 for r in self.rows if r[2] is None)} skipped")
        return 1 if failed else 0


def iter_lines(path: Path) -> Iterator[bytes]:
    if path.name.endswith(".zst"):
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
                        yield buf[:nl]
                        buf = buf[nl + 1:]
                if buf:
                    yield buf
    else:
        with open(path, "rb") as fh:
            for line in fh:
                yield line.rstrip(b"\n")


def corpus_files(domain_dir: Path) -> tuple[list[Path], list[Path]]:
    train = sorted(p for p in domain_dir.iterdir()
                   if p.name.startswith("train_") and ".jsonl" in p.name
                   and not p.name.endswith(".tmp"))
    val = sorted(p for p in domain_dir.iterdir()
                 if p.name.startswith("val_") and ".jsonl" in p.name
                 and not p.name.endswith(".tmp"))
    return train, val


def scan(root: Path, res: Results) -> None:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if manifest is None:
        res.add("--", f"manifest.json present in {root}", False, "missing")
        return

    domain_names = list(manifest["domains"])
    present = [n for n in domain_names if (root / n).is_dir() and corpus_files(root / n)[0]]
    blocked = {n for n, d in manifest["domains"].items()
               if d.get("status") == "blocked" or not d.get("files")}

    # ---- C1 -------------------------------------------------------------
    bad = []
    for name in domain_names:
        d = root / name
        if not d.is_dir():
            bad.append(f"{name}: no directory")
            continue
        tr, va = corpus_files(d)
        if not tr:
            bad.append(f"{name}: no train file")
        if not va:
            bad.append(f"{name}: no val file")
    checked = [n for n in domain_names if n not in blocked]
    bad_checked = [b for b in bad if b.split(":")[0] not in blocked]
    res.add("C1", "every domain has >=1 train and >=1 val file",
            not bad_checked,
            "; ".join(bad_checked) if bad_checked else
            (f"{len(checked)} domains ok"
             + (f"; {len(blocked)} blocked domain(s) skipped: {sorted(blocked)}" if blocked else "")))

    # ---- per-line checks: C2 C3 C6 C7, feeding C4 C5 C8 ------------------
    all_ids: set[str] = set()
    total_lines = 0
    c2_bad: list[str] = []
    c3_bad: list[str] = []
    c6_bad: list[str] = []
    c7_bad: list[str] = []
    c5_bad: list[str] = []
    c8_bad: list[str] = []
    c10_bad: list[str] = []
    disk_bytes: dict[str, int] = {}

    for name in present:
        ddir = root / name
        tr, va = corpus_files(ddir)
        min_bytes = MIN_DOC_BYTES_BY_DOMAIN.get(name, MIN_DOC_BYTES)
        hashes: set[bytes] = set()
        train_ids: set[str] = set()
        val_ids: set[str] = set()
        dup_hash = 0
        disk_bytes[name] = 0

        for path in tr + va:
            size = path.stat().st_size
            disk_bytes[name] += size
            if size > MAX_FILE_BYTES:
                c10_bad.append(f"{path.name} is {size / 1e6:.0f} MB")
            is_val = path.name.startswith("val_")
            for lineno, raw in enumerate(iter_lines(path), 1):
                if not raw.strip():
                    continue
                total_lines += 1
                try:
                    obj = json.loads(raw)
                except Exception as exc:
                    if len(c2_bad) < 5:
                        c2_bad.append(f"{name}/{path.name}:{lineno} {type(exc).__name__}")
                    continue
                if not isinstance(obj, dict) or set(obj) != EXPECTED_KEYS:
                    if len(c2_bad) < 5:
                        c2_bad.append(f"{name}/{path.name}:{lineno} keys={sorted(obj)}"
                                      if isinstance(obj, dict) else
                                      f"{name}/{path.name}:{lineno} not an object")
                    continue
                text, source, doc_id = obj["text"], obj["source"], obj["id"]
                if source != name and len(c3_bad) < 5:
                    c3_bad.append(f"{name}/{path.name}:{lineno} source={source!r}")
                if not isinstance(text, str):
                    if len(c6_bad) < 5:
                        c6_bad.append(f"{name}/{path.name}:{lineno} text is not str")
                    continue
                try:
                    enc = text.encode("utf-8")
                except UnicodeEncodeError:
                    if len(c6_bad) < 5:
                        c6_bad.append(f"{name}/{path.name}:{lineno} not encodable as UTF-8")
                    continue
                if "\r" in text or "\x00" in text:
                    if len(c6_bad) < 5:
                        c6_bad.append(f"{name}/{path.name}:{lineno} contains CR or NUL")
                if len(enc) < min_bytes and len(c7_bad) < 5:
                    c7_bad.append(f"{name}/{path.name}:{lineno} {len(enc)} B < {min_bytes}")
                h = hashlib.sha256(enc).digest()
                if h in hashes:
                    dup_hash += 1
                else:
                    hashes.add(h)
                all_ids.add(doc_id)
                (val_ids if is_val else train_ids).add(doc_id)

        overlap = train_ids & val_ids
        if overlap:
            c5_bad.append(f"{name}: {len(overlap)} shared ids")
        if dup_hash:
            c8_bad.append(f"{name}: {dup_hash} duplicate texts")
        del hashes, train_ids, val_ids

    res.add("C2", "every line is JSON with exactly text/source/id",
            not c2_bad, "; ".join(c2_bad) if c2_bad else f"{total_lines:,} lines")
    res.add("C3", "source field equals the directory name",
            not c3_bad, "; ".join(c3_bad) if c3_bad else f"{len(present)} domains")
    res.add("C4", "ids are globally unique",
            len(all_ids) == total_lines,
            f"{len(all_ids):,} unique ids vs {total_lines:,} lines")
    res.add("C5", "train and val ids are disjoint per domain",
            not c5_bad, "; ".join(c5_bad) if c5_bad else f"{len(present)} domains")
    res.add("C6", "text is valid UTF-8 with no CR or NUL",
            not c6_bad, "; ".join(c6_bad) if c6_bad else f"{total_lines:,} lines")
    res.add("C7", "text is at least 200 bytes (terminal_docs 50)",
            not c7_bad, "; ".join(c7_bad) if c7_bad else f"{total_lines:,} lines")
    res.add("C8", "no duplicate text within a domain",
            not c8_bad, "; ".join(c8_bad) if c8_bad else f"{len(present)} domains")

    # ---- C9: share, renormalized over the domains actually built ---------
    built = [n for n in domain_names if n not in blocked and n in disk_bytes]
    total_actual = sum(disk_bytes[n] for n in built)
    total_target_share = sum(manifest["domains"][n]["target_share"] for n in built)
    c9_bad = []
    detail_bits = []
    for name in built:
        actual = 100.0 * disk_bytes[name] / total_actual if total_actual else 0.0
        expect = 100.0 * manifest["domains"][name]["target_share"] / total_target_share
        if abs(actual - expect) > SHARE_TOLERANCE_PP:
            c9_bad.append(f"{name}: {actual:.1f}% vs {expect:.1f}%")
        detail_bits.append(f"{name} {actual:.1f}/{expect:.1f}")
    note = ""
    if blocked:
        note = f" (shares renormalized over built domains; blocked: {sorted(blocked)})"
    res.add("C9", "domain byte share is within 5 points of target",
            not c9_bad if built else None,
            ("; ".join(c9_bad) if c9_bad else ", ".join(detail_bits)) + note)

    # ---- C10 ------------------------------------------------------------
    res.add("C10", "no jsonl file exceeds 512 MB",
            not c10_bad, "; ".join(c10_bad) if c10_bad else f"{len(present)} domains")

    # ---- C11 ------------------------------------------------------------
    c11_bad = []
    for name in built:
        recorded = manifest["domains"][name]["actual_bytes"]
        if recorded != disk_bytes[name]:
            c11_bad.append(f"{name}: manifest {recorded} vs disk {disk_bytes[name]}")
    recorded_total = manifest["total"]["bytes"]
    disk_total = sum(disk_bytes.values())
    if recorded_total != disk_total:
        c11_bad.append(f"total: manifest {recorded_total} vs disk {disk_total}")
    res.add("C11", "manifest byte counts match the files on disk",
            not c11_bad, "; ".join(c11_bad) if c11_bad else f"{disk_total:,} bytes")


def check_smoke(root: Path, smoke: Path, res: Results) -> None:
    if not smoke.is_dir():
        res.add("C12", "smoke subset exists and mirrors the corpus", False, f"{smoke} missing")
        return
    problems = []
    main_manifest = json.loads((root / "manifest.json").read_text())
    built = [n for n, d in main_manifest["domains"].items() if d.get("files")]
    for name in built:
        d = smoke / name
        if not d.is_dir():
            problems.append(f"{name}: missing")
            continue
        tr, va = corpus_files(d)
        if not tr:
            problems.append(f"{name}: no train file")
        if not va:
            problems.append(f"{name}: no val file")
    if not (smoke / "manifest.json").exists():
        problems.append("no manifest.json")
    total = sum(p.stat().st_size for p in smoke.rglob("*.jsonl*") if p.is_file())
    size_ok = SMOKE_MIN_BYTES <= total <= SMOKE_MAX_BYTES
    if not size_ok:
        problems.append(f"total {total / 1e6:.1f} MB outside 30-80 MB")
    res.add("C12", "smoke subset exists and mirrors the corpus",
            not problems, "; ".join(problems) if problems else f"{total / 1e6:.1f} MB, "
            f"{len(built)} domains")


def main(argv: Optional[list[str]] = None) -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Verify a built corpus.")
    ap.add_argument("--root", type=Path, default=here)
    ap.add_argument("--smoke", type=Path, default=None)
    args = ap.parse_args(argv)

    root: Path = args.root.resolve()
    smoke: Path = (args.smoke or root.parent / "corpus_smoke").resolve()
    print(f"corpus: {root}")
    print(f"smoke : {smoke}")
    print()

    res = Results()
    scan(root, res)
    check_smoke(root, smoke, res)
    return res.report()


if __name__ == "__main__":
    sys.exit(main())
