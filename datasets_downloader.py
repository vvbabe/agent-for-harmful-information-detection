#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Datasets downloader: reads configs/datasets.yaml, downloads into datasets/<name>/,
supports multithreaded segmented downloads with HTTP Range and resume, and
converts CSV/JSON/JSONL/ZIP/TAR to unified JSONL format.

CLI:
  python datasets_downloader.py --config configs/datasets.yaml --workers 4

Notes:
- For sites needing auth (e.g., Kaggle), this script leaves placeholders and
  will skip with a warning unless proper URLs are provided.
- PCAP conversion to JSONL is stubbed (requires domain-specific parsing). We
  write a placeholder wrapper with file references.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import io
import json
import os
import re
import shutil
import tarfile
import tempfile
import threading
import urllib.parse
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests  # type: ignore
import yaml  # type: ignore
from tqdm import tqdm  # type: ignore

from utils.logger import get_logger

logger = get_logger("datasets")

CHUNK_SIZE = 1 << 20  # 1 MiB


@dataclass
class Dataset:
    name: str
    urls: List[str]
    dtype: str
    target_dir: Path
    notes: Optional[str] = None
    files: List[Path] = field(default_factory=list)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_config(path: str) -> List[Dataset]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    datasets: List[Dataset] = []
    for name, cfg in data.items():
        urls = cfg.get("urls", []) or []
        dtype = cfg.get("type", "unknown")
        target_dir = Path("datasets") / name
        notes = cfg.get("notes")
        datasets.append(Dataset(name=name, urls=urls, dtype=dtype, target_dir=target_dir, notes=notes))
    return datasets


def supports_range(url: str) -> bool:
    try:
        r = requests.head(url, timeout=10, allow_redirects=True)
        return r.headers.get("Accept-Ranges", "").lower() == "bytes"
    except Exception:
        return False


def download_with_resume(url: str, dest: Path) -> Path:
    """Download a file with resume if server supports range; otherwise do full download."""
    ensure_dir(dest.parent)
    temp = dest.with_suffix(dest.suffix + ".part")

    if url.startswith("file://"):
        src = Path(urllib.parse.urlparse(url).path)
        shutil.copy2(src, dest)
        return dest

    headers = {}
    mode = "wb"
    pos = 0
    if temp.exists():
        pos = temp.stat().st_size
        headers["Range"] = f"bytes={pos}-"
        mode = "ab"

    if pos > 0 and not supports_range(url):
        # cannot resume, restart
        temp.unlink(missing_ok=True)
        pos = 0
        headers.pop("Range", None)
        mode = "wb"

    with requests.get(url, stream=True, headers=headers, timeout=30) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", "0"))
        total = pos + total if pos else total
        with open(temp, mode) as f, tqdm(total=total, initial=pos, unit="B", unit_scale=True, desc=dest.name) as pbar:
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                f.write(chunk)
                pbar.update(len(chunk))
    temp.rename(dest)
    return dest


def extract_archive(path: Path, out_dir: Path) -> List[Path]:
    extracted: List[Path] = []
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            zf.extractall(out_dir)
            extracted = [out_dir / n for n in zf.namelist() if not n.endswith("/")]
    elif tarfile.is_tarfile(path):
        with tarfile.open(path) as tf:
            def is_within_directory(directory, target):
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
                prefix = os.path.commonprefix([abs_directory, abs_target])
                return prefix == abs_directory

            def safe_extract(tar, path="."):
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
                tar.extractall(path)
            safe_extract(tf, out_dir)
            extracted = [out_dir / m.name for m in tf.getmembers() if m.isfile()]
    else:
        extracted = [path]
    return extracted


def convert_to_jsonl(paths: List[Path], out_path: Path, payload_field: str = "payload") -> int:
    ensure_dir(out_path.parent)
    count = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for p in paths:
            lower = p.suffix.lower()
            if lower in {".jsonl"}:
                with open(p, "r", encoding="utf-8", errors="ignore") as fin:
                    for line in fin:
                        if not line.strip():
                            continue
                        fout.write(line if line.endswith("\n") else line + "\n")
                        count += 1
            elif lower in {".json"}:
                with open(p, "r", encoding="utf-8", errors="ignore") as fin:
                    data = json.load(fin)
                if isinstance(data, list):
                    for i, obj in enumerate(data):
                        out = normalize_record(obj, i)
                        json.dump(out, fout, ensure_ascii=False)
                        fout.write("\n")
                        count += 1
                elif isinstance(data, dict):
                    out = normalize_record(data, 0)
                    json.dump(out, fout, ensure_ascii=False)
                    fout.write("\n")
                    count += 1
            elif lower in {".csv"}:
                with open(p, "r", encoding="utf-8", errors="ignore") as fin:
                    reader = csv.DictReader(fin)
                    for i, row in enumerate(reader):
                        out = normalize_record(row, i)
                        json.dump(out, fout, ensure_ascii=False)
                        fout.write("\n")
                        count += 1
            elif lower in {".txt", ""}:
                # Plain text files (e.g., SpamAssassin emails). Infer label from path parts.
                try:
                    label_guess = None
                    parts = [s.lower() for s in p.parts]
                    if any("spam" in s for s in parts):
                        label_guess = "spam"
                    elif any("ham" in s for s in parts):
                        label_guess = "ham"
                    with open(p, "r", encoding="utf-8", errors="ignore") as fin:
                        txt = fin.read()
                    obj = {"id": str(count + 1), "text": txt, "label": label_guess}
                    json.dump(obj, fout, ensure_ascii=False)
                    fout.write("\n")
                    count += 1
                except Exception:
                    # fallback to path reference
                    obj = {"id": str(count + 1), payload_field: str(p), "label": None}
                    json.dump(obj, fout, ensure_ascii=False)
                    fout.write("\n")
                    count += 1
            elif lower in {".pcap", ".pcapng"}:
                # Stub: PCAP parsing requires scapy/pyshark and domain logic.
                # Here we just reference file path as payload to keep pipeline unified.
                obj = {"id": str(count + 1), payload_field: str(p), "label": None}
                json.dump(obj, fout, ensure_ascii=False)
                fout.write("\n")
                count += 1
            else:
                # Heuristic: small files likely text (e.g., SpamAssassin emails with hash suffix)
                try:
                    if p.is_file() and p.stat().st_size <= 2 * 1024 * 1024:
                        label_guess = None
                        parts = [s.lower() for s in p.parts]
                        if any("spam" in s for s in parts):
                            label_guess = "spam"
                        elif any("ham" in s for s in parts):
                            label_guess = "ham"
                        with open(p, "r", encoding="utf-8", errors="ignore") as fin:
                            txt = fin.read()
                        obj = {"id": str(count + 1), "text": txt, "label": label_guess}
                    else:
                        obj = {"id": str(count + 1), payload_field: str(p), "label": None}
                except Exception:
                    obj = {"id": str(count + 1), payload_field: str(p), "label": None}
                json.dump(obj, fout, ensure_ascii=False)
                fout.write("\n")
                count += 1
    logger.info("Converted %d records -> %s", count, out_path)
    return count


def normalize_record(obj: Dict, idx: int) -> Dict:
    # Try map common fields to a unified structure
    id_ = str(obj.get("id") or obj.get("ID") or obj.get("index") or idx + 1)
    # known text keys across datasets
    text = (
        obj.get("text")
        or obj.get("comment_text")
        or obj.get("tweet")
        or obj.get("payload")
        or obj.get("content")
        or obj.get("message")
    )
    label = obj.get("label") or obj.get("class") or obj.get("target")
    return {"id": id_, "text": text, "label": label}


def process_dataset(ds: Dataset, workers: int = 4) -> Path:
    ensure_dir(ds.target_dir)
    logger.info("Processing dataset: %s", ds.name)

    downloaded: List[Path] = []

    def task(url: str) -> Optional[Path]:
        try:
            fname = os.path.basename(urllib.parse.urlparse(url).path) or "index.html"
            dest = ds.target_dir / fname
            out = download_with_resume(url, dest)
            logger.info("Downloaded %s -> %s", url, out)
            return out
        except Exception as e:
            logger.warning("Skip %s: %s", url, e)
            return None

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(task, u) for u in ds.urls]
        for fut in cf.as_completed(futures):
            p = fut.result()
            if p:
                downloaded.append(p)

    # Extract archives if any
    expanded: List[Path] = []
    for p in downloaded:
        outdir = ds.target_dir / (p.stem + "_extracted")
        expanded.extend(extract_archive(p, outdir))

    candidates = expanded or downloaded

    # Convert to JSONL
    out_jsonl = ds.target_dir / f"{ds.name}.jsonl"
    convert_to_jsonl(candidates, out_jsonl)
    return out_jsonl


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Datasets downloader and converter to JSONL")
    ap.add_argument("--config", default="configs/datasets.yaml", help="Path to datasets.yaml")
    ap.add_argument("--workers", type=int, default=4, help="Download worker threads")
    ap.add_argument("--dataset", default=None, help="Only process a specific dataset name")
    args = ap.parse_args(list(argv) if argv is not None else None)

    datasets = load_config(args.config)
    if args.dataset:
        datasets = [d for d in datasets if d.name == args.dataset]
        if not datasets:
            logger.error("Dataset not found in config: %s", args.dataset)
            return 2

    for ds in datasets:
        process_dataset(ds, workers=args.workers)
    logger.info("All done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
