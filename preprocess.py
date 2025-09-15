#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Text preprocessing utilities and CLI.
Cleans HTML, emojis, and redundant whitespaces; normalizes to UTF-8; reads/writes JSONL.
"""

from __future__ import annotations

import argparse
import html
import io
import json
import os
import re
import sys
from typing import Dict, Iterable, Iterator

import chardet  # type: ignore
from bs4 import BeautifulSoup  # type: ignore

from utils.logger import get_logger

logger = get_logger("preprocess")


def strip_html(text: str) -> str:
    if not text:
        return ""
    # Unescape first, then remove tags
    text = html.unescape(text)
    soup = BeautifulSoup(text, "lxml")
    return soup.get_text(" ").strip()


_EMOJI_RE = re.compile(
    "[\U00010000-\U0010ffff]",  # astral plane emoji and symbols
    flags=re.UNICODE,
)


def strip_emoji(text: str) -> str:
    if not text:
        return ""
    return _EMOJI_RE.sub("", text)


_WS_RE = re.compile(r"\s+")


def normalize_ws(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def clean_text(text: str) -> str:
    text = strip_html(text)
    text = strip_emoji(text)
    text = normalize_ws(text)
    return text


def detect_open(path: str, mode: str) -> io.TextIOBase:
    """Open a file with detected encoding for reading text, or utf-8 for writing."""
    if "r" in mode and "b" not in mode:
        with open(path, "rb") as fb:
            raw = fb.read(8192)
            enc = chardet.detect(raw).get("encoding") or "utf-8"
        return open(path, mode, encoding=enc, errors="ignore")
    return open(path, mode, encoding="utf-8")


def iter_jsonl(fp: io.TextIOBase) -> Iterator[Dict]:
    for line in fp:
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            logger.warning("Skip invalid JSON line: %s", line[:200])


def process_jsonl(input_path: str, output_path: str, text_field: str = "text") -> int:
    """
    Read JSONL, clean text_field, write cleaned JSONL. Returns count of processed rows.
    """
    processed = 0
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with detect_open(input_path, "r") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for obj in iter_jsonl(fin):
            if text_field not in obj:
                logger.debug("Missing text field '%s' in: %s", text_field, obj)
                continue
            obj[text_field] = clean_text(str(obj[text_field]))
            json.dump(obj, fout, ensure_ascii=False)
            fout.write("\n")
            processed += 1
    logger.info("Cleaned %d records -> %s", processed, output_path)
    return processed


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Clean JSONL texts")
    p.add_argument("--input", required=True, help="Input JSONL file path")
    p.add_argument("--output", required=True, help="Output JSONL file path")
    p.add_argument("--text-field", default="text", help="Field name containing text")
    args = p.parse_args(list(argv) if argv is not None else None)

    try:
        count = process_jsonl(args.input, args.output, args.text_field)
        logger.info("Done: %d rows", count)
        return 0
    except Exception as e:
        logger.exception("Failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
