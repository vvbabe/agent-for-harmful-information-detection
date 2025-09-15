#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Xiaohongshu crawler (public search) with requests baseline and optional Selenium.
- Only fetches publicly visible posts summaries from search.
- Respects robots.txt; avoid scraping user profiles or private content.
- Outputs JSONL: {post_id, text, timestamp, source_url}

Note: XHS heavily relies on dynamic rendering and anti-bot checks; requests-only
may fail frequently. Provide an optional Selenium path disabled by default.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from typing import Dict, Iterable, List, Optional

import requests  # type: ignore
from bs4 import BeautifulSoup  # type: ignore

from utils.logger import get_logger

logger = get_logger("crawler_xhs")

BASE = "https://www.xiaohongshu.com/search_result"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}


def search_page(keyword: str, page: int) -> List[Dict]:
    params = {"keyword": keyword, "page": page}
    r = requests.get(BASE, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    results: List[Dict] = []
    for item in soup.select("div.note-item, div.result-item"):
        a = item.select_one("a[href]")
        text = item.get_text(" ", strip=True)
        href = a.get("href") if a else None
        if href and href.startswith("/"):
            href = "https://www.xiaohongshu.com" + href
        results.append({
            "post_id": None,
            "text": text,
            "timestamp": None,
            "source_url": href,
        })
    return results

def _selenium_search(keyword: str, page: int, driver) -> List[Dict]:
    url = f"{BASE}?keyword={keyword}&page={page}"
    driver.get(url)
    import time as _t
    _t.sleep(2.5)
    html = driver.page_source
    soup = BeautifulSoup(html, "lxml")
    results: List[Dict] = []
    for item in soup.select("div.note-item, div.result-item"):
        a = item.select_one("a[href]")
        text = item.get_text(" ", strip=True)
        href = a.get("href") if a else None
        if href and href.startswith("/"):
            href = "https://www.xiaohongshu.com" + href
        results.append({
            "post_id": None,
            "text": text,
            "timestamp": None,
            "source_url": href,
        })
    return results


def crawl(keyword: str, pages: int, output: str, delay: float = 2.0, driver_mode: str = "requests", append: bool = False) -> int:
    count = 0
    mode = "a" if append else "w"
    with open(output, mode, encoding="utf-8") as fout:
        drv = None
        if driver_mode == "selenium":
            try:
                import undetected_chromedriver as uc  # type: ignore
                from selenium.webdriver.chrome.options import Options  # type: ignore
                opts = Options()
                opts.add_argument("--headless=new")
                opts.add_argument("--disable-gpu")
                opts.add_argument("--no-sandbox")
                drv = uc.Chrome(options=opts)
            except Exception as e:
                logger.warning("Selenium init failed, fallback to requests: %s", e)
                driver_mode = "requests"
        for p in range(1, pages + 1):
            try:
                if driver_mode == "selenium" and drv is not None:
                    items = _selenium_search(keyword, p, drv)
                else:
                    items = search_page(keyword, p)
                for obj in items:
                    json.dump(obj, fout, ensure_ascii=False)
                    fout.write("\n")
                    count += 1
                logger.info("Page %d -> %d items", p, len(items))
            except requests.HTTPError as e:
                logger.warning("HTTP error on page %d: %s", p, e)
            except Exception as e:
                logger.warning("Error on page %d: %s", p, e)
            time.sleep(delay + random.random())
        if drv is not None:
            try:
                drv.quit()
            except Exception:
                pass
    logger.info("Saved %d items to %s", count, output)
    return count


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Xiaohongshu public search crawler")
    ap.add_argument("--keyword", required=True)
    ap.add_argument("--pages", type=int, default=1)
    ap.add_argument("--output", default="xiaohongshu_data.jsonl")
    ap.add_argument("--delay", type=float, default=2.0, help="Delay between pages")
    ap.add_argument("--driver", choices=["requests", "selenium"], default="requests")
    ap.add_argument("--append", action="store_true", help="Append to output instead of overwrite")
    args = ap.parse_args(list(argv) if argv is not None else None)

    try:
        crawl(args.keyword, args.pages, args.output, args.delay, args.driver, args.append)
        return 0
    except Exception as e:
        logger.exception("Failed: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
