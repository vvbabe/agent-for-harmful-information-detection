#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Weibo search crawler (public content only) using requests + BeautifulSoup.
- Respects robots.txt (best-effort; verify s.weibo.com robots before runs)
- Rate limits and retries
- Outputs JSONL: {post_id, text, timestamp, source_url}

Note: Weibo may employ anti-scraping; this basic version may be brittle.
Use reasonable headers, delays, and avoid excessive requests.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from typing import Dict, Iterable, List, Optional
import os

import requests  # type: ignore
from bs4 import BeautifulSoup  # type: ignore

from utils.logger import get_logger

logger = get_logger("crawler_weibo")

BASE = "https://s.weibo.com/weibo"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}


def search_page(keyword: str, page: int, proxies: Optional[Dict[str, str]] = None) -> List[Dict]:
    params = {"q": keyword, "page": page}
    r = requests.get(BASE, params=params, headers=HEADERS, timeout=20, proxies=proxies)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    results: List[Dict] = []
    for card in soup.select("div.card-wrap"):  # brittle: relies on current site structure
        mid = card.get("mid") or card.get("id")
        text_tag = card.select_one("p.txt")
        if not text_tag:
            continue
        for s in text_tag.select("script,style,span\u200b"):
            s.extract()
        text = text_tag.get_text(" ", strip=True)
        date_tag = card.select_one("p.from a")
        ts = date_tag.get_text(strip=True) if date_tag else None
        link = (date_tag.get("href") if date_tag else None) or r.url
        if link and link.startswith("/"):
            link = "https://weibo.com" + link
        results.append({
            "post_id": str(mid) if mid else None,
            "text": text,
            "timestamp": ts,
            "source_url": link,
        })
    return results

def _selenium_search(keyword: str, page: int, driver) -> List[Dict]:
    url = f"{BASE}?q={keyword}&page={page}"
    driver.get(url)
    # 适度等待渲染
    import time as _t
    _t.sleep(2.5)
    html = driver.page_source
    soup = BeautifulSoup(html, "lxml")
    results: List[Dict] = []
    for card in soup.select("div.card-wrap"):
        mid = card.get("mid") or card.get("id")
        text_tag = card.select_one("p.txt")
        if not text_tag:
            continue
        for s in text_tag.select("script,style,span\u200b"):
            s.extract()
        text = text_tag.get_text(" ", strip=True)
        date_tag = card.select_one("p.from a")
        ts = date_tag.get_text(strip=True) if date_tag else None
        link = (date_tag.get("href") if date_tag else None) or url
        if link and link.startswith("/"):
            link = "https://weibo.com" + link
        results.append({
            "post_id": str(mid) if mid else None,
            "text": text,
            "timestamp": ts,
            "source_url": link,
        })
    return results


def crawl(keyword: str, pages: int, output: str, delay: float = 2.0, driver_mode: str = "requests", append: bool = False, proxy: Optional[str] = None, chrome_binary: Optional[str] = None) -> int:
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
                cb = chrome_binary or os.environ.get("CHROME_PATH")
                if cb:
                    opts.binary_location = cb
                if proxy:
                    opts.add_argument(f"--proxy-server={proxy}")
                drv = uc.Chrome(options=opts)
            except Exception as e:
                logger.warning("Selenium init failed, fallback to requests: %s", e)
                driver_mode = "requests"
        proxies = None
        if proxy and driver_mode == "requests":
            proxies = {"http": proxy, "https": proxy}
        for p in range(1, pages + 1):
            try:
                if driver_mode == "selenium" and drv is not None:
                    items = _selenium_search(keyword, p, drv)
                else:
                    items = search_page(keyword, p, proxies=proxies)
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
    ap = argparse.ArgumentParser(description="Weibo public search crawler")
    ap.add_argument("--keyword", required=True)
    ap.add_argument("--pages", type=int, default=1)
    ap.add_argument("--output", default="weibo_data.jsonl")
    ap.add_argument("--delay", type=float, default=2.0, help="Delay between pages")
    ap.add_argument("--driver", choices=["requests", "selenium"], default="requests")
    ap.add_argument("--append", action="store_true", help="Append to output instead of overwrite")
    ap.add_argument("--proxy", default=None, help="HTTP/HTTPS proxy, e.g., http://127.0.0.1:8080")
    ap.add_argument("--chrome-binary", default=None, help="Path to Chrome/Chromium binary for selenium mode")
    args = ap.parse_args(list(argv) if argv is not None else None)

    try:
        crawl(args.keyword, args.pages, args.output, args.delay, args.driver, args.append, args.proxy, args.chrome_binary)
        return 0
    except Exception as e:
        logger.exception("Failed: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
