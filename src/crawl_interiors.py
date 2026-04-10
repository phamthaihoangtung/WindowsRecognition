"""
Crawl interior-with-windows images from Unsplash and Pexels.

Usage:
    python src/crawl_interiors.py [--source {unsplash,pexels,all}] [--out data/interiors_crawled]
                                  [--target 100] [--min-mp 20] [--max-mp 30]

API keys required in .env:
    UNSPLASH_ACCESS_KEY
    PEXELS_API_KEY
"""

import argparse
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import dotenv_values
from PIL import Image, UnidentifiedImageError

DEFAULT_QUERIES = [
    # "interior room window",
    # "living room large window",
    # "bedroom window daylight",
    # "kitchen window interior",
    # "office interior window",
    "interior design"
]


def passes_filter(width: int, height: int, min_mp: float, max_mp: float) -> bool:
    mp = width * height / 1e6
    return width > height and min_mp <= mp <= max_mp


def verify_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, Exception):
        return False


# ---------------------------------------------------------------------------
# Unsplash
# ---------------------------------------------------------------------------

def crawl_unsplash(
    key: str,
    queries: list[str],
    out_dir: Path,
    target: int,
    min_mp: float,
    max_mp: float,
    saved: list[Path],
) -> None:
    base = "https://api.unsplash.com/search/photos"
    headers = {"Authorization": f"Client-ID {key}"}

    for query in queries:
        if len(saved) >= target:
            break
        page = 1
        while len(saved) < target:
            params = {
                "query": query,
                "orientation": "landscape",
                "per_page": 30,
                "page": page,
            }
            resp = requests.get(base, headers=headers, params=params, timeout=30)
            if resp.status_code == 429:
                print("Unsplash rate-limited, sleeping 60s…")
                time.sleep(60)
                continue
            if not resp.ok:
                print(f"Unsplash error {resp.status_code} for query '{query}'")
                break
            data = resp.json()
            results = data.get("results", [])
            if not results:
                break  # no more pages

            for photo in results:
                if len(saved) >= target:
                    break
                w, h = photo.get("width", 0), photo.get("height", 0)
                if not passes_filter(w, h, min_mp, max_mp):
                    continue
                url = photo["urls"]["raw"]
                photo_id = photo["id"]
                dest = out_dir / f"unsplash_{photo_id}.jpg"
                if dest.exists():
                    continue
                try:
                    img_resp = requests.get(url, timeout=60)
                    img_resp.raise_for_status()
                    dest.write_bytes(img_resp.content)
                    if verify_image(dest):
                        saved.append(dest)
                        print(f"[unsplash] {len(saved)}/{target}  {dest.name}  ({w}x{h})")
                    else:
                        dest.unlink(missing_ok=True)
                except Exception as e:
                    print(f"  download error: {e}")
                    dest.unlink(missing_ok=True)
                time.sleep(0.5)  # be polite to the CDN

            if data.get("total_pages", 0) <= page:
                break
            page += 1
            time.sleep(1)  # between search pages


# ---------------------------------------------------------------------------
# Pexels
# ---------------------------------------------------------------------------

def crawl_pexels(
    key: str,
    queries: list[str],
    out_dir: Path,
    target: int,
    min_mp: float,
    max_mp: float,
    saved: list[Path],
) -> None:
    base = "https://api.pexels.com/v1/search"
    headers = {"Authorization": key}

    for query in queries:
        if len(saved) >= target:
            break
        page = 1
        while len(saved) < target:
            params = {
                "query": query,
                "orientation": "landscape",
                "size": "large",  # ≥~24 MP on Pexels server-side pre-filter
                "per_page": 80,
                "page": page,
            }
            resp = requests.get(base, headers=headers, params=params, timeout=30)
            if resp.status_code == 429:
                print("Pexels rate-limited, sleeping 60s…")
                time.sleep(60)
                continue
            if not resp.ok:
                print(f"Pexels error {resp.status_code} for query '{query}'")
                break
            data = resp.json()
            photos = data.get("photos", [])
            if not photos:
                break

            for photo in photos:
                if len(saved) >= target:
                    break
                w, h = photo.get("width", 0), photo.get("height", 0)
                if not passes_filter(w, h, min_mp, max_mp):
                    continue
                url = photo["src"]["original"]
                photo_id = photo["id"]
                dest = out_dir / f"pexels_{photo_id}.jpg"
                if dest.exists():
                    continue
                try:
                    img_resp = requests.get(url, timeout=60)
                    img_resp.raise_for_status()
                    dest.write_bytes(img_resp.content)
                    if verify_image(dest):
                        saved.append(dest)
                        print(f"[pexels]   {len(saved)}/{target}  {dest.name}  ({w}x{h})")
                    else:
                        dest.unlink(missing_ok=True)
                except Exception as e:
                    print(f"  download error: {e}")
                    dest.unlink(missing_ok=True)
                time.sleep(0.5)  # be polite to the CDN

            # Pexels paginates via next_page field
            if not data.get("next_page"):
                break
            page += 1
            time.sleep(1)  # between search pages


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl interior-with-windows images.")
    parser.add_argument("--source", choices=["unsplash", "pexels", "all"], default="all")
    parser.add_argument("--out", default="data/interiors_crawled")
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--min-mp", type=float, default=20.0)
    parser.add_argument("--max-mp", type=float, default=30.0)
    parser.add_argument("--queries", nargs="+", default=DEFAULT_QUERIES)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = dotenv_values(".env")
    saved: list[Path] = []

    use_unsplash = args.source in ("unsplash", "all")
    use_pexels = args.source in ("pexels", "all")

    # Split target equally when both sources are active
    if use_unsplash and use_pexels:
        unsplash_target = args.target // 2
        pexels_target = args.target
    else:
        unsplash_target = args.target
        pexels_target = args.target

    if use_unsplash:
        key = env.get("UNSPLASH_ACCESS_KEY") or os.environ.get("UNSPLASH_ACCESS_KEY")
        if not key:
            print("ERROR: UNSPLASH_ACCESS_KEY not set in .env", file=sys.stderr)
            if not use_pexels:
                sys.exit(1)
        else:
            crawl_unsplash(key, args.queries, out_dir, unsplash_target, args.min_mp, args.max_mp, saved)

    if use_pexels and len(saved) < pexels_target:
        key = env.get("PEXELS_API_KEY") or os.environ.get("PEXELS_API_KEY")
        if not key:
            print("ERROR: PEXELS_API_KEY not set in .env", file=sys.stderr)
            if not use_unsplash:
                sys.exit(1)
        else:
            crawl_pexels(key, args.queries, out_dir, pexels_target, args.min_mp, args.max_mp, saved)

    print(f"\nDone. {len(saved)} images saved to {out_dir}/")


if __name__ == "__main__":
    main()
