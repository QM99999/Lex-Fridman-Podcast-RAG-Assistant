"""Step 0 - Fetch a Lex Fridman transcript page from the official site.

Takes a Lex Fridman URL (the episode page, e.g.
https://lexfridman.com/jensen-huang/ , or the transcript page
https://lexfridman.com/jensen-huang-transcript/ ) and downloads the
transcript HTML into data/raw, so step 1 (01_clean_raw.py) can process it
exactly like the manually-saved transcripts.

Usage:
    python src/00_fetch_transcript.py --url https://lexfridman.com/peter-steinberger/
    python src/00_fetch_transcript.py --url https://lexfridman.com/peter-steinberger-transcript/
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import httpx
from lxml import html as lxml_html

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
PODCAST_RE = re.compile(r"Podcast #(\d+)")


def safe_filename(text: str) -> str:
    """Keep a readable name but drop characters illegal in file names."""
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "-", text).strip().rstrip(".")
    return safe or "unknown"


def transcript_candidates(url: str) -> list[str]:
    """Return URLs to try, transcript page first when a slug is available."""
    parsed = urlparse(url)
    slug = parsed.path.strip("/").split("/")[0] if parsed.path else ""
    if slug:
        base = f"https://lexfridman.com/{slug}"
        if slug.endswith("-transcript"):
            return [f"{base}/"]
        return [f"{base}-transcript/", f"{base}/"]
    return [url]


def page_info(text: str) -> tuple[str | None, int | None]:
    """Return (entry-title, episode number) from a transcript page."""
    root = lxml_html.fromstring(text)
    title = None
    for h1 in root.xpath("//h1[contains(@class,'entry-title')]"):
        title = " ".join(h1.text_content().split())
        break
    episode = None
    if title:
        m = PODCAST_RE.search(title)
        if m:
            episode = int(m.group(1))
    return title, episode


def existing_episodes(out_dir: Path) -> set[int]:
    eps: set[int] = set()
    for f in out_dir.glob("*.html"):
        head = f.read_text(encoding="utf-8", errors="ignore")[:3000]
        m = PODCAST_RE.search(head)
        if m:
            eps.add(int(m.group(1)))
    return eps


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a Lex Fridman transcript page into data/raw.")
    parser.add_argument("--url", required=True, help="Lex Fridman episode or transcript URL")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "data/raw"),
                        help="folder for downloaded transcript HTML (default: data/raw)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for url in transcript_candidates(args.url):
        print(f"fetching {url}")
        try:
            r = httpx.get(url, follow_redirects=True, timeout=60, headers=HEADERS)
            r.raise_for_status()
        except httpx.HTTPError as exc:
            print(f"  !! download failed: {exc}")
            continue
        if "ts-segment" not in r.text:
            print("  !! not a transcript page (no dialogue segments) - trying next URL")
            continue

        title, episode = page_info(r.text)
        if episode is not None and episode in existing_episodes(out_dir):
            print(f"episode #{episode} already exists in {out_dir} - skipping")
            return

        name = safe_filename(title or f"episode-{episode or 'unknown'}") + ".html"
        path = out_dir / name
        path.write_text(r.text, encoding="utf-8")
        print(f"saved -> {path}")
        print(f"episode: #{episode if episode is not None else '?'} | title: {title}")
        print("next: run the pipeline with steps clean -> faq -> index "
              "(faq generates Q&A for the new episode, costs LLM tokens)")
        return

    print("ERROR: no transcript page found for that URL.\n"
          "Only episodes with an official transcript page are supported, e.g.\n"
          "https://lexfridman.com/<guest>-transcript/ (older episodes often have none).",
          file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
