"""Step 1 - Clean Lex Fridman transcript HTML files into per-episode JSON files.

The pipeline scans every .html file in a raw data folder (or a single file),
parses the shared Lex Fridman transcript page format, and writes one lean JSON
file per episode named "<episode-number>-<guest>.json".

Each file groups dialogue segments under their chapter title:

    {
      "episode": 494,
      "guest": "Jensen Huang",
      "chapters": [
        {"title": "Introduction", "segments": [
          {"speaker": "Lex Fridman", "timestamp": "00:00:00", "text": "..."}
        ]}
      ]
    }

Usage:
    python src/01_clean_raw.py
    python src/01_clean_raw.py --input data/raw --output data/processed
    python src/01_clean_raw.py --input data/raw/lex_jensen_transcript.html
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from lxml import html as lxml_html

CHAPTER_H2 = re.compile(r"^chapter\d+_", re.IGNORECASE)
TIMESTAMP_RE = re.compile(r"\((\d{2}):(\d{2}):(\d{2})\)")
T_SECONDS_RE = re.compile(r"[?&;]t=(\d+)")


def _clean(text: str) -> str:
    """Collapse whitespace and newlines into single spaces."""
    return " ".join(text.split())


def _first_text(node) -> str:
    return _clean(node.text_content()) if node is not None else ""


def _seconds_to_clock(seconds: int) -> str:
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _safe_filename(text: str) -> str:
    """Keep a readable name but drop characters illegal in file names."""
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "-", text).strip().rstrip(".")
    return safe or "unknown"


def _pick_span(node, class_name: str):
    """Return the first descendant span with the given class name."""
    for span in node.iter("span"):
        if class_name in span.get("class", "").split():
            return span
    return None


def parse_timestamp(text: str, href: str):
    """Extract a clock string from a timestamp span (fallback to the t= link)."""
    m = TIMESTAMP_RE.search(text or "")
    if m:
        h, mi, s = (int(g) for g in m.groups())
        return f"{h:02d}:{mi:02d}:{s:02d}"
    m = T_SECONDS_RE.search(href or "")
    if m:
        return _seconds_to_clock(int(m.group(1)))
    return None


def parse_episode(root) -> dict:
    """Extract episode number and guest name from the title."""
    episode: dict = {}
    for title in root.xpath("//h1[contains(@class,'entry-title')]"):
        episode["title"] = _first_text(title)

    title = episode.get("title", "")
    m = re.search(r"Podcast #(\d+)", title)
    if m:
        episode["episode"] = int(m.group(1))
    m = re.search(r"Transcript for\s+(.+?):", title)
    if m:
        episode["guest"] = m.group(1).strip()
    return episode


def parse_document(root):
    """Walk the document in order and collect chapters and segments."""
    chapters: list[dict] = []
    segments: list[dict] = []
    current_chapter = None

    for node in root.iter():
        if node.tag == "h2":
            node_id = node.get("id", "")
            if CHAPTER_H2.match(node_id):
                current_chapter = {"id": node_id, "title": _first_text(node)}
                chapters.append(current_chapter)
            continue
        if node.tag != "div" or "ts-segment" not in node.get("class", "").split():
            continue

        name_node = _pick_span(node, "ts-name")
        time_node = _pick_span(node, "ts-timestamp")
        text_node = _pick_span(node, "ts-text")
        if text_node is None:
            continue

        href = ""
        link = node.xpath(".//span[contains(@class,'ts-timestamp')]//a")
        if link:
            href = link[0].get("href", "")

        segments.append(
            {
                "speaker": _first_text(name_node) or "Unknown",
                "timestamp": parse_timestamp(_first_text(time_node), href),
                "text": _first_text(text_node),
                "chapter_id": current_chapter["id"] if current_chapter else None,
                "chapter_title": current_chapter["title"] if current_chapter else None,
            }
        )
    return chapters, segments


def detect_guest(segments: list[dict], fallback: str | None) -> str | None:
    """Identify the guest: the most frequent speaker who is not the host."""
    counts: dict[str, int] = {}
    for seg in segments:
        speaker = seg["speaker"]
        if speaker and speaker != "Lex Fridman":
            counts[speaker] = counts.get(speaker, 0) + 1
    if not counts:
        return fallback
    return max(counts, key=counts.get)


def build_episode(source_file: Path) -> dict:
    """Parse one raw transcript file into a chapter-grouped episode record."""
    root = lxml_html.parse(str(source_file)).getroot()
    meta = parse_episode(root)
    chapters, segments = parse_document(root)

    groups = {c["id"]: {"title": c["title"], "segments": []} for c in chapters}
    for seg in segments:
        key = seg["chapter_id"] or "<no-chapter>"
        if key not in groups:
            groups[key] = {"title": seg["chapter_title"] or "Unknown", "segments": []}
        groups[key]["segments"].append(
            {
                "speaker": seg["speaker"],
                "timestamp": seg["timestamp"],
                "text": seg["text"],
            }
        )

    return {
        "episode": meta.get("episode"),
        "guest": detect_guest(segments, meta.get("guest")),
        "chapters": list(groups.values()),
    }


def episode_filename(episode: dict) -> str:
    """File name for an episode: <episode-number>-<guest>.json."""
    number = episode.get("episode")
    guest = _safe_filename(episode.get("guest") or "unknown")
    return f"{number}-{guest}.json"


def discover_inputs(input_arg: str) -> list[Path]:
    """Resolve --input to a list of .html files (file or directory)."""
    path = Path(input_arg)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.glob("*.html"))
    raise SystemExit(f"Input path not found: {input_arg}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean Lex Fridman transcripts into per-episode JSON files.")
    parser.add_argument("--input", default="data/raw", help="HTML file or folder with raw transcripts")
    parser.add_argument("--output", default="data/processed", help="Output folder for per-episode JSON files")
    args = parser.parse_args()

    files = discover_inputs(args.input)
    if not files:
        raise SystemExit(f"No .html files found in: {args.input}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[tuple[Path, dict]] = []
    for source_file in files:
        episode = build_episode(source_file)
        out_path = output_dir / episode_filename(episode)
        out_path.write_text(json.dumps(episode, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append((out_path, episode))

    print(f"Processed {len(written)} episode(s) -> {output_dir.resolve()}")
    for out_path, ep in written:
        segments = sum(len(c["segments"]) for c in ep["chapters"])
        print(f"  #{ep.get('episode')} {ep.get('guest'):<16} chapters={len(ep['chapters']):<3} segments={segments:<4}")
        print(f"      -> {out_path.name}")


if __name__ == "__main__":
    main()

