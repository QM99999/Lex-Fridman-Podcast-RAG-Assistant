"""Step 2 - Extract Q&A pairs (FAQ) from cleaned transcripts using an LLM.

Reads the cleaned episode JSON files from data/processed and, for every chapter,
asks an LLM (OpenAI or DeepSeek, OpenAI-compatible APIs) to extract grounded Q&A
pairs from the dialogue. Chapters of an episode are processed in parallel
(--workers) and results are saved incrementally with per-batch granularity, so an
interrupted run only redoes the batches that failed instead of whole chapters.
Already-complete episode files are skipped; only new or incomplete episodes are
processed. Use --force to regenerate everything.

The FAQ dataset serves two purposes downstream:
  * retrieval evaluation (golden questions + grounded answers)
  * knowledge-base entries in question-answer form

Configuration (env vars or .env file):
    02_MODEL            model used by this script (primary), e.g. gpt-4o-mini
    --provider          optional CLI override: openai | deepseek
                        (default: inferred from 02_MODEL name, then openai)
    OPENAI_API_KEY      required when provider=openai
    DEEPSEEK_API_KEY    required when provider=deepseek

Usage:
    python src/02_generate_faq.py
    python src/02_generate_faq.py --force            # regenerate from scratch
    python src/02_generate_faq.py --provider deepseek
    python src/02_generate_faq.py --max-words 2000   # batch size in words
    python src/02_generate_faq.py --workers 6        # chapters processed in parallel

Note: when resuming an interrupted run, keep --max-words unchanged so batch
boundaries (and therefore the batch-level resume state) stay consistent.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROVIDERS = {
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "base_url_default": "https://api.openai.com/v1",
        "model_env": "OPENAI_MODEL",
        "model_default": "gpt-4o-mini",
    },
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "base_url_default": "https://api.deepseek.com",
        "model_env": "DEEPSEEK_MODEL",
        "model_default": "deepseek-v4-flash",
    },
}


def load_dotenv(path: Path) -> None:
    """Tiny .env loader (no extra dependency)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


load_dotenv(PROJECT_ROOT / ".env")

SYSTEM_PROMPT = """You convert podcast transcript dialogue into high-quality Q&A pairs.

You receive an episode (with its guest), a chapter title, and timestamped dialogue
segments in the form "speaker | timestamp | text". Turn the dialogue into Q&A pairs
a listener would ask.

Rules:
1. Identify Q&A rounds: consecutive turns by the same role that form one question
   or one answer may be merged into one pair.
2. Split multi-question turns: if one turn asks several distinct questions, split
   them into separate Q&A pairs, matching each answer by order or semantics.
3. Split long answers: if an answer exceeds ~300 words and contains independent
   knowledge points, split it into multiple Q&A pairs.
4. Boundary cases: if answers to several questions are too tangled to split cleanly,
   keep one Q&A pair and set boundary to "merge".
5. Questions: a single natural standalone question, one topic per question; rewrite
   in your own words, do NOT copy sentences verbatim. NO personal names and NO
   "you"/"I" referring to the guest.
6. Answers: self-contained (understandable without the surrounding context), 2-4
   sentences, keep facts/numbers/opinions, drop small talk, at most 500 tokens.
   NO personal names - refer to the person neutrally ("the guest", "he"). The
   guest identity is tracked in metadata, not in the text.
7. source_timestamps: list the exact timestamps (as shown in the input) of the
   segments the answer is based on.
8. Skip: openings/summarizing monologues, pure politeness (e.g. "Yeah, that's
   right"), and answers shorter than 10 words.
9. Output ONLY valid JSON: {"qa_pairs": [
      {"question": "...", "answer": "...", "source_timestamps": ["00:12:34"],
       "boundary": "ok" | "merge" | "long_split", "skipped": false}
   ]}
   For skipped dialogue use: {"skipped": true, "reason": "..."}"""
USER_TEMPLATE = """Episode #{episode} with guest {guest}
Chapter: {chapter}

Dialogue segments (speaker | timestamp | text):
{dialogue}"""


def batch_segments(segments: list[dict], max_words: int) -> list[list[dict]]:
    """Split a chapter's segments into batches not exceeding max_words each."""
    batches, current, words = [], [], 0
    for seg in segments:
        w = len(seg["text"].split())
        if current and words + w > max_words:
            batches.append(current)
            current, words = [], 0
        current.append(seg)
        words += w
    if current:
        batches.append(current)
    return batches


def extract_json(content: str):
    """Parse a model response into a dict, trying raw JSON, then markdown fences."""
    if not content:
        return None
    content = content.strip()
    try:
        data = json.loads(content)
        return data if isinstance(data, dict) else {"qa_pairs": data}
    except json.JSONDecodeError:
        pass
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass
    start, end = content.find("{"), content.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(content[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def call_llm(client, model: str, episode: dict, chapter: str, batch: list[dict], retries: int):
    """Call the LLM for one batch; returns a list of raw items or None on total failure."""
    dialogue = "\n".join(f"{s['speaker']} | {s['timestamp']} | {s['text']}" for s in batch)
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_TEMPLATE.format(episode=episode.get("episode"), guest=episode.get("guest"), chapter=chapter, dialogue=dialogue)},
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            data = extract_json(resp.choices[0].message.content)
            if data is None:
                raise ValueError("could not parse JSON from model output")
            items = data.get("qa_pairs")
            if items is None:
                for key in ("faqs", "pairs", "results"):
                    if isinstance(data.get(key), list):
                        items = data[key]
                        break
            if isinstance(items, list):
                return items
            return []
        except Exception as exc:  # noqa: BLE001 - retry any transient failure
            if attempt == retries:
                print(f"    !! batch failed after {retries} attempts: {exc}")
                return None
            wait = 2 ** attempt
            print(f"    retry {attempt}/{retries} after {wait}s ({exc})")
            time.sleep(wait)
    return None


def process_batch(client, model: str, episode: dict, chapter: str, batch: list[dict], retries: int):
    """Extract QA items from one batch; returns (qa_dicts, stats_delta, failed)."""
    stats = {"ok": 0, "merge": 0, "long_split": 0, "skipped": 0}
    items = call_llm(client, model, episode, chapter, batch, retries)
    qas = []
    if items is None:
        return qas, stats, True  # batch failed entirely
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("skipped"):
            stats["skipped"] += 1
            continue
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if not question or not answer or len(question) < 8 or len(answer) < 20:
            stats["skipped"] += 1
            continue
        boundary = item.get("boundary", "ok")
        boundary = boundary if boundary in ("merge", "long_split") else "ok"
        stats[boundary] += 1
        timestamps = item.get("source_timestamps", [])
        if isinstance(timestamps, str):
            timestamps = [timestamps]
        qas.append({
            "question": question,
            "answer": answer,
            "source_timestamps": [str(t) for t in timestamps][:6],
            "boundary": boundary,
            "est_tokens": (len(question) + len(answer) + 3) // 4,
        })
    return qas, stats, False


def resolve_provider(cli_provider: str | None, model_env: str) -> str:
    """Decide the provider: CLI arg > inferred from model name > openai."""
    if cli_provider:
        return cli_provider
    if "deepseek" in model_env.lower():
        return "deepseek"
    return "openai"


def save_payload(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def process_chapter(client, model, episode: dict, ci: int, title: str,
                    segments: list[dict], max_words: int, retries: int,
                    existing_faqs: list[dict], done_batches: list[int]) -> dict:
    """Process the pending batches of one chapter; returns a mergeable result.

    existing_faqs are Q&A pairs already saved for this chapter from a previous
    run (they seed the local id counter and are not returned again).
    """
    batches = batch_segments(segments, max_words)
    done = set(done_batches)
    stats = {"ok": 0, "merge": 0, "long_split": 0, "skipped": 0}
    new_faqs: list[dict] = []
    newly_done: list[int] = []
    failed: list[int] = []
    local_idx = len(existing_faqs) + 1
    for bi, batch in enumerate(batches):
        if bi in done:
            continue
        qas, delta, batch_failed = process_batch(client, model, episode, title, batch, retries)
        for key, value in delta.items():
            stats[key] += value
        if batch_failed:
            failed.append(bi)
            continue
        newly_done.append(bi)
        for qa in qas:
            qa["id"] = f"{episode.get('episode')}-{ci:02d}-{local_idx:03d}"
            local_idx += 1
            qa["chapter"] = title
            qa["guest"] = episode.get("guest")
            qa["episode"] = episode.get("episode")
            new_faqs.append(qa)
    return {
        "title": title,
        "ci": ci,
        "new_faqs": new_faqs,
        "done_batches": newly_done,
        "failed_batches": failed,
        "stats": stats,
        "all_ok": not failed,
    }


def merge_chapter_result(faqs: list[dict], stats: dict, processed: set[str],
                         chapter_state: dict, result: dict) -> None:
    """Merge one chapter result into episode-level state (single-threaded)."""
    title = result["title"]
    prev = chapter_state.get(title, {})
    done_batches = sorted(set(prev.get("done_batches", [])) | set(result["done_batches"]))
    for key, value in result["stats"].items():
        stats[key] = stats.get(key, 0) + value
    faqs.extend(result["new_faqs"])
    if result["all_ok"]:
        processed.add(title)
        chapter_state.pop(title, None)
    else:
        chapter_state[title] = {
            "index": result["ci"],
            "done_batches": done_batches,
            "failed_batches": sorted(result["failed_batches"]),
        }


def build_payload(episode: dict, provider: str, model: str, faqs: list[dict],
                  processed: set[str], chapter_state: dict, stats: dict) -> dict:
    """Assemble the episode payload, with faqs sorted by chapter order."""
    tokens = [f["est_tokens"] for f in faqs]
    if tokens:
        stats["est_tokens_median"] = sorted(tokens)[len(tokens) // 2]
        stats["est_tokens_max"] = max(tokens)
        stats["est_over_500"] = sum(1 for t in tokens if t > 500)
    order = {c["title"]: i for i, c in enumerate(episode["chapters"])}
    faqs = sorted(faqs, key=lambda q: (order.get(q.get("chapter"), 999), q.get("id", "")))
    return {
        "episode": episode.get("episode"),
        "guest": episode.get("guest"),
        "provider": provider,
        "model": model,
        "faq_count": len(faqs),
        "processed_chapters": sorted(processed),
        "chapters": chapter_state,
        "stats": stats,
        "faqs": faqs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Q&A pairs from cleaned transcripts with an LLM.")
    parser.add_argument("--input", default=str(PROJECT_ROOT / "data/processed"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "data/faq"))
    parser.add_argument("--provider", choices=sorted(PROVIDERS), default=None,
                        help="openai | deepseek (default: inferred from 02_MODEL, then openai)")
    parser.add_argument("--force", action="store_true", help="regenerate existing files from scratch")
    parser.add_argument("--max-words", type=int, default=2500, help="max dialogue words per LLM batch")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=900, help="HTTP timeout in seconds per API call")
    parser.add_argument("--workers", type=int, default=6,
                        help="max chapters processed in parallel per episode")
    args = parser.parse_args()

    model_override = os.environ.get("02_MODEL", "")
    provider = resolve_provider(args.provider, model_override)
    cfg = PROVIDERS[provider]

    api_key = os.environ.get(cfg["api_key_env"], "")
    if not api_key:
        raise SystemExit(
            f"{cfg['api_key_env']} is not set. Put it in the environment or in the .env file."
        )
    model = model_override or os.environ.get(cfg["model_env"], cfg["model_default"])
    base_url = os.environ.get(cfg["base_url_env"], cfg["base_url_default"])

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=args.timeout)
    print(f"provider={provider} model={model} base_url={base_url} workers={args.workers}")

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_files = sorted(input_dir.glob("*.json"))
    if not episode_files:
        raise SystemExit(f"No cleaned episode files found in {input_dir}")

    for episode_file in episode_files:
        episode = json.loads(episode_file.read_text(encoding="utf-8"))
        out_file = output_dir / f"{episode_file.stem}.faq.json"

        chapter_titles = {c["title"] for c in episode["chapters"] if c.get("segments")}
        faqs: list[dict] = []
        processed: set[str] = set()
        stats: dict = {}
        chapter_state: dict = {}
        if out_file.exists() and not args.force:
            resume = json.loads(out_file.read_text(encoding="utf-8"))
            processed = set(resume.get("processed_chapters", []))
            stats = dict(resume.get("stats", {}))
            faqs = list(resume.get("faqs", []))
            chapter_state = dict(resume.get("chapters", {}))
            if chapter_titles <= processed:
                print(f"skip (already generated): {out_file.name}")
                continue
            # Old-format faqs carry no batch state -> regenerate their chapters
            known = processed | set(chapter_state)
            legacy = [q for q in faqs if q.get("chapter") not in known]
            if legacy:
                print(f"  drop {len(legacy)} legacy faq(s) without batch state "
                      f"(their chapters will be regenerated)")
                faqs = [q for q in faqs if q.get("chapter") in known]
            print(f"resume (incomplete) #{episode.get('episode')} {episode.get('guest')} "
                  f"({len(faqs)} faqs, {len(processed)}/{len(chapter_titles)} chapters)")
        else:
            print(f"episode #{episode.get('episode')} {episode.get('guest')} "
                  f"({len(episode['chapters'])} chapters, new)")

        chapters = episode["chapters"]
        pending = [
            (ci, c) for ci, c in enumerate(chapters, 1)
            if c.get("segments") and c["title"] not in processed
        ]
        if not pending:
            save_payload(out_file, build_payload(episode, provider, model, faqs, processed, chapter_state, stats))
            print(f"  wrote {out_file.name} ({len(faqs)} faqs)")
            continue

        failed_chapters: list[str] = []

        def finalize_chapter(future) -> None:
            result = future.result()
            merge_chapter_result(faqs, stats, processed, chapter_state, result)
            save_payload(out_file, build_payload(episode, provider, model, faqs, processed, chapter_state, stats))
            title = result["title"]
            chapter_total = sum(1 for q in faqs if q.get("chapter") == title)
            tag = "ok" if result["all_ok"] else "!! partial"
            print(f"  [{result['ci']}/{len(chapters)}] {title}: {chapter_total} faqs total ({tag})")
            if not result["all_ok"]:
                failed_chapters.append(title)

        if args.workers > 1 and len(pending) > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                future_map = {}
                for ci, chapter in pending:
                    title = chapter["title"]
                    start_faqs = [q for q in faqs if q.get("chapter") == title]
                    start_done = list(chapter_state.get(title, {}).get("done_batches", []))
                    future = pool.submit(process_chapter, client, model, episode, ci, title,
                                         chapter["segments"], args.max_words, args.retries,
                                         start_faqs, start_done)
                    future_map[future] = (ci, title)
                for future in as_completed(future_map):
                    finalize_chapter(future)
        else:
            for ci, chapter in pending:
                title = chapter["title"]
                start_faqs = [q for q in faqs if q.get("chapter") == title]
                start_done = list(chapter_state.get(title, {}).get("done_batches", []))
                result = process_chapter(client, model, episode, ci, title, chapter["segments"],
                                         args.max_words, args.retries, start_faqs, start_done)
                merge_chapter_result(faqs, stats, processed, chapter_state, result)
                save_payload(out_file, build_payload(episode, provider, model, faqs, processed, chapter_state, stats))
                chapter_total = sum(1 for q in faqs if q.get("chapter") == title)
                tag = "ok" if result["all_ok"] else "!! partial"
                print(f"  [{ci}/{len(chapters)}] {title}: {chapter_total} faqs total ({tag})")
                if not result["all_ok"]:
                    failed_chapters.append(title)

        print(f"  wrote {out_file.name} ({len(faqs)} faqs)")
        if failed_chapters:
            print(f"  !! {len(failed_chapters)} chapter(s) have failed batches "
                  f"(will retry on next run): {', '.join(failed_chapters)}")

    print("\nDone.")


if __name__ == "__main__":
    main()




