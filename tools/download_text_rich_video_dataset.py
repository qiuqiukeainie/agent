from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vector_engine import SUPPORTED_VIDEO_TYPES, VectorEngine


COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "qjc-agent-text-rich-video-dataset/1.0 (student multimodal asset library)"
SUPPORTED_MIME = {"video/webm", "video/mp4", "video/quicktime", "video/x-msvideo", "video/x-matroska"}
TEXT_RICH_QUERIES = [
    "interview",
    "oral history interview",
    "lecture",
    "presentation",
    "conference talk",
    "library narrator",
    "Wikipedia video",
    "subtitle",
    "captioned",
    "museum narrator",
    "city public libraries narrator",
    "educational video",
]
PREFERRED_TITLE_WORDS = {
    "interview",
    "oral",
    "history",
    "lecture",
    "presentation",
    "conference",
    "talk",
    "library",
    "narrator",
    "wikipedia",
    "subtitle",
    "caption",
    "museum",
    "education",
}
BLOCKED_TITLE_WORDS = {
    "death",
    "bodycam",
    "police",
    "penis",
    "surgery",
    "blood",
    "war",
    "violence",
    "execution",
    "corpse",
    "nude",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download text-rich videos suitable for OCR/subtitle extraction.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--dataset-dir", default="data/datasets/text_rich_videos")
    parser.add_argument("--max-total", type=int, default=16)
    parser.add_argument("--max-mb", type=float, default=30.0)
    parser.add_argument("--search-limit", type=int, default=35)
    parser.add_argument("--download-delay", type=float, default=1.0)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    upload_dir = Path(args.data_dir) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    engine = VectorEngine(args.data_dir)

    manifest = []
    seen_urls = set()
    for query in TEXT_RICH_QUERIES:
        if len(manifest) >= args.max_total:
            break
        for candidate in search_commons(query, args.search_limit):
            if len(manifest) >= args.max_total:
                break
            if candidate["url"] in seen_urls:
                continue
            if not is_usable_candidate(candidate, args.max_mb):
                continue

            target = dataset_dir / safe_filename(candidate["title"], candidate["url"])
            if not target.exists():
                print(f"downloading {query}: {candidate['title']} -> {target.name}")
                try:
                    download_file(candidate["url"], target)
                except Exception as exc:
                    print(f"skipped download {candidate['title']}: {exc}")
                    continue
                time.sleep(args.download_delay)

            upload_target = upload_dir / target.name
            if not upload_target.exists():
                shutil.copy2(target, upload_target)

            try:
                asset = engine.add_asset(upload_target, datetime.now().isoformat(timespec="seconds"), library="public")
                tags = sorted({"video", "commons", "text_rich", *query.split(), *preferred_tags(candidate["title"])})
                engine.set_asset_tags(asset.id, tags, source="commons_text_rich_dataset", confidence=0.84)
                engine.fuse_asset_tags(asset.id)
            except Exception as exc:
                print(f"failed to index {target.name}: {exc}")
                continue

            seen_urls.add(candidate["url"])
            manifest.append(
                {
                    "query": query,
                    "asset_id": asset.id,
                    "filename": asset.filename,
                    "source_title": candidate["title"],
                    "source_url": candidate["url"],
                    "mime": candidate["mime"],
                    "size": candidate["size"],
                }
            )

    manifest_path = dataset_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"downloaded_or_indexed={len(manifest)} manifest={manifest_path}")
    print(engine.library_stats())


def search_commons(query: str, limit: int) -> list[dict]:
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": f"filetype:video {query}",
        "gsrnamespace": "6",
        "gsrlimit": str(limit),
        "prop": "imageinfo",
        "iiprop": "url|size|mime",
        "format": "json",
        "formatversion": "2",
    }
    url = f"{COMMONS_API}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=45) as response:
        payload = json.loads(response.read().decode("utf-8"))

    rows = []
    for page in payload.get("query", {}).get("pages", []):
        info = (page.get("imageinfo") or [{}])[0]
        rows.append(
            {
                "title": page.get("title", ""),
                "url": info.get("url", ""),
                "size": int(info.get("size") or 0),
                "mime": info.get("mime", ""),
            }
        )
    return sorted(rows, key=score_candidate, reverse=True)


def score_candidate(candidate: dict) -> int:
    title = candidate.get("title", "").lower()
    score = 0
    for word in PREFERRED_TITLE_WORDS:
        if word in title:
            score += 3
    if "webm" in candidate.get("mime", ""):
        score += 1
    return score


def is_usable_candidate(candidate: dict, max_mb: float) -> bool:
    title = candidate["title"].lower()
    if any(word in title for word in BLOCKED_TITLE_WORDS):
        return False
    if candidate["mime"] not in SUPPORTED_MIME:
        return False
    suffix = Path(urllib.parse.urlparse(candidate["url"]).path).suffix.lower()
    if suffix not in SUPPORTED_VIDEO_TYPES:
        return False
    if candidate["size"] <= 0 or candidate["size"] > max_mb * 1024 * 1024:
        return False
    return True


def preferred_tags(title: str) -> set[str]:
    lower = title.lower()
    return {word for word in PREFERRED_TITLE_WORDS if word in lower}


def download_file(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=180) as response:
        with target.open("wb") as file:
            shutil.copyfileobj(response, file)


def safe_filename(title: str, url: str) -> str:
    suffix = Path(urllib.parse.urlparse(url).path).suffix.lower()
    stem = title.removeprefix("File:")
    stem = Path(stem).stem
    stem = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "_", stem).strip("_")
    stem = stem[:90] or "commons_text_video"
    return f"commons_text_{stem}{suffix}"


if __name__ == "__main__":
    main()
