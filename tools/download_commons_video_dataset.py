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
USER_AGENT = "qjc-agent-video-dataset/1.0 (student multimodal asset library)"
DEFAULT_TOPICS = {
    "animals": ["cat", "dog", "bird"],
    "nature": ["waterfall", "ocean waves", "mountain landscape"],
    "city": ["city street", "traffic", "train station"],
    "sports": ["basketball", "running", "football"],
    "food": ["cooking", "coffee", "fruit"],
    "people": ["people walking", "dance", "classroom"],
    "transport": ["train", "bus", "bicycle"],
    "indoor": ["office", "museum", "library"],
    "education": ["classroom", "lecture", "student"],
    "performance": ["dance", "music performance", "stage"],
}
SUPPORTED_MIME = {"video/webm", "video/mp4", "video/quicktime", "video/x-msvideo", "video/x-matroska"}
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
    parser = argparse.ArgumentParser(description="Download a small broad video dataset from Wikimedia Commons.")
    parser.add_argument("--data-dir", default="data", help="Project data directory.")
    parser.add_argument("--dataset-dir", default="data/datasets/commons_videos", help="Download directory.")
    parser.add_argument("--per-topic", type=int, default=2, help="Videos to keep for each broad topic.")
    parser.add_argument("--max-total", type=int, default=24, help="Maximum videos to download/import.")
    parser.add_argument("--max-mb", type=float, default=15.0, help="Skip videos larger than this size.")
    parser.add_argument("--search-limit", type=int, default=25, help="Commons candidates per query.")
    parser.add_argument("--download-delay", type=float, default=1.5, help="Delay between downloads.")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    engine = VectorEngine(args.data_dir)
    upload_dir = Path(args.data_dir) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    downloaded_urls = set()
    total = 0
    for topic, queries in DEFAULT_TOPICS.items():
        topic_count = 0
        for query in queries:
            if topic_count >= args.per_topic or total >= args.max_total:
                break
            for candidate in search_commons(query, args.search_limit):
                if topic_count >= args.per_topic or total >= args.max_total:
                    break
                if candidate["url"] in downloaded_urls:
                    continue
                if not is_usable_candidate(candidate, args.max_mb):
                    continue

                target = dataset_dir / safe_filename(candidate["title"], candidate["url"])
                if not target.exists():
                    print(f"downloading {topic}: {candidate['title']} -> {target.name}")
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
                    asset = engine.add_asset(upload_target, datetime.now().isoformat(timespec="seconds"))
                    tags = sorted({"video", "commons", topic, *query.split()})
                    engine.set_asset_tags(asset.id, tags, source="commons_dataset", confidence=0.82)
                except Exception as exc:
                    print(f"failed to index {target.name}: {exc}")
                    continue

                downloaded_urls.add(candidate["url"])
                topic_count += 1
                total += 1
                manifest.append(
                    {
                        "topic": topic,
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
        "gsrsearch": f"filetype:video filemime:video/webm {query}",
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
    return rows


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


def download_file(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        with target.open("wb") as file:
            shutil.copyfileobj(response, file)


def safe_filename(title: str, url: str) -> str:
    suffix = Path(urllib.parse.urlparse(url).path).suffix.lower()
    stem = title.removeprefix("File:")
    stem = Path(stem).stem
    stem = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "_", stem).strip("_")
    stem = stem[:90] or "commons_video"
    return f"commons_{stem}{suffix}"


if __name__ == "__main__":
    main()
