from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vector_engine import SUPPORTED_VIDEO_TYPES, VectorEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Import local video files into the asset library.")
    parser.add_argument("--videos", required=True, help="Path to a directory containing video files.")
    parser.add_argument("--limit", type=int, default=100, help="Maximum number of videos to import.")
    parser.add_argument("--data-dir", default="data", help="Project data directory.")
    parser.add_argument("--library", default="public", choices=["public", "personal"], help="Target library.")
    parser.add_argument("--frame-count", type=int, default=3, help="Frames sampled per new video.")
    parser.add_argument(
        "--reindex-existing",
        action="store_true",
        help="Recompute vectors for videos that are already in the library.",
    )
    parser.add_argument(
        "--tag-source",
        default="video_dataset",
        help="Optional tag source used for imported videos.",
    )
    parser.add_argument(
        "--tags",
        default="video",
        help="Comma-separated tags added to imported videos. Use an empty string to skip.",
    )
    args = parser.parse_args()

    video_dir = Path(args.videos).expanduser().resolve()
    if not video_dir.exists():
        raise SystemExit(f"video directory does not exist: {video_dir}")

    engine = VectorEngine(args.data_dir)
    existing_filenames = {asset.filename for asset in engine.assets}
    upload_dir = Path(args.data_dir) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    indexed = 0
    copied = 0
    skipped = 0
    failed = []
    for source in sorted(video_dir.iterdir()):
        if indexed >= args.limit:
            break
        if source.suffix.lower() not in SUPPORTED_VIDEO_TYPES:
            continue
        if source.name in existing_filenames and not args.reindex_existing:
            skipped += 1
            continue
        target = upload_dir / source.name
        if target.exists():
            skipped += 1
        else:
            shutil.copy2(source, target)
            copied += 1
        try:
            asset = engine.add_video(
                target,
                datetime.now().isoformat(timespec="seconds"),
                frame_count=args.frame_count,
                library=args.library,
            )
            existing_filenames.add(asset.filename)
            tags = [tag.strip() for tag in args.tags.split(",") if tag.strip()]
            if tags:
                engine.set_asset_tags(asset.id, tags, source=args.tag_source, confidence=0.82)
                engine.fuse_asset_tags(asset.id)
            indexed += 1
        except Exception as exc:
            failed.append({"filename": source.name, "error": str(exc)})

    print(f"indexed={indexed} copied={copied} already_present={skipped} failed={failed}")
    print(engine.library_stats())


if __name__ == "__main__":
    main()
