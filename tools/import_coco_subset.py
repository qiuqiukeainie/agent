from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vector_engine import SUPPORTED_IMAGE_TYPES, SUPPORTED_VIDEO_TYPES, VectorEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Import a local image/video subset into the demo asset library.")
    parser.add_argument("--images", required=True, help="Path to an image/video directory. Kept as --images for compatibility.")
    parser.add_argument("--limit", type=int, default=200, help="Maximum number of assets to import.")
    parser.add_argument("--data-dir", default="data", help="Project data directory.")
    args = parser.parse_args()

    image_dir = Path(args.images).expanduser().resolve()
    if not image_dir.exists():
        raise SystemExit(f"image directory does not exist: {image_dir}")

    engine = VectorEngine(args.data_dir)
    upload_dir = Path(args.data_dir) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    imported = 0
    skipped = 0
    for source in sorted(image_dir.iterdir()):
        if imported + skipped >= args.limit:
            break
        if source.suffix.lower() not in SUPPORTED_IMAGE_TYPES | SUPPORTED_VIDEO_TYPES:
            continue
        if (upload_dir / source.name).exists():
            skipped += 1
            continue

        target = unique_path(upload_dir / source.name)
        shutil.copy2(source, target)
        engine.add_asset(target, datetime.now().isoformat(timespec="seconds"))
        imported += 1

    print(f"imported {imported} assets into {upload_dir}; skipped {skipped} existing assets")


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    counter = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


if __name__ == "__main__":
    main()
