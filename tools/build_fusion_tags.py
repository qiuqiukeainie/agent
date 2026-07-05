from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vector_engine import VectorEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified fusion tags from visual, OCR/ASR, person and metadata tags.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--library", choices=["all", "personal", "public"], default="all")
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit.")
    args = parser.parse_args()

    engine = VectorEngine(args.data_dir)
    assets = engine.list_assets()
    if args.library != "all":
        assets = [asset for asset in assets if asset.get("library") == args.library]
    if args.limit > 0:
        assets = assets[: args.limit]

    written = 0
    failed = []
    for asset in assets:
        try:
            result = engine.fuse_asset_tags(asset["id"])
            written += result["written_count"]
        except Exception as exc:
            failed.append({"asset_id": asset["id"], "error": str(exc)})

    print({"assets": len(assets), "written_tags": written, "failed": failed[:20]})


if __name__ == "__main__":
    main()
