from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import ENGINE, ensure_thumbnail


def main() -> None:
    parser = argparse.ArgumentParser(description="Prebuild cached thumbnails for existing assets.")
    parser.add_argument("--library", choices=["all", "personal", "public"], default="all")
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit.")
    args = parser.parse_args()

    assets = ENGINE.list_assets()
    if args.library != "all":
        assets = [asset for asset in assets if asset.get("library") == args.library]
    if args.limit > 0:
        assets = assets[: args.limit]

    built = 0
    failed = []
    for asset in assets:
        try:
            ensure_thumbnail(asset["id"])
            built += 1
        except Exception as exc:
            failed.append({"id": asset["id"], "error": str(exc)})

    print({"built": built, "failed": failed[:20], "total": len(assets)})


if __name__ == "__main__":
    main()
