from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vector_engine import VectorEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Import COCO instance category tags into SQLite metadata.")
    parser.add_argument(
        "--instances",
        default="data/datasets/coco/annotations/instances_val2017.json",
        help="Path to COCO instances_val2017.json.",
    )
    parser.add_argument("--data-dir", default="data", help="Project data directory.")
    args = parser.parse_args()

    engine = VectorEngine(args.data_dir)
    imported_ids = {asset.id for asset in engine.assets}
    raw = json.loads(Path(args.instances).read_text(encoding="utf-8"))
    categories = {item["id"]: item["name"] for item in raw["categories"]}
    image_tags: dict[str, set[str]] = defaultdict(set)

    for annotation in raw["annotations"]:
        asset_id = f"{annotation['image_id']:012d}"
        if asset_id in imported_ids:
            image_tags[asset_id].add(categories[annotation["category_id"]])

    tagged_assets = 0
    tag_edges = 0
    for asset_id, tags in image_tags.items():
        engine.metadata.set_asset_tags(asset_id, sorted(tags), source="coco_instances", confidence=1.0)
        tagged_assets += 1
        tag_edges += len(tags)

    print(f"tagged_assets={tagged_assets} tag_edges={tag_edges}")


if __name__ == "__main__":
    main()
