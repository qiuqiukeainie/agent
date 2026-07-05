from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vector_engine import VectorEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate image retrieval on imported COCO validation captions.")
    parser.add_argument(
        "--captions",
        default="data/datasets/coco/annotations/captions_val2017.json",
        help="Path to COCO captions_val2017.json.",
    )
    parser.add_argument("--data-dir", default="data", help="Project data directory.")
    parser.add_argument("--limit", type=int, default=100, help="Maximum number of caption queries to evaluate.")
    parser.add_argument("--top-k", type=int, default=10, help="Maximum rank to retrieve.")
    parser.add_argument("--out", default="data/eval/coco_retrieval_report.json", help="Output report path.")
    args = parser.parse_args()

    engine = VectorEngine(args.data_dir)
    captions = load_captions(Path(args.captions))
    imported_filenames = {asset.filename for asset in engine.assets}
    samples = [
        sample
        for sample in captions
        if coco_filename(sample["image_id"]) in imported_filenames
    ][: args.limit]

    if not samples:
        raise SystemExit("No evaluable COCO captions found for the current imported assets.")

    ranks = []
    examples = []
    for sample in samples:
        expected_filename = coco_filename(sample["image_id"])
        result = engine.agent_search(sample["caption"], limit=args.top_k)
        filenames = [item["filename"] for item in result["results"]]
        rank = filenames.index(expected_filename) + 1 if expected_filename in filenames else None
        ranks.append(rank)
        examples.append(
            {
                "query": sample["caption"],
                "expected": expected_filename,
                "rank": rank,
                "top_results": filenames[:5],
            }
        )

    report = {
        "query_count": len(samples),
        "asset_count": len(engine.assets),
        "backend": engine.vector_index.backend,
        "recall_at_1": recall_at(ranks, 1),
        "recall_at_5": recall_at(ranks, 5),
        "recall_at_10": recall_at(ranks, 10),
        "mrr_at_10": mrr_at(ranks, 10),
        "examples": examples[:20],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def load_captions(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw["annotations"]


def coco_filename(image_id: int) -> str:
    return f"{image_id:012d}.jpg"


def recall_at(ranks: list[int | None], k: int) -> float:
    hits = sum(1 for rank in ranks if rank is not None and rank <= k)
    return round(hits / len(ranks), 4)


def mrr_at(ranks: list[int | None], k: int) -> float:
    total = 0.0
    for rank in ranks:
        if rank is not None and rank <= k:
            total += 1.0 / rank
    return round(total / len(ranks), 4)


if __name__ == "__main__":
    main()
