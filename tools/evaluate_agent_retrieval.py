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
    parser = argparse.ArgumentParser(description="Evaluate Agent retrieval with a small labeled query set.")
    parser.add_argument("--queries", required=True, help="JSON file with query, relevant_ids, optional library fields.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--profiles", nargs="+", default=["clip_only", "clip_tags", "agent_v3"])
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    query_rows = json.loads(Path(args.queries).read_text(encoding="utf-8"))
    engine = VectorEngine(args.data_dir)
    report = engine.evaluate_retrieval(query_rows, limit=args.limit, ks=args.k, profiles=args.profiles)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
