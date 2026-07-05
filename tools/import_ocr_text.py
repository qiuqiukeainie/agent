from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vector_engine import VectorEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Import OCR text rows into SQLite metadata.")
    parser.add_argument("--csv", required=True, help="CSV with columns: asset_id,text,confidence")
    parser.add_argument("--engine", default="manual-ocr", help="OCR engine name.")
    parser.add_argument("--data-dir", default="data", help="Project data directory.")
    args = parser.parse_args()

    engine = VectorEngine(args.data_dir)
    count = 0
    with Path(args.csv).open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            asset_id = (row.get("asset_id") or "").strip()
            text = (row.get("text") or "").strip()
            if not asset_id or not text:
                continue
            confidence_text = (row.get("confidence") or "").strip()
            confidence = float(confidence_text) if confidence_text else None
            engine.metadata.upsert_ocr_text(asset_id, text, args.engine, confidence)
            count += 1

    print(f"imported_ocr_rows={count}")


if __name__ == "__main__":
    main()
