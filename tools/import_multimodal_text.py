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
    parser = argparse.ArgumentParser(description="Import OCR/subtitle/ASR text signals and refresh fusion tags.")
    parser.add_argument("csv_path", help="CSV with columns: asset_id,text_type,text,engine,confidence")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--no-fuse", action="store_true", help="Do not refresh fusion tags after import.")
    args = parser.parse_args()

    engine = VectorEngine(args.data_dir)
    imported = 0
    failed = []
    with Path(args.csv_path).open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            try:
                confidence = row.get("confidence")
                result = engine.add_text_signal(
                    asset_id=row.get("asset_id", ""),
                    text=row.get("text", ""),
                    text_type=row.get("text_type", "ocr"),
                    engine=row.get("engine", "csv-import"),
                    confidence=float(confidence) if confidence else None,
                    auto_fuse=not args.no_fuse,
                )
                imported += 1 if result["text_length"] else 0
            except Exception as exc:
                failed.append({"asset_id": row.get("asset_id", ""), "error": str(exc)})

    print({"imported": imported, "failed": failed[:20]})


if __name__ == "__main__":
    main()
