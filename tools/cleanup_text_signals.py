from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vector_engine import VectorEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove low-confidence OCR/ASR text signals and refresh fusion tags.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--engine", default="subtitle_ocr:easyocr")
    parser.add_argument("--below", type=float, default=0.45)
    args = parser.parse_args()

    db_path = Path(args.data_dir) / "assets.db"
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT asset_id
            FROM ocr_texts
            WHERE engine = ? AND confidence IS NOT NULL AND confidence < ?
            """,
            (args.engine, args.below),
        ).fetchall()
        conn.execute(
            """
            DELETE FROM ocr_texts
            WHERE engine = ? AND confidence IS NOT NULL AND confidence < ?
            """,
            (args.engine, args.below),
        )

    engine = VectorEngine(args.data_dir)
    refreshed = []
    for (asset_id,) in rows:
        engine.fuse_asset_tags(asset_id)
        refreshed.append(asset_id)
    print({"removed_assets": refreshed, "count": len(refreshed)})


if __name__ == "__main__":
    main()
