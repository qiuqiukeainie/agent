from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vector_engine import SUPPORTED_IMAGE_TYPES, SUPPORTED_VIDEO_TYPES


@dataclass
class TextSignal:
    asset_id: str
    text_type: str
    text: str
    engine: str
    confidence: float | None


class OcrEngine:
    name = "ocr_unavailable"

    def available(self) -> bool:
        return False

    def read_image(self, image: Image.Image) -> tuple[str, float | None]:
        return "", None


class TesseractCliOcr(OcrEngine):
    name = "tesseract"

    def __init__(self, languages: str = "chi_sim+eng") -> None:
        self.languages = languages
        self.executable = shutil.which("tesseract")

    def available(self) -> bool:
        return self.executable is not None

    def read_image(self, image: Image.Image) -> tuple[str, float | None]:
        if not self.executable:
            return "", None
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "frame.png"
            output_base = Path(tmp) / "ocr"
            image.save(image_path)
            completed = subprocess.run(
                [self.executable, str(image_path), str(output_base), "-l", self.languages, "--psm", "6"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
            )
            if completed.returncode != 0:
                return "", None
            output_path = output_base.with_suffix(".txt")
            if not output_path.exists():
                return "", None
            return clean_text(output_path.read_text(encoding="utf-8", errors="replace")), None


class EasyOcrEngine(OcrEngine):
    name = "easyocr"

    def __init__(self, languages: list[str] | None = None, model_dir: Path | None = None, max_side: int = 960) -> None:
        self.languages = languages or ["ch_sim", "en"]
        self.model_dir = model_dir or ROOT / "data" / "models" / "easyocr"
        self.max_side = max_side
        self.reader = None

    def available(self) -> bool:
        try:
            import easyocr  # noqa: F401
        except Exception:
            return False
        return True

    def read_image(self, image: Image.Image) -> tuple[str, float | None]:
        import easyocr

        if self.reader is None:
            self.model_dir.mkdir(parents=True, exist_ok=True)
            self.reader = easyocr.Reader(
                self.languages,
                gpu=False,
                model_storage_directory=str(self.model_dir),
                user_network_directory=str(self.model_dir),
            )
        image = resize_for_ocr(image, self.max_side)
        rows = self.reader.readtext(image_to_array(image), detail=1, paragraph=False)
        texts = []
        scores = []
        for row in rows:
            if len(row) >= 3:
                texts.append(str(row[1]))
                scores.append(float(row[2]))
            elif len(row) >= 2:
                texts.append(str(row[1]))
        return clean_text(" ".join(texts)), average(scores)


class PaddleOcrEngine(OcrEngine):
    name = "paddleocr"

    def __init__(self) -> None:
        self.reader = None

    def available(self) -> bool:
        try:
            import paddleocr  # noqa: F401
        except Exception:
            return False
        return True

    def read_image(self, image: Image.Image) -> tuple[str, float | None]:
        from paddleocr import PaddleOCR

        if self.reader is None:
            self.reader = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
        rows = self.reader.ocr(image_to_array(image), cls=True)
        texts = []
        scores = []
        for page in rows or []:
            for item in page or []:
                if len(item) >= 2 and len(item[1]) >= 2:
                    texts.append(str(item[1][0]))
                    scores.append(float(item[1][1]))
        return clean_text(" ".join(texts)), average(scores)


class WhisperAsr:
    def __init__(self, model_name: str = "tiny", model_dir: Path | None = None) -> None:
        self.model_name = model_name
        self.model_dir = model_dir or ROOT / "data" / "models" / "whisper"
        self.model = None
        self.backend = None

    def available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401

            self.backend = "faster_whisper"
            return True
        except Exception:
            pass
        try:
            import whisper  # noqa: F401

            self.backend = "whisper"
            return True
        except Exception:
            return False

    def transcribe(self, path: Path) -> tuple[str, float | None]:
        if self.backend == "faster_whisper":
            from faster_whisper import WhisperModel

            if self.model is None:
                self.model_dir.mkdir(parents=True, exist_ok=True)
                self.model = WhisperModel(
                    self.model_name,
                    device="cpu",
                    compute_type="int8",
                    download_root=str(self.model_dir),
                )
            segments, _ = self.model.transcribe(str(path))
            return clean_text(" ".join(segment.text for segment in segments)), None
        if self.backend == "whisper":
            import whisper

            if self.model is None:
                self.model = whisper.load_model(self.model_name)
            result = self.model.transcribe(str(path))
            return clean_text(str(result.get("text", ""))), None
        return "", None


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract OCR/subtitle/ASR text signals for C module.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--library", choices=["all", "personal", "public"], default="all")
    parser.add_argument("--kind", choices=["all", "image", "video"], default="all")
    parser.add_argument("--filename-contains", default="")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--ocr-engine", choices=["auto", "tesseract", "easyocr", "paddleocr", "none"], default="auto")
    parser.add_argument("--asr-engine", choices=["auto", "whisper", "none"], default="auto")
    parser.add_argument("--asr-model", default="tiny")
    parser.add_argument("--video-frames", type=int, default=2)
    parser.add_argument("--min-confidence", type=float, default=0.45)
    parser.add_argument("--output", default="data/text_signals.csv")
    parser.add_argument("--import-signals", action="store_true", help="Write extracted signals into SQLite and refresh fusion tags.")
    parser.add_argument("--capabilities", action="store_true", help="Only print available OCR/ASR capability status.")
    args = parser.parse_args()

    capability = capability_status()
    if args.capabilities:
        print(json.dumps(capability, ensure_ascii=False, indent=2))
        return

    ocr = build_ocr_engine(args.ocr_engine)
    ocr_enabled = args.ocr_engine != "none"
    asr = build_asr_engine(args.asr_engine, args.asr_model)
    asr_enabled = args.asr_engine != "none"
    assets = select_assets(
        load_assets(Path(args.data_dir)),
        args.library,
        args.kind,
        args.filename_contains,
        args.offset,
        args.limit,
    )
    signals: list[TextSignal] = []
    skipped = []

    for asset in assets:
        path = resolve_asset_path(asset)
        if not path.exists():
            skipped.append({"asset_id": asset["id"], "reason": "file_missing"})
            continue
        try:
            if asset["kind"] == "image":
                if not ocr.available():
                    skipped.append({"asset_id": asset["id"], "reason": f"ocr_engine_unavailable:{ocr.name}"})
                    continue
                text, confidence = ocr.read_image(load_image(path))
                append_signal(signals, asset["id"], "ocr", text, ocr.name, confidence, args.min_confidence)
            elif asset["kind"] == "video":
                if ocr_enabled and ocr.available():
                    subtitle_texts = []
                    scores = []
                    for frame in sample_video_frames(path, args.video_frames):
                        text, confidence = ocr.read_image(frame)
                        if text:
                            subtitle_texts.append(text)
                        if confidence is not None:
                            scores.append(confidence)
                    append_signal(
                        signals,
                        asset["id"],
                        "subtitle_ocr",
                        dedupe_texts(subtitle_texts),
                        ocr.name,
                        average(scores),
                        args.min_confidence,
                    )
                elif ocr_enabled:
                    skipped.append({"asset_id": asset["id"], "reason": f"subtitle_ocr_unavailable:{ocr.name}"})
                if asr_enabled and asr.available():
                    text, confidence = asr.transcribe(path)
                    append_signal(signals, asset["id"], "asr", text, f"whisper:{asr.backend}", confidence, args.min_confidence)
                elif asr_enabled:
                    skipped.append({"asset_id": asset["id"], "reason": "asr_unavailable"})
        except Exception as exc:
            skipped.append({"asset_id": asset["id"], "reason": str(exc)})

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output_path, signals)
    imported = 0
    if args.import_signals and signals:
        imported = import_signals(args.data_dir, signals)

    print(
        json.dumps(
            {
                "assets": len(assets),
                "signals": len(signals),
                "imported": imported,
                "output": str(output_path),
                "capabilities": capability,
                "skipped": skipped[:30],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def capability_status() -> dict:
    return {
        "tesseract_cli": shutil.which("tesseract") is not None,
        "easyocr": module_available("easyocr"),
        "paddleocr": module_available("paddleocr"),
        "whisper": module_available("whisper"),
        "faster_whisper": module_available("faster_whisper"),
        "av": module_available("av"),
    }


def module_available(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def build_ocr_engine(name: str) -> OcrEngine:
    candidates: list[OcrEngine]
    if name == "auto":
        candidates = [PaddleOcrEngine(), EasyOcrEngine(), TesseractCliOcr()]
    elif name == "paddleocr":
        candidates = [PaddleOcrEngine()]
    elif name == "easyocr":
        candidates = [EasyOcrEngine()]
    elif name == "tesseract":
        candidates = [TesseractCliOcr()]
    else:
        candidates = [OcrEngine()]
    available = next((engine for engine in candidates if engine.available()), None)
    if available is not None:
        return available
    missing = OcrEngine()
    missing.name = f"{name}:no_engine"
    return missing


def build_asr_engine(name: str, model_name: str) -> WhisperAsr:
    asr = WhisperAsr(model_name=model_name)
    if name == "none":
        asr.backend = None
        return asr
    asr.available()
    return asr


def load_assets(data_dir: Path) -> list[dict]:
    index_path = data_dir / "index.json"
    if not index_path.exists():
        return []
    return json.loads(index_path.read_text(encoding="utf-8"))


def select_assets(
    assets: list[dict],
    library: str,
    kind: str,
    filename_contains: str,
    offset: int,
    limit: int,
) -> list[dict]:
    rows = []
    filename_contains = filename_contains.lower().strip()
    for asset in assets:
        if library != "all" and asset.get("library") != library:
            continue
        if kind != "all" and asset.get("kind") != kind:
            continue
        if filename_contains and filename_contains not in str(asset.get("filename", "")).lower():
            continue
        rows.append(asset)
    rows = rows[max(offset, 0):]
    return rows[:limit] if limit > 0 else rows


def resolve_asset_path(asset: dict) -> Path:
    path = Path(asset.get("path", ""))
    return path if path.is_absolute() else ROOT / path


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def resize_for_ocr(image: Image.Image, max_side: int) -> Image.Image:
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image.convert("RGB")
    scale = max_side / longest
    return image.resize((max(int(width * scale), 1), max(int(height * scale), 1)), Image.Resampling.LANCZOS).convert("RGB")


def sample_video_frames(path: Path, frame_count: int) -> Iterable[Image.Image]:
    try:
        import av
    except Exception:
        return []
    frames = []
    try:
        with av.open(str(path)) as container:
            stream = next((item for item in container.streams if item.type == "video"), None)
            if stream is None:
                return []
            wanted = max(frame_count, 1)
            indexes = {0}
            decoded = 0
            for frame in container.decode(stream):
                if decoded % max(1, 30 // wanted) == 0:
                    frames.append(frame.to_image().convert("RGB"))
                    if len(frames) >= wanted:
                        break
                decoded += 1
    except Exception:
        return []
    return frames


def image_to_array(image: Image.Image):
    import numpy as np

    return np.asarray(image.convert("RGB"))


def append_signal(
    signals: list[TextSignal],
    asset_id: str,
    text_type: str,
    text: str,
    engine: str,
    confidence: float | None,
    min_confidence: float,
) -> None:
    text = clean_text(text)
    if len(text) < 2:
        return
    if confidence is not None and confidence < min_confidence:
        return
    signals.append(TextSignal(asset_id, text_type, text, engine, confidence))


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    text = re.sub(r"([^\w\u4e00-\u9fff])\1{2,}", r"\1", text)
    return text[:2000]


def dedupe_texts(texts: list[str]) -> str:
    seen = set()
    result = []
    for text in texts:
        text = clean_text(text)
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return " ".join(result)


def average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def write_csv(path: Path, signals: list[TextSignal]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["asset_id", "text_type", "text", "engine", "confidence"])
        writer.writeheader()
        for signal in signals:
            writer.writerow(
                {
                    "asset_id": signal.asset_id,
                    "text_type": signal.text_type,
                    "text": signal.text,
                    "engine": signal.engine,
                    "confidence": "" if signal.confidence is None else round(signal.confidence, 4),
                }
            )


def import_signals(data_dir: str, signals: list[TextSignal]) -> int:
    from vector_engine import VectorEngine

    engine = VectorEngine(data_dir)
    imported = 0
    for signal in signals:
        engine.add_text_signal(
            asset_id=signal.asset_id,
            text=signal.text,
            text_type=signal.text_type,
            engine=signal.engine,
            confidence=signal.confidence,
            auto_fuse=True,
        )
        imported += 1
    return imported


if __name__ == "__main__":
    main()
