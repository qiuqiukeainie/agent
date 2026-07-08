from __future__ import annotations

import json
import mimetypes
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import uuid
from datetime import datetime
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
from PIL import Image, ImageOps

from vector_engine import (
    SUPPORTED_DOCUMENT_TYPES,
    SUPPORTED_IMAGE_TYPES,
    SUPPORTED_VIDEO_TYPES,
    VectorEngine,
    cosine_similarity,
)
from chat_engine import route_chat, session_store


ROOT = Path(__file__).parent.resolve()
STATIC_DIR = ROOT / "static"
DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
THUMB_DIR = DATA_DIR / "thumbs"
VIDEO_SAMPLE_DIR = DATA_DIR / "video_samples"
ENGINE = VectorEngine(str(DATA_DIR))
TASKS: dict[str, dict] = {}
TASK_LOCK = threading.Lock()
UPLOAD_TAG_VECTOR_CACHE: dict[str, list[np.ndarray]] | None = None
UPLOAD_TAG_LOCK = threading.Lock()

UPLOAD_TAG_PROMPTS = {
    "person": ["a person", "people", "portrait photo"],
    "group": ["a group of people", "group photo"],
    "food": ["food", "meal", "dish"],
    "coffee": ["coffee cup", "coffee drink"],
    "fruit": ["fruit", "fresh fruit"],
    "city": ["city street", "urban scene"],
    "street": ["street scene", "road in a city"],
    "building": ["building architecture", "architecture"],
    "nature": ["nature landscape", "outdoor natural scene"],
    "sky": ["blue sky", "sky outdoors"],
    "mountain": ["mountain landscape", "mountains"],
    "ocean": ["ocean waves", "sea waves"],
    "waterfall": ["waterfall", "falling water in nature"],
    "flower": ["flower", "flowers"],
    "cat": ["a cat", "a pet cat"],
    "dog": ["a dog", "a pet dog"],
    "car": ["car", "vehicle"],
    "bus": ["bus", "public bus"],
    "train": ["train", "railway train"],
    "sports": ["sports activity", "people playing sports"],
    "basketball": ["basketball game", "a person playing basketball"],
    "night": ["night scene", "dark night"],
    "snow": ["snow", "snowy scene"],
    "office": ["office room", "workplace"],
    "library": ["library", "bookshelves in a library"],
    "stage": ["stage performance", "performance on stage"],
}


class AppHandler(SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            return self.serve_file(STATIC_DIR / "index.html", no_cache=True)
        if parsed.path == "/e":
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", "/search")
            self.end_headers()
            return
        if parsed.path in {"/search", "/library", "/tags", "/people", "/tasks", "/system"}:
            return self.serve_file(STATIC_DIR / "e-index.html", no_cache=True)
        if parsed.path == "/api/assets":
            params = parse_qs(parsed.query)
            return self.send_json(list_assets_response(params))
        asset_match = re.fullmatch(r"/api/asset/([^/]+)", parsed.path)
        if asset_match:
            try:
                return self.send_json(ENGINE.asset_detail(unquote(asset_match.group(1))))
            except Exception as exc:
                return self.send_json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
        if parsed.path == "/api/asset":
            asset_id = parse_qs(parsed.query).get("id", [""])[0]
            try:
                return self.send_json({"asset": ENGINE.asset_detail(asset_id)})
            except Exception as exc:
                return self.send_json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
        if parsed.path == "/api/tag-assets":
            params = parse_qs(parsed.query)
            tag = params.get("tag", [""])[0]
            library = params.get("library", ["all"])[0]
            return self.send_json({"assets": ENGINE.tag_browser(tag=tag, library=library, limit=300).get("assets", [])})
        if parsed.path == "/api/status":
            return self.send_json(ENGINE.model_status())
        if parsed.path == "/api/stats":
            return self.send_json(ENGINE.library_stats())
        if parsed.path == "/api/duplicates":
            return self.send_json({"groups": ENGINE.duplicate_groups()})
        if parsed.path == "/api/search-logs":
            return self.send_json({"logs": ENGINE.recent_search_logs()})
        if parsed.path == "/api/agent-recommendations":
            return self.send_json(ENGINE.agent_recommendations())
        if parsed.path == "/api/evaluate-retrieval":
            params = parse_qs(parsed.query)
            limit = parse_positive_int(params.get("limit", ["20"])[0], default=20, minimum=1, maximum=80)
            return self.send_json(load_retrieval_evaluation(limit=limit))
        if parsed.path == "/api/tags":
            params = parse_qs(parsed.query)
            library = params.get("library", ["all"])[0]
            tag = params.get("tag", [""])[0]
            limit = parse_positive_int(params.get("limit", ["120"])[0], default=120, minimum=20, maximum=300)
            return self.send_json(ENGINE.tag_browser(tag=tag, library=library, limit=limit))
        if parsed.path == "/api/tasks":
            return self.send_json({"tasks": list_tasks()})
        if parsed.path == "/api/persons":
            library = parse_qs(parsed.query).get("library", ["personal"])[0]
            return self.send_json({"persons": ENGINE.list_persons(library)})
        if parsed.path == "/api/search":
            params = parse_qs(parsed.query)
            query = params.get("q", [""])[0]
            limit = parse_positive_int(params.get("limit", ["36"])[0], default=36, minimum=1, maximum=80)
            library = params.get("library", ["all"])[0]
            return self.send_json(ENGINE.agent_search(query, limit=limit, library=library))
        if parsed.path == "/api/chat/sessions":
            return self.send_json({"sessions": session_store.list_sessions()})
        if parsed.path == "/api/document-search":
            params = parse_qs(parsed.query)
            query = params.get("q", [""])[0]
            limit = parse_positive_int(params.get("limit", ["12"])[0], default=12, minimum=1, maximum=50)
            library = params.get("library", ["all"])[0]
            return self.send_json(ENGINE.document_search(query, limit=limit, library=library))
        if parsed.path.startswith("/uploads/"):
            filename = Path(parsed.path).name
            return self.serve_file(UPLOAD_DIR / filename)
        if parsed.path.startswith("/thumbs/"):
            asset_id = parse_thumbnail_asset_id(parsed.path)
            try:
                return self.serve_file(ensure_thumbnail(asset_id), no_cache=False)
            except Exception:
                return self.send_error(HTTPStatus.NOT_FOUND)
        if parsed.path.startswith("/static/"):
            return self.serve_file(STATIC_DIR / Path(parsed.path).name, no_cache=True)
        if parsed.path.startswith("/assets/"):
            return self.serve_file(STATIC_DIR / "assets" / Path(parsed.path).name, no_cache=True)
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/upload":
            return self.handle_upload()
        tag_match = re.fullmatch(r"/api/asset/([^/]+)/tags", parsed.path)
        if tag_match:
            payload = self.read_json_body()
            try:
                result = ENGINE.set_asset_tags(
                    asset_id=unquote(tag_match.group(1)),
                    tags=[str(item) for item in payload.get("tags", [])],
                    source="manual",
                    confidence=1.0,
                    replace_source=True,
                )
                ENGINE.fuse_asset_tags(unquote(tag_match.group(1)))
            except Exception as exc:
                return self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return self.send_json({"ok": True, "result": result})
        if parsed.path == "/api/assets/batch":
            payload = self.read_json_body()
            try:
                return self.send_json(batch_operation(payload))
            except Exception as exc:
                return self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        if parsed.path == "/api/tasks":
            payload = self.read_json_body()
            task = submit_processing_task({"mode": payload.get("type", "ocr")})
            return self.send_json({"task_id": task["id"], "task": task})
        if parsed.path == "/api/persons/merge":
            payload = self.read_json_body()
            person_ids = [str(item) for item in payload.get("person_ids", []) if item]
            if len(person_ids) < 2:
                return self.send_json({"error": "person_ids requires at least two ids"}, HTTPStatus.BAD_REQUEST)
            try:
                result = ENGINE.merge_persons(person_ids[0], person_ids[1:])
            except Exception as exc:
                return self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return self.send_json({"ok": True, "result": result})
        if parsed.path == "/api/reindex-metadata":
            ENGINE.reindex_metadata()
            return self.send_json({"ok": True, "stats": ENGINE.library_stats()})
        if parsed.path == "/api/rebuild-index":
            return self.send_json(ENGINE.rebuild_index_from_uploads())
        if parsed.path == "/api/rebuild-document-chunks":
            payload = self.read_json_body()
            try:
                result = ENGINE.rebuild_document_chunks(
                    limit=parse_optional_positive_int(payload.get("limit"), default=None, minimum=1, maximum=500),
                    library=str(payload.get("library", "all")),
                )
            except Exception as exc:
                return self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return self.send_json({"ok": True, "result": result, "stats": ENGINE.library_stats()})
        if parsed.path == "/api/import-video-samples":
            return self.send_json(import_video_samples())
        if parsed.path == "/api/rebuild-persons":
            return self.send_json(rebuild_person_clusters())
        if parsed.path == "/api/repair-person-tags":
            payload = self.read_json_body()
            try:
                result = ENGINE.repair_person_tags(str(payload.get("library", "personal")))
            except Exception as exc:
                return self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return self.send_json({"ok": True, "result": result, "stats": ENGINE.library_stats()})
        if parsed.path == "/api/delete-asset":
            payload = self.read_json_body()
            try:
                result = ENGINE.delete_asset(str(payload.get("asset_id", "")), delete_file=bool(payload.get("delete_file", True)))
            except Exception as exc:
                return self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return self.send_json({"ok": True, "result": result, "stats": ENGINE.library_stats()})
        if parsed.path == "/api/favorite":
            payload = self.read_json_body()
            try:
                result = ENGINE.set_favorite(
                    asset_id=str(payload.get("asset_id", "")),
                    favorite=bool(payload.get("favorite", True)),
                )
            except Exception as exc:
                return self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return self.send_json({"ok": True, "result": result})
        if parsed.path == "/api/apply-archive-suggestion":
            payload = self.read_json_body()
            try:
                result = ENGINE.apply_archive_suggestion(
                    asset_id=str(payload.get("asset_id", "")),
                    suggestion_key=str(payload.get("suggestion_key", "")),
                )
            except Exception as exc:
                return self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return self.send_json({"ok": True, "result": result})
        if parsed.path == "/api/merge-persons":
            payload = self.read_json_body()
            try:
                result = ENGINE.merge_persons(
                    target_person_id=str(payload.get("target_person_id", "")),
                    source_person_ids=[str(item) for item in payload.get("source_person_ids", [])],
                    library=str(payload.get("library", "personal")),
                )
            except Exception as exc:
                return self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return self.send_json({"ok": True, "result": result, "stats": ENGINE.library_stats()})
        if parsed.path == "/api/delete-person":
            payload = self.read_json_body()
            try:
                result = ENGINE.delete_person(
                    person_id=str(payload.get("person_id", "")),
                    library=str(payload.get("library", "personal")),
                )
            except Exception as exc:
                return self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return self.send_json({"ok": True, "result": result, "stats": ENGINE.library_stats()})
        if parsed.path == "/api/remove-asset-person":
            payload = self.read_json_body()
            try:
                result = ENGINE.remove_asset_person(
                    asset_id=str(payload.get("asset_id", "")),
                    person_id=str(payload.get("person_id", "")),
                    library=str(payload.get("library", "personal")),
                )
            except Exception as exc:
                return self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return self.send_json({"ok": True, "result": result, "stats": ENGINE.library_stats()})
        if parsed.path == "/api/rename-person":
            payload = self.read_json_body()
            try:
                result = ENGINE.rename_person(
                    person_id=str(payload.get("person_id", "")),
                    display_name=str(payload.get("display_name", "")),
                    library=str(payload.get("library", "personal")),
                )
            except Exception as exc:
                return self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return self.send_json({"ok": True, "result": result})
        if parsed.path == "/api/cleanup-persons":
            payload = self.read_json_body()
            try:
                result = ENGINE.cleanup_person_clusters(str(payload.get("library", "personal")))
            except Exception as exc:
                return self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return self.send_json({"ok": True, "result": result, "stats": ENGINE.library_stats()})
        if parsed.path == "/api/ocr-text":
            payload = self.read_json_body()
            try:
                result = ENGINE.add_ocr_text(
                    asset_id=str(payload.get("asset_id", "")),
                    text=str(payload.get("text", "")),
                    engine=str(payload.get("engine", "external-ocr")),
                    confidence=payload.get("confidence"),
                )
            except Exception as exc:
                return self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return self.send_json({"ok": True, "result": result})
        if parsed.path == "/api/multimodal-text":
            payload = self.read_json_body()
            try:
                result = ENGINE.add_text_signal(
                    asset_id=str(payload.get("asset_id", "")),
                    text=str(payload.get("text", "")),
                    text_type=str(payload.get("text_type", "ocr")),
                    engine=str(payload.get("engine", "external")),
                    confidence=payload.get("confidence"),
                    auto_fuse=bool(payload.get("auto_fuse", True)),
                )
            except Exception as exc:
                return self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return self.send_json({"ok": True, "result": result})
        if parsed.path == "/api/asset-tags":
            payload = self.read_json_body()
            try:
                results = write_asset_tags(payload)
            except Exception as exc:
                return self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return self.send_json({"ok": True, "results": results})
        if parsed.path == "/api/fuse-tags":
            payload = self.read_json_body()
            try:
                asset_ids = [str(item) for item in payload.get("asset_ids", [])]
                if not asset_ids and payload.get("asset_id"):
                    asset_ids = [str(payload.get("asset_id"))]
                results = [ENGINE.fuse_asset_tags(asset_id) for asset_id in asset_ids]
            except Exception as exc:
                return self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return self.send_json({"ok": True, "results": results})
        if parsed.path == "/api/document-qa":
            payload = self.read_json_body()
            try:
                result = ENGINE.document_qa(
                    question=str(payload.get("question", "")),
                    top_k=parse_optional_positive_int(payload.get("top_k"), default=6, minimum=1, maximum=12) or 6,
                    library=str(payload.get("library", "all")),
                )
            except Exception as exc:
                return self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return self.send_json({"ok": True, "result": result})
        if parsed.path == "/api/process-assets":
            payload = self.read_json_body()
            task = submit_processing_task(payload)
            return self.send_json({"ok": True, "task": task})
        if parsed.path == "/api/chat":
            payload = self.read_json_body()
            try:
                limit = parse_positive_int(str(payload.get("limit", "36")), default=36, minimum=1, maximum=80)
                result = route_chat(
                    query=str(payload.get("query", "")),
                    session_id=payload.get("session_id") or None,
                    library=str(payload.get("library", "all")),
                    engine=ENGINE,
                    limit=limit,
                )
            except Exception as exc:
                return self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return self.send_json(result)
        if parsed.path == "/api/chat/clear":
            payload = self.read_json_body()
            session_id = str(payload.get("session_id", ""))
            if session_id:
                session_store.delete_session(session_id)
            return self.send_json({"ok": True})
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        asset_match = re.fullmatch(r"/api/asset/([^/]+)", parsed.path)
        if asset_match:
            try:
                result = ENGINE.delete_asset(unquote(asset_match.group(1)), delete_file=True)
            except Exception as exc:
                return self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return self.send_json({"ok": True, "result": result, "stats": ENGINE.library_stats()})
        self.send_error(HTTPStatus.NOT_FOUND)

    def handle_upload(self) -> None:
        content_type = self.headers.get("Content-Type", "")
        boundary_match = re.search("boundary=(.+)", content_type)
        if not boundary_match:
            return self.send_json({"error": "缺少上传边界"}, HTTPStatus.BAD_REQUEST)

        boundary = boundary_match.group(1).encode()
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        uploaded = []
        upload_library = extract_multipart_field(body, boundary, "library", default="personal")

        for part in body.split(b"--" + boundary):
            if b"Content-Disposition" not in part or b"filename=" not in part:
                continue
            header, _, file_data = part.partition(b"\r\n\r\n")
            filename_match = re.search(rb'filename="([^"]+)"', header)
            if not filename_match:
                continue

            original_name = Path(filename_match.group(1).decode("utf-8", errors="ignore")).name
            safe_name = safe_filename(original_name)
            target = unique_path(UPLOAD_DIR / safe_name)
            file_data = file_data.rstrip(b"\r\n")
            target.write_bytes(file_data)

            try:
                asset = ENGINE.add_asset(target, datetime.now().isoformat(timespec="seconds"), library=upload_library)
                tag_result = tag_uploaded_asset(asset)
                uploaded.append({**asset_payload(asset), "tag_result": tag_result})
            except Exception as exc:
                target.unlink(missing_ok=True)
                return self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        self.send_json({"uploaded": uploaded})

    def serve_file(self, path: Path, no_cache: bool = False) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        range_header = self.headers.get("Range")
        if range_header:
            return self.serve_file_range(path, range_header, no_cache=no_cache)
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(path.stat().st_size))
        if no_cache:
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("Pragma", "no-cache")
        self.end_headers()
        with path.open("rb") as file:
            shutil.copyfileobj(file, self.wfile)

    def serve_file_range(self, path: Path, range_header: str, no_cache: bool = False) -> None:
        file_size = path.stat().st_size
        match = re.match(r"bytes=(\d*)-(\d*)", range_header)
        if not match:
            self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            return

        start_text, end_text = match.groups()
        if start_text:
            start = int(start_text)
            end = int(end_text) if end_text else file_size - 1
        else:
            suffix_length = int(end_text) if end_text else file_size
            start = max(file_size - suffix_length, 0)
            end = file_size - 1

        end = min(end, file_size - 1)
        if start < 0 or start > end or start >= file_size:
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{file_size}")
            self.end_headers()
            return

        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        content_length = end - start + 1
        self.send_response(HTTPStatus.PARTIAL_CONTENT)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.send_header("Content-Length", str(content_length))
        if no_cache:
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("Pragma", "no-cache")
        self.end_headers()

        with path.open("rb") as file:
            file.seek(start)
            remaining = content_length
            while remaining > 0:
                chunk = file.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    break
                remaining -= len(chunk)

    def send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw)

    def log_message(self, format: str, *args: object) -> None:
        return


def import_video_samples() -> dict:
    VIDEO_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    imported = []
    skipped = []
    for source in VIDEO_SAMPLE_DIR.glob("*.mp4"):
        target = UPLOAD_DIR / source.name
        if not target.exists():
            shutil.copy2(source, target)
        asset = ENGINE.add_asset(target, datetime.now().isoformat(timespec="seconds"), library="public")
        imported.append(asset_payload(asset))
    return {"imported": imported, "skipped": skipped, "asset_count": len(ENGINE.assets)}


def parse_thumbnail_asset_id(path: str) -> str:
    value = path.removeprefix("/thumbs/")
    if value.lower().endswith(".jpg"):
        value = value[:-4]
    return unquote(value)


def ensure_thumbnail(asset_id: str, size: int = 360) -> Path:
    asset = ENGINE.get_asset(asset_id)
    if asset is None:
        raise ValueError(f"unknown asset_id: {asset_id}")

    source = Path(asset.path)
    if not source.is_absolute():
        source = ROOT / source
    if not source.exists() or not source.is_file():
        raise ValueError(f"missing asset file: {asset_id}")

    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    target = THUMB_DIR / f"{safe_filename(asset.id)}.jpg"
    if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
        return target

    suffix = source.suffix.lower()
    if asset.kind == "image" and suffix in SUPPORTED_IMAGE_TYPES:
        with Image.open(source) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            thumb = ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    elif asset.kind == "video" and suffix in SUPPORTED_VIDEO_TYPES:
        frame = extract_representative_video_frame(source)
        if frame is None:
            raise ValueError(f"cannot decode video thumbnail: {asset_id}")
        thumb = ImageOps.fit(frame.convert("RGB"), (size, size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    elif asset.kind == "document" and suffix in SUPPORTED_DOCUMENT_TYPES:
        thumb = make_document_thumbnail(asset.filename, size=size)
    else:
        raise ValueError(f"unsupported thumbnail type: {asset.kind}")

    thumb.save(target, format="JPEG", quality=82, optimize=True)
    return target


def make_document_thumbnail(filename: str, size: int = 360) -> Image.Image:
    image = Image.new("RGB", (size, size), "#f8fbfa")
    try:
        from PIL import ImageDraw, ImageFont

        draw = ImageDraw.Draw(image)
        suffix = Path(filename).suffix.upper().removeprefix(".") or "DOC"
        font_large = ImageFont.load_default(size=42)
        font_small = ImageFont.load_default(size=16)
        draw.rectangle([28, 28, size - 28, size - 28], outline="#8fb5a8", width=3)
        draw.text((size / 2, size / 2 - 32), suffix, fill="#1d6f62", font=font_large, anchor="mm")
        stem = Path(filename).stem[:28]
        draw.text((size / 2, size - 70), stem, fill="#344054", font=font_small, anchor="mm")
    except Exception:
        pass
    return image


def extract_representative_video_frame(path: Path) -> Image.Image | None:
    try:
        import av
    except Exception:
        return None
    try:
        with av.open(str(path)) as container:
            stream = next((item for item in container.streams if item.type == "video"), None)
            if stream is None:
                return None
            if stream.duration and stream.time_base:
                target = int(stream.duration * 0.35)
                try:
                    container.seek(target, stream=stream)
                except Exception:
                    pass
            for frame in container.decode(stream):
                return frame.to_image().convert("RGB")
    except Exception:
        return None
    return None


def submit_processing_task(payload: dict) -> dict:
    task_id = uuid.uuid4().hex[:12]
    task = {
        "task_id": task_id,
        "status": "queued",
        "actions": payload.get("actions") or ["thumbnail", "fusion_tags"],
        "library": str(payload.get("library", "all")),
        "total": 0,
        "done": 0,
        "failed": 0,
        "errors": [],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    with TASK_LOCK:
        TASKS[task_id] = task
    thread = threading.Thread(target=run_processing_task, args=(task_id, payload), daemon=True)
    thread.start()
    return task


def run_processing_task(task_id: str, payload: dict) -> None:
    actions = set(payload.get("actions") or ["thumbnail", "fusion_tags"])
    if "text_signals" in actions:
        run_text_signal_task(task_id, payload)
        actions.discard("text_signals")
        if not actions:
            return

    asset_ids = [str(item) for item in payload.get("asset_ids", []) if item]
    library = str(payload.get("library", "all"))
    assets = ENGINE.list_assets()
    if asset_ids:
        wanted = set(asset_ids)
        assets = [asset for asset in assets if asset["id"] in wanted]
    elif library in {"personal", "public"}:
        assets = [asset for asset in assets if asset.get("library") == library]

    update_task(task_id, status="running", total=len(assets))
    for asset in assets:
        try:
            if "thumbnail" in actions:
                ensure_thumbnail(asset["id"])
            if "fusion_tags" in actions:
                ENGINE.fuse_asset_tags(asset["id"])
            increment_task(task_id, done=1)
        except Exception as exc:
            increment_task(task_id, failed=1, error={"asset_id": asset.get("id"), "error": str(exc)})
    update_task(task_id, status="done")


def run_text_signal_task(task_id: str, payload: dict) -> None:
    script = ROOT / "tools" / "extract_text_signals.py"
    output = DATA_DIR / f"text_signals_{task_id}.csv"
    command = [
        sys.executable,
        str(script),
        "--data-dir",
        str(DATA_DIR),
        "--library",
        str(payload.get("library", "all")),
        "--kind",
        str(payload.get("kind", "video")),
        "--offset",
        str(int(payload.get("offset", 0))),
        "--limit",
        str(int(payload.get("limit", 6))),
        "--ocr-engine",
        str(payload.get("ocr_engine", "easyocr")),
        "--asr-engine",
        str(payload.get("asr_engine", "none")),
        "--asr-model",
        str(payload.get("asr_model", "tiny")),
        "--video-frames",
        str(int(payload.get("video_frames", 2))),
        "--min-confidence",
        str(float(payload.get("min_confidence", 0.45))),
        "--output",
        str(output),
        "--import-signals",
    ]
    update_task(task_id, status="running", total=1, detail="text_signals")
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=int(payload.get("timeout", 900)),
    )
    if completed.returncode == 0:
        increment_task(task_id, done=1)
        update_task(task_id, status="done", output=str(output), stdout=completed.stdout[-4000:])
    else:
        increment_task(
            task_id,
            failed=1,
            error={
                "action": "text_signals",
                "returncode": completed.returncode,
                "stderr": (completed.stderr or completed.stdout)[-1000:],
            },
        )
        update_task(task_id, status="failed", output=str(output))


def update_task(task_id: str, **changes) -> None:
    with TASK_LOCK:
        task = TASKS.get(task_id)
        if not task:
            return
        task.update(changes)
        task["updated_at"] = datetime.now().isoformat(timespec="seconds")


def increment_task(task_id: str, done: int = 0, failed: int = 0, error: dict | None = None) -> None:
    with TASK_LOCK:
        task = TASKS.get(task_id)
        if not task:
            return
        task["done"] += done
        task["failed"] += failed
        if error and len(task["errors"]) < 20:
            task["errors"].append(error)
        task["updated_at"] = datetime.now().isoformat(timespec="seconds")


def list_tasks() -> list[dict]:
    with TASK_LOCK:
        tasks = sorted(TASKS.values(), key=lambda item: item["created_at"], reverse=True)[:30]
    return [task_payload_for_e(task) for task in tasks]


def task_payload_for_e(task: dict) -> dict:
    total = int(task.get("total") or 0)
    done = int(task.get("done") or 0)
    failed = int(task.get("failed") or 0)
    raw_status = str(task.get("status") or "queued")
    status_map = {
        "queued": "pending",
        "running": "running",
        "done": "completed",
        "failed": "failed",
    }
    progress = 0 if total <= 0 else round(min((done + failed) / total * 100, 100), 1)
    return {
        **task,
        "type": task.get("detail") or ",".join(task.get("actions") or []),
        "status": status_map.get(raw_status, raw_status),
        "progress": progress,
        "message": f"done={done}, failed={failed}, total={total}",
    }


def load_retrieval_evaluation(limit: int = 20) -> dict:
    query_path = DATA_DIR / "eval_queries.json"
    template_path = DATA_DIR / "eval_queries.example.json"
    source_path = query_path if query_path.exists() else template_path
    if not source_path.exists():
        return {
            "source": str(query_path),
            "template": str(template_path),
            "warning": "missing evaluation query file",
            "limit": limit,
            "profiles": [],
        }
    query_rows = json.loads(source_path.read_text(encoding="utf-8"))
    report = ENGINE.evaluate_retrieval(query_rows, limit=limit)
    labeled_count = sum(1 for item in query_rows if item.get("relevant_ids"))
    report.update(
        {
            "source": str(source_path),
            "template": str(template_path),
            "query_count": len(query_rows),
            "labeled_query_count": labeled_count,
            "warning": "" if labeled_count else "评测集还没有填写 relevant_ids，指标会显示为 0。",
        }
    )
    return report


def rebuild_person_clusters() -> dict:
    script = ROOT / "tools" / "build_person_clusters.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--data-dir", str(DATA_DIR), "--library", "personal"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
    )
    ENGINE.load()
    ENGINE.reindex_metadata()
    if completed.returncode != 0:
        return {
            "ok": False,
            "error": completed.stderr.strip() or completed.stdout.strip(),
            "returncode": completed.returncode,
        }
    repair_result = ENGINE.repair_person_tags("personal")
    return {
        "ok": True,
        "stdout": completed.stdout.strip(),
        "persons": ENGINE.list_persons("personal"),
        "repair": repair_result,
        "stats": ENGINE.library_stats(),
    }


def extract_multipart_field(body: bytes, boundary: bytes, name: str, default: str = "") -> str:
    pattern = re.compile(rb'name="' + re.escape(name.encode("utf-8")) + rb'"')
    for part in body.split(b"--" + boundary):
        if b"Content-Disposition" not in part or b"filename=" in part:
            continue
        header, _, value = part.partition(b"\r\n\r\n")
        if not pattern.search(header):
            continue
        return value.strip().decode("utf-8", errors="ignore") or default
    return default


def write_asset_tags(payload: dict) -> list[dict]:
    asset_id = str(payload.get("asset_id", "")).strip()
    raw_tags = payload.get("tags", [])
    default_source = str(payload.get("source", "external")).strip() or "external"
    default_confidence = float(payload.get("confidence", 1.0))
    replace_source = bool(payload.get("replace_source", False))
    if not isinstance(raw_tags, list):
        raise ValueError("tags must be a list")

    grouped: dict[tuple[str, float], list[str]] = {}
    for item in raw_tags:
        if isinstance(item, str):
            tag_name = item
            source = default_source
            confidence = default_confidence
        elif isinstance(item, dict):
            tag_name = str(item.get("name", ""))
            source = str(item.get("source", default_source)).strip() or default_source
            confidence = float(item.get("confidence", default_confidence))
        else:
            raise ValueError("each tag must be a string or object")

        tag_name = tag_name.strip()
        if not tag_name:
            continue
        confidence = max(0.0, min(1.0, confidence))
        grouped.setdefault((source, confidence), []).append(tag_name)

    if not grouped and not replace_source:
        raise ValueError("tags must contain at least one non-empty tag")

    if not grouped and replace_source:
        return [ENGINE.set_asset_tags(asset_id, [], source=default_source, confidence=default_confidence, replace_source=True)]

    return [
        ENGINE.set_asset_tags(
            asset_id,
            tags,
            source=source,
            confidence=confidence,
            replace_source=replace_source and source == default_source,
        )
        for (source, confidence), tags in grouped.items()
    ]


def list_assets_response(params: dict[str, list[str]]) -> dict:
    assets = ENGINE.list_assets()
    kind = params.get("kind", ["all"])[0] or "all"
    library = params.get("library", ["all"])[0] or "all"
    if kind != "all":
        assets = [asset for asset in assets if asset.get("kind") == kind]
    if library != "all":
        assets = [asset for asset in assets if asset.get("library") == library]
    total = len(assets)
    offset = parse_positive_int(params.get("offset", ["0"])[0], default=0, minimum=0, maximum=max(total, 0))
    limit = parse_positive_int(params.get("limit", [str(total or 300)])[0], default=total or 300, minimum=1, maximum=500)
    return {"assets": assets[offset : offset + limit], "total": total}


def batch_operation(payload: dict) -> dict:
    action = str(payload.get("action", "")).strip().lower()
    asset_ids = [str(item) for item in payload.get("asset_ids", []) if item]
    tags = [str(item) for item in payload.get("tags", []) if str(item).strip()]
    affected = 0
    for asset_id in asset_ids:
        if action == "tag":
            ENGINE.set_asset_tags(asset_id, tags, source="manual", confidence=1.0, replace_source=False)
            ENGINE.fuse_asset_tags(asset_id)
            affected += 1
        elif action == "favorite":
            ENGINE.set_favorite(asset_id, True)
            affected += 1
        elif action == "archive":
            ENGINE.set_asset_tags(asset_id, ["archived"], source="manual", confidence=1.0, replace_source=False)
            ENGINE.fuse_asset_tags(asset_id)
            affected += 1
        elif action == "delete":
            ENGINE.delete_asset(asset_id, delete_file=True)
            affected += 1
        else:
            raise ValueError(f"unsupported batch action: {action}")
    return {"ok": True, "affected": affected}


def asset_payload(asset) -> dict:
    return {
        "id": asset.id,
        "filename": asset.filename,
        "url": f"/uploads/{asset.filename}",
        "kind": asset.kind,
        "library": asset.library,
        "width": asset.width,
        "height": asset.height,
    }


def tag_uploaded_asset(asset) -> dict:
    base_tags = [asset.kind, asset.library]
    if asset.kind == "image":
        base_tags.append("image_material")
    if asset.kind == "video":
        base_tags.extend(["video", "video_material"])
    if asset.kind == "document":
        base_tags.extend(["document", "document_material"])
    ENGINE.set_asset_tags(asset.id, base_tags, source="system", confidence=1.0, replace_source=True)

    scene_tags = [] if asset.kind == "document" else generate_upload_scene_tags(asset)
    if scene_tags:
        ENGINE.set_asset_tags(
            asset.id,
            scene_tags,
            source="clip_zero_shot_scene",
            confidence=0.82,
            replace_source=True,
        )
    fusion = ENGINE.fuse_asset_tags(asset.id)
    return {
        "system_tags": base_tags,
        "scene_tags": scene_tags,
        "fusion_tags": fusion.get("tags", []),
    }


def generate_upload_scene_tags(asset, top_k: int = 4, threshold: float = 0.24, margin: float = 0.035) -> list[str]:
    tag_vectors = get_upload_tag_vectors()
    asset_vector = np.array(asset.vector, dtype=np.float32)
    scored = []
    for tag, prompt_vectors in tag_vectors.items():
        score = max(cosine_similarity(asset_vector, prompt_vector) for prompt_vector in prompt_vectors)
        scored.append((score, tag))
    scored.sort(reverse=True)
    if not scored:
        return []
    best_score = scored[0][0]
    selected = [
        tag
        for score, tag in scored[:top_k]
        if score >= threshold and score >= best_score - margin
    ]
    return selected


def get_upload_tag_vectors() -> dict[str, list[np.ndarray]]:
    global UPLOAD_TAG_VECTOR_CACHE
    if UPLOAD_TAG_VECTOR_CACHE is not None:
        return UPLOAD_TAG_VECTOR_CACHE
    with UPLOAD_TAG_LOCK:
        if UPLOAD_TAG_VECTOR_CACHE is None:
            UPLOAD_TAG_VECTOR_CACHE = {
                tag: [ENGINE.encoder.encode_text(prompt) for prompt in prompts]
                for tag, prompts in UPLOAD_TAG_PROMPTS.items()
            }
    return UPLOAD_TAG_VECTOR_CACHE


def safe_filename(name: str) -> str:
    stem = Path(name).stem or "asset"
    suffix = Path(name).suffix.lower()
    stem = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "_", stem).strip("_") or "asset"
    return f"{stem}{suffix}"


def unique_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        return path
    counter = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def parse_positive_int(value: str, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def parse_optional_positive_int(value: object, default: int | None, minimum: int, maximum: int) -> int | None:
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def main() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "0.0.0.0")
    server = ThreadingHTTPServer((host, port), AppHandler)
    lan_ip = get_lan_ip()
    print(f"服务已启动：http://127.0.0.1:{port}")
    if host in {"0.0.0.0", "::"} and lan_ip:
        print(f"同一局域网访问：http://{lan_ip}:{port}")
    print(f"服务已启动：http://127.0.0.1:{port}")
    server.serve_forever()


def get_lan_ip() -> str | None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return None


if __name__ == "__main__":
    main()
