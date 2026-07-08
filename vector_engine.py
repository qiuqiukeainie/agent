from __future__ import annotations

import json
import math
import re
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np
from PIL import Image, ImageStat

from agent_engine import QueryPlan, SearchAgent
from metadata_store import MetadataStore
from vector_index import VectorIndex

try:
    from document_module import (
        DocumentChunk as BDocumentChunk,
        SemanticRetriever as BSemanticRetriever,
        answer_from_chunks as b_answer_from_chunks,
        chunk_document as b_chunk_document,
        generate_document_tags as b_generate_document_tags,
        parse_document as b_parse_document,
    )
except Exception:
    BDocumentChunk = None
    BSemanticRetriever = None
    b_parse_document = None
    b_chunk_document = None
    b_generate_document_tags = None
    b_answer_from_chunks = None


SUPPORTED_IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SUPPORTED_VIDEO_TYPES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
SUPPORTED_DOCUMENT_TYPES = {".pdf", ".docx", ".pptx", ".txt", ".md"}
SEASON_MONTHS = {
    "\u6625\u5929": {3, 4, 5},
    "\u590f\u5929": {6, 7, 8},
    "\u79cb\u5929": {9, 10, 11},
    "\u51ac\u5929": {12, 1, 2},
}


@dataclass
class Asset:
    id: str
    filename: str
    path: str
    kind: str
    library: str
    created_at: str
    width: int
    height: int
    vector: list[float]
    vector_model: str = "lite-visual-v1"


class Encoder(Protocol):
    name: str

    def encode_image(self, image: Image.Image) -> np.ndarray:
        ...

    def encode_text(self, query: str) -> np.ndarray:
        ...


class LiteEncoder:
    name = "lite-visual-v1"

    def encode_image(self, image: Image.Image) -> np.ndarray:
        return image_to_vector(image)

    def encode_text(self, query: str) -> np.ndarray:
        return text_to_vector(query)


class ClipEncoder:
    name = "clip-vit-base-patch32"

    def __init__(self) -> None:
        import clip
        import torch

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        cache_dir = Path.cwd() / ".model_cache" / "clip"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.model, self.preprocess = clip.load(
            "ViT-B/32",
            device=self.device,
            download_root=str(cache_dir),
            jit=False,
        )
        self.tokenize = clip.tokenize
        self.model.eval()

    def encode_image(self, image: Image.Image) -> np.ndarray:
        inputs = self.preprocess(image).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            features = self.model.encode_image(inputs)
        return normalize(features[0].detach().cpu().numpy().astype(np.float32))

    def encode_text(self, query: str) -> np.ndarray:
        prompt = expand_chinese_query(query)
        inputs = self.tokenize([prompt], truncate=True).to(self.device)
        with self.torch.no_grad():
            features = self.model.encode_text(inputs)
        return normalize(features[0].detach().cpu().numpy().astype(np.float32))


class VectorEngine:
    def __init__(self, storage_dir: str = "data") -> None:
        self.storage_dir = Path(storage_dir)
        self.upload_dir = self.storage_dir / "uploads"
        self.index_path = self.storage_dir / "index.json"
        self.document_chunk_index_path = self.storage_dir / "document_chunk_index.json"
        self.person_ignore_path = self.storage_dir / "person_ignore.json"
        self.person_merge_rules_path = self.storage_dir / "person_merge_rules.json"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.encoder = build_encoder()
        self.agent = SearchAgent()
        self.metadata = MetadataStore(self.storage_dir / "assets.db")
        self.assets: list[Asset] = []
        self.document_chunk_vectors: dict[str, list[float]] = {}
        self.document_semantic_retriever = None
        self.document_semantic_signature = ""
        self.document_semantic_error = ""
        self.vector_index = VectorIndex()
        self.index_asset_positions: list[int] = []
        self.load()
        self.rebuild_stale_assets()
        self.rebuild_vector_index()
        self.reindex_metadata()

    def load(self) -> None:
        if not self.index_path.exists():
            self.assets = []
            self.load_document_chunk_index()
            return
        raw_assets = json.loads(self.index_path.read_text(encoding="utf-8"))
        self.assets = [
            Asset(
                **{
                    **item,
                    "library": item.get("library", infer_library_from_filename(item.get("filename", ""))),
                    "vector_model": item.get("vector_model", "lite-visual-v1"),
                }
            )
            for item in raw_assets
        ]
        self.load_document_chunk_index()

    def save(self) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        payload = [asdict(item) for item in self.assets]
        self.index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_document_chunk_index(self) -> None:
        if not self.document_chunk_index_path.exists():
            self.document_chunk_vectors = {}
            return
        try:
            raw = json.loads(self.document_chunk_index_path.read_text(encoding="utf-8"))
        except Exception:
            self.document_chunk_vectors = {}
            return
        self.document_chunk_vectors = {
            str(chunk_id): [float(value) for value in vector]
            for chunk_id, vector in raw.items()
            if isinstance(vector, list)
        }

    def save_document_chunk_index(self) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        known_chunk_ids = {chunk["chunk_id"] for chunk in self.metadata.list_document_chunks()}
        self.document_chunk_vectors = {
            chunk_id: vector
            for chunk_id, vector in self.document_chunk_vectors.items()
            if chunk_id in known_chunk_ids
        }
        self.document_chunk_index_path.write_text(
            json.dumps(self.document_chunk_vectors, ensure_ascii=False),
            encoding="utf-8",
        )

    def add_image(self, source_path: Path, created_at: str, library: str = "public") -> Asset:
        suffix = source_path.suffix.lower()
        if suffix not in SUPPORTED_IMAGE_TYPES:
            raise ValueError(f"unsupported image format: {suffix}")

        image_id = source_path.stem
        with Image.open(source_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            vector = self.encoder.encode_image(image)

        asset = Asset(
            id=image_id,
            filename=source_path.name,
            path=str(source_path.as_posix()),
            kind="image",
            library=normalize_library(library),
            created_at=created_at,
            width=width,
            height=height,
            vector=vector.tolist(),
            vector_model=self.encoder.name,
        )
        self.assets = [item for item in self.assets if item.id != image_id]
        self.assets.append(asset)
        self.save()
        self.rebuild_vector_index()
        self.metadata.sync_asset(asset, Path.cwd())
        return asset

    def add_asset(self, source_path: Path, created_at: str, library: str = "public") -> Asset:
        suffix = source_path.suffix.lower()
        if suffix in SUPPORTED_IMAGE_TYPES:
            return self.add_image(source_path, created_at, library=library)
        if suffix in SUPPORTED_VIDEO_TYPES:
            return self.add_video(source_path, created_at, library=library)
        if suffix in SUPPORTED_DOCUMENT_TYPES:
            return self.add_document(source_path, created_at, library=library)
        raise ValueError(f"unsupported asset format: {suffix}")

    def add_document(self, source_path: Path, created_at: str, library: str = "public") -> Asset:
        suffix = source_path.suffix.lower()
        if suffix not in SUPPORTED_DOCUMENT_TYPES:
            raise ValueError(f"unsupported document format: {suffix}")
        parsed_document = parse_document_with_b_module(source_path)
        text = parsed_document.get("text") or extract_document_text(source_path)
        if not text.strip():
            raise ValueError(f"document text extraction produced no text: {source_path.name}")
        vector = self.encoder.encode_text(text[:1200])
        asset = Asset(
            id=source_path.stem,
            filename=source_path.name,
            path=str(source_path.as_posix()),
            kind="document",
            library=normalize_library(library),
            created_at=created_at,
            width=0,
            height=0,
            vector=vector.tolist(),
            vector_model=self.encoder.name,
        )
        self.assets = [item for item in self.assets if item.id != asset.id]
        self.assets.append(asset)
        self.save()
        self.rebuild_vector_index()
        self.metadata.sync_asset(asset, Path.cwd())
        self.metadata.set_asset_tags(asset.id, ["document", suffix.removeprefix(".")], source="system", confidence=1.0)
        self.metadata.upsert_ocr_text(asset.id, text[:20000], engine=f"document_text:{suffix.removeprefix('.')}", confidence=0.95)
        self.index_document_chunks(asset, text, suffix.removeprefix("."), parsed_document=parsed_document)
        self.fuse_asset_tags(asset.id)
        return asset

    def add_video(self, source_path: Path, created_at: str, frame_count: int = 8, library: str = "public") -> Asset:
        suffix = source_path.suffix.lower()
        if suffix not in SUPPORTED_VIDEO_TYPES:
            raise ValueError(f"unsupported video format: {suffix}")

        frames = extract_video_frames(source_path, frame_count)
        if not frames:
            raise ValueError("video decoding failed: no readable frames")

        vectors = [self.encoder.encode_image(frame) for frame in frames]
        vector = normalize(np.mean(np.vstack(vectors), axis=0).astype(np.float32))
        width, height = frames[0].size
        asset = Asset(
            id=source_path.stem,
            filename=source_path.name,
            path=str(source_path.as_posix()),
            kind="video",
            library=normalize_library(library),
            created_at=created_at,
            width=width,
            height=height,
            vector=vector.tolist(),
            vector_model=self.encoder.name,
        )
        self.assets = [item for item in self.assets if item.id != asset.id]
        self.assets.append(asset)
        self.save()
        self.rebuild_vector_index()
        self.metadata.sync_asset(asset, Path.cwd())
        self.metadata.set_asset_tags(asset.id, ["video"], source="system", confidence=1.0)
        return asset

    def rebuild_index_from_uploads(self, limit: int | None = None) -> dict:
        existing_filenames = {asset.filename for asset in self.assets}
        imported = 0
        skipped = 0
        failed: list[dict] = []

        for path in iter_existing_assets(self.upload_dir):
            if limit is not None and imported >= limit:
                break
            if path.name in existing_filenames:
                skipped += 1
                continue
            try:
                created_at = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
                self.add_asset(path, created_at)
                existing_filenames.add(path.name)
                imported += 1
            except Exception as exc:
                failed.append({"filename": path.name, "error": str(exc)})

        self.reindex_metadata()
        return {
            "imported": imported,
            "skipped": skipped,
            "failed": failed[:20],
            "asset_count": len(self.assets),
        }

    def index_document_chunks(
        self,
        asset: Asset,
        text: str,
        suffix: str = "document",
        parsed_document: dict | None = None,
    ) -> list[dict]:
        chunks = build_document_chunks_from_b_module(asset, parsed_document) if parsed_document else []
        if not chunks:
            chunks = build_document_chunks(asset, text, suffix)
        for chunk in chunks:
            embedding_text = str(chunk.get("embedding_text") or chunk.get("text") or "")
            self.document_chunk_vectors[chunk["chunk_id"]] = self.encoder.encode_text(embedding_text[:1600]).tolist()
        self.metadata.replace_document_chunks(asset.id, chunks)
        self.save_document_chunk_index()
        document_tags = build_document_tags_from_b_module(parsed_document) if parsed_document else []
        if not document_tags:
            document_tags = build_document_level_tags(asset.filename, chunks, suffix)
        if document_tags:
            self.set_asset_tags(
                asset.id,
                document_tags,
                source="document_chunk",
                confidence=0.86,
                replace_source=True,
            )
        return chunks

    def rebuild_document_chunks(self, limit: int | None = None, library: str = "all") -> dict:
        library_filter = normalize_library_filter(library)
        processed = 0
        failed: list[dict] = []
        for asset in self.assets:
            if asset.kind != "document":
                continue
            if library_filter != "all" and asset.library != library_filter:
                continue
            if limit is not None and processed >= limit:
                break
            try:
                path = Path(asset.path)
                if not path.is_absolute():
                    path = Path.cwd() / path
                parsed_document = parse_document_with_b_module(path, doc_id=asset.id)
                text = parsed_document.get("text") or self.metadata.get_text_by_type(asset.id, {"document_text"}) or extract_document_text(path)
                self.index_document_chunks(
                    asset,
                    text,
                    path.suffix.lower().removeprefix(".") or "document",
                    parsed_document=parsed_document,
                )
                self.fuse_asset_tags(asset.id)
                processed += 1
            except Exception as exc:
                failed.append({"asset_id": asset.id, "filename": asset.filename, "error": str(exc)})
        return {
            "processed": processed,
            "failed": failed[:20],
            "document_chunk_count": self.metadata.count_document_chunks(),
        }

    def ensure_document_chunks(self) -> None:
        changed = False
        for asset in self.assets:
            if asset.kind != "document":
                continue
            chunks = self.metadata.list_document_chunks(asset.id)
            if not chunks:
                try:
                    path = Path(asset.path)
                    if not path.is_absolute():
                        path = Path.cwd() / path
                    parsed_document = parse_document_with_b_module(path, doc_id=asset.id)
                    text = parsed_document.get("text") or self.metadata.get_text_by_type(asset.id, {"document_text"}) or extract_document_text(path)
                    chunks = self.index_document_chunks(
                        asset,
                        text,
                        path.suffix.lower().removeprefix(".") or "document",
                        parsed_document=parsed_document,
                    )
                    changed = True
                except Exception:
                    continue
            for chunk in chunks:
                chunk_id = str(chunk.get("chunk_id") or "")
                if chunk_id and chunk_id not in self.document_chunk_vectors:
                    embedding_text = str(chunk.get("embedding_text") or chunk.get("text") or "")
                    self.document_chunk_vectors[chunk_id] = self.encoder.encode_text(embedding_text[:1600]).tolist()
                    changed = True
        if changed:
            self.save_document_chunk_index()

    def document_search(self, query: str, limit: int = 12, library: str = "all") -> dict:
        query = str(query or "").strip()
        if not query:
            return {"query": query, "results": [], "mode": "document_chunk_hybrid"}
        self.ensure_document_chunks()
        library_filter = normalize_library_filter(library)
        content_query = normalize_document_content_query(query)
        if not content_query:
            return {
                "query": query,
                "expanded_query": "",
                "mode": "document_chunk_hybrid_skipped_empty_content",
                "retriever_error": "",
                "results": [],
                "total_chunks": self.metadata.count_document_chunks(),
            }
        expanded_query = expand_document_query(content_query)
        asset_by_id = {asset.id: asset for asset in self.assets}
        candidate_chunks = []
        for chunk in self.metadata.list_document_chunks():
            asset = asset_by_id.get(str(chunk.get("asset_id") or ""))
            if asset is None or asset.kind != "document":
                continue
            if library_filter != "all" and asset.library != library_filter:
                continue
            candidate_chunks.append((asset, chunk))

        b_scores, b_error = self.document_semantic_scores(expanded_query, candidate_chunks, top_k=max(limit * 6, 40))
        use_b_retriever = bool(b_scores)
        query_vector = None if use_b_retriever else self.encoder.encode_text(expanded_query[:1600])
        rows = []
        for asset, chunk in candidate_chunks:
            chunk_id = str(chunk.get("chunk_id") or "")
            if use_b_retriever:
                if chunk_id not in b_scores:
                    continue
                vector_score = b_scores[chunk_id]
            else:
                raw_vector = self.document_chunk_vectors.get(chunk_id)
                if not raw_vector or query_vector is None:
                    continue
                chunk_vector = np.array(raw_vector, dtype=np.float32)
                if chunk_vector.shape != query_vector.shape:
                    continue
                vector_score = cosine_similarity(query_vector, chunk_vector)
            keyword, keyword_hits = document_keyword_match(expanded_query, chunk)
            original_keyword, original_keyword_hits = document_keyword_match(content_query, chunk)
            if original_keyword > keyword:
                keyword = original_keyword
                keyword_hits = original_keyword_hits
            title_score = keyword_score(content_query, asset.filename)
            final_score = 0.78 * vector_score + 0.16 * keyword + 0.06 * title_score
            if keyword_hits:
                final_score += min(len(keyword_hits) * 0.035, 0.14)
            if len(str(chunk.get("text") or "")) < 30:
                final_score -= 0.12
            rows.append(
                {
                    "asset_id": asset.id,
                    "chunk_id": chunk["chunk_id"],
                    "filename": asset.filename,
                    "url": f"/uploads/{asset.filename}",
                    "library": asset.library,
                    "chunk_index": chunk["chunk_index"],
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                    "section_title": chunk.get("section_title") or "",
                    "summary": chunk.get("summary") or summarize_text_snippet(chunk.get("text") or ""),
                    "snippet": highlight_document_snippet(chunk.get("text") or "", query),
                    "text": chunk.get("text") or "",
                    "keywords": chunk.get("keywords") or [],
                    "score": round(float(final_score), 4),
                    "vector_score": round(float(vector_score), 4),
                    "keyword_score": round(float(keyword), 4),
                    "match_reason": build_document_match_reason(keyword_hits, chunk),
                }
            )
        rows.sort(key=lambda item: item["score"], reverse=True)
        return {
            "query": query,
            "expanded_query": expanded_query,
            "mode": "B.semantic_retriever_hybrid" if use_b_retriever else "document_chunk_hybrid_fallback",
            "retriever_error": b_error,
            "results": rows[:limit],
            "total_chunks": self.metadata.count_document_chunks(),
        }

    def document_semantic_scores(
        self,
        query: str,
        candidate_chunks: list[tuple[Asset, dict]],
        top_k: int,
    ) -> tuple[dict[str, float], str]:
        if BSemanticRetriever is None or BDocumentChunk is None:
            return {}, "B SemanticRetriever is unavailable"
        if not candidate_chunks:
            return {}, ""
        signature = "|".join(
            f"{chunk.get('chunk_id')}:{len(str(chunk.get('text') or ''))}"
            for _asset, chunk in candidate_chunks
        )
        try:
            if self.document_semantic_retriever is None or self.document_semantic_signature != signature:
                b_chunks = [
                    BDocumentChunk(
                        chunk_id=str(chunk.get("chunk_id") or ""),
                        doc_id=str(chunk.get("asset_id") or ""),
                        chunk_index=int(chunk.get("chunk_index") or index),
                        chunk_type=str(chunk.get("chunk_type") or "text"),
                        page_start=chunk.get("page_start"),
                        page_end=chunk.get("page_end"),
                        section_id=str(chunk.get("section_id") or ""),
                        section_title=str(chunk.get("section_title") or ""),
                        heading_path=list(chunk.get("heading_path") or []),
                        text=str(chunk.get("text") or ""),
                        summary=str(chunk.get("summary") or ""),
                        embedding_text=str(chunk.get("embedding_text") or chunk.get("text") or ""),
                    )
                    for index, (_asset, chunk) in enumerate(candidate_chunks)
                    if str(chunk.get("text") or "").strip()
                ]
                retriever = BSemanticRetriever()
                retriever.index(b_chunks)
                self.document_semantic_retriever = retriever
                self.document_semantic_signature = signature
                self.document_semantic_error = ""
            results = self.document_semantic_retriever.search(query, top_k=top_k, min_score=0.0, min_chunk_chars=20)
            return {chunk.chunk_id: float(score) for chunk, score in results}, ""
        except Exception as exc:
            self.document_semantic_error = str(exc)
            return {}, str(exc)

    def document_qa(self, question: str, top_k: int = 6, library: str = "all") -> dict:
        top_k = max(1, min(int(top_k or 6), 12))
        search_result = self.document_search(question, limit=top_k, library=library)
        chunks = search_result["results"]
        if not chunks:
            return {
                "question": question,
                "answer": "未在当前素材库文档中找到可靠答案。",
                "sources": [],
                "mode": "extractive_document_qa",
            }
        reliable = [chunk for chunk in chunks if chunk["score"] >= max(chunks[0]["score"] - 0.08, 0.0)]
        selected = reliable[: min(len(reliable), 5)] or chunks[:3]
        if b_answer_from_chunks is not None:
            try:
                answer = b_answer_from_chunks(question, document_chunk_context_for_b_qa(selected))
                if answer.get("answer"):
                    return {
                        "question": question,
                        "answer": answer.get("answer", ""),
                        "sources": selected,
                        "source_refs": answer.get("sources", []),
                        "mode": "B.document_module.extractive_qa",
                    }
            except Exception:
                pass
        answer_lines = []
        for chunk in selected:
            section = f"《{chunk['filename']}》"
            if chunk.get("section_title"):
                section += f"的“{chunk['section_title']}”部分"
            answer_lines.append(f"{section}提到：{chunk['summary']}")
        return {
            "question": question,
            "answer": "\n".join(answer_lines),
            "sources": selected,
            "mode": "extractive_document_qa",
            "note": "当前为本地抽取式问答，只依据素材库命中文档片段组织答案；后续可接入 Dify/LLM 做更自然的归纳。",
        }

    def search(self, query: str, limit: int = 12, library: str = "all") -> list[dict]:
        return self.agent_search(query, limit, library=library)["results"]

    def agent_search(self, query: str, limit: int = 12, library: str = "all") -> dict:
        expanded_query = self.expand_person_names(query, library=library)
        plan = self.agent.build_plan(expanded_query)
        return self.agent_search_with_plan(plan, limit=limit, library=library)

    def agent_search_with_plan(self, plan: QueryPlan, limit: int = 12, library: str = "all") -> dict:
        intent_strength = document_intent_strength(plan)
        visual_limit = max(limit * 4, 36)
        visual_results = self.search_with_plan(plan, visual_limit, library=library)
        visual_results = [item for item in visual_results if item.get("kind") != "document"]
        document_limit = max(min(limit, 12), 6)
        document_results = []
        document_payload = {
            "mode": "document_branch_skipped_by_media_filter",
            "total_chunks": self.metadata.count_document_chunks(),
            "retriever_error": "",
            "results": [],
        }
        if plan.executable_filters.get("kind") not in {"image", "video"}:
            document_payload = self.document_search(plan.raw_query, limit=document_limit, library=library)
            document_results = [
                self.document_result_to_asset_result(item, plan)
                for item in document_payload.get("results", [])
            ]
        if intent_strength < 0.35:
            document_results = document_results[: max(1, min(3, limit // 3 or 1))]
        merged = merge_search_results(visual_results, document_results, limit)
        apply_display_scores(merged)
        results = merged
        plan_dict = plan.to_dict()
        if document_payload is not None:
            plan_dict["document_retrieval"] = {
                "mode": document_payload.get("mode"),
                "total_chunks": document_payload.get("total_chunks"),
                "retriever_error": document_payload.get("retriever_error"),
                "top_chunks": [
                    {
                        "filename": item.get("filename"),
                        "section_title": item.get("section_title"),
                        "score": item.get("score"),
                    }
                    for item in document_payload.get("results", [])[:3]
                ],
            }
            if "B.document_semantic_recall" not in plan_dict["recall_routes"]:
                plan_dict["recall_routes"] = [*plan_dict["recall_routes"], "B.document_semantic_recall"]
        return {"plan": plan_dict, "results": results}

    def document_result_to_asset_result(self, item: dict, plan: QueryPlan) -> dict:
        asset = self.get_asset(str(item.get("asset_id") or ""))
        if asset is None:
            created_at = ""
            width = 0
            height = 0
            tags = []
        else:
            created_at = asset.created_at
            width = asset.width
            height = asset.height
            tags = self.metadata.get_asset_tags(asset.id)
        raw_score = float(item.get("score") or 0.0)
        intent_strength = document_intent_strength(plan)
        if intent_strength >= 0.85:
            score = 0.42 + raw_score * 0.42
        elif intent_strength >= 0.55:
            score = 0.30 + raw_score * 0.34
        else:
            score = 0.04 + raw_score * 0.52
        keyword_score_value = float(item.get("keyword_score") or 0.0)
        score += min(keyword_score_value * 0.10, 0.10)
        if raw_score < 0.24 and keyword_score_value < 0.12:
            score -= 0.08
        if raw_score < 0.18 and keyword_score_value < 0.20:
            score -= 0.06
        score = max(0.0, min(score, 0.92))
        return {
            "id": item.get("asset_id"),
            "filename": item.get("filename"),
            "url": item.get("url"),
            "kind": "document",
            "library": item.get("library", asset.library if asset else "personal"),
            "created_at": created_at,
            "width": width,
            "height": height,
            "score": round(score, 4),
            "clip_score": 0.0,
            "raw_clip_score": round(raw_score, 4),
            "best_prompt": "B.document_semantic_retriever",
            "tag_score": round(float(item.get("keyword_score") or 0.0), 4),
            "tag_hits": list(item.get("keywords") or [])[:6],
            "tags": tags[:8],
            "ocr_score": round(float(item.get("keyword_score") or 0.0), 4),
            "ocr_hits": [],
            "metadata_score": 0.0,
            "constraint_score": 0.0,
            "constraint_hits": [item.get("section_title")] if item.get("section_title") else [],
            "constraint_misses": [],
            "constraint_penalty": 0.0,
            "relation_score": 0.0,
            "relation_margin": 0.0,
            "relation_hits": [],
            "geometry_relation_score": 0.0,
            "geometry_relation_hits": [],
            "filename_score": 0.0,
            "metadata_hits": [],
            "deferred_conditions": plan.unresolved_conditions,
            "document_chunk": {
                "chunk_id": item.get("chunk_id"),
                "section_title": item.get("section_title"),
                "summary": item.get("summary"),
                "snippet": item.get("snippet"),
                "score": item.get("score"),
                "vector_score": item.get("vector_score"),
                "keyword_score": item.get("keyword_score"),
                "intent_strength": round(intent_strength, 3),
            },
            "explanation": [
                f"B 文档语义模型命中：{item.get('section_title') or item.get('filename')}",
                item.get("match_reason") or "chunk 语义与查询接近",
            ],
        }

    def expand_person_names(self, query: str, library: str = "all") -> str:
        query = str(query or "")
        library_filter = normalize_library_filter(library)
        libraries = ["personal", "public"] if library_filter == "all" else [library_filter]
        additions = []
        lowered = query.lower()
        for item_library in libraries:
            for person_id, names in self.metadata.person_name_map(item_library).items():
                for name in [names.get("display_name"), names.get("alias")]:
                    clean_name = str(name or "").strip()
                    if clean_name and clean_name.lower() in lowered:
                        additions.append(person_id)
        if additions:
            return " ".join([query, *sorted(set(additions))])
        return query

    def search_with_plan(
        self,
        plan: QueryPlan,
        limit: int = 12,
        library: str = "all",
        profile: str = "agent_v3",
        log_search: bool = True,
    ) -> list[dict]:
        if not self.assets:
            return []

        library = normalize_library_filter(library)
        strict_library = plan.strict_conditions.get("library")
        if library == "all" and strict_library in {"public", "personal"}:
            library = str(strict_library)
        query_vectors = [self.encoder.encode_text(item) for item in plan.semantic_queries]
        relation_vectors = [self.encoder.encode_text(item) for item in plan.relation_queries]
        negative_relation_vectors = [self.encoder.encode_text(item) for item in plan.negative_relation_queries]
        interaction_vectors = build_interaction_vectors(self.encoder, plan)
        candidates = self.recall_candidates(plan, query_vectors, max(limit * 8, 64))
        rows = []
        for asset_index, clip_score, best_prompt in candidates:
            asset = self.assets[asset_index]
            if asset.vector_model != self.encoder.name:
                continue
            if library != "all" and asset.library != library:
                continue
            if plan.executable_filters.get("kind") and asset.kind != plan.executable_filters["kind"]:
                continue

            metadata_score, metadata_hits = metadata_match(plan, asset)
            tags = self.metadata.get_asset_tags(asset.id)
            text_signals = self.metadata.list_text_signals(asset.id)
            faces = self.metadata.get_asset_faces(asset.id)
            tag_score, tag_hits = tag_match(plan, tags)
            ocr_score, ocr_hits = ocr_match(plan, self.metadata.get_ocr_text(asset.id))
            constraint_score, constraint_hits, constraint_penalty, constraint_misses = constraint_match(
                plan,
                asset,
                tags,
                text_signals,
                faces,
            )
            filename_score = keyword_score(plan.raw_query, asset.filename)
            relation_score, relation_hits, relation_margin = relation_match(
                plan,
                best_prompt,
                np.array(asset.vector, dtype=np.float32),
                relation_vectors,
                negative_relation_vectors,
            )
            geometry_score, geometry_hits = geometric_relation_match(plan, faces)
            if geometry_hits:
                relation_score = max(relation_score, geometry_score)
                relation_hits = sorted({*relation_hits, *geometry_hits})
            score = weighted_search_score(
                profile=profile,
                clip_score=clip_score,
                tag_score=tag_score,
                metadata_score=metadata_score,
                filename_score=filename_score,
                ocr_score=ocr_score,
                relation_score=relation_score,
            )
            if profile == "agent_v3":
                interaction_boost, interaction_hits, interaction_misses = interaction_frame_match(
                    plan,
                    np.array(asset.vector, dtype=np.float32),
                    tags,
                    interaction_vectors,
                )
                if interaction_hits:
                    constraint_hits = [*constraint_hits, *interaction_hits]
                if interaction_misses:
                    constraint_misses = [*constraint_misses, *interaction_misses]
                score = score + 0.12 * constraint_score - constraint_penalty
                score = score + interaction_boost

            rows.append(
                {
                    "id": asset.id,
                    "filename": asset.filename,
                    "url": f"/uploads/{asset.filename}",
                    "kind": asset.kind,
                    "library": asset.library,
                    "created_at": asset.created_at,
                    "width": asset.width,
                    "height": asset.height,
                    "score": round(float(score), 4),
                    "clip_score": round(float(clip_score), 4),
                    "raw_clip_score": round(float(clip_score), 4),
                    "best_prompt": best_prompt,
                    "tag_score": round(float(tag_score), 4),
                    "tag_hits": tag_hits,
                    "tags": tags[:8],
                    "ocr_score": round(float(ocr_score), 4),
                    "ocr_hits": ocr_hits,
                    "metadata_score": round(float(metadata_score), 4),
                    "constraint_score": round(float(constraint_score), 4),
                    "constraint_hits": constraint_hits,
                    "constraint_misses": constraint_misses,
                    "constraint_penalty": round(float(constraint_penalty), 4),
                    "relation_score": round(float(relation_score), 4),
                    "relation_margin": round(float(relation_margin), 4),
                    "relation_hits": relation_hits,
                    "geometry_relation_score": round(float(geometry_score), 4),
                    "geometry_relation_hits": geometry_hits,
                    "filename_score": round(float(filename_score), 4),
                    "metadata_hits": metadata_hits,
                    "deferred_conditions": plan.unresolved_conditions,
                    "explanation": explain_search_match(
                        clip_score=clip_score,
                        tag_hits=tag_hits,
                        ocr_hits=ocr_hits,
                        metadata_hits=metadata_hits,
                        constraint_hits=constraint_hits,
                        constraint_misses=constraint_misses,
                        relation_hits=relation_hits,
                        filename_score=filename_score,
                        best_prompt=best_prompt,
                    ),
                }
            )

        rows.sort(key=lambda item: item["score"], reverse=True)
        apply_display_scores(rows)
        if log_search:
            self.metadata.log_search(plan.raw_query, plan.to_dict(), rows[:limit], self.vector_index.backend)
        return rows[:limit]

    def recall_candidates(
        self,
        plan: QueryPlan,
        query_vectors: list[np.ndarray],
        recall_limit: int,
    ) -> list[tuple[int, float, str]]:
        scores: dict[int, tuple[float, str]] = {}
        for query, vector in zip(plan.semantic_queries, query_vectors):
            for hit in self.vector_index.search(vector, recall_limit):
                best = scores.get(hit.index)
                if best is None or hit.score > best[0]:
                    scores[hit.index] = (hit.score, query)

        asset_index_by_id = {asset.id: index for index, asset in enumerate(self.assets)}
        for term in query_terms(plan):
            for asset_id in self.metadata.asset_ids_for_tag(term, library="all", limit=recall_limit):
                index = asset_index_by_id.get(asset_id)
                if index is None:
                    continue
                best = scores.get(index)
                tag_prompt = f"tag:{term}"
                tag_recall_score = 0.16 if plan.unresolved_conditions.get("relations") else 0.22
                if best is None:
                    scores[index] = (tag_recall_score, tag_prompt)
                elif best[1].startswith("tag:") and tag_recall_score > best[0]:
                    scores[index] = (tag_recall_score, tag_prompt)

        if not scores:
            for index, asset in enumerate(self.assets):
                if asset.vector_model == self.encoder.name:
                    image_vector = np.array(asset.vector, dtype=np.float32)
                    score, prompt = best_vector_score(plan.semantic_queries, query_vectors, image_vector)
                    scores[index] = (score, prompt)

        ranked = sorted(scores.items(), key=lambda item: item[1][0], reverse=True)
        return [(index, score, prompt) for index, (score, prompt) in ranked[:recall_limit]]

    def model_status(self) -> dict:
        return {
            "vector_model": self.encoder.name,
            "asset_count": len(self.assets),
            "indexed_count": sum(1 for item in self.assets if item.vector_model == self.encoder.name),
            "agent": "search-orchestrator-v5",
            "agent_mode": "deepseek-llm-chat",
            "agent_type": "multi_turn_llm_search_orchestrator",
            "chat_llm": "deepseek-compatible",
            "vector_index": self.vector_index.backend,
            "supported_video_types": sorted(SUPPORTED_VIDEO_TYPES),
            "supported_document_types": sorted(SUPPORTED_DOCUMENT_TYPES),
            "document_chunk_count": self.metadata.count_document_chunks(),
            "document_module": "B.document_module" if b_parse_document is not None else "fallback",
            "routes": [
                "intent_parse",
                "condition_split",
                "query_rewrite",
                "clip_recall",
                "tag_text_recall",
                "metadata_filter",
                "fusion_rerank",
                "relation_contrast_rerank",
                "interaction_frame_rerank",
                "document_chunk_search",
                "document_extract_qa",
                "strict_condition_rerank",
                "negative_condition_penalty",
                "clarification_hints",
                "multi_turn_chat",
            ],
        }

    def library_stats(self) -> dict:
        stats = self.metadata.stats()
        stats["json_index_assets"] = len(self.assets)
        return stats

    def duplicate_groups(self, limit: int = 20) -> list[dict]:
        return self.metadata.duplicate_groups(limit)

    def recent_search_logs(self, limit: int = 20) -> list[dict]:
        return self.metadata.recent_search_logs(limit)

    def add_ocr_text(
        self,
        asset_id: str,
        text: str,
        engine: str = "external-ocr",
        confidence: float | None = None,
    ) -> dict:
        known_ids = {asset.id for asset in self.assets}
        if asset_id not in known_ids:
            raise ValueError(f"unknown asset_id: {asset_id}")
        self.metadata.upsert_ocr_text(asset_id, text, engine, confidence)
        return {"asset_id": asset_id, "engine": engine, "text_length": len(text)}

    def add_text_signal(
        self,
        asset_id: str,
        text: str,
        text_type: str = "ocr",
        engine: str = "external",
        confidence: float | None = None,
        auto_fuse: bool = True,
    ) -> dict:
        text_type = normalize_text_type(text_type)
        result = self.add_ocr_text(
            asset_id=asset_id,
            text=text,
            engine=f"{text_type}:{engine}",
            confidence=confidence,
        )
        fused = self.fuse_asset_tags(asset_id) if auto_fuse else None
        return {**result, "text_type": text_type, "fusion": fused}

    def set_asset_tags(
        self,
        asset_id: str,
        tags: list[str],
        source: str = "external",
        confidence: float = 1.0,
        replace_source: bool = False,
    ) -> dict:
        asset = self.get_asset(asset_id)
        if asset is None:
            raise ValueError(f"unknown asset_id: {asset_id}")
        clean_tags = sorted({tag.strip().lower() for tag in tags if tag and tag.strip()})
        if replace_source:
            self.metadata.delete_asset_tags(asset_id, source)
        if clean_tags:
            self.metadata.set_asset_tags(asset_id, clean_tags, source=source, confidence=confidence)
        return {
            "asset_id": asset_id,
            "source": source,
            "confidence": confidence,
            "tags": self.metadata.get_asset_tags(asset_id),
            "written_count": len(clean_tags),
        }

    def fuse_asset_tags(self, asset_id: str) -> dict:
        asset = self.get_asset(asset_id)
        if asset is None:
            raise ValueError(f"unknown asset_id: {asset_id}")

        tag_rows = self.metadata.get_asset_tag_rows(asset_id)
        source_tags = {
            row["name"].strip().lower()
            for row in tag_rows
            if row.get("source") != "fusion" and row.get("name")
        }
        ocr_tags = extract_text_tags(self.metadata.get_ocr_text(asset_id))
        person_ids = self.metadata.get_asset_face_person_ids(asset_id)
        fused = set(source_tags)
        fused.update(ocr_tags)
        fused.add(asset.kind)
        fused.add(asset.library)
        if asset.kind == "video":
            fused.add("video_material")
        if asset.kind == "image":
            fused.add("image_material")
        if person_ids:
            fused.add("has_face")
            fused.update(person_ids)

        clean_tags = sorted(tag for tag in fused if is_useful_tag(tag))[:40]
        return self.set_asset_tags(
            asset_id,
            clean_tags,
            source="fusion",
            confidence=0.88,
            replace_source=True,
        )

    def get_asset(self, asset_id: str) -> Asset | None:
        return next((asset for asset in self.assets if asset.id == asset_id), None)

    def asset_detail(self, asset_id: str) -> dict:
        asset = self.get_asset(asset_id)
        if asset is None:
            raise ValueError(f"unknown asset_id: {asset_id}")
        tag_rows = self.metadata.get_asset_tag_rows(asset.id)
        manual_tags = [row["name"] for row in tag_rows if row["source"] == "manual"]
        text_signals = self.metadata.list_text_signals(asset.id)
        asr_text = " ".join(item["text"] for item in text_signals if item["text_type"] == "asr")
        visual_text = " ".join(
            item["text"] for item in text_signals if item["text_type"] in {"ocr", "subtitle_ocr", "manual_text", "document_text"}
        )
        return {
            "id": asset.id,
            "filename": asset.filename,
            "url": f"/uploads/{asset.filename}",
            "kind": asset.kind,
            "created_at": asset.created_at,
            "width": asset.width,
            "height": asset.height,
            "library": asset.library,
            "vector_model": asset.vector_model,
            "tags": self.metadata.get_asset_tags(asset.id),
            "manual_tags": manual_tags,
            "tag_rows": tag_rows,
            "favorite": self.metadata.has_asset_tag(asset.id, "favorite", source="favorite"),
            "faces": self.metadata.get_asset_faces(asset.id),
            "ocr_text": self.metadata.get_ocr_text(asset.id),
            "asr_text": asr_text,
            "visual_text": visual_text,
            "text_signals": text_signals,
            "document_chunks": self.metadata.list_document_chunks(asset.id)[:20] if asset.kind == "document" else [],
            "text_summary": summarize_text_signals(text_signals),
            "similar_assets": self.similar_assets(asset.id, limit=8, library=asset.library),
            "archive_suggestions": self.archive_suggestions(asset.id),
            "health": self.asset_health(asset.id),
        }

    def asset_health(self, asset_id: str) -> dict:
        asset = self.get_asset(asset_id)
        if asset is None:
            raise ValueError(f"unknown asset_id: {asset_id}")
        tags = self.metadata.get_asset_tags(asset.id)
        tag_rows = self.metadata.get_asset_tag_rows(asset.id)
        faces = self.metadata.get_asset_faces(asset.id)
        text_signals = self.metadata.list_text_signals(asset.id)
        issues = []
        actions = []
        score = 100

        if len(tags) <= 3:
            score -= 18
            issues.append("标签较少，语义检索和自动归档依据不足。")
            actions.append("补充人工标签或重新融合标签")
        if not any(row.get("source") == "auto_archive" for row in tag_rows):
            score -= 10
            issues.append("尚未采纳自动归档建议。")
            actions.append("在详情页采纳 Agent 归档建议")
        if asset.kind == "video" and not text_signals:
            score -= 24
            issues.append("视频还没有 OCR/ASR 文本信号。")
            actions.append("加入视频字幕 OCR 或音频转写任务")
        if asset.kind == "image" and asset.library == "personal" and not faces and has_people_tag(tags):
            score -= 14
            issues.append("个人库人物素材缺少人脸分组。")
            actions.append("重建个人库人物聚类")
        if min(asset.width, asset.height) < 320:
            score -= 12
            issues.append("素材分辨率偏低，展示或复用价值可能有限。")
            actions.append("标记为低清素材或人工复核")
        if self.metadata.has_asset_tag(asset.id, "favorite", source="favorite"):
            score += 4
        score = max(0, min(score, 100))
        if score >= 82:
            level = "good"
            label = "健康"
        elif score >= 60:
            level = "medium"
            label = "可用"
        else:
            level = "weak"
            label = "待整理"
        return {
            "score": score,
            "level": level,
            "label": label,
            "issues": issues or ["素材信息较完整，可以正常检索和归档。"],
            "actions": unique_text(actions),
        }

    def agent_recommendations(self, limit: int = 6) -> dict:
        assets = self.assets
        videos_without_text = []
        low_tag_assets = []
        no_archive_assets = []
        personal_people_candidates = []
        weak_health = []

        for asset in assets:
            tags = self.metadata.get_asset_tags(asset.id)
            tag_rows = self.metadata.get_asset_tag_rows(asset.id)
            text_signals = self.metadata.list_text_signals(asset.id)
            if asset.kind == "video" and not text_signals:
                videos_without_text.append(asset)
            if len(tags) <= 3:
                low_tag_assets.append(asset)
            if not any(row.get("source") == "auto_archive" for row in tag_rows):
                no_archive_assets.append(asset)
            if asset.library == "personal" and has_people_tag(tags) and not self.metadata.get_asset_faces(asset.id):
                personal_people_candidates.append(asset)
            health = self.asset_health(asset.id)
            if health["score"] < 60:
                weak_health.append({"asset": asset, "health": health})

        recommendations = [
            build_agent_task_recommendation(
                key="video_text_signals",
                title="为视频补充字幕 OCR / 文本信号",
                priority=92,
                count=len(videos_without_text),
                reason="视频没有文本信号时，无法发挥 C 模块的字幕提取、音频转写和文本检索能力。",
                payload={
                    "actions": ["text_signals"],
                    "library": "public",
                    "kind": "video",
                    "offset": 0,
                    "limit": min(max(len(videos_without_text), 1), 10),
                    "ocr_engine": "easyocr",
                    "asr_engine": "none",
                    "video_frames": 2,
                    "min_confidence": 0.45,
                },
            ),
            build_agent_task_recommendation(
                key="fusion_tags",
                title="重新融合低标签素材",
                priority=76,
                count=len(low_tag_assets),
                reason="标签过少会让语义检索、标签浏览和自动归档都变弱。",
                payload={
                    "actions": ["fusion_tags"],
                    "asset_ids": [asset.id for asset in low_tag_assets[:40]],
                },
            ),
            build_agent_task_recommendation(
                key="archive_suggestions",
                title="复核未归档素材",
                priority=68,
                count=len(no_archive_assets),
                reason="未采纳自动归档建议的素材，可以通过详情页进入统一归档标签体系。",
                payload=None,
            ),
            build_agent_task_recommendation(
                key="personal_face_clusters",
                title="重建个人库人物聚类",
                priority=64,
                count=len(personal_people_candidates),
                reason="个人库中疑似人物素材需要人物分组，后续才能把名字绑定到 person_id。",
                payload=None,
            ),
            build_agent_task_recommendation(
                key="weak_health_review",
                title="人工复核低健康度素材",
                priority=58,
                count=len(weak_health),
                reason="低健康度素材通常缺少标签、文本信号或归档信息，适合集中整理。",
                payload=None,
            ),
        ]
        recommendations = [item for item in recommendations if item["count"] > 0]
        recommendations.sort(key=lambda item: item["priority"], reverse=True)
        avg_health = round(sum(self.asset_health(asset.id)["score"] for asset in assets) / max(len(assets), 1), 1)
        return {
            "summary": {
                "asset_count": len(assets),
                "average_health": avg_health,
                "weak_count": len(weak_health),
                "video_without_text_count": len(videos_without_text),
                "low_tag_count": len(low_tag_assets),
            },
            "recommendations": recommendations[:limit],
        }

    def evaluate_retrieval(
        self,
        query_rows: list[dict],
        limit: int = 20,
        ks: list[int] | None = None,
        profiles: list[str] | None = None,
    ) -> dict:
        import statistics
        import time

        ks = ks or [1, 5, 10]
        profiles = profiles or ["clip_only", "clip_tags", "agent_v3"]
        profile_reports = []
        for profile in profiles:
            evaluations = []
            latencies = []
            for row in query_rows:
                query = str(row.get("query", "")).strip()
                if not query:
                    continue
                relevant_ids = {str(item) for item in row.get("relevant_ids", [])}
                library = normalize_library_filter(str(row.get("library", "all")))
                expanded_query = self.expand_person_names(query, library=library)
                plan = self.agent.build_plan(expanded_query)
                started = time.perf_counter()
                results = self.search_with_plan(
                    plan,
                    limit=limit,
                    library=library,
                    profile=profile,
                    log_search=False,
                )
                latency_ms = (time.perf_counter() - started) * 1000
                latencies.append(latency_ms)
                ranked_ids = [item["id"] for item in results]
                rank = first_relevant_rank(ranked_ids, relevant_ids)
                evaluations.append(
                    {
                        "query": query,
                        "expanded_query": expanded_query,
                        "library": library,
                        "relevant_ids": sorted(relevant_ids),
                        "ranked_ids": ranked_ids,
                        "top_result": ranked_ids[0] if ranked_ids else None,
                        "first_relevant_rank": rank,
                        "recall": {f"R@{k}": recall_at_k(ranked_ids, relevant_ids, k) for k in ks},
                        "mrr": 0.0 if rank is None else 1.0 / rank,
                        "latency_ms": round(latency_ms, 2),
                    }
                )
            profile_reports.append(
                {
                    "profile": profile,
                    "summary": {
                        "query_count": len(evaluations),
                        "labeled_query_count": sum(1 for item in evaluations if item["relevant_ids"]),
                        "recall": {
                            f"R@{k}": round(statistics.mean(item["recall"][f"R@{k}"] for item in evaluations), 4)
                            if evaluations
                            else 0.0
                            for k in ks
                        },
                        "mrr": round(statistics.mean(item["mrr"] for item in evaluations), 4) if evaluations else 0.0,
                        "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
                    },
                    "evaluations": evaluations,
                }
            )
        return {
            "limit": limit,
            "ks": ks,
            "profiles": profile_reports,
        }

    def archive_suggestions(self, asset_id: str, limit: int = 4) -> list[dict]:
        asset = self.get_asset(asset_id)
        if asset is None:
            raise ValueError(f"unknown asset_id: {asset_id}")
        tags = set(self.metadata.get_asset_tags(asset.id))
        faces = self.metadata.get_asset_faces(asset.id)
        text_signals = self.metadata.list_text_signals(asset.id)
        text_types = {item.get("text_type") for item in text_signals}
        text_blob = " ".join(item.get("text", "") for item in text_signals).lower()

        candidates = [
            build_archive_candidate(
                key="archive_text_rich_video",
                title="文本丰富视频",
                folder="视频素材 / 字幕与转写",
                score=score_archive_rule(
                    asset.kind == "video",
                    bool(text_signals),
                    "subtitle" in tags or "caption" in tags or "asr" in tags or "transcript" in tags,
                ),
                reasons=archive_reasons(
                    asset.kind == "video",
                    "视频素材",
                    bool(text_signals),
                    "已有 OCR/ASR 文本信号",
                    "subtitle" in tags or "caption" in tags or "asr" in tags or "transcript" in tags,
                    "命中字幕/转写相关标签",
                ),
                action_tags=["archive:text-rich-video", "text_rich", "subtitle_material"],
            ),
            build_archive_candidate(
                key="archive_interview_talk",
                title="访谈 / 演讲素材",
                folder="视频素材 / 访谈演讲",
                score=score_archive_rule(
                    asset.kind == "video",
                    bool(tags & {"interview", "talk", "lecture", "presentation", "speaker", "conversation"}),
                    any(word in text_blob for word in ["interview", "speaker", "talk", "presentation", "lecture"]),
                ),
                reasons=archive_reasons(
                    asset.kind == "video",
                    "视频素材",
                    bool(tags & {"interview", "talk", "lecture", "presentation", "speaker", "conversation"}),
                    "标签包含访谈、演讲或说话场景",
                    any(word in text_blob for word in ["interview", "speaker", "talk", "presentation", "lecture"]),
                    "文本内容出现访谈/演讲关键词",
                ),
                action_tags=["archive:interview-talk", "interview_material", "talking_head"],
            ),
            build_archive_candidate(
                key="archive_people",
                title="人物素材",
                folder="个人相册 / 人物",
                score=score_archive_rule(
                    bool(faces),
                    "has_face" in tags or bool(tags & {"person", "people", "portrait"}),
                    asset.library == "personal",
                ),
                reasons=archive_reasons(
                    bool(faces),
                    f"检测到 {len(faces)} 张人脸",
                    "has_face" in tags or bool(tags & {"person", "people", "portrait"}),
                    "标签包含人物或人像信息",
                    asset.library == "personal",
                    "来自个人库，适合进入人物相册归档",
                ),
                action_tags=["archive:people", "people_material", "portrait_material"],
            ),
            build_archive_candidate(
                key="archive_food",
                title="美食素材",
                folder="生活素材 / 美食",
                score=score_archive_rule(bool(tags & {"food", "meal", "coffee", "fruit", "pizza", "cake", "cooking"})),
                reasons=archive_reasons(
                    bool(tags & {"food", "meal", "coffee", "fruit", "pizza", "cake", "cooking"}),
                    "标签包含食物、饮品或烹饪信息",
                ),
                action_tags=["archive:food", "food_material"],
            ),
            build_archive_candidate(
                key="archive_nature",
                title="风景自然素材",
                folder="公共素材 / 风景自然",
                score=score_archive_rule(bool(tags & {"landscape", "nature", "mountain", "ocean", "waterfall", "lake", "sky", "flower"})),
                reasons=archive_reasons(
                    bool(tags & {"landscape", "nature", "mountain", "ocean", "waterfall", "lake", "sky", "flower"}),
                    "标签包含自然风景信息",
                ),
                action_tags=["archive:nature", "nature_material", "landscape_material"],
            ),
            build_archive_candidate(
                key="archive_city",
                title="城市建筑素材",
                folder="公共素材 / 城市建筑",
                score=score_archive_rule(bool(tags & {"city", "street", "building", "architecture", "traffic", "vehicle"})),
                reasons=archive_reasons(
                    bool(tags & {"city", "street", "building", "architecture", "traffic", "vehicle"}),
                    "标签包含城市、街道、建筑或交通信息",
                ),
                action_tags=["archive:city", "city_material", "urban_material"],
            ),
        ]
        rows = [item for item in candidates if item["confidence"] >= 0.34]
        rows.sort(key=lambda item: item["confidence"], reverse=True)
        if not rows:
            rows.append(
                build_archive_candidate(
                    key="archive_review_needed",
                    title="待人工复核",
                    folder="待整理 / 信息不足",
                    score=0.32,
                    reasons=["当前素材标签和文本信号较少，建议补充标签后再自动归档。"],
                    action_tags=["archive:review-needed", "needs_review"],
                )
            )
        return rows[:limit]

    def apply_archive_suggestion(self, asset_id: str, suggestion_key: str) -> dict:
        suggestions = self.archive_suggestions(asset_id, limit=12)
        suggestion = next((item for item in suggestions if item["key"] == suggestion_key), None)
        if suggestion is None:
            raise ValueError(f"unknown archive suggestion: {suggestion_key}")
        result = self.set_asset_tags(
            asset_id,
            suggestion["action_tags"],
            source="auto_archive",
            confidence=float(suggestion["confidence"]),
            replace_source=True,
        )
        fusion = self.fuse_asset_tags(asset_id)
        return {"suggestion": suggestion, "tags": result["tags"], "fusion": fusion}

    def set_favorite(self, asset_id: str, favorite: bool) -> dict:
        if self.get_asset(asset_id) is None:
            raise ValueError(f"unknown asset_id: {asset_id}")
        if favorite:
            self.set_asset_tags(asset_id, ["favorite"], source="favorite", confidence=1.0, replace_source=True)
        else:
            self.metadata.delete_asset_tags(asset_id, "favorite")
        self.fuse_asset_tags(asset_id)
        return {"asset_id": asset_id, "favorite": favorite}

    def tag_browser(self, tag: str | None = None, library: str = "all", limit: int = 120) -> dict:
        library = normalize_library_filter(library)
        facets = self.metadata.tag_facets(limit=limit, library=library)
        for facet in facets:
            facet["count"] = facet.get("asset_count", 0)
        selected_tag = (tag or "").strip().lower()
        assets = []
        if selected_tag:
            asset_ids = set(self.metadata.asset_ids_for_tag(selected_tag, library=library, limit=160))
            assets = [item for item in self.list_assets() if item["id"] in asset_ids]
        return {"tags": facets, "selected_tag": selected_tag, "assets": assets}

    def similar_assets(self, asset_id: str, limit: int = 8, library: str = "all") -> list[dict]:
        asset = self.get_asset(asset_id)
        if asset is None:
            raise ValueError(f"unknown asset_id: {asset_id}")
        library = normalize_library_filter(library)
        base_vector = np.array(asset.vector, dtype=np.float32)
        rows = []
        for candidate in self.assets:
            if candidate.id == asset.id or candidate.vector_model != self.encoder.name:
                continue
            if library != "all" and candidate.library != library:
                continue
            score = cosine_similarity(base_vector, np.array(candidate.vector, dtype=np.float32))
            rows.append(
                {
                    "id": candidate.id,
                    "filename": candidate.filename,
                    "url": f"/uploads/{candidate.filename}",
                    "kind": candidate.kind,
                    "library": candidate.library,
                    "width": candidate.width,
                    "height": candidate.height,
                    "similarity": round(float(score), 4),
                    "tags": self.metadata.get_asset_tags(candidate.id)[:6],
                }
            )
        rows.sort(key=lambda item: item["similarity"], reverse=True)
        return rows[:limit]

    def list_persons(self, library: str = "personal") -> list[dict]:
        persons = self.metadata.list_persons(normalize_library(library))
        for person in persons:
            for asset in person.get("assets", []):
                asset["url"] = f"/uploads/{asset['filename']}"
        return persons

    def delete_asset(self, asset_id: str, delete_file: bool = True) -> dict:
        asset = self.get_asset(asset_id)
        if asset is None:
            raise ValueError(f"unknown asset_id: {asset_id}")
        self.assets = [item for item in self.assets if item.id != asset_id]
        for chunk in self.metadata.list_document_chunks(asset_id):
            self.document_chunk_vectors.pop(str(chunk.get("chunk_id") or ""), None)
        self.save()
        self.save_document_chunk_index()
        self.rebuild_vector_index()
        self.metadata.delete_asset(asset_id)
        removed_file = False
        if delete_file:
            path = Path(asset.path)
            if not path.is_absolute():
                path = Path.cwd() / path
            if path.exists() and path.is_file():
                path.unlink()
                removed_file = True
        return {"asset_id": asset_id, "removed_file": removed_file}

    def merge_persons(self, target_person_id: str, source_person_ids: list[str], library: str = "personal") -> dict:
        library = normalize_library(library)
        affected_asset_ids = self.metadata.get_person_asset_ids(
            [target_person_id, *source_person_ids],
            library,
        )
        affected_assets = self.metadata.merge_persons(target_person_id, source_person_ids, library)
        self.add_person_merge_rule(affected_asset_ids)
        self.refresh_person_tags(affected_assets, library=library)
        return {
            "target_person_id": target_person_id,
            "merged_person_ids": [item for item in source_person_ids if item != target_person_id],
            "affected_assets": affected_assets,
            "persons": self.list_persons(library),
        }

    def delete_person(self, person_id: str, library: str = "personal") -> dict:
        library = normalize_library(library)
        affected_assets = self.metadata.delete_person(person_id, library)
        self.refresh_person_tags(affected_assets, library=library)
        return {
            "person_id": person_id,
            "affected_assets": affected_assets,
            "persons": self.list_persons(library),
        }

    def remove_asset_person(self, asset_id: str, person_id: str, library: str = "personal") -> dict:
        library = normalize_library(library)
        affected_assets = self.metadata.remove_asset_person(asset_id, person_id, library)
        self.refresh_person_tags(affected_assets, library=library)
        return {
            "asset_id": asset_id,
            "person_id": person_id,
            "persons": self.list_persons(library),
        }

    def rename_person(self, person_id: str, display_name: str, library: str = "personal") -> dict:
        library = normalize_library(library)
        self.metadata.update_person_name(person_id, display_name, library)
        affected_assets = self.metadata.get_person_asset_ids([person_id], library)
        self.refresh_person_tags(affected_assets, library=library)
        return {
            "person_id": person_id,
            "display_name": display_name.strip(),
            "affected_assets": affected_assets,
            "persons": self.list_persons(library),
        }

    def cleanup_person_clusters(self, library: str = "personal") -> dict:
        affected_assets = self.metadata.cleanup_person_clusters(normalize_library(library))
        self.refresh_person_tags(affected_assets, library=library)
        return {
            "affected_assets": affected_assets,
            "persons": self.list_persons(library),
        }

    def refresh_person_tags(self, asset_ids: list[str], library: str = "personal") -> None:
        person_names = self.metadata.person_name_map(normalize_library(library))
        for asset_id in sorted(set(asset_ids)):
            if self.get_asset(asset_id) is None:
                continue
            person_ids = self.metadata.get_asset_face_person_ids(asset_id)
            person_labels = []
            for person_id in person_ids:
                names = person_names.get(person_id, {})
                person_labels.extend(
                    str(name).strip()
                    for name in [names.get("display_name"), names.get("alias")]
                    if name and str(name).strip()
                )
            self.set_asset_tags(
                asset_id,
                sorted({"has_face", *person_ids, *person_labels}) if person_ids else [],
                source="person_cluster",
                confidence=0.9,
                replace_source=True,
            )
            self.fuse_asset_tags(asset_id)

    def repair_person_tags(self, library: str = "personal") -> dict:
        library = normalize_library(library)
        self.metadata.delete_asset_tags_by_source("person_cluster")
        face_asset_ids = self.metadata.face_asset_ids(library)
        self.refresh_person_tags(face_asset_ids, library=library)
        target_assets = [asset.id for asset in self.assets if asset.library == library]
        for asset_id in target_assets:
            if asset_id not in set(face_asset_ids):
                self.fuse_asset_tags(asset_id)
        return {
            "library": library,
            "face_asset_count": len(face_asset_ids),
            "refreshed_asset_count": len(target_assets),
            "persons": self.list_persons(library),
        }

    def add_person_ignored_assets(self, asset_ids: list[str]) -> None:
        ignored = set(self.load_person_ignored_assets())
        ignored.update(asset_id for asset_id in asset_ids if asset_id)
        self.person_ignore_path.write_text(
            json.dumps(sorted(ignored), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load_person_ignored_assets(self) -> list[str]:
        if not self.person_ignore_path.exists():
            return []
        try:
            return json.loads(self.person_ignore_path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def add_person_merge_rule(self, asset_ids: list[str]) -> None:
        clean_ids = sorted({asset_id for asset_id in asset_ids if asset_id})
        if len(clean_ids) < 2:
            return
        rules = self.load_person_merge_rules()
        existing = {tuple(rule.get("asset_ids", [])) for rule in rules}
        key = tuple(clean_ids)
        if key not in existing:
            rules.append({"asset_ids": clean_ids})
            self.person_merge_rules_path.write_text(
                json.dumps(rules, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def load_person_merge_rules(self) -> list[dict]:
        if not self.person_merge_rules_path.exists():
            return []
        try:
            raw = json.loads(self.person_merge_rules_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        if not isinstance(raw, list):
            return []
        rules = []
        for item in raw:
            asset_ids = item.get("asset_ids") if isinstance(item, dict) else None
            if not isinstance(asset_ids, list):
                continue
            clean_ids = sorted({str(asset_id) for asset_id in asset_ids if asset_id})
            if len(clean_ids) >= 2:
                rules.append({"asset_ids": clean_ids})
        return rules

    def reindex_metadata(self) -> None:
        self.metadata.sync_assets(self.assets, Path.cwd())

    def rebuild_vector_index(self) -> None:
        vectors = []
        ids = []
        for index, asset in enumerate(self.assets):
            if asset.vector_model != self.encoder.name:
                continue
            vector = np.array(asset.vector, dtype=np.float32)
            if vector.size == 0:
                continue
            vectors.append(vector)
            ids.append(index)
        self.vector_index.build(vectors, ids)

    def rebuild_stale_assets(self) -> None:
        changed = False
        rebuilt: list[Asset] = []
        for asset in self.assets:
            if asset.vector_model == self.encoder.name:
                rebuilt.append(asset)
                continue

            path = Path(asset.path)
            if not path.is_absolute():
                path = Path.cwd() / path
            if not path.exists():
                rebuilt.append(asset)
                continue

            with Image.open(path) as image:
                image = image.convert("RGB")
                asset.vector = self.encoder.encode_image(image).tolist()
                asset.vector_model = self.encoder.name
                asset.width, asset.height = image.size
                rebuilt.append(asset)
                changed = True

        self.assets = rebuilt
        if changed:
            self.save()

    def list_assets(self) -> list[dict]:
        return [
            {
                "id": item.id,
                "filename": item.filename,
                "url": f"/uploads/{item.filename}",
                "kind": item.kind,
                "library": item.library,
                "created_at": item.created_at,
                "width": item.width,
                "height": item.height,
                "favorite": self.metadata.has_asset_tag(item.id, "favorite", source="favorite"),
                "tags": self.metadata.get_asset_tags(item.id)[:8],
            }
            for item in reversed(self.assets)
        ]


def best_vector_score(queries: list[str], query_vectors: list[np.ndarray], image_vector: np.ndarray) -> tuple[float, str]:
    if not query_vectors:
        return 0.0, ""
    if any(vector.shape != image_vector.shape for vector in query_vectors):
        return 0.0, ""
    scored = [
        (cosine_similarity(vector, image_vector), queries[index])
        for index, vector in enumerate(query_vectors)
    ]
    score, prompt = max(scored, key=lambda item: item[0])
    return score, prompt


def normalize_library(value: str) -> str:
    value = str(value or "").strip().lower()
    return value if value in {"public", "personal"} else "public"


def normalize_library_filter(value: str) -> str:
    value = str(value or "").strip().lower()
    return value if value in {"all", "public", "personal"} else "all"


def normalize_text_type(value: str) -> str:
    value = str(value or "").strip().lower()
    aliases = {
        "subtitle": "subtitle_ocr",
        "subtitles": "subtitle_ocr",
        "document": "document_text",
        "doc": "document_text",
        "speech": "asr",
        "audio": "asr",
        "whisper": "asr",
    }
    value = aliases.get(value, value)
    return value if value in {"ocr", "subtitle_ocr", "asr", "manual_text", "document_text"} else "ocr"


def build_archive_candidate(
    key: str,
    title: str,
    folder: str,
    score: float,
    reasons: list[str],
    action_tags: list[str],
) -> dict:
    return {
        "key": key,
        "title": title,
        "folder": folder,
        "confidence": round(max(0.0, min(float(score), 0.98)), 4),
        "reasons": reasons,
        "action_tags": action_tags,
    }


def score_archive_rule(*signals: bool) -> float:
    if not signals:
        return 0.0
    matched = sum(1 for item in signals if item)
    if matched == 0:
        return 0.0
    return min(0.28 + matched * 0.22, 0.94)


def archive_reasons(*pairs: object) -> list[str]:
    reasons = []
    for index in range(0, len(pairs), 2):
        if bool(pairs[index]):
            reasons.append(str(pairs[index + 1]))
    return reasons


def build_agent_task_recommendation(
    key: str,
    title: str,
    priority: int,
    count: int,
    reason: str,
    payload: dict | None,
) -> dict:
    return {
        "key": key,
        "title": title,
        "priority": priority,
        "count": count,
        "reason": reason,
        "payload": payload,
    }


def has_people_tag(tags: list[str]) -> bool:
    return bool(set(tags) & {"has_face", "person", "people", "portrait", "people_material", "portrait_material"})


def unique_text(values: Iterable[str]) -> list[str]:
    seen = set()
    rows = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            rows.append(value)
    return rows


def summarize_text_signals(signals: list[dict]) -> dict:
    grouped = {
        "asr": {"label": "音频转写", "count": 0, "summary": "", "engines": []},
        "visual": {"label": "画面/字幕 OCR", "count": 0, "summary": "", "engines": []},
        "manual": {"label": "人工文本", "count": 0, "summary": "", "engines": []},
    }
    buckets = {"asr": [], "visual": [], "manual": []}
    for signal in signals:
        text_type = signal.get("text_type", "ocr")
        if text_type == "asr":
            key = "asr"
        elif text_type in {"manual_text", "document_text"}:
            key = "manual"
        else:
            key = "visual"
        grouped[key]["count"] += 1
        engine = signal.get("engine")
        if engine and engine not in grouped[key]["engines"]:
            grouped[key]["engines"].append(engine)
        buckets[key].append(signal.get("text", ""))

    for key, texts in buckets.items():
        grouped[key]["summary"] = compact_text(" ".join(texts), max_chars=260)
    return grouped


def compact_text(text: str, max_chars: int = 260) -> str:
    clean = " ".join((text or "").split())
    if len(clean) <= max_chars:
        return clean
    sentence_end = max(
        clean.rfind(".", 0, max_chars),
        clean.rfind("。", 0, max_chars),
        clean.rfind("!", 0, max_chars),
        clean.rfind("?", 0, max_chars),
    )
    if sentence_end >= 90:
        return clean[: sentence_end + 1]
    return clean[:max_chars].rstrip() + "..."


def infer_library_from_filename(filename: str) -> str:
    filename = filename.lower()
    public_prefixes = ("commons_",)
    public_video_samples = (
        "red_square_motion",
        "blue_sky_grass_motion",
        "yellow_ball_motion",
        "green_block_motion",
    )
    if filename.startswith(public_prefixes) or any(filename.startswith(item) for item in public_video_samples):
        return "public"
    return "public"


def metadata_match(plan: QueryPlan, asset: Asset) -> tuple[float, list[str]]:
    expected = 0
    matched = 0
    hits = []
    year = plan.executable_filters.get("year")
    season = plan.executable_filters.get("season")

    if year:
        expected += 1
        if created_year(asset.created_at) == year:
            matched += 1
            hits.append(str(year))
    if season:
        expected += 1
        if created_month(asset.created_at) in SEASON_MONTHS.get(str(season), set()):
            matched += 1
            hits.append(str(season))
    if not expected:
        return 0.0, []
    return matched / expected, hits


def constraint_match(
    plan: QueryPlan,
    asset: Asset,
    tags: list[str],
    text_signals: list[dict],
    faces: list[dict],
) -> tuple[float, list[str], float, list[str]]:
    expected = 0
    matched = 0
    hits: list[str] = []
    misses: list[str] = []
    penalty = 0.0
    tag_text = " ".join(tags).lower()
    filename = asset.filename.lower()
    text_blob = " ".join(str(item.get("text", "")) for item in text_signals).lower()
    signal_types = {str(item.get("text_type", "")).lower() for item in text_signals}

    people_count = plan.strict_conditions.get("people_count")
    if isinstance(people_count, int) and people_count > 0:
        expected += 1
        detected_people = len({str(face.get("person_id", "")) for face in faces if face.get("person_id")})
        if detected_people >= people_count:
            matched += 1
            hits.append(f"人物数量>={people_count}")
        else:
            misses.append(f"人物数量不足{people_count}")
            penalty += 0.08

    text_signal_aliases = {
        "ocr": {"ocr", "subtitle_ocr"},
        "subtitle": {"subtitle_ocr", "ocr"},
        "asr": {"asr"},
        "document": {"document_text", "manual_text"},
    }
    for signal in plan.unresolved_conditions.get("text_signals", []):
        expected += 1
        allowed = text_signal_aliases.get(signal, {signal})
        tag_hit = signal in tag_text or ("subtitle" in tag_text and signal == "subtitle")
        if signal_types & allowed or tag_hit:
            matched += 1
            hits.append(f"文本信号:{signal}")
        else:
            misses.append(f"缺少文本信号:{signal}")
            penalty += 0.05

    for quality in plan.unresolved_conditions.get("quality", []):
        if quality == "exclude_dark":
            dark_hit = any(word in f"{tag_text} {filename}" for word in ["dark", "night", "low_light", "underexposed", "太暗"])
            if dark_hit:
                misses.append("排除太暗")
                penalty += 0.1
            else:
                hits.append("未命中太暗排除项")
        elif quality in {"clear", "clear_subject"}:
            expected += 1
            if any(word in tag_text for word in ["clear", "sharp", "portrait", "person", "people", "subject"]):
                matched += 1
                hits.append(f"质量:{quality}")

    object_terms = plan.unresolved_conditions.get("objects", [])
    object_alias_hits = 0
    missing_important_objects = []
    for obj in object_terms:
        aliases = tag_aliases_for_query_term(obj)
        if aliases and any(alias in tag_text for alias in aliases):
            object_alias_hits += 1
            hits.append(f"物体:{obj}")
        elif obj in {"手机", "电话", "智能手机"}:
            missing_important_objects.append(obj)
    if object_terms:
        expected += min(len(object_terms), 3)
        matched += min(object_alias_hits, 3)
    if missing_important_objects:
        misses.append(f"缺少关键物体:{'、'.join(unique_text(missing_important_objects))}")
        penalty += 0.08

    negative_terms = []
    for values in plan.negative_conditions.values():
        negative_terms.extend(values)
    for term in unique_text(str(item).lower() for item in negative_terms if str(item).strip()):
        if term and (term in tag_text or term in filename or term in text_blob):
            misses.append(f"排除:{term}")
            penalty += 0.14

    if asset.kind == "document" and plan.interaction_frames and not wants_document_results(plan):
        misses.append("视觉交互查询降低文档权重")
        penalty += 0.35
    elif asset.kind == "document" and not wants_document_results(plan):
        misses.append("非文档查询降低文档权重")
        penalty += 0.18
    if asset.kind != "document" and wants_document_results(plan):
        misses.append("文档查询降低非文档权重")
        penalty += 0.12

    if not expected:
        score = 0.0
    else:
        score = matched / expected
    return max(0.0, min(score, 1.0)), hits[:8], min(penalty, 0.45), misses[:8]


def wants_document_results(plan: QueryPlan) -> bool:
    if plan.executable_filters.get("kind") == "document":
        return True
    return "document" in plan.unresolved_conditions.get("media", []) or "document" in plan.unresolved_conditions.get("text_signals", [])


def document_intent_strength(plan: QueryPlan) -> float:
    query = str(plan.raw_query or "").lower()
    score = 0.0
    if plan.executable_filters.get("kind") == "document":
        score += 0.85
    if "document" in plan.unresolved_conditions.get("media", []):
        score += 0.65
    if "document" in plan.unresolved_conditions.get("text_signals", []):
        score += 0.35
    strong_words = [
        "文档",
        "文章",
        "报告",
        "资料",
        "论文",
        "计划书",
        "方案",
        "正文",
        "大意",
        "主旨",
        "段落",
        "document",
        "article",
        "report",
        "paper",
        "proposal",
    ]
    weak_words = ["关于", "有关", "有关于", "提到", "讲述", "内容", "关键词", "总结", "摘要"]
    visual_words = ["照片", "图片", "图像", "视频", "合影", "相册", "相片", "photo", "image", "video"]
    if any(word in query for word in strong_words):
        score += 0.65
    if any(word in query for word in weak_words):
        score += 0.22
    if any(word in query for word in visual_words):
        score -= 0.55
    return max(0.0, min(score, 1.0))


def should_use_document_branch(plan: QueryPlan) -> bool:
    if document_intent_strength(plan) >= 0.55:
        return True
    query = str(plan.raw_query or "").lower()
    document_words = [
        "文档",
        "文章",
        "报告",
        "资料",
        "正文",
        "主旨",
        "关键词",
        "讲述",
        "提到",
        "关于",
        "有关于",
        "内容",
        "article",
        "document",
        "report",
    ]
    visual_words = ["照片", "图片", "图像", "视频", "合影", "相片", "photo", "image", "video"]
    return any(word in query for word in document_words) and not any(word in query for word in visual_words)


def merge_search_results(visual_results: list[dict], document_results: list[dict], limit: int) -> list[dict]:
    merged_by_id: dict[str, dict] = {}
    for row in [*document_results, *visual_results]:
        asset_id = str(row.get("id") or "")
        if not asset_id:
            continue
        existing = merged_by_id.get(asset_id)
        if existing is None or float(row.get("score") or 0.0) > float(existing.get("score") or 0.0):
            merged_by_id[asset_id] = row
    merged = sorted(merged_by_id.values(), key=lambda item: float(item.get("score") or 0.0), reverse=True)
    return merged[:limit]


def interaction_prompt_boost(plan: QueryPlan, best_prompt: str, tags: list[str]) -> tuple[float, list[str]]:
    prompt = (best_prompt or "").lower()
    tag_text = " ".join(tags).lower()
    actions = set(plan.unresolved_conditions.get("actions", []))
    objects = set(plan.unresolved_conditions.get("objects", []))
    wants_phone = bool(objects & {"手机", "电话", "智能手机"})
    if not wants_phone:
        return 0.0, []
    phone_aliases = tag_aliases_for_query_term("手机") | tag_aliases_for_query_term("电话")
    if not any(alias in tag_text for alias in phone_aliases):
        return 0.0, []
    if actions & {"打电话", "通话", "接电话"}:
        if any(phrase in prompt for phrase in ["talking on the phone", "phone call", "making a phone call", "calling"]):
            return 0.065, ["动作提示:打电话"]
        if any(phrase in prompt for phrase in ["holding a mobile phone", "using a smartphone", "mobile phone"]):
            return 0.025, ["物体提示:手机"]
    if actions & {"拿着", "手持", "举着"}:
        if "holding" in prompt and any(phone in prompt for phone in ["phone", "smartphone"]):
            return 0.055, ["动作提示:手持手机"]
    return 0.0, []


def build_interaction_vectors(encoder: Encoder, plan: QueryPlan) -> list[dict[str, object]]:
    rows = []
    for frame in getattr(plan, "interaction_frames", []):
        positive_prompts = [str(item) for item in frame.get("positive_prompts", []) if str(item).strip()]
        negative_prompts = [str(item) for item in frame.get("negative_prompts", []) if str(item).strip()]
        rows.append(
            {
                "frame": frame,
                "positive_vectors": [encoder.encode_text(prompt) for prompt in positive_prompts],
                "negative_vectors": [encoder.encode_text(prompt) for prompt in negative_prompts],
            }
        )
    return rows


def interaction_frame_match(
    plan: QueryPlan,
    image_vector: np.ndarray,
    tags: list[str],
    interaction_vectors: list[dict[str, object]],
) -> tuple[float, list[str], list[str]]:
    if not interaction_vectors:
        return 0.0, [], []
    tag_text = " ".join(tags).lower()
    total_boost = 0.0
    hits = []
    misses = []
    for item in interaction_vectors:
        frame = item["frame"]
        action = str(frame.get("action", "interaction"))
        obj = str(frame.get("object", ""))
        aliases = tag_aliases_for_query_term(obj)
        has_object = not obj or any(alias in tag_text for alias in aliases)
        has_actor = any(alias in tag_text for alias in {"person", "people", "man", "woman"})
        positive = max((cosine_similarity(image_vector, vector) for vector in item.get("positive_vectors", [])), default=0.0)
        negative = max((cosine_similarity(image_vector, vector) for vector in item.get("negative_vectors", [])), default=positive)
        margin = positive - negative
        if has_actor and has_object and margin >= -0.005:
            boost = 0.04 + min(max((margin + 0.005) / 0.08, 0.0), 1.0) * 0.06
            total_boost += boost
            hits.append(f"交互:{action}:{obj or 'object'}")
        elif frame.get("requires_object", True) and not has_object:
            total_boost -= 0.055
            misses.append(f"缺少交互物体:{obj}")
        elif frame.get("requires_actor", True) and not has_actor:
            total_boost -= 0.045
            misses.append(f"缺少交互主体:{obj or action}")
        elif margin < -0.02:
            total_boost -= 0.035
            misses.append(f"动作关系弱:{action}")
    return max(min(total_boost, 0.16), -0.16), hits[:6], misses[:6]


def weighted_search_score(
    profile: str,
    clip_score: float,
    tag_score: float,
    metadata_score: float,
    filename_score: float,
    ocr_score: float,
    relation_score: float,
) -> float:
    if profile == "clip_only":
        return float(clip_score)
    if profile == "clip_tags":
        return float(0.82 * clip_score + 0.18 * tag_score)
    return float(
        0.63 * clip_score
        + 0.14 * tag_score
        + 0.06 * metadata_score
        + 0.09 * filename_score
        + 0.03 * ocr_score
        + 0.05 * relation_score
    )


def first_relevant_rank(ranked_ids: list[str], relevant_ids: set[str]) -> int | None:
    if not relevant_ids:
        return None
    for index, asset_id in enumerate(ranked_ids, start=1):
        if asset_id in relevant_ids:
            return index
    return None


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    hits = set(ranked_ids[:k]) & relevant_ids
    return len(hits) / len(relevant_ids)


def relation_match(
    plan: QueryPlan,
    best_prompt: str,
    image_vector: np.ndarray,
    relation_vectors: list[np.ndarray],
    negative_relation_vectors: list[np.ndarray],
) -> tuple[float, list[str], float]:
    relations = plan.unresolved_conditions.get("relations", [])
    if not relations:
        return 0.0, [], 0.0
    prompt = (best_prompt or "").lower()
    hits = [
        relation
        for relation in relations
        if relation.lower() in prompt or relation_prompt_text(relation).lower() in prompt
    ]
    margin = relation_margin_score(image_vector, relation_vectors, negative_relation_vectors)
    contrast_score = relation_margin_to_score(margin)
    if hits:
        return min(contrast_score + 0.15, 1.0), hits, margin
    partial_hits = []
    for relation in relations:
        parts = relation_prompt_text(relation).lower().split()
        if len(parts) >= 3 and all(part in prompt for part in parts):
            partial_hits.append(relation)
    if partial_hits:
        return min(contrast_score + 0.08, 1.0), partial_hits, margin
    return contrast_score, [], margin


def relation_margin_score(
    image_vector: np.ndarray,
    relation_vectors: list[np.ndarray],
    negative_relation_vectors: list[np.ndarray],
) -> float:
    if not relation_vectors:
        return 0.0
    positive = max(cosine_similarity(image_vector, vector) for vector in relation_vectors)
    negative = max((cosine_similarity(image_vector, vector) for vector in negative_relation_vectors), default=positive)
    return float(positive - negative)


def relation_margin_to_score(margin: float) -> float:
    if margin <= -0.03:
        return 0.0
    if margin >= 0.08:
        return 1.0
    return max(0.0, min((margin + 0.03) / 0.11, 1.0))


def relation_prompt_text(relation: str) -> str:
    text = re.sub(
        r"person[_-]?(\d+)",
        lambda match: f"person {int(match.group(1))}",
        relation,
        flags=re.IGNORECASE,
    )
    return text.replace("left_of", "left of").replace("right_of", "right of")


def geometric_relation_match(plan: QueryPlan, faces: list[dict]) -> tuple[float, list[str]]:
    relations = plan.unresolved_conditions.get("relations", [])
    if not relations or not faces:
        return 0.0, []
    face_by_person = {}
    for face in faces:
        person_id = str(face.get("person_id", "")).lower()
        if person_id:
            face_by_person.setdefault(person_id, []).append(face)

    hits = []
    for relation in relations:
        parsed = parse_person_relation(relation)
        if parsed is None:
            continue
        subject, predicate, target = parsed
        subject_faces = face_by_person.get(subject, [])
        target_faces = face_by_person.get(target, [])
        if not subject_faces or not target_faces:
            continue
        if any(check_bbox_relation(s_face["bbox"], predicate, t_face["bbox"]) for s_face in subject_faces for t_face in target_faces):
            hits.append(relation)
    if not hits:
        return 0.0, []
    return 1.0, hits


def parse_person_relation(relation: str) -> tuple[str, str, str] | None:
    match = re.fullmatch(r"person[_-]?(\d+)\s+(.+?)\s+person[_-]?(\d+)", relation.strip(), flags=re.IGNORECASE)
    if not match:
        return None
    subject = f"person_{match.group(1).zfill(4)}"
    predicate = match.group(2).strip()
    target = f"person_{match.group(3).zfill(4)}"
    return subject, predicate, target


def check_bbox_relation(subject_bbox: list[float], predicate: str, target_bbox: list[float]) -> bool:
    sx1, sy1, sx2, sy2 = [float(item) for item in subject_bbox]
    tx1, ty1, tx2, ty2 = [float(item) for item in target_bbox]
    subject_center = ((sx1 + sx2) / 2, (sy1 + sy2) / 2)
    target_center = ((tx1 + tx2) / 2, (ty1 + ty2) / 2)
    subject_area = max((sx2 - sx1) * (sy2 - sy1), 1.0)
    target_area = max((tx2 - tx1) * (ty2 - ty1), 1.0)
    tolerance = max((subject_area**0.5 + target_area**0.5) * 0.08, 6.0)

    if predicate == "left_of":
        return subject_center[0] < target_center[0] - tolerance
    if predicate == "right_of":
        return subject_center[0] > target_center[0] + tolerance
    if predicate in {"on", "above"}:
        return subject_center[1] < target_center[1] - tolerance
    if predicate in {"under", "below"}:
        return subject_center[1] > target_center[1] + tolerance
    if predicate == "inside":
        return sx1 >= tx1 and sy1 >= ty1 and sx2 <= tx2 and sy2 <= ty2
    if predicate == "beside":
        return abs(subject_center[0] - target_center[0]) > tolerance
    return False


def tag_match(plan: QueryPlan, tags: list[str]) -> tuple[float, list[str]]:
    if not tags:
        return 0.0, []
    query_tokens = query_terms(plan)
    tag_set = {tag.lower() for tag in tags}
    alias_terms = set()
    for values in plan.unresolved_conditions.values():
        for value in values:
            alias_terms.update(tag_aliases_for_query_term(str(value)))
    searchable_tags = set()
    for tag in tag_set:
        searchable_tags.add(tag)
        searchable_tags.add(tag.replace(" ", "_"))
        searchable_tags.add(tag.replace(" ", "-"))
    hits = sorted(tag for tag in tag_set if tag in query_tokens or tag in alias_terms)
    alias_variants = set()
    for alias in alias_terms:
        alias_variants.add(alias)
        alias_variants.add(alias.replace(" ", "_"))
        alias_variants.add(alias.replace(" ", "-"))
    hits.extend(sorted(tag for tag in tag_set if tag in alias_variants and tag not in hits))
    if not hits:
        return 0.0, []
    denominator = max(min(len(query_tokens), 8), 1)
    return min(len(hits) / denominator, 1.0), hits[:8]


def tag_aliases_for_query_term(term: str) -> set[str]:
    normalized = str(term or "").strip().lower()
    stable_aliases = {
        "\u624b\u673a": {"cell phone", "mobile phone", "smartphone", "phone", "cell_phone", "mobile_phone"},
        "\u667a\u80fd\u624b\u673a": {"cell phone", "mobile phone", "smartphone", "phone"},
        "\u7535\u8bdd": {"cell phone", "mobile phone", "smartphone", "phone", "telephone"},
        "\u4eba": {"person", "people", "man", "woman"},
        "\u7537\u4eba": {"person", "man"},
        "\u5973\u4eba": {"person", "woman"},
        "\u5b69\u5b50": {"person", "child"},
        "\u7535\u8111": {"computer", "laptop", "laptop computer"},
        "\u7b14\u8bb0\u672c\u7535\u8111": {"laptop", "laptop computer"},
        "\u8f66": {"car", "vehicle", "bus", "truck"},
        "\u6c7d\u8f66": {"car", "vehicle"},
        "\u81ea\u884c\u8f66": {"bicycle", "bike"},
        "\u6469\u6258\u8f66": {"motorcycle"},
        "\u732b": {"cat"},
        "\u72d7": {"dog"},
        "\u62ab\u8428": {"pizza"},
        "\u4e09\u660e\u6cbb": {"sandwich"},
        "\u86cb\u7cd5": {"cake"},
        "\u5496\u5561": {"coffee"},
        "\u7bee\u7403": {"basketball", "ball"},
        "\u8db3\u7403": {"football", "soccer ball", "ball"},
        "\u4e66": {"book"},
        "\u98df\u7269": {"food", "meal", "pizza", "sandwich", "cake", "fruit"},
        "\u996e\u6599": {"drink", "beverage", "cup", "coffee"},
        "\u676f\u5b50": {"cup"},
        "\u7897": {"bowl"},
        "\u9910\u684c": {"dining table", "table"},
    }
    if normalized in stable_aliases:
        return stable_aliases[normalized]
    aliases = {
        "手机": {"cell phone", "mobile phone", "smartphone", "phone", "cell_phone", "mobile_phone"},
        "智能手机": {"cell phone", "mobile phone", "smartphone", "phone"},
        "电话": {"cell phone", "mobile phone", "smartphone", "phone", "telephone"},
        "人": {"person", "people", "man", "woman"},
        "男人": {"person", "man"},
        "女人": {"person", "woman"},
        "电脑": {"computer", "laptop", "laptop computer"},
        "笔记本电脑": {"laptop", "laptop computer"},
        "车": {"car", "vehicle", "bus", "truck"},
        "汽车": {"car", "vehicle"},
        "猫": {"cat"},
        "狗": {"dog"},
    }
    return aliases.get(normalized, {normalized} if re.fullmatch(r"[a-z][a-z0-9 _-]+", normalized) else set())


def ocr_match(plan: QueryPlan, text: str) -> tuple[float, list[str]]:
    if not text:
        return 0.0, []
    lowered = text.lower()
    hits = sorted(term for term in query_terms(plan) if len(term) >= 2 and term in lowered)
    if not hits:
        return 0.0, []
    return min(len(hits) / max(len(query_terms(plan)), 1), 1.0), hits[:8]


def query_terms(plan: QueryPlan) -> set[str]:
    text = " ".join([plan.raw_query, *plan.semantic_queries]).lower()
    text = f"{text} {expand_chinese_query(text)}".lower()
    tokens = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", text))
    for values in plan.unresolved_conditions.values():
        for value in values:
            if re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_-]+", value):
                tokens.add(value.lower())
            for person_id in re.findall(r"person\s+(\d+)", str(value), flags=re.IGNORECASE):
                tokens.add(f"person_{person_id.zfill(4)}")
            for person_id in re.findall(r"person[_-]?(\d+)", str(value), flags=re.IGNORECASE):
                tokens.add(f"person_{person_id.zfill(4)}")
    aliases = {
        "people": {"person", "people"},
        "group": {"person", "people"},
        "photo": set(),
        "picture": set(),
        "image": set(),
        "food": {"pizza", "sandwich", "cake", "banana", "apple", "orange", "broccoli", "carrot"},
        "cooking": {"food", "cook"},
        "coffee": {"food", "drink"},
        "fruit": {"food"},
        "vehicle": {"car", "bus", "truck", "motorcycle", "bicycle", "train"},
        "traffic": {"car", "bus", "truck", "vehicle"},
        "pet": {"cat", "dog"},
        "animal": {"cat", "dog", "bird"},
        "nature": {"waterfall", "ocean", "mountain", "landscape"},
        "waterfall": {"nature"},
        "ocean": {"sea", "waves", "nature"},
        "sports": {"basketball", "football", "running"},
        "basketball": {"sports", "ball"},
        "dance": {"people", "person"},
        "interview": {"talk", "speaker", "conversation", "guest"},
        "presentation": {"lecture", "talk", "conference", "speaker", "slides"},
        "lecture": {"presentation", "talk", "speaker"},
        "subtitle": {"caption", "text", "asr"},
        "caption": {"subtitle", "text"},
        "conference": {"presentation", "talk", "meeting"},
        "meeting": {"conference", "talk"},
        "journalism": {"interview", "story"},
    }
    expanded = set(tokens)
    for token in tokens:
        expanded.update(aliases.get(token, set()))
    return {token for token in expanded if token not in {"photo", "picture", "image", "a", "the", "of", "at", "in", "on"}}


def extract_text_tags(text: str) -> set[str]:
    if not text:
        return set()
    lowered = text.lower()
    tokens = [
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", lowered)
        if token not in TEXT_STOPWORDS and not token.endswith("n't")
    ]
    tags = {token for token, _ in Counter(tokens).most_common(30)}
    chinese_tokens = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    tags.update(chinese_tokens[:20])
    aliases = {
        "咖啡": "coffee",
        "美食": "food",
        "食物": "food",
        "风景": "landscape",
        "天空": "sky",
        "城市": "city",
        "街道": "street",
        "人物": "person",
    }
    for key, value in aliases.items():
        if key in text:
            tags.add(value)
    return tags


def is_useful_tag(tag: str) -> bool:
    if not tag or len(tag) < 2:
        return False
    if len(tag) > 40:
        return False
    return tag not in {"none", "null", "undefined", "image", "photo"}


TEXT_STOPWORDS = {
    "about",
    "add",
    "after",
    "again",
    "all",
    "also",
    "and",
    "are",
    "because",
    "before",
    "being",
    "better",
    "between",
    "but",
    "can",
    "could",
    "did",
    "didn",
    "does",
    "done",
    "each",
    "every",
    "for",
    "far",
    "feel",
    "final",
    "from",
    "going",
    "good",
    "get",
    "got",
    "great",
    "had",
    "has",
    "have",
    "here",
    "how",
    "into",
    "just",
    "like",
    "lot",
    "many",
    "make",
    "more",
    "much",
    "not",
    "only",
    "out",
    "pretty",
    "over",
    "really",
    "seem",
    "sort",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "they",
    "this",
    "time",
    "trying",
    "was",
    "were",
    "what",
    "when",
    "which",
    "with",
    "work",
    "would",
    "you",
    "your",
}


def apply_display_scores(rows: list[dict]) -> None:
    if not rows:
        return
    raw_scores = [float(row["score"]) for row in rows]
    top = max(raw_scores)
    bottom = min(raw_scores)
    spread = max(top - bottom, 1e-6)
    for rank, row in enumerate(rows):
        raw_score = max(0.0, min(float(row["score"]), 1.0))
        relative = (raw_score - bottom) / spread
        rank_prior = max(0.0, 1.0 - rank * 0.035)
        display = 0.18 + 0.62 * raw_score + 0.12 * relative + 0.08 * rank_prior
        row["display_score"] = round(max(0.0, min(display, 0.92)), 4)


def explain_search_match(
    clip_score: float,
    tag_hits: list[str],
    ocr_hits: list[str],
    metadata_hits: list[str],
    constraint_hits: list[str],
    constraint_misses: list[str],
    relation_hits: list[str],
    filename_score: float,
    best_prompt: str,
) -> list[str]:
    reasons = []
    if clip_score >= 0.18:
        if best_prompt.startswith("tag:"):
            reasons.append(f"标签召回命中：{best_prompt.removeprefix('tag:')}")
        else:
            prompt = f"（{best_prompt}）" if best_prompt else ""
            reasons.append(f"视觉语义与查询相似{prompt}")
    if tag_hits:
        reasons.append(f"标签命中：{'、'.join(tag_hits[:5])}")
    if relation_hits:
        reasons.append(f"关系提示命中：{'、'.join(relation_hits[:3])}")
    if ocr_hits:
        reasons.append(f"文本信号命中：{'、'.join(ocr_hits[:5])}")
    if metadata_hits:
        reasons.append(f"元数据条件命中：{'、'.join(metadata_hits[:5])}")
    if constraint_hits:
        reasons.append(f"结构化条件命中：{'、'.join(constraint_hits[:4])}")
    if constraint_misses:
        reasons.append(f"结构化条件降权：{'、'.join(constraint_misses[:3])}")
    if filename_score >= 0.5:
        reasons.append("文件名与查询关键词匹配")
    return reasons[:4] or ["主要依据 CLIP 跨模态语义相似度排序"]


def created_year(value: str) -> int | None:
    try:
        return datetime.fromisoformat(value).year
    except ValueError:
        return None


def created_month(value: str) -> int | None:
    try:
        return datetime.fromisoformat(value).month
    except ValueError:
        return None


def image_to_vector(image: Image.Image) -> np.ndarray:
    small = image.resize((128, 128))
    arr = np.asarray(small, dtype=np.float32) / 255.0
    means = arr.mean(axis=(0, 1))
    stds = arr.std(axis=(0, 1))

    hsv = small.convert("HSV")
    hsv_arr = np.asarray(hsv, dtype=np.float32)
    hue_hist, _ = np.histogram(hsv_arr[:, :, 0], bins=12, range=(0, 255), density=True)
    sat_hist, _ = np.histogram(hsv_arr[:, :, 1], bins=6, range=(0, 255), density=True)
    val_hist, _ = np.histogram(hsv_arr[:, :, 2], bins=6, range=(0, 255), density=True)

    gray = small.convert("L")
    gray_arr = np.asarray(gray, dtype=np.float32) / 255.0
    gx = np.abs(np.diff(gray_arr, axis=1)).mean()
    gy = np.abs(np.diff(gray_arr, axis=0)).mean()
    contrast = ImageStat.Stat(gray).stddev[0] / 255.0
    aspect = image.width / max(image.height, 1)
    aspect_feature = math.tanh((aspect - 1.0) / 2.0)

    feature = np.concatenate(
        [
            means,
            stds,
            hue_hist,
            sat_hist,
            val_hist,
            np.array([gx, gy, contrast, aspect_feature], dtype=np.float32),
        ]
    ).astype(np.float32)
    return normalize(feature)


def text_to_vector(query: str) -> np.ndarray:
    query = query.lower()
    feature = np.zeros(34, dtype=np.float32)
    color_map = {
        ("\u7ea2", "red", "\u665a\u971e", "\u5915\u9633", "\u706b"): [0.9, 0.2, 0.15],
        ("\u7eff", "green", "\u8349\u5730", "\u68ee\u6797", "\u6811"): [0.2, 0.75, 0.25],
        ("\u84dd", "blue", "\u5929\u7a7a", "\u6d77", "\u6e56", "\u6c34"): [0.2, 0.35, 0.9],
        ("\u767d", "white", "\u96ea", "\u51ac\u5929", "\u51ac\u5b63"): [0.9, 0.9, 0.85],
        ("\u9ed1", "black", "\u591c\u665a", "\u591c\u666f"): [0.08, 0.08, 0.1],
        ("\u9ec4", "yellow", "\u91d1\u8272", "\u9633\u5149"): [0.9, 0.75, 0.2],
    }
    for words, rgb in color_map.items():
        if any(word in query for word in words):
            feature[:3] += np.array(rgb, dtype=np.float32)

    if any(word in query for word in ["\u660e\u4eae", "\u6674\u5929", "\u9633\u5149", "\u767d\u5929"]):
        feature[29] += 1.0
    if any(word in query for word in ["\u6697", "\u591c\u665a", "\u591c\u666f", "\u9ed1"]):
        feature[29] -= 0.5
    if any(word in query for word in ["\u590d\u6742", "\u5efa\u7b51", "\u57ce\u5e02", "\u8857\u9053", "\u4eba\u7fa4"]):
        feature[30:33] += 0.8
    if any(word in query for word in ["\u6a2a\u5c4f", "\u98ce\u666f", "\u5168\u666f"]):
        feature[33] += 0.7
    if any(word in query for word in ["\u7ad6\u5c4f", "\u81ea\u62cd", "\u4eba\u50cf"]):
        feature[33] -= 0.7

    if not np.any(feature):
        feature[:3] = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        feature[29:33] = 0.3
    return normalize(feature)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(normalize(left), normalize(right)))


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def keyword_score(query: str, filename: str) -> float:
    query = query.lower().strip()
    filename = filename.lower()
    if not query:
        return 0.0
    tokens = [token for token in re_split_query(query) if token]
    if not tokens:
        return 1.0 if query in filename else 0.0
    return sum(1 for token in tokens if token in filename) / len(tokens)


def re_split_query(query: str) -> list[str]:
    separators = ["_", "-", "\uff0c", "\u3002", "\u3001", "\u7684", "\u5728", "\u548c", "\u4e0e", "\u8ddf"]
    for separator in separators:
        query = query.replace(separator, " ")
    return query.split()


def build_encoder() -> Encoder:
    try:
        return ClipEncoder()
    except Exception as exc:
        print(f"CLIP unavailable, fallback to lite visual encoder: {exc}")
        return LiteEncoder()


def expand_chinese_query(query: str) -> str:
    translations = {
        "\u84dd\u5929": "blue sky",
        "\u5929\u7a7a": "sky",
        "\u6d77\u8fb9": "beach seaside ocean",
        "\u5927\u6d77": "ocean sea",
        "\u96ea": "snow",
        "\u51ac\u5929": "winter snow",
        "\u591c\u666f": "night city lights",
        "\u57ce\u5e02": "city street urban",
        "\u8857\u9053": "street city urban",
        "\u4ea4\u901a": "traffic vehicles city street",
        "\u5efa\u7b51": "building architecture",
        "\u6545\u5bab": "Forbidden City palace Chinese traditional architecture",
        "\u5408\u5f71": "group photo people portrait",
        "\u81ea\u62cd": "selfie portrait person",
        "\u4eba\u50cf": "portrait person",
        "\u805a\u9910": "dinner restaurant food people",
        "\u98df\u7269": "food meal",
        "\u7f8e\u98df": "food meal",
        "\u505a\u996d": "cooking food kitchen",
        "\u70f9\u996a": "cooking food kitchen",
        "\u5496\u5561": "coffee drink cup",
        "\u6c34\u679c": "fruit food",
        "\u732b": "cat",
        "\u72d7": "dog",
        "\u9e1f": "bird animal",
        "\u52a8\u7269": "animal",
        "\u8f66": "car vehicle",
        "\u706b\u8f66": "train railway",
        "\u7bee\u7403": "basketball sports ball",
        "\u8db3\u7403": "football soccer sports ball",
        "\u8df3\u821e": "dance people dancing",
        "\u8bbf\u8c08": "interview talk conversation guest speaker",
        "\u91c7\u8bbf": "interview journalism reporter guest",
        "\u5bf9\u8bdd": "conversation interview talk",
        "\u6f14\u8bb2": "lecture presentation talk speaker",
        "\u8bb2\u5ea7": "lecture presentation education speaker",
        "\u6c47\u62a5": "presentation slides conference talk",
        "\u4f1a\u8bae": "conference meeting presentation discussion",
        "\u5b57\u5e55": "subtitle caption text speech",
        "\u8bed\u97f3": "speech audio transcript asr",
        "\u8f6c\u5199": "transcript speech text asr",
        "\u89e3\u8bf4": "narrator narration speech",
        "\u82b1": "flower",
        "\u5c71": "mountain",
        "\u7011\u5e03": "waterfall nature water",
        "\u6d77\u6d6a": "ocean waves sea",
        "\u6e56": "lake water",
        "\u5c0f\u5b69": "child kid",
        "\u5b9d\u5b9d": "baby child",
        "\u8fd0\u52a8": "sports running",
    }
    words = [english for chinese, english in translations.items() if chinese in query]
    if not words:
        return query
    return f"{query}. {' '.join(words)}"


def parse_document_with_b_module(path: Path, doc_id: str | None = None) -> dict:
    if b_parse_document is None:
        return {}
    try:
        result = b_parse_document(str(path), doc_id=doc_id)
        if getattr(result, "status", "ok") == "failed":
            return {}
        chunks = b_chunk_document(result) if b_chunk_document is not None else []
        tags = b_generate_document_tags(result, chunks) if b_generate_document_tags is not None else []
        return {
            "provider": "B.document_module",
            "result": result,
            "chunks": chunks,
            "tags": tags,
            "text": getattr(result, "text", "") or "",
            "title": getattr(result, "title", "") or "",
            "format": getattr(result, "format", "") or path.suffix.lower().removeprefix("."),
            "pages": getattr(result, "pages", 0) or 0,
        }
    except Exception:
        return {}


def build_document_chunks_from_b_module(asset: Asset, parsed_document: dict | None) -> list[dict]:
    if not parsed_document:
        return []
    raw_chunks = parsed_document.get("chunks") or []
    chunks = []
    for index, raw in enumerate(raw_chunks):
        text = str(getattr(raw, "text", "") or "")
        if not text.strip():
            continue
        chunk_id = str(getattr(raw, "chunk_id", "") or f"{asset.id}_chunk_{index:04d}")
        if not chunk_id.startswith(asset.id):
            chunk_id = f"{asset.id}_{chunk_id}"
        section_title = str(getattr(raw, "section_title", "") or "")
        heading_path = list(getattr(raw, "heading_path", []) or [])
        summary = str(getattr(raw, "summary", "") or summarize_text_snippet(text, max_chars=180))
        embedding_text = str(getattr(raw, "embedding_text", "") or "")
        if not embedding_text:
            embedding_text = "\n".join(item for item in [asset.filename, section_title, summary, text] if item)
        chunks.append(
            {
                "asset_id": asset.id,
                "chunk_id": chunk_id,
                "chunk_index": int(getattr(raw, "chunk_index", index) or index),
                "page_start": getattr(raw, "page_start", None),
                "page_end": getattr(raw, "page_end", None),
                "section_title": section_title,
                "heading_path": heading_path,
                "chunk_type": str(getattr(raw, "chunk_type", "") or "text"),
                "text": text,
                "summary": summary,
                "embedding_text": embedding_text,
                "keywords": extract_document_keywords(" ".join([section_title, summary, text])),
            }
        )
    return chunks


def build_document_tags_from_b_module(parsed_document: dict | None) -> list[str]:
    if not parsed_document:
        return []
    tags = []
    for tag in parsed_document.get("tags") or []:
        name = str(getattr(tag, "name", "") or "").strip()
        confidence = float(getattr(tag, "confidence", 0.0) or 0.0)
        if name and confidence >= 0.45 and is_useful_tag(name):
            tags.append(name)
    fmt = str(parsed_document.get("format") or "").strip()
    if fmt:
        tags.append(fmt)
    tags.append("document")
    return sorted(set(tags))[:28]


def document_chunk_context_for_b_qa(chunks: list[dict]) -> list[dict]:
    contexts = []
    for chunk in chunks:
        contexts.append(
            {
                "chunk_id": chunk.get("chunk_id", ""),
                "doc_id": chunk.get("asset_id", ""),
                "filename": chunk.get("filename", ""),
                "page": chunk.get("page_start"),
                "section_title": chunk.get("section_title", ""),
                "text": chunk.get("text") or chunk.get("snippet") or "",
                "summary": chunk.get("summary", ""),
            }
        )
    return contexts


def build_document_chunks(asset: Asset, text: str, suffix: str) -> list[dict]:
    paragraphs = normalize_document_paragraphs(text)
    chunks: list[dict] = []
    current: list[str] = []
    heading_path: list[str] = []
    current_title = ""
    target_chars = 650
    hard_limit = 1000

    def flush() -> None:
        nonlocal current
        chunk_text = "\n".join(current).strip()
        if not chunk_text:
            current = []
            return
        chunk_index = len(chunks)
        summary = summarize_text_snippet(chunk_text, max_chars=180)
        keywords = extract_document_keywords(" ".join([asset.filename, current_title, chunk_text]))
        embedding_text = "\n".join(
            item
            for item in [
                asset.filename,
                current_title,
                " ".join(keywords[:12]),
                summary,
                chunk_text,
            ]
            if item
        )
        chunks.append(
            {
                "asset_id": asset.id,
                "chunk_id": f"{asset.id}_chunk_{chunk_index:04d}",
                "chunk_index": chunk_index,
                "page_start": None,
                "page_end": None,
                "section_title": current_title,
                "heading_path": heading_path[-4:],
                "chunk_type": "text",
                "text": chunk_text,
                "summary": summary,
                "embedding_text": embedding_text,
                "keywords": keywords,
            }
        )
        overlap = tail_overlap(chunk_text, 90)
        current = [overlap] if overlap and len(chunk_text) > hard_limit else []

    for paragraph in paragraphs:
        heading = parse_document_heading(paragraph, suffix)
        if heading:
            if current:
                flush()
            current_title = heading
            heading_path.append(heading)
            continue
        if len(paragraph) > hard_limit:
            if current:
                flush()
            for piece in split_long_paragraph(paragraph, target_chars, hard_limit):
                current.append(piece)
                flush()
            continue
        tentative_len = len("\n".join([*current, paragraph]))
        if current and tentative_len > target_chars:
            flush()
        current.append(paragraph)
    if current:
        flush()

    if not chunks and text.strip():
        clean = text.strip()
        chunks.append(
            {
                "asset_id": asset.id,
                "chunk_id": f"{asset.id}_chunk_0000",
                "chunk_index": 0,
                "page_start": None,
                "page_end": None,
                "section_title": "",
                "heading_path": [],
                "chunk_type": "text",
                "text": clean,
                "summary": summarize_text_snippet(clean),
                "embedding_text": clean,
                "keywords": extract_document_keywords(clean),
            }
        )
    return chunks


def normalize_document_paragraphs(text: str) -> list[str]:
    lines = [line.strip() for line in str(text or "").splitlines()]
    paragraphs: list[str] = []
    buffer: list[str] = []
    for line in lines:
        if not line:
            if buffer:
                paragraphs.append(" ".join(buffer).strip())
                buffer = []
            continue
        if parse_document_heading(line, ""):
            if buffer:
                paragraphs.append(" ".join(buffer).strip())
                buffer = []
            paragraphs.append(line)
            continue
        if len(line) <= 30 and re.match(r"^(\d+[\.\u3001]|\u7b2c.+[\u7ae0\u8282]|[一二三四五六七八九十]+[\u3001.])", line):
            if buffer:
                paragraphs.append(" ".join(buffer).strip())
                buffer = []
            paragraphs.append(line)
            continue
        buffer.append(line)
    if buffer:
        paragraphs.append(" ".join(buffer).strip())
    return [item for item in paragraphs if item]


def parse_document_heading(paragraph: str, suffix: str) -> str:
    text = paragraph.strip()
    if not text:
        return ""
    md_match = re.match(r"^(#{1,6})\s+(.+)$", text)
    if md_match:
        return md_match.group(2).strip()[:80]
    if len(text) <= 48 and re.match(r"^(\d+(\.\d+)*[\.\u3001]?\s+|第.+[章节]\s*|[一二三四五六七八九十]+[\u3001.]\s*)", text):
        return re.sub(r"\s+", " ", text)[:80]
    if suffix in {"pptx"} and len(text) <= 36:
        return text
    return ""


def split_long_paragraph(text: str, target_chars: int, hard_limit: int) -> list[str]:
    sentences = re.split(r"(?<=[。！？.!?])\s*", text)
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if not sentence:
            continue
        if current and len(current) + len(sentence) > target_chars:
            pieces.append(current.strip())
            current = ""
        if len(sentence) > hard_limit:
            for index in range(0, len(sentence), target_chars):
                pieces.append(sentence[index : index + target_chars].strip())
        else:
            current += sentence
    if current.strip():
        pieces.append(current.strip())
    return pieces


def tail_overlap(text: str, max_chars: int) -> str:
    clean = " ".join(text.split())
    if len(clean) <= max_chars:
        return ""
    return clean[-max_chars:]


def summarize_text_snippet(text: str, max_chars: int = 220) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= max_chars:
        return clean
    split_at = max(
        clean.rfind("。", 0, max_chars),
        clean.rfind(".", 0, max_chars),
        clean.rfind("！", 0, max_chars),
        clean.rfind("？", 0, max_chars),
        clean.rfind(";", 0, max_chars),
    )
    if split_at >= 80:
        return clean[: split_at + 1]
    return clean[:max_chars].rstrip() + "..."


def extract_document_keywords(text: str, limit: int = 16) -> list[str]:
    clean = str(text or "")
    english = [
        token.lower()
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", clean)
        if token.lower() not in TEXT_STOPWORDS
    ]
    chinese = [
        token
        for token in re.findall(r"[\u4e00-\u9fff]{2,8}", clean)
        if token not in {"我们", "这个", "一个", "可以", "进行", "相关", "内容", "文档", "素材"}
    ]
    ranked = []
    seen = set()
    for token, _ in Counter([*english, *chinese]).most_common(limit * 2):
        if token not in seen and is_useful_tag(token):
            ranked.append(token)
            seen.add(token)
        if len(ranked) >= limit:
            break
    return ranked


def build_document_level_tags(filename: str, chunks: list[dict], suffix: str) -> list[str]:
    text = " ".join(
        [
            filename,
            suffix,
            *[str(keyword) for chunk in chunks[:8] for keyword in chunk.get("keywords", [])],
            *[str(chunk.get("section_title") or "") for chunk in chunks[:12]],
        ]
    )
    tags = set(extract_document_keywords(text, limit=14))
    tags.update({"document", suffix})
    if any(term in text for term in ["教育", "课程", "教学", "education", "learning"]):
        tags.update({"education", "教学"})
    if any(term in text for term in ["计算机", "人工智能", "算法", "computer", "software", "ai"]):
        tags.update({"computer", "计算机"})
    if any(term in text for term in ["春天", "花", "柳树", "温暖", "spring"]):
        tags.update({"春天", "spring"})
    return sorted(tag for tag in tags if is_useful_tag(tag))[:24]


def normalize_document_content_query(query: str) -> str:
    cleaned = str(query or "")
    format_words = [
        "的文档",
        "文档",
        "文章",
        "报告",
        "资料",
        "正文",
        "文件",
        "论文",
        "计划书",
        "方案",
        "pdf",
        "docx",
        "pptx",
        "markdown",
        "md",
        "document",
        "article",
        "report",
        "paper",
        "素材",
        "资源",
        "文件",
        "找点",
        "找",
        "搜索",
        "检索",
        "查找",
        "随便",
        "一些",
        "点",
    ]
    for word in format_words:
        cleaned = re.sub(re.escape(word), " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" 的关于有关有关于")
    return cleaned


def expand_document_query(query: str) -> str:
    expanded = expand_chinese_query(query)
    semantic_expansions = {
        "春天": "春季 花开 柳树 温暖 万物复苏 spring blossom warm season",
        "春季": "春天 花开 柳树 温暖 spring",
        "计算机教育": "信息技术课程 编程 教学 人工智能 computer education curriculum programming",
        "计算机": "computer software algorithm programming information technology",
        "教育": "teaching learning curriculum school education",
        "答辩": "presentation report ppt slide defense project",
        "项目": "system design implementation experiment evaluation",
        "接口": "api endpoint request response integration",
        "ocr": "text extraction subtitle image text",
        "asr": "speech transcript audio whisper",
    }
    additions = [value for key, value in semantic_expansions.items() if key.lower() in query.lower()]
    if additions:
        return " ".join([expanded, *additions])
    return expanded


def document_keyword_match(query: str, chunk: dict) -> tuple[float, list[str]]:
    text = " ".join(
        [
            str(chunk.get("section_title") or ""),
            str(chunk.get("summary") or ""),
            str(chunk.get("text") or ""),
            " ".join(str(item) for item in chunk.get("keywords") or []),
        ]
    ).lower()
    terms = extract_document_keywords(query, limit=24)
    terms.extend(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{1,}", query.lower()))
    terms.extend(re.findall(r"[\u4e00-\u9fff]{2,8}", query))
    terms.extend(chinese_query_terms(query))
    unique_terms = []
    for term in terms:
        if term and term not in unique_terms:
            unique_terms.append(term)
    hits = [term for term in unique_terms if term.lower() in text]
    if not unique_terms:
        return 0.0, []
    return min(len(hits) / min(len(unique_terms), 10), 1.0), hits[:10]


def chinese_query_terms(query: str) -> list[str]:
    terms = []
    for block in re.findall(r"[\u4e00-\u9fff]{2,}", str(query or "")):
        for size in [4, 3, 2]:
            if len(block) < size:
                continue
            for index in range(0, len(block) - size + 1):
                term = block[index : index + size]
                if term not in {"什么", "一下", "这个", "那个", "需要", "可以", "文章"}:
                    terms.append(term)
    return unique_text(terms)


def highlight_document_snippet(text: str, query: str, max_chars: int = 260) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= max_chars:
        return clean
    terms = extract_document_keywords(query, limit=8)
    positions = [clean.lower().find(term.lower()) for term in terms if clean.lower().find(term.lower()) >= 0]
    if positions:
        center = min(positions)
        start = max(center - max_chars // 3, 0)
        end = min(start + max_chars, len(clean))
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(clean) else ""
        return prefix + clean[start:end].strip() + suffix
    return clean[:max_chars].rstrip() + "..."


def build_document_match_reason(keyword_hits: list[str], chunk: dict) -> str:
    reasons = []
    section = str(chunk.get("section_title") or "")
    if section:
        reasons.append(f"命中章节：{section}")
    if keyword_hits:
        reasons.append("关键词/语义词：" + "、".join(keyword_hits[:6]))
    if not reasons:
        reasons.append("chunk 向量与查询语义接近")
    return "；".join(reasons)


def extract_document_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return read_text_file(path)
    if suffix == ".docx":
        return extract_docx_text(path)
    if suffix == ".pptx":
        return extract_pptx_text(path)
    if suffix == ".pdf":
        return extract_pdf_text(path)
    return ""


def read_text_file(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ["utf-8", "utf-8-sig", "gbk", "latin-1"]:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def extract_docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.startswith("word/") and name.endswith(".xml")]
        chunks = [xml_text_nodes(archive.read(name).decode("utf-8", errors="ignore")) for name in names]
    return clean_document_text("\n".join(chunks))


def extract_pptx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        names = sorted(name for name in archive.namelist() if name.startswith("ppt/slides/slide") and name.endswith(".xml"))
        chunks = [xml_text_nodes(archive.read(name).decode("utf-8", errors="ignore")) for name in names]
    return clean_document_text("\n".join(chunks))


def extract_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception as exc:
        raise ValueError("PDF text extraction requires pypdf. Install it or use DOCX/TXT first.") from exc
    reader = PdfReader(str(path))
    chunks = []
    for page in reader.pages:
        chunks.append(page.extract_text() or "")
    return clean_document_text("\n".join(chunks))


def xml_text_nodes(xml: str) -> str:
    values = re.findall(r"<[^:/>\s]+:t[^>]*>(.*?)</[^:/>\s]+:t>", xml, flags=re.DOTALL)
    if not values:
        values = re.findall(r"<t[^>]*>(.*?)</t>", xml, flags=re.DOTALL)
    return "\n".join(html_unescape(strip_xml_tags(value)) for value in values)


def strip_xml_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


def html_unescape(text: str) -> str:
    return (
        text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&apos;", "'")
    )


def clean_document_text(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def iter_existing_images(upload_dir: Path) -> Iterable[Path]:
    for path in upload_dir.iterdir():
        if path.suffix.lower() in SUPPORTED_IMAGE_TYPES:
            yield path


def iter_existing_assets(upload_dir: Path) -> Iterable[Path]:
    for path in upload_dir.iterdir():
        if path.suffix.lower() in SUPPORTED_IMAGE_TYPES | SUPPORTED_VIDEO_TYPES | SUPPORTED_DOCUMENT_TYPES:
            yield path


def extract_video_frames(path: Path, frame_count: int = 8) -> list[Image.Image]:
    try:
        import av
    except Exception as exc:
        raise ValueError("?????????????? av ????") from exc

    if frame_count <= 0:
        return []

    frames: list[Image.Image] = []
    try:
        with av.open(str(path)) as container:
            stream = next((item for item in container.streams if item.type == "video"), None)
            if stream is None:
                return []

            duration = None
            if stream.duration and stream.time_base:
                duration = float(stream.duration * stream.time_base)
            elif container.duration:
                duration = float(container.duration / 1_000_000)

            if duration and duration > 0:
                sample_seconds = np.linspace(0, max(duration * 0.9, 0), num=frame_count)
                for second in sample_seconds:
                    try:
                        container.seek(int(second / float(stream.time_base)), stream=stream, backward=True)
                    except Exception:
                        container.seek(int(second * 1_000_000), backward=True)
                    for frame in container.decode(stream):
                        frames.append(frame.to_image().convert("RGB"))
                        break
                    if len(frames) >= frame_count:
                        break

            if not frames:
                for frame in container.decode(stream):
                    frames.append(frame.to_image().convert("RGB"))
                    if len(frames) >= frame_count:
                        break
    except Exception as exc:
        raise ValueError(f"video decoding failed: {exc}") from exc

    if not frames:
        return []
    return frames[:frame_count]
