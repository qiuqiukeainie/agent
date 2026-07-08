"""
文档理解、切片检索与文档问答模块。

将文档从"整篇一个向量"升级为"多切片、多标签、可定位、可解释、可问答"的多模态素材单元。

快速使用:

    from document_module import parse_document, chunk_document, generate_document_tags, answer_from_chunks

    result = parse_document("example.md")
    chunks = chunk_document(result)
    tags = generate_document_tags(result, chunks)
    answer = answer_from_chunks("这个问题", [chunk_to_dict(c) for c in chunks])
"""

from .schema import (
    DocumentSection,
    DocumentParseResult,
    DocumentChunk,
    DocumentTag,
    DocumentModuleError,
    DocumentParseError,
    DocumentUnsupportedError,
)

from .parser import parse_document
from .chunker import chunk_document
from .tagger import generate_document_tags, build_tag_context
from .qa import answer_from_chunks
from .llm_client import LLMClient, HeuristicLLMClient
from .retriever import SemanticRetriever

__all__ = [
    # 核心函数
    "parse_document",
    "chunk_document",
    "generate_document_tags",
    "build_tag_context",
    "answer_from_chunks",
    # 数据结构
    "DocumentSection",
    "DocumentParseResult",
    "DocumentChunk",
    "DocumentTag",
    # LLM 客户端
    "LLMClient",
    "HeuristicLLMClient",
    # 语义检索
    "SemanticRetriever",
    # 异常
    "DocumentModuleError",
    "DocumentParseError",
    "DocumentUnsupportedError",
]
