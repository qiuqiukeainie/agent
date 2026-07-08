"""
文档模块数据结构定义。

所有核心数据结构集中在此，方便 parser / chunker / tagger / qa 统一引用。
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DocumentSection:
    """文档中的一个章节/区块。

    对于 MD/DOCX：对应一个标题块。
    对于 PDF：对应页内的段落组。
    对于 PPTX：对应一页幻灯片。
    """

    section_id: str
    title: str                          # 章节标题
    heading_path: list[str] = field(default_factory=list)  # 完整标题路径
    page_start: Optional[int] = None    # 起始页码 (1-indexed)
    page_end: Optional[int] = None      # 结束页码
    text: str = ""                      # 该 section 的完整文本


@dataclass
class DocumentParseResult:
    """文档解析结果。"""

    doc_id: str
    filename: str
    format: str                         # "md" | "txt" | "docx" | "pdf" | "pptx"
    title: str
    pages: int                          # 总页数（对 TXT/MD 估算为 1）
    text: str                           # 全文文本
    sections: list[DocumentSection] = field(default_factory=list)
    status: str = "ok"                  # "ok" | "empty_text" | "partial" | "failed"
    warnings: list[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class DocumentChunk:
    """文档切片。"""

    chunk_id: str
    doc_id: str
    chunk_index: int
    chunk_type: str = "text"            # "text" | "code" | "heading_only" | "table"
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section_id: str = ""
    section_title: str = ""
    heading_path: list[str] = field(default_factory=list)
    text: str = ""
    summary: str = ""
    embedding_text: str = ""


@dataclass
class DocumentTag:
    """文档标签。"""

    name: str
    type: str                           # "topic" | "domain" | "content" | "genre" | "usage"
    confidence: float
    evidence: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 异常定义
# ---------------------------------------------------------------------------

class DocumentModuleError(Exception):
    """文档模块基础异常。"""
    pass


class DocumentParseError(DocumentModuleError):
    """文档解析失败。"""
    pass


class DocumentUnsupportedError(DocumentModuleError):
    """不支持的文档格式。"""
    pass
