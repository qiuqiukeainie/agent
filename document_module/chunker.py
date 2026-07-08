"""
文档切片器。

将 DocumentParseResult 的 sections 切分为 DocumentChunk 列表。

切片优先级：标题边界 > 段落边界 > 句子边界 > 固定长度兜底
目标长度：~500 中文字，范围 300-800，最长 1200
重叠：~100 字
"""

import re
from typing import Optional

from .schema import DocumentChunk, DocumentParseResult
from .llm_client import LLMClient


# ---------------------------------------------------------------------------
# 公开接口
# ---------------------------------------------------------------------------

def chunk_document(
    parse_result: DocumentParseResult,
    llm_client: Optional[LLMClient] = None,
    target_chars: int = 500,
    min_chars: int = 300,
    max_chars: int = 800,
    hard_max_chars: int = 1200,
    overlap_chars: int = 100,
) -> list[DocumentChunk]:
    """将解析结果切分为 chunk 列表。

    Args:
        parse_result: 解析结果。
        llm_client: 可选的 LLM 客户端，用于生成摘要。无则用前 80 字截断。
        target_chars: 目标字数。
        min_chars: 最小字数（低于此尽量合并）。
        max_chars: 最大字数（尽量不超过）。
        hard_max_chars: 硬上限（代码块和表格可超过）。
        overlap_chars: 相邻 chunk 重叠字数。

    Returns:
        DocumentChunk 列表。
    """
    chunks: list[DocumentChunk] = []
    chunk_counter = 0

    for section in parse_result.sections:
        if not section.text.strip():
            continue

        section_chunks = _chunk_section(
            section=section,
            doc_id=parse_result.doc_id,
            chunk_counter=chunk_counter,
            target_chars=target_chars,
            min_chars=min_chars,
            max_chars=max_chars,
            hard_max_chars=hard_max_chars,
            overlap_chars=overlap_chars,
        )
        chunks.extend(section_chunks)
        chunk_counter += len(section_chunks)

    # 生成摘要 & embedding_text
    _fill_summaries(chunks, llm_client)

    return chunks


# ---------------------------------------------------------------------------
# 内部实现
# ---------------------------------------------------------------------------

def _chunk_section(
    section,
    doc_id: str,
    chunk_counter: int,
    target_chars: int,
    min_chars: int,
    max_chars: int,
    hard_max_chars: int,
    overlap_chars: int,
) -> list[DocumentChunk]:
    """切分单个 section。"""

    section_title = " > ".join(section.heading_path) if section.heading_path else section.title

    # 极短 section → 单 chunk
    text = section.text.strip()
    if len(text) <= target_chars:
        return [_make_chunk(
            doc_id=doc_id,
            chunk_index=chunk_counter,
            section=section,
            section_title=section_title,
            text=text,
            chunk_type="text",
        )]

    # 先按段落边界切
    paragraphs = _split_paragraphs(text)

    # 对过长段落按句子切
    atoms: list[str] = []
    for para in paragraphs:
        if len(para) > max_chars:
            atoms.extend(_split_sentences_long(para, max_chars))
        else:
            atoms.append(para)

    # 按目标长度聚合
    raw_chunks = _aggregate_atoms(atoms, target_chars, min_chars, hard_max_chars, overlap_chars)

    # 组装 DocumentChunk
    chunks: list[DocumentChunk] = []
    for i, (ctext, ctype) in enumerate(raw_chunks):
        chunks.append(_make_chunk(
            doc_id=doc_id,
            chunk_index=chunk_counter + i,
            section=section,
            section_title=section_title,
            text=ctext,
            chunk_type=ctype,
        ))

    return chunks


def _make_chunk(
    doc_id: str,
    chunk_index: int,
    section,
    section_title: str,
    text: str,
    chunk_type: str = "text",
) -> DocumentChunk:
    """构造一个 DocumentChunk。"""
    return DocumentChunk(
        chunk_id=f"{doc_id}_chunk_{chunk_index:04d}",
        doc_id=doc_id,
        chunk_index=chunk_index,
        chunk_type=chunk_type,
        page_start=section.page_start,
        page_end=section.page_end,
        section_id=section.section_id,
        section_title=section_title,
        heading_path=list(section.heading_path),
        text=text,
        summary="",
        embedding_text="",
    )


def _split_paragraphs(text: str) -> list[str]:
    """按双换行 / 单换行切段落。"""
    # 先按双换行
    parts = re.split(r'\n\s*\n', text)
    result: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        result.append(p)
    return result


def _split_sentences_long(text: str, max_chars: int) -> list[str]:
    """对过长文本按句子切分，句子仍超长则按固定长度切。"""
    raw = re.split(r'(?<=[。！？])\s*', text)
    sentences = [s.strip() for s in raw if s.strip()]

    result: list[str] = []
    for sent in sentences:
        if len(sent) <= max_chars:
            result.append(sent)
        else:
            # 按逗号、分号二次切
            sub = re.split(r'(?<=[，,；;])\s*', sent)
            sub = [s.strip() for s in sub if s.strip()]
            for ss in sub:
                if len(ss) <= max_chars:
                    result.append(ss)
                else:
                    # 固定长度兜底
                    for k in range(0, len(ss), max_chars):
                        result.append(ss[k:k + max_chars])
    return result


def _aggregate_atoms(
    atoms: list[str],
    target_chars: int,
    min_chars: int,
    hard_max_chars: int,
    overlap_chars: int,
) -> list[tuple[str, str]]:
    """将原子文本段聚合成目标大小的 chunk。

    Returns:
        [(text, chunk_type), ...]
    """
    if not atoms:
        return []

    chunks: list[tuple[str, str]] = []
    buf: list[str] = []
    buf_len = 0

    def _flush(overlap_text: str = "") -> Optional[tuple[str, str]]:
        nonlocal buf, buf_len
        if not buf:
            return None

        # 如果有 overlap_text，把它加到最前面
        if overlap_text and buf:
            buf[0] = overlap_text + buf[0]

        text = "\n\n".join(buf)
        text = text.strip()

        # 判断类型
        ctype = "text"
        if buf and buf[0].startswith("```"):
            ctype = "code"

        buf = []
        buf_len = 0
        return (text, ctype)

    for atom in atoms:
        atom_len = len(atom)

        # 代码块标记 → 直接作为独立 chunk
        if atom.strip().startswith("```"):
            if buf:
                result = _flush()
                if result:
                    chunks.append(result)
            chunks.append((atom, "code"))
            continue

        # 代码块过长的处理：标记为 code 类型
        if atom_len > hard_max_chars and buf:
            result = _flush()
            if result:
                chunks.append(result)

        if atom_len > hard_max_chars:
            chunks.append((atom, "code" if _looks_like_code(atom) else "text"))
            continue

        # 正常累积
        if buf_len + atom_len <= target_chars:
            buf.append(atom)
            buf_len += atom_len
        elif buf_len < min_chars:
            # 还没到最小长度，继续加
            buf.append(atom)
            buf_len += atom_len
        else:
            # 当前 buf 已经够大了，flush 并保留 overlap
            prev_text = "\n\n".join(buf) if buf else ""
            overlap = ""
            if len(prev_text) > overlap_chars:
                overlap = prev_text[-overlap_chars:]

            result = _flush()
            if result:
                chunks.append(result)

            buf.append(atom)
            buf_len = len(atom)
            # 新 chunk 开头拼接 overlap
            if overlap and buf:
                buf[0] = overlap + buf[0]
                buf_len = len(buf[0])

    # 收尾
    result = _flush()
    if result:
        chunks.append(result)

    return chunks


def _looks_like_code(text: str) -> bool:
    """启发式判断文本是否为代码。"""
    lines = text.split("\n")
    code_indicators = 0
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("//"):
            continue
        if re.match(r'^\s*(def |class |import |from |if |for |while |return |function |const |let |var )', stripped):
            code_indicators += 1
        if any(c in stripped for c in ['{', '}', '(', ')', ';', '=']):
            code_indicators += 0.5
    return code_indicators >= 3


def _fill_summaries(chunks: list[DocumentChunk], llm_client: Optional[LLMClient]) -> None:
    """为每个 chunk 填充 summary 和 embedding_text。"""
    for chunk in chunks:
        if llm_client:
            chunk.summary = llm_client.summarize(
                chunk.text,
                context={
                    "section_title": chunk.section_title,
                    "page": chunk.page_start,
                    "heading_path": chunk.heading_path,
                },
            )
        else:
            # 无 LLM：取前 80 字
            chunk.summary = chunk.text[:80].replace("\n", " ").strip()

        # 构造 embedding_text
        parts = []
        if chunk.section_title:
            parts.append(chunk.section_title)
        if chunk.summary:
            parts.append(chunk.summary)
        parts.append(chunk.text)
        chunk.embedding_text = "\n".join(parts)
