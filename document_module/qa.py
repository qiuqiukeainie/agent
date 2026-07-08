"""
文档问答模块。

基于检索到的 chunk 回答问题。遵循"无引用，不回答"原则。
"""

from typing import Optional

from .llm_client import LLMClient, HeuristicLLMClient


def answer_from_chunks(
    question: str,
    chunks: list[dict],
    llm_client: Optional[LLMClient] = None,
    max_chunks: int = 5,
    max_context_chars: int = 6000,
) -> dict:
    """基于检索到的文档 chunk 回答问题。

    Args:
        question: 用户问题。
        chunks: 简化版 chunk 列表。每个 dict 需包含：
                chunk_id / doc_id / filename / page / section_title / text / summary。
        llm_client: LLM 客户端，默认 HeuristicLLMClient。
        max_chunks: 最多使用的 chunk 数。
        max_context_chars: 上下文最大字数。

    Returns:
        {"answer": "...", "sources": [...]}
    """
    if llm_client is None:
        llm_client = HeuristicLLMClient()

    # 二次限制：最多 max_chunks 个，总字数不超过 max_context_chars
    limited: list[dict] = []
    total_chars = 0
    for ctx in chunks[:max_chunks]:
        text_len = len(ctx.get("text", ""))
        if total_chars + text_len > max_context_chars:
            break
        limited.append(ctx)
        total_chars += text_len

    return llm_client.answer(question, limited)
