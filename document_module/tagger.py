"""
文档标签生成器。

包含：
- build_tag_context(): 构造标签生成的上下文字符串
- generate_document_tags(): 生成分层文档标签
"""

from typing import Optional

import jieba

from .schema import DocumentChunk, DocumentParseResult, DocumentTag
from .llm_client import LLMClient, HeuristicLLMClient


# ---------------------------------------------------------------------------
# 公开接口
# ---------------------------------------------------------------------------

def build_tag_context(
    parse_result: DocumentParseResult,
    chunks: list[DocumentChunk],
    max_chars: int = 4000,
) -> str:
    """从解析结果和 chunk 构造标签生成上下文。

    内容包括：
    1. 文档标题
    2. 文件名和格式
    3. section_title 列表
    4. 前 3 个 chunk 的 summary + 少量 text
    5. 各章节 summary
    6. 高频候选关键词

    Args:
        parse_result: 解析结果。
        chunks: 切片列表。
        max_chars: 上下文最大字数。

    Returns:
        拼接好的上下文字符串。
    """
    parts: list[str] = []

    # 1. 文档标题
    parts.append(f"标题：{parse_result.title}")

    # 2. 文件名和格式
    parts.append(f"文件：{parse_result.filename}（{parse_result.format}）")

    # 3. section_title 列表
    titles = []
    for sec in parse_result.sections:
        if sec.title:
            path = " > ".join(sec.heading_path) if sec.heading_path else sec.title
            titles.append(path)
    if titles:
        parts.append(f"章节：{' / '.join(titles[:20])}")

    # 4. 前 3 个 chunk
    for i, chunk in enumerate(chunks[:3]):
        text_snippet = chunk.text[:150].replace("\n", " ")
        parts.append(f"[片段{i + 1}] {chunk.summary} | {text_snippet}")

    # 5. 各章节 summary（去重）
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.section_title and chunk.section_title not in seen:
            seen.add(chunk.section_title)
            summary = chunk.summary[:60] if chunk.summary else ""
            text_snippet = chunk.text[:100].replace("\n", " ")
            parts.append(f"[章节摘要] {chunk.section_title}: {summary} | {text_snippet}")

    # 6. 高频候选关键词
    full_text = parse_result.text
    if full_text:
        keywords = _extract_keywords(full_text, top_n=20)
        if keywords:
            parts.append(f"高频词：{', '.join(keywords)}")

    # 拼接并截断
    context = "\n".join(parts)
    if len(context) > max_chars:
        # 尽量保留前面的信息（标题、章节 > 片段 > 摘要）
        context = context[:max_chars]

    return context


def generate_document_tags(
    parse_result: DocumentParseResult,
    chunks: list[DocumentChunk],
    llm_client: Optional[LLMClient] = None,
    max_tags: int = 12,
) -> list[DocumentTag]:
    """为文档生成分层标签。

    Args:
        parse_result: 解析结果。
        chunks: 切片列表。
        llm_client: LLM 客户端。默认使用 HeuristicLLMClient。
        max_tags: 标签数量上限。

    Returns:
        DocumentTag 列表，按 confidence 降序。
    """
    if llm_client is None:
        llm_client = HeuristicLLMClient()

    context = build_tag_context(parse_result, chunks)
    raw_tags = llm_client.generate_tags(context, max_tags=max_tags)

    # 后处理：去空白 / 统一大小写 / 长度限制 / 同义合并
    processed = _post_process_tags(raw_tags, parse_result, chunks)

    return processed[:max_tags]


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------

# 通用停用词（与 llm_client 互补的标签专用过滤）
_TAG_BLACKLIST = {
    "本文", "我们", "介绍", "内容", "研究", "进行", "说明", "相关",
    "一个", "这个", "进行", "通过", "使用", "目前", "可以", "需要",
    "已经", "还有", "一些", "不同", "主要", "其中", "之间", "以及",
    "对于", "方面", "问题", "情况", "方法", "过程", "部分", "作用",
    "影响", "发展", "形成", "具有", "比较", "表示", "包括", "成为",
    "不是", "非常", "就是", "可能", "应该", "必须", "能够",
    "本章", "本节", "下一页", "上一页", "下一页", "返回",
    "首页", "目录", "前言", "结论", "参考文献", "致谢", "附录",
}


def _extract_keywords(text: str, top_n: int = 20) -> list[str]:
    """用 jieba 提取高频关键词。"""
    # 构建停用词并集
    stop_words = _TAG_BLACKLIST | {
        '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都',
        '一', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着',
        '它', '她', '他', '这', '那', '些', '所', '为', '但', '还', '只',
        '被', '把', '从', '让', '对', '向', '与', '及', '其', '中', '等',
        '之', '已', '将', '能', '更', '最', '啊', '吧', '呢', '吗', '嘛',
        '得', '地', '过',
    }

    words = list(jieba.cut(text))
    freq: dict[str, int] = {}
    for w in words:
        w = w.strip()
        if len(w) >= 2 and w not in stop_words:
            freq[w] = freq.get(w, 0) + 1

    sorted_words = sorted(freq, key=freq.get, reverse=True)
    return sorted_words[:top_n]


def _post_process_tags(
    raw_tags: list[dict],
    parse_result: DocumentParseResult,
    chunks: list[DocumentChunk],
) -> list[DocumentTag]:
    """对原始标签做后处理：去重、去黑名单、合并相似、限制长度。"""
    result: list[DocumentTag] = []

    seen_names: set[str] = set()
    seen_lower: set[str] = set()

    for tag in raw_tags:
        name = tag.get("name", "").strip()
        if not name:
            continue

        # 黑名单过滤
        if name in _TAG_BLACKLIST:
            continue

        # 长度限制（标签不宜太长）
        if len(name) > 15:
            continue

        # 去重（大小写不敏感）
        lower = name.lower()
        if lower in seen_lower:
            continue

        # 数值分数不应是标签
        if name.replace(".", "").replace("%", "").isdigit():
            continue

        seen_lower.add(lower)
        seen_names.add(name)

        confidence = min(max(tag.get("confidence", 0.0), 0.0), 1.0)
        if confidence < 0.45:
            continue

        result.append(DocumentTag(
            name=name,
            type=tag.get("type", "topic"),
            confidence=confidence,
            evidence=tag.get("evidence", [])[:3],
        ))

    return sorted(result, key=lambda t: t.confidence, reverse=True)
