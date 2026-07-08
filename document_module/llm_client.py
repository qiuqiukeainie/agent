"""
LLM 客户端抽象接口与离线实现。

第一版使用 HeuristicLLMClient（纯规则，零外部依赖）。
后续可接入 ExternalLLMClient 调用 Dify / OpenAI 兼容接口。
"""

import json
import os
import re
from typing import Optional

import jieba


class LLMClient:
    """LLM 能力抽象接口。

    子类需要实现 summarize / generate_tags / answer 三个方法。
    """

    def summarize(self, text: str, max_chars: int = 80, context: Optional[dict] = None) -> str:
        """为单个 chunk 生成短摘要。

        Args:
            text: chunk 正文。
            max_chars: 摘要最大字数。
            context: 可选上下文字典，可包含 filename / section_title / page / heading_path。

        Returns:
            不超过 max_chars 的中文短摘要。
        """
        raise NotImplementedError

    def generate_tags(self, context: str, max_tags: int = 12) -> list[dict]:
        """从文档上下文中生成标签。

        Args:
            context: build_tag_context() 构造的上下文字符串。
            max_tags: 标签数量上限。

        Returns:
            [{"name": "...", "type": "...", "confidence": 0.86, "evidence": [...]}, ...]
        """
        raise NotImplementedError

    def answer(self, question: str, contexts: list[dict]) -> dict:
        """基于文档片段回答问题。

        Args:
            question: 用户问题。
            contexts: 简化版 chunk 列表，至少包含 chunk_id / text / summary / section_title。

        Returns:
            {"answer": "...", "sources": [...]}
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 离线实现
# ---------------------------------------------------------------------------

# 中文停用词表 —— 包含常见无意义词和文档中不需要的元描述词
_STOP_WORDS: set[str] = set()

def _load_stop_words() -> set[str]:
    """延迟加载停用词，避免模块导入时开销。"""
    global _STOP_WORDS
    if _STOP_WORDS:
        return _STOP_WORDS

    _STOP_WORDS = {
        '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一',
        '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '它', '她', '他',
        '没有', '看', '好', '自己', '这', '那', '些', '所', '为', '但',
        '还', '只', '被', '把', '从', '让', '对', '向', '与', '及', '其',
        '中', '等', '之', '已', '将', '能', '更', '最',
        '啊', '吧', '呢', '吗', '嘛', '哦', '嗯', '哈', '呀', '哇', '啦',
        '得', '地', '过',
        '本文', '我们', '介绍', '内容', '研究', '进行', '说明', '相关',
        '一个', '这个', '通过', '使用', '目前', '可以', '需要', '已经',
        '还有', '一些', '不同', '主要', '其中', '之间', '以及', '对于',
        '方面', '问题', '情况', '方法', '过程', '部分', '作用', '影响',
        '发展', '形成', '具有', '比较', '表示', '包括', '成为', '不是',
        '非常', '就是', '因为', '所以', '此外', '最后', '首先', '其次',
        '然后', '接着', '另外', '同时', '一般', '一定', '一直', '一样',
        '可能', '应该', '必须', '能够', '不能', '不会', '是否', '怎样',
        '如何', '多少', '什么', '怎么', '它们', '这是', '那是', '大家',
        '每个', '很多', '很少', '总是', '经常', '有时', '偶尔',
        '通常', '往往', '仍然', '还是', '以前', '以后', '现在', '将来',
        '以前', '之后', '之前', '当中',
        '采用', '基于', '利用', '提出', '分析', '结果', '实验', '数据',
        '模型', '系统', '用户', '提供', '支持', '实现', '设计',
        '做', '办', '弄', '搞', '干', '来', '去', '想', '知道',
    }
    return _STOP_WORDS


class HeuristicLLMClient(LLMClient):
    """基于规则的离线 LLM 客户端。

    摘要：取原文前几句 / 含高频关键词的句子。
    标签：jieba 分词 + 领域词表匹配 + TF 统计。
    问答：关键词重合度打分 → 返回最高分片段。

    Args:
        term_dict_path: 领域词表 JSON 文件路径。默认读取 resources/term_dict.json。
    """

    def __init__(self, term_dict_path: Optional[str] = None):
        self._stop_words = _load_stop_words()

        if term_dict_path is None:
            term_dict_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "resources", "term_dict.json",
            )

        self.term_dict: dict[str, list[str]] = {"tech": [], "domain": [], "genre": [], "usage": []}
        if os.path.exists(term_dict_path):
            with open(term_dict_path, "r", encoding="utf-8") as f:
                self.term_dict = json.load(f)

        # 构建 词 → type 的反向索引
        self._term_to_type: dict[str, str] = {}
        for type_name, terms in self.term_dict.items():
            for term in terms:
                self._term_to_type[term] = type_name

    # ------------------------------------------------------------------
    # summarize
    # ------------------------------------------------------------------

    def summarize(self, text: str, max_chars: int = 80, context: Optional[dict] = None) -> str:
        """启发式摘要：优先取含高频词的关键句，兜底取前 max_chars 字。"""
        if not text or not text.strip():
            return ""

        text = text.strip()

        # 已经足够短，直接返回
        if len(text) <= max_chars:
            return text

        sentences = _split_sentences(text)
        if not sentences:
            return text[:max_chars]

        # 计算词频，找 top keywords
        word_freq = _compute_word_freq(text, self._stop_words)
        top_keywords = sorted(word_freq, key=word_freq.get, reverse=True)[:10]

        # 为每个句子打分
        def score(sent: str, idx: int) -> float:
            s = 0.0
            for kw in top_keywords:
                if kw in sent:
                    s += 1.0
            s += max(0, 1.5 - idx * 0.15)  # 位次奖励
            if len(sent) < 8:
                s -= 1.5                    # 太短惩罚
            if len(sent) > max_chars:
                s -= 0.5                    # 太长轻微惩罚
            return s

        ranked = sorted(enumerate(sentences), key=lambda x: score(x[1], x[0]), reverse=True)

        # 贪心选取不超过 max_chars 的句子组合
        parts: list[str] = []
        current = 0
        for _, sent in ranked:
            if current + len(sent) + 1 <= max_chars:
                parts.append(sent)
                current += len(sent) + 1
            else:
                break

        if parts:
            return "。".join(parts) + "。"

        return text[:max_chars]

    # ------------------------------------------------------------------
    # generate_tags
    # ------------------------------------------------------------------

    def generate_tags(self, context: str, max_tags: int = 12) -> list[dict]:
        """基于 jieba 分词 + 词表匹配生成标签。"""
        if not context or not context.strip():
            return []

        words = list(jieba.cut(context))
        candidates = [
            w.strip() for w in words
            if len(w.strip()) >= 2 and w.strip() not in self._stop_words
        ]

        # 单字和双字组合统计
        freq: dict[str, int] = {}
        for w in candidates:
            freq[w] = freq.get(w, 0) + 1

        # 2-gram 短语补充
        for i in range(len(candidates) - 1):
            bigram = candidates[i] + candidates[i + 1]
            if len(bigram) <= 8:
                freq[bigram] = freq.get(bigram, 0) + 1

        if not freq:
            return []

        total_chars = len(context)
        max_freq = max(freq.values())

        tags: list[dict] = []
        for word, count in freq.items():
            tag_type = self._term_to_type.get(word, "topic")

            # confidence = 词频分 + 覆盖度分 + 词表奖励
            freq_score = (count / max_freq) * 0.4
            coverage = min((len(word) * count) / max(total_chars, 1), 1.0) * 0.3
            dict_bonus = 0.3 if word in self._term_to_type else 0.0
            confidence = min(freq_score + coverage + dict_bonus, 1.0)

            if confidence >= 0.45:
                evidence = _find_evidence(word, context)
                tags.append({
                    "name": word,
                    "type": tag_type,
                    "confidence": round(confidence, 2),
                    "evidence": evidence,
                })

        # 去重 & 排序
        seen: set[str] = set()
        unique: list[dict] = []
        for t in sorted(tags, key=lambda t: t["confidence"], reverse=True):
            if t["name"] not in seen:
                seen.add(t["name"])
                unique.append(t)

        # 每类最多 3 个，总数不超过 max_tags
        type_counts: dict[str, int] = {}
        result: list[dict] = []
        for t in unique:
            tp = t["type"]
            if type_counts.get(tp, 0) < 3 and len(result) < max_tags:
                result.append(t)
                type_counts[tp] = type_counts.get(tp, 0) + 1

        return result

    # ------------------------------------------------------------------
    # answer
    # ------------------------------------------------------------------

    def answer(self, question: str, contexts: list[dict]) -> dict:
        """基于关键词匹配的文档问答。"""
        empty = {"answer": "未在当前素材库文档中找到可靠答案。", "sources": []}

        if not question or not contexts:
            return empty

        # 从问题中提取关键词
        q_words = list(jieba.cut(question))
        q_keywords = [
            w.strip() for w in q_words
            if len(w.strip()) >= 2 and w.strip() not in self._stop_words
        ]
        if not q_keywords:
            q_keywords = [question.strip()]

        # 对每个 context 打分
        scored: list[tuple[int, list[str], dict]] = []
        for ctx in contexts:
            text = (ctx.get("text", "") + " " +
                    ctx.get("summary", "") + " " +
                    ctx.get("section_title", ""))
            score = 0
            matched: list[str] = []
            for kw in q_keywords:
                if kw in text:
                    score += 1
                    matched.append(kw)
            # 标题命中额外加分
            for kw in q_keywords:
                if kw in ctx.get("section_title", ""):
                    score += 2

            if score > 0:
                scored.append((score, matched, ctx))

        scored.sort(key=lambda x: x[0], reverse=True)

        if not scored or scored[0][0] < 1:
            return empty

        top = scored[:3]
        answer_parts: list[str] = []
        sources: list[dict] = []

        for score, _matched, ctx in top:
            text = ctx.get("text", "")
            snippet = text[:200] if text else ""
            if len(text) > 200:
                snippet += "..."

            sources.append({
                "chunk_id": ctx.get("chunk_id", ""),
                "doc_id": ctx.get("doc_id", ""),
                "filename": ctx.get("filename", ""),
                "page": ctx.get("page"),
                "section_title": ctx.get("section_title", ""),
                "snippet": snippet,
            })

            summary = ctx.get("summary", "")
            if summary:
                answer_parts.append(f"根据文档片段：{summary}")
            else:
                answer_parts.append(f"根据文档片段：{snippet}")

        return {
            "answer": "\n\n".join(answer_parts),
            "sources": sources,
        }


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """按中文标点 + 换行切句。"""
    raw = re.split(r'[。！？\n]{1,2}', text)
    return [s.strip() for s in raw if s.strip()]


def _compute_word_freq(text: str, stop_words: set[str]) -> dict[str, int]:
    """对文本做 jieba 分词并统计词频（过滤停用词）。"""
    words = list(jieba.cut(text))
    freq: dict[str, int] = {}
    for w in words:
        w = w.strip()
        if len(w) >= 2 and w not in stop_words:
            freq[w] = freq.get(w, 0) + 1
    return freq


def _find_evidence(tag_name: str, context: str, max_items: int = 3, max_len: int = 40) -> list[str]:
    """从上下文中寻找支持标签的证据短语。"""
    fragments = re.split(r'[。！？\n,，、；;]', context)
    evidence: list[str] = []

    # 精确包含
    for frag in fragments:
        frag = frag.strip()
        if tag_name in frag and len(frag) <= max_len and frag not in evidence:
            evidence.append(frag)
        if len(evidence) >= max_items:
            return evidence

    # 模糊匹配：字符重叠 ≥ 50%
    tag_chars = set(tag_name)
    for frag in fragments:
        frag = frag.strip()
        if len(frag) <= max_len and frag not in evidence:
            overlap = len(tag_chars & set(frag))
            if len(tag_chars) > 0 and overlap / len(tag_chars) >= 0.5:
                evidence.append(frag)
        if len(evidence) >= max_items:
            break

    return evidence
