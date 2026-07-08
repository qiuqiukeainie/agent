"""
语义检索器 — 路径一实现。

基于 sentence-transformers 将文本编码为向量，通过余弦相似度做语义匹配。
解决纯关键词匹配无法识别"风褪去了刺骨的凉" = "春天"的问题。
"""

import os
import numpy as np
from typing import Optional

from .schema import DocumentChunk


class SemanticRetriever:
    """语义检索器。

    使用多语言 embedding 模型将查询和 chunk 文本映射到同一向量空间，
    通过余弦相似度排序，实现"大意匹配"而非关键词匹配。

    用法:
        retriever = SemanticRetriever()
        retriever.index(chunks)
        results = retriever.search("描写春天的文章", top_k=10)
        # results: [(DocumentChunk, score), ...]
    """

    # 默认模型路径（本地）
    _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DEFAULT_MODEL = os.path.join(_ROOT, "models")

    # 备选在线模型:
    # "paraphrase-multilingual-MiniLM-L12-v2"
    # "shibing624/text2vec-base-chinese"
    # "BAAI/bge-small-zh-v1.5"

    def __init__(self, model_name: Optional[str] = None):
        """
        Args:
            model_name: HuggingFace 模型名，默认用多语言 MiniLM。
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model = None                # 延迟加载
        self._chunks: list[DocumentChunk] = []
        self._embeddings: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def index(self, chunks: list[DocumentChunk]) -> None:
        """对 chunk 列表建立向量索引。

        对每个 chunk 的 embedding_text 编码并缓存，后续 search() 直接使用。
        """
        if not chunks:
            self._chunks = []
            self._embeddings = None
            return

        texts = [c.embedding_text for c in chunks]
        print(f"[Retriever] 正在为 {len(texts)} 个 chunk 生成向量...")
        self._embeddings = self.encode(texts)
        self._chunks = list(chunks)
        print(f"[Retriever] 向量维度: {self._embeddings.shape[1]}")

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0,
        min_chunk_chars: int = 20,
    ) -> list[tuple[DocumentChunk, float]]:
        """语义搜索。

        Args:
            query: 查询文本。
            top_k: 返回数量。
            min_score: 最低分数阈值（0-1），低于此值的结果丢弃。
            min_chunk_chars: 最短 chunk 字数，过短的 chunk 跳过（避免"谢谢！"等干扰）。

        Returns:
            [(chunk, score), ...] 按分数降序排列。
        """
        if self._embeddings is None or not self._chunks:
            return []

        query_vec = self.encode([query])
        scores = _cosine_similarity(query_vec, self._embeddings)[0]

        # 排序取 top_k，跳过过短 chunk
        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True,
        )

        results: list[tuple[DocumentChunk, float]] = []
        for idx, score in ranked:
            if score < min_score:
                continue
            chunk = self._chunks[idx]
            if len(chunk.text) < min_chunk_chars:
                continue
            if len(results) >= top_k:
                break
            results.append((chunk, float(score)))

        return results

    def encode(self, texts: list[str]) -> np.ndarray:
        """将文本列表编码为向量矩阵。

        Args:
            texts: 文本列表。

        Returns:
            shape=(len(texts), dim) 的 numpy 数组。
        """
        self._ensure_model()
        # show_progress_bar=False 避免输出干扰
        return self._model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,   # 归一化，便于直接点积=余弦相似度
        )

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """延迟加载模型（首次调用 encode 时才下载/加载）。"""
        if self._model is not None:
            return
        print(f"[Retriever] 加载模型: {self.model_name}")
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.model_name)
        print("[Retriever] 模型就绪")


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """计算两组归一化向量的余弦相似度。

    如果向量已归一化，cos_sim = 点积。
    """
    return np.dot(a, b.T)
