# B 文档模块第二轮技术细化答复

本文档用于回答 B 关于文档模块第二轮问题，重点明确 `LLMClient` Schema、离线实现策略、切片算法细节、`sections/chunks` 关系、demo 形态和依赖管理。

总体原则：

```text
第一版优先稳定、离线可跑、接口清晰。
LLM 是增强项，不是硬依赖。
解析和切片必须确定性强，方便 A/D 入库、向量化和调试。
```

## A. LLMClient 接口 Schema

建议 B 在 `document_module/schema.py` 中先定义稳定数据结构。推荐使用 `dataclass`，也可以使用 `TypedDict`。

### 1. summarize

函数签名：

```python
def summarize(text: str, *, max_chars: int = 80, context: dict | None = None) -> str:
    ...
```

输入说明：

- `text`：单个 chunk 的正文，不是整篇文档。
- `context`：可选上下文，建议包含 `filename / section_title / page / heading_path`。
- 如果要增强摘要质量，可以由调用方传入已经拼好的上下文，但摘要主体仍以 chunk text 为准。

返回要求：

- 返回不超过 `max_chars` 的中文短摘要。
- 默认建议 `max_chars=80`。
- 摘要应描述 chunk 主要内容，不要输出“本文介绍了”这种空泛句。
- `HeuristicLLMClient` 做不好抽象摘要时，允许返回：

```text
chunk 前 80 字的清洗截断文本
```

不要返回空字符串，除非原文为空。

### 2. generate_tags

函数签名：

```python
def generate_tags(context: str, *, max_tags: int = 12) -> list[DocumentTag]:
    ...
```

返回结构：

```json
{
  "name": "计算机教育",
  "type": "topic",
  "confidence": 0.86,
  "evidence": ["信息技术课程", "编程教学", "课堂实践"]
}
```

标签类型：

```text
topic：主题，如 计算机教育、春天描写
domain：领域，如 人工智能、教育、文学
content：具体内容，如 OCR、ASR、CLIP、接口规范
genre：文体，如 说明文、报告、散文、PPT
usage：用途，如 答辩材料、学习资料、项目文档
```

是否五类都必须生成：

```text
不强制。
按文档内容生成即可。
一篇纯散文可以没有 domain。
一份接口文档可能没有 genre 之外的文学文体。
```

第一版约束：

- 总标签数控制在 5-12 个。
- 每类最多 3 个。
- `confidence < 0.45` 的标签不输出。
- 同名标签去重。

`HeuristicLLMClient` 下如何区分标签类型：

```text
可以退化，但不要只输出无类型标签。
```

建议启发式：

- 命中技术词表：`content`
- 命中领域词表：`domain`
- 命中文体词表：`genre`
- 命中用途词表：`usage`
- 高频核心短语：`topic`

### 3. answer

函数签名：

```python
def answer(question: str, contexts: list[dict]) -> dict:
    ...
```

`contexts` 使用简化版 chunk，不要求传完整对象，但必须包含来源信息：

```json
{
  "chunk_id": "asset_xxx_chunk_0001",
  "doc_id": "asset_xxx",
  "filename": "api-contracts.md",
  "page": 3,
  "section_title": "接口说明 > OCR",
  "text": "原文片段...",
  "summary": "片段摘要..."
}
```

返回结构：

```json
{
  "answer": "OCR 文本可以通过 POST /api/multimodal-text 写入。",
  "sources": [
    {
      "chunk_id": "asset_xxx_chunk_0001",
      "doc_id": "asset_xxx",
      "filename": "api-contracts.md",
      "page": 3,
      "section_title": "接口说明 > OCR",
      "snippet": "POST /api/multimodal-text 用于写入 OCR..."
    }
  ]
}
```

找不到答案时：

```json
{
  "answer": "未在当前素材库文档中找到可靠答案。",
  "sources": []
}
```

原则：

```text
无引用，不回答。
不要让 LLM 脱离文档自由发挥。
```

## B. HeuristicLLMClient 实现策略

第一版选择“中等离线可用”方案，不要只做极简截断。

推荐实现：

| 能力 | 第一版策略 |
|---|---|
| 摘要 | 优先取标题 + 关键句；没有关键句时取前 80 字 |
| 标签 | jieba 分词 + 停用词过滤 + 领域词表 + 简单 TF 统计 |
| 问答 | 根据问题关键词/BM25-like 分数选择最相关 chunk，返回片段式回答 |

不要求第一版引入 KeyBERT、text2vec、TextRank4ZH 这类较重依赖。

摘要规则建议：

```text
1. 如果 chunk 有 summary，直接清洗截断。
2. 否则按句号/问号/感叹号切句。
3. 优先选择包含高频关键词的句子。
4. 兜底返回 text[:80]。
```

标签规则建议：

```text
1. jieba 分词。
2. 去停用词。
3. 合并技术词，如 OCR / ASR / CLIP / Agent / Faiss。
4. 根据领域词表打 type。
5. 根据词频、标题命中、章节覆盖计算 confidence。
```

问答规则建议：

```text
1. 从 question 提取关键词。
2. 对 contexts 计算关键词重合分。
3. 选择 top 1-3 chunk。
4. 如果分数过低，返回“未找到可靠答案”。
5. 如果命中，返回“根据文档片段：...”加原文摘要。
```

## C. 切片算法具体实现

### PDF

第一版使用方案 A：

```text
按页切；每页内按段落切；段落太长按句子切。
```

不建议第一版做复杂标题视觉检测。PDF 标题结构不稳定，容易误判。

规则：

- 每页保留 `page`。
- 空页跳过，并记录到 `skipped_pages`。
- 如果一页文本很短，可以和下一页合并，但必须保留 `page_start/page_end`。

### DOCX

第一版：

```text
优先识别 Word 样式 Heading 1/2/3。
如果没有样式，再用启发式识别短标题行。
```

启发式标题规则：

- 单独一行。
- 长度小于 40 字。
- 以 `第x章`、`1.`、`1.1`、`一、`、`（一）` 开头。
- 或者行尾没有句号且下一段明显较长。

不要求做字体大小、加粗等视觉检测，python-docx 里处理这些会增加复杂度。

### PPTX

第一版：

```text
一页幻灯片一个 chunk。
```

如果幻灯片只有标题、正文为空：

- 如果标题有意义，生成 chunk，`chunk_type="slide_title"`。
- 如果标题也为空，跳过。

过渡页可以保留，因为它可能对章节结构有帮助。

## D. section_title 与 heading_path

保留编号，不要剥离编号。

例如：

```text
["项目介绍", "背景", "1.1 相关研究", "方法 A"]
```

原因：

- 编号对定位有帮助。
- 前端展示和报告引用更清晰。
- 后续跳转页码/章节时更稳定。

`section_title` 由 `heading_path` 拼接：

```text
项目介绍 > 背景 > 1.1 相关研究 > 方法 A
```

标题层级不连续时：

```text
按出现顺序和层级尽力维护。
H1 跳到 H3 时，中间 H2 不强行补虚拟标题。
```

没有标题的段落：

```text
归属于上一标题。
```

不要归给下一个标题，因为文本流上它属于当前上下文。

## E. parse_document 的 sections 结构

选择方案 A：

```text
parse 阶段尽量拆成 sections；
chunker 在 sections 基础上切 chunk。
```

二者关系：

```text
DocumentParseResult
  -> sections
    -> chunks
```

`sections` 粒度：

- 对 MD/DOCX：一个标题块对应一个 section。
- 对 PDF：默认一页或一页内的伪段落组对应一个 section。
- 对 PPTX：一页 slide 对应一个 section。

section 示例：

```json
{
  "section_id": "sec_0003",
  "title": "2.2 方法",
  "heading_path": ["第二章", "2.2 方法"],
  "page_start": 12,
  "page_end": 13,
  "text": "..."
}
```

chunk 应保留其来源 section：

```json
{
  "chunk_id": "asset_xxx_chunk_0007",
  "section_id": "sec_0003",
  "section_title": "第二章 > 2.2 方法",
  "page": 12,
  "text": "..."
}
```

## F. Demo 脚本形态

第一版 demo 建议生成 Markdown 报告，兼顾可检查和可展示。

脚本：

```text
python demo_document_module.py --input samples --output demo_report.md
```

报告内容：

- 每个文档的解析状态。
- sections 数量。
- chunks 数量。
- 前 3 个 chunk 示例。
- 文档标签。
- 示例查询结果。
- 示例问答结果。
- warnings/errors。

同时可以在终端打印简短摘要，但主要产物是 `demo_report.md`。

不要求第一版做交互式问答。

## G. 依赖管理

B 模块建议单独提供：

```text
document_module/requirements.txt
```

第一版依赖建议：

```text
python-docx
python-pptx
pypdf
jieba
```

PDF 如果 `pypdf` 效果不够，再可选增加：

```text
pdfplumber
```

不建议第一版加入：

```text
scikit-learn
text2vec
KeyBERT
textrank4zh
aspose
PyMuPDF
```

原因：

- 依赖重。
- 安装不稳定。
- 部分库有协议或商业授权问题。
- 第一版目标是稳定集成，不是追求最强 NLP 效果。

如果 B 想做增强版，可以把重依赖放到：

```text
requirements-optional.txt
```

## 最终执行口径

B 第一版按以下标准实现：

```text
离线可跑
Schema 固定
sections -> chunks 两层结构
chunk 有 embedding_text
标签有 type/confidence/evidence
问答必须有 sources
demo 输出 Markdown 报告
依赖尽量轻
```

第一版不追求“像 ChatGPT 一样会写答案”，而是追求：

```text
文档可以稳定解析、切片、打标签、定位命中片段，并能给 A/D 后续向量检索和前端展示提供可靠结构。
```

