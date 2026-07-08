# 文档理解、切片检索与文档问答模块

## 模块目标

把文档从"整篇一个向量"升级为"多切片、多标签、可定位、可解释、可问答"的多模态素材单元。

## 支持格式

| 格式 | 支持程度 | 说明 |
|------|----------|------|
| `.md` | ✅ 必须 | 按 Markdown 标题层级解析 |
| `.txt` | ✅ 必须 | 自动编码检测（UTF-8 / GBK / UTF-16） |
| `.docx` | ✅ 必须 | 识别 Word 样式标题 + 启发式短标题检测 |
| `.pdf` | 🟡 尽力 | 使用 pypdf，仅支持可提取文本的 PDF |
| `.pptx` | 🟡 尽力 | 每页幻灯片一个 chunk |

明确不支持：图片型 PDF OCR、复杂表格结构还原、公式提取、SmartArt 解析。

## 依赖安装

```bash
pip install -r requirements.txt
```

依赖清单（轻量，离线可用）：

- `python-docx` — DOCX 解析
- `python-pptx` — PPTX 解析
- `pypdf` — PDF 文本提取
- `jieba` — 中文分词（纯词典，非模型）

## 核心函数

### 1. 解析文档

```python
from document_module import parse_document

result = parse_document("example.md")
# result.doc_id, result.filename, result.format, result.title
# result.pages, result.text, result.sections, result.status
```

返回 `DocumentParseResult`。

### 2. 文档切片

```python
from document_module import parse_document, chunk_document

result = parse_document("example.md")
chunks = chunk_document(result)  # 可选传入 llm_client 参数
# 每个 chunk: chunk_id, chunk_type, text, summary, embedding_text, ...
```

切片策略：
- 优先级：标题边界 > 段落边界 > 句子边界 > 固定长度兜底
- 目标 500 中文字，范围 300-800，硬上限 1200
- 相邻 chunk 重叠 ~100 字
- 代码块单独成 chunk（`chunk_type="code"`）

### 3. 生成文档标签

```python
from document_module import generate_document_tags, build_tag_context

tags = generate_document_tags(result, chunks)
# 返回 list[DocumentTag]: name, type, confidence, evidence
```

标签类型：`topic` / `domain` / `content` / `genre` / `usage`
数量：5-12 个，每类最多 3 个，confidence < 0.45 不输出。

### 4. 文档问答

```python
from document_module import answer_from_chunks

# contexts 为简化版 chunk 列表
contexts = [
    {
        "chunk_id": c.chunk_id,
        "doc_id": c.doc_id,
        "filename": result.filename,
        "page": c.page_start,
        "section_title": c.section_title,
        "text": c.text,
        "summary": c.summary,
    }
    for c in chunks
]

answer = answer_from_chunks("OCR 接口怎么调用？", contexts)
# answer["answer"], answer["sources"]
```

原则：无引用，不回答。

## 输入输出示例

### parse_document 输出

```json
{
  "doc_id": "api-guide",
  "filename": "api-guide.md",
  "format": "md",
  "title": "API 接口文档",
  "pages": 1,
  "text": "# API 接口文档\n\n## OCR 接口\n...",
  "sections": [
    {
      "section_id": "sec_0001",
      "title": "API 接口文档",
      "heading_path": ["API 接口文档"],
      "page_start": null,
      "page_end": null,
      "text": "..."
    }
  ],
  "status": "ok",
  "warnings": [],
  "error": null
}
```

### chunk_document 输出

```json
[
  {
    "chunk_id": "api-guide_chunk_0000",
    "doc_id": "api-guide",
    "chunk_index": 0,
    "chunk_type": "text",
    "page_start": null,
    "page_end": null,
    "section_id": "sec_0002",
    "section_title": "API 接口文档 > OCR 接口",
    "heading_path": ["API 接口文档", "OCR 接口"],
    "text": "POST /api/multimodal-text 用于写入 OCR 文本...",
    "summary": "本段介绍 OCR 文本写入接口的参数和调用方式。",
    "embedding_text": "API 接口文档 > OCR 接口\n本段介绍 OCR 文本写入接口的参数和调用方式。\nPOST /api/multimodal-text..."
  }
]
```

### generate_document_tags 输出

```json
[
  {
    "name": "计算机教育",
    "type": "topic",
    "confidence": 0.86,
    "evidence": ["信息技术课程", "编程教学", "课堂实践"]
  },
  {
    "name": "OCR",
    "type": "content",
    "confidence": 0.92,
    "evidence": ["OCR 文本写入接口", "POST /api/multimodal-text"]
  }
]
```

### answer_from_chunks 输出

```json
{
  "answer": "根据文档片段：OCR 文本可以通过 POST /api/multimodal-text 写入。",
  "sources": [
    {
      "chunk_id": "api-guide_chunk_0003",
      "doc_id": "api-guide",
      "filename": "api-guide.md",
      "page": null,
      "section_title": "API 接口文档 > OCR 接口",
      "snippet": "POST /api/multimodal-text 用于写入 OCR..."
    }
  ]
}
```

## 切片策略

目标 500 中文字，允许范围 300-800 字，硬上限 1200 字。

- 标题边界 > 段落边界 > 句子边界 > 固定长度兜底
- 相邻 chunk 重叠 100 字
- MD 代码块单独成 chunk，标记 `chunk_type="code"`
- PPTX 仅标题页标记 `chunk_type="heading_only"`
- DOCX 表格标记 `chunk_type="table"`

## 标签生成策略

第一版使用 `HeuristicLLMClient`（纯规则）：

1. `build_tag_context()` 构造 3000-5000 字的上下文字符串
2. jieba 分词 → 停用词过滤 → 词频统计
3. 与 `resources/term_dict.json` 中的领域词表匹配确定标签类型
4. 综合词频、覆盖度、词表命中计算 confidence
5. 后处理：去重、去黑名单、类型限数

后续可通过 `ExternalLLMClient` 增强。

## 问答策略

第一版使用关键词匹配：

1. 从 question 中用 jieba 提取关键词
2. 对 contexts 计算关键词重合分
3. 标题命中额外加分
4. 选 top-3 chunk，分数过低返回"未找到可靠答案"
5. 返回原文摘要 + 引用来源

## 错误处理

- 解析函数返回 `DocumentParseResult.status` 标识状态
- 可能的 status：`"ok"` / `"empty_text"` / `"partial"` / `"failed"`
- 严重错误抛出 `DocumentParseError` / `DocumentUnsupportedError`
- 函数内部尽量不崩溃，通过 warnings 和 error 字段报告问题

## 已知限制

- PDF 仅支持可提取文本的页面，扫描件/图片型 PDF 返回空页
- DOCX 的 SmartArt、嵌入对象不支持
- PPTX 的图表、动画内容不支持
- 公式提取不支持
- 第一版启发式标签质量低于 LLM 方案
- 问答基于关键词匹配，不支持复杂推理

## Demo 运行方式

```bash
# 安装依赖
pip install -r requirements.txt

# 运行 demo（使用内置示例文档）
python demo_document_module.py

# 指定输入目录和输出文件
python demo_document_module.py --input ./my_docs --output report.md

# 指定自定义查询
python demo_document_module.py --query "OCR 接口怎么调用"
```

## 目录结构

```text
document_module/
  __init__.py      # 入口
  schema.py        # 数据结构定义
  parser.py        # 文档解析器
  chunker.py       # 文档切片器
  tagger.py        # 标签生成器
  qa.py            # 文档问答
  llm_client.py    # LLM 客户端抽象与实现
  resources/
    term_dict.json # 领域词表
  requirements.txt # 依赖清单
  README.md        # 本文档
```

## 协作边界

B 负责：文档解析、切片、摘要、标签、问答逻辑。
A/D 负责：入库、向量化、搜索融合、前端展示和评测。
