# B 方向任务说明：文档理解、切片检索与文档问答

本文档用于给 B 分配“多模态文档能力”相关任务。当前系统已经支持上传 `.pdf / .docx / .pptx / .md / .txt`，但文档处理仍然偏基础：主要是提取整篇文本，生成一个整体向量和少量标签。

这种方式存在两个明显问题：

- 长文档内容太杂，容易在很多查询中都被匹配到，出现“文档总是排在前面”的情况。
- 当前文档标签偏随意，很多标签只是简单词频或零散关键词，不能代表文档主题。

因此，B 接下来负责把文档模块升级为：

```text
文档切片检索 + 代表性文档标签 + 基于素材库的文档问答
```

## 一、核心目标

把文档从“整篇一个向量”升级为“多切片、多标签、可定位、可解释、可问答”的文档理解模块。

目标效果：

用户搜索：

```text
我想搜一篇描写春天的文章
```

系统应能匹配到和“春天、季节、自然、景色、植物、温暖、生机”等语义相关的文章，而不是只匹配“春天”两个字。

用户搜索：

```text
有关于计算机教育的文章
```

系统应能匹配到主题接近“计算机、教育、教学、课程、信息技术、学习、编程课堂”等内容的文档。

用户搜索：

```text
OCR 接口怎么调用？
```

系统应返回：

- 文档名
- 命中页码或段落
- 命中摘要
- 匹配原因
- 可引用的原文片段

## 二、具体任务

### 1. 文档解析

需要支持以下格式：

```text
.pdf
.docx
.pptx
.md
.txt
```

需要提取：

- 文档标题
- 正文文本
- 页码 / 段落编号 / 幻灯片编号
- 标题层级或章节结构
- 基础元数据

不同格式建议：

- PDF：优先按页解析。
- DOCX：按标题和段落解析。
- PPTX：按幻灯片解析。
- MD：按 Markdown 标题层级解析。
- TXT：按段落或固定长度解析。

### 2. 文档切片

不要再整篇文档只生成一个向量。每篇文档应切成多个 chunk。

切片示例：

```json
{
  "doc_id": "asset_xxx",
  "chunk_id": "asset_xxx_chunk_0001",
  "chunk_index": 1,
  "page": 3,
  "section_title": "OCR 接口说明",
  "text": "POST /api/multimodal-text ...",
  "summary": "本段介绍 OCR 文本写入接口的参数和调用方式"
}
```

切片原则：

- 每个 chunk 不宜太短，否则语义不完整。
- 每个 chunk 不宜太长，否则主题混杂。
- 建议中文长度为 300-800 字左右。
- 保留页码、段落、标题信息。
- PPTX 每页幻灯片可以作为一个 chunk。
- MD/DOCX 尽量按标题结构切。

### 3. 每个切片生成向量

后续搜索时，不再只匹配整篇文档，而是：

```text
用户查询
  ↓
查询向量
  ↓
匹配文档 chunk 向量
  ↓
聚合同一篇文档下的高分 chunk
  ↓
返回文档 + 命中片段
```

这样可以解决长文档“什么都沾一点”的问题，也能让搜索结果更具体。

### 4. 文档标签生成

当前文档标签不够代表性。后续标签应分层生成，而不是只抽取高频词。

建议标签类型：

```text
主题标签：计算机教育、春天描写、接口规范、项目分工
领域标签：人工智能、教育、文学、软件工程、多媒体
内容标签：OCR、ASR、CLIP、Agent、向量检索
文体标签：说明文、论文、报告、散文、接口文档、演示稿
用途标签：答辩材料、项目文档、学习资料、素材说明
```

输出示例：

```json
{
  "doc_id": "asset_xxx",
  "tags": [
    {"name": "计算机教育", "type": "topic", "confidence": 0.86},
    {"name": "信息技术课程", "type": "domain", "confidence": 0.78},
    {"name": "教学实践", "type": "content", "confidence": 0.74},
    {"name": "文章", "type": "genre", "confidence": 0.65}
  ]
}
```

要求：

- 标签要能代表整篇文档主题。
- 标签数量控制在 5-12 个。
- 不要把大量无意义高频词当标签。
- 中文文档优先生成中文标签。
- 技术英文词如 `CLIP / OCR / ASR / Agent / Faiss` 可以保留英文。

### 5. 支持大意匹配

文档模块不能只做关键词匹配，要支持语义大意匹配。

示例一：

```text
我想搜一篇描写春天的文章
```

不一定要求文档中出现“春天”，只要内容包含：

```text
花开、柳树、温暖、季节、自然、景色、生机
```

也应能被召回。

示例二：

```text
有关于计算机教育的文章
```

应能匹配：

```text
信息技术教学、编程课程、计算机基础、教育信息化、课堂实践
```

实现上可以结合：

- 文档 chunk 向量检索
- 主题标签辅助
- chunk 摘要辅助
- 查询扩展词辅助

### 6. 文档问答

支持基于素材库文档的问答。

示例：

```text
用户：OCR 接口怎么调用？
```

系统流程：

```text
1. 检索相关文档 chunk
2. 选取 top-k 片段
3. 基于片段生成回答
4. 返回引用来源
```

返回示例：

```json
{
  "answer": "OCR 文本可以通过 POST /api/multimodal-text 写入，参数包括 asset_id、text_type、text、engine、confidence 等。",
  "sources": [
    {
      "doc_id": "docs_api_contracts",
      "filename": "api-contracts.md",
      "chunk_id": "docs_api_contracts_chunk_0007",
      "page": null,
      "section_title": "C：OCR、ASR 与标签融合",
      "snippet": "POST /api/multimodal-text 用于写入 OCR、ASR、字幕、文档等文本信号..."
    }
  ]
}
```

注意：回答必须基于检索到的文档片段，不要凭空编造。

## 三、建议模块结构

B 可以先做成 Python 模块，不需要直接开 HTTP 端口。A/D 后端统一调用，便于集成。

建议目录：

```text
document_module/
  __init__.py
  parser.py
  chunker.py
  tagger.py
  qa.py
  schema.py
  README.md
```

## 四、Python 模块接口要求

### 1. 解析文档

```python
parse_document(path: str) -> DocumentParseResult
```

返回：

```json
{
  "doc_id": "xxx",
  "filename": "xxx.docx",
  "format": "docx",
  "title": "文档标题",
  "pages": 12,
  "text": "全文文本",
  "sections": [
    {
      "section_id": "sec_001",
      "title": "接口说明",
      "page": 3,
      "text": "..."
    }
  ]
}
```

### 2. 文档切片

```python
chunk_document(parse_result: DocumentParseResult) -> list[DocumentChunk]
```

返回：

```json
[
  {
    "chunk_id": "asset_xxx_chunk_0001",
    "doc_id": "asset_xxx",
    "chunk_index": 1,
    "page": 1,
    "section_title": "项目背景",
    "text": "...",
    "summary": "本段介绍项目背景和研究意义"
  }
]
```

### 3. 生成文档标签

```python
generate_document_tags(parse_result, chunks) -> list[DocumentTag]
```

返回：

```json
[
  {
    "name": "计算机教育",
    "type": "topic",
    "confidence": 0.86,
    "evidence": ["信息技术课程", "编程教学", "课堂实践"]
  }
]
```

### 4. 文档问答

```python
answer_from_chunks(question: str, chunks: list[DocumentChunk]) -> dict
```

返回：

```json
{
  "answer": "根据文档内容，...",
  "sources": [
    {
      "chunk_id": "asset_xxx_chunk_0003",
      "doc_id": "asset_xxx",
      "page": 2,
      "section_title": "相关章节",
      "snippet": "..."
    }
  ]
}
```

## 五、建议由 A/D 集成的 HTTP 接口

### 1. 写入文档切片

```text
POST /api/document-chunks
```

请求：

```json
{
  "asset_id": "asset_xxx",
  "chunks": [
    {
      "chunk_id": "asset_xxx_chunk_0001",
      "chunk_index": 1,
      "page": 1,
      "section_title": "项目背景",
      "text": "...",
      "summary": "..."
    }
  ]
}
```

### 2. 写入文档标签

```text
POST /api/document-tags
```

请求：

```json
{
  "asset_id": "asset_xxx",
  "tags": [
    {
      "name": "计算机教育",
      "type": "topic",
      "confidence": 0.86,
      "evidence": ["信息技术课程", "编程教学"]
    }
  ]
}
```

### 3. 文档切片搜索

```text
GET /api/document-search?q=计算机教育&limit=10
```

返回：

```json
{
  "query": "计算机教育",
  "results": [
    {
      "asset_id": "asset_xxx",
      "filename": "example.docx",
      "chunk_id": "asset_xxx_chunk_0004",
      "page": 2,
      "section_title": "信息技术教学",
      "score": 0.82,
      "summary": "本段讨论计算机课程在教育场景中的应用。",
      "snippet": "..."
    }
  ]
}
```

### 4. 文档问答

```text
POST /api/document-qa
```

请求：

```json
{
  "question": "OCR 接口怎么调用？",
  "library": "all",
  "top_k": 5
}
```

返回：

```json
{
  "answer": "OCR 文本可以通过 POST /api/multimodal-text 写入...",
  "sources": [
    {
      "asset_id": "asset_xxx",
      "filename": "api-contracts.md",
      "chunk_id": "asset_xxx_chunk_0007",
      "page": null,
      "section_title": "OCR 接口",
      "snippet": "..."
    }
  ]
}
```

## 六、验收标准

B 的阶段性成果至少应满足：

- 能解析 `.md / .txt / .docx`，PDF/PPTX 可作为进阶。
- 能把文档切成 chunk。
- 每个 chunk 至少包含 `chunk_id / page / section_title / text / summary`。
- 能生成 5-12 个代表性文档标签。
- 能支持“描写春天的文章”“计算机教育文章”这种大意匹配。
- 搜索结果能返回命中文档片段，而不是只返回整篇文档。
- 文档问答能给出回答和引用来源。
- 提供 `README.md`，说明输入、输出和调用示例。

## 七、协作边界

B 负责：

```text
文档解析
文档切片
文档主题标签
文档摘要
文档问答逻辑
```

A/D 负责：

```text
接入后端 API
写入 SQLite
生成 chunk 向量
接入主搜索调度器
前端展示搜索结果
评测文档检索效果
```

B 不需要直接改 Web 前端，也不需要自己维护主数据库。B 只需要提供稳定的 Python 模块和清晰输出格式。

## 八、一句话目标

```text
你的目标不是“把文档读出来”，而是把文档变成可检索、可定位、可解释、可问答的多模态素材单元。
```

