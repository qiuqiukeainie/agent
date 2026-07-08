# 团队接口契约

本文档用于 A/B/C/D/E 成员联调。除上传接口外，HTTP 接口默认使用 JSON；所有返回建议包含可读的 `error` 字段，方便前端展示。

## 通用字段

- `asset_id`：素材唯一 ID，前端、向量索引、元数据表、人脸结果都用它关联。
- `filename`：上传后的文件名，不保证等于用户原始文件名。
- `kind`：素材类型，当前包括 `image`、`video`、`document`。
- `library`：素材库类型，当前包括 `public` 和 `personal`。
- `tags`：统一标签列表，可来自视觉、OCR、ASR、人脸、人工编辑、文档文本。
- `confidence`：置信度，范围建议为 `0.0-1.0`。
- `source`：标签或文本来源，例如 `clip`、`ocr`、`asr`、`face`、`manual`、`document`。

## A：检索与 Agent

### GET `/api/search`

用途：单次自然语言检索入口，由 Agent 完成意图解析、条件拆解、多路召回和重排。不保留对话上下文。

查询参数：

- `q`：用户查询文本。
- `limit`：返回数量，默认前端常用 `36`。
- `library`：`all/public/personal`。

返回核心字段：

```json
{
  “query”: “person_0001 在海边”,
  “plan”: {
    “intent”: “search”,
    “conditions”: {},
    “query_rewrites”: [],
    “trace”: []
  },
  “results”: [
    {
      “id”: “asset_xxx”,
      “filename”: “demo.jpg”,
      “kind”: “image”,
      “library”: “personal”,
      “display_score”: 0.73,
      “tag_hits”: [],
      “ocr_hits”: [],
      “metadata_hits”: [],
      “explanation”: []
    }
  ]
}
```

约定：

- 前端只展示 `display_score`，不要直接解释原始 CLIP 相似度。
- Agent 新增能力时，应写入 `plan.trace`，便于答辩展示”系统不是黑盒”。
- 如果查询包含人物别名，后端负责映射到 `person_id`，前端不做别名解析。

### POST `/api/chat`

用途：多轮对话检索入口。与 `/api/search` 的区别是它会维护会话上下文，支持用户通过多轮对话逐步精炼检索条件（如”只要视频””不要风景””换成海边”）。

请求体：

```json
{
  “query”: “找春天的照片”,
  “session_id”: “a1b2c3d4e5f6”,
  “library”: “all”,
  “limit”: 36
}
```

- `session_id`：可选。首次对话不传，后端会创建并返回新 session；后续轮次传入以保持上下文。
- `library`：`all/public/personal`，默认 `all`。
- `limit`：返回结果上限，默认 36，最大 80。

返回核心字段：

```json
{
  “session_id”: “a1b2c3d4e5f6”,
  “results”: [],
  “plan”: {
    “intent”: “chat”,
    “chat_agent”: {
      “provider”: “deepseek”,
      “model”: “deepseek-chat”,
      “search_query”: “春天的照片”,
      “reason”: “新对话，直接检索”,
      “llm_intent”: “search”
    },
    “chat_intent”: “search”,
    “chat_entities”: {
      “people”: [],
      “locations”: [],
      “scenes”: [“春天”],
      “objects”: [],
      “actions”: [],
      “time”: [],
      “media_type”: null,
      “library”: “all”,
      “negatives”: []
    }
  },
  “should_clarify”: false,
  “clarification_text”: “”,
  “answer”: “正在检索春天的照片。找到 12 个结果。”,
  “search_query”: “春天的照片”,
  “llm_used”: true
}
```

语义分析流程：

1. 前端发送 query + session_id。
2. 后端调用 DeepSeek LLM 做**意图分类**（search/refine/clarify/chat）、**实体提取**（人物、地点、场景、物体、动作、时间、媒体类型、否定条件）、**上下文合并**（将简短补全合并为完整 search_query）。
3. LLM 返回的 `entities` 注入 `SearchAgent.build_plan()` 的 QueryPlan，补充规则解析的盲区。
4. `search_query` 经人物名扩展后进入多路召回和融合重排。
5. 结果经对话级否定过滤（`apply_chat_negative_filters`）后返回。

约定：

- `should_clarify` 为 true 时前端应展示 `clarification_text`，引导用户补充条件。
- 连续 30 分钟无活动后会话自动过期。
- 前端对话检索结果**仅在对话框内展示**，不更新主界面结果网格。

### GET `/api/chat/sessions`

用途：列出当前所有活跃会话。

返回：

```json
{
  “sessions”: [
    {
      “session_id”: “a1b2c3d4e5f6”,
      “library”: “all”,
      “message_count”: 6,
      “created_at”: “2026-07-08T21:30:00”,
      “last_active”: “2026-07-08T21:35:00”
    }
  ]
}
```

### POST `/api/chat/clear`

用途：清除指定会话及其历史消息。

请求体：

```json
{
  “session_id”: “a1b2c3d4e5f6”
}
```

### GET `/api/status`

用途：系统健康状态。

关键字段：

- `vector_model`
- `asset_count`
- `indexed_count`
- `agent`
- `routes`
- `supported_document_types`

## B：人脸模块

B 的模块应优先以 Python 模块方式提供，不直接开端口。A/D 在后端 Pipeline 中调用它，这样部署更简单，也避免多个服务互相抢端口。

建议模块输出：

```json
{
  "asset_id": "asset_xxx",
  "faces": [
    {
      "face_id": "face_xxx",
      "embedding": [0.01, 0.02],
      "bbox": [x1, y1, x2, y2],
      "det_score": 0.98,
      "person_id": "person_0001"
    }
  ]
}
```

后端已提供的人物相关 HTTP 接口：

- GET `/api/persons?library=personal`：查看人物分组。
- POST `/api/rebuild-persons`：重建人物聚类。
- POST `/api/rename-person`：给人物组命名。
- POST `/api/merge-persons`：合并人物组。
- POST `/api/delete-person`：删除人物组标签，不删除素材文件。
- POST `/api/remove-asset-person`：从当前素材移除某个人物命中。
- POST `/api/repair-person-tags`：清理旧人物标签并按当前聚类结果重写。
- POST `/api/cleanup-persons`：清理无效人物分组。

人物命名规则：

- 系统内部 ID 保持 `person_0001` 这种稳定代号。
- 用户可设置显示名，例如“俊”“陈”。
- 搜索时应同时支持 `person_0001` 和显示名。

## C：OCR、ASR 与标签融合

### POST `/api/multimodal-text`

用途：写入 OCR、ASR、字幕、文档等文本信号。

请求示例：

```json
{
  "asset_id": "asset_xxx",
  "text_type": "ocr",
  "text": "画面中的文字",
  "engine": "easyocr",
  "confidence": 0.86,
  "language": "zh"
}
```

`text_type` 建议取值：

- `ocr`：画面文字或字幕 OCR。
- `asr`：音频转写。
- `subtitle`：字幕文件或字幕轨。
- `document`：文档正文。

### POST `/api/ocr-text`

用途：兼容已有 OCR 写入接口。新功能优先使用 `/api/multimodal-text`，除非只需要简单 OCR 文本写入。

### POST `/api/fuse-tags`

用途：根据 OCR/ASR/视觉/文档文本等信号生成统一标签。

约定：

- C 输出文本，A/D 负责入库和检索融合。
- OCR 和 ASR 必须区分 `text_type`，不要混成一个字段。
- 低置信文本可以入库，但标签置信度应降低。

## D：数据、索引、归档与评测

### GET `/api/assets`

用途：素材库列表。

### GET `/api/asset?id=...`

用途：素材详情，包括标签、人脸、文本信号、相似素材、归档建议。

### POST `/api/asset-tags`

用途：写入人工标签或外部模块生成的标签。

请求示例：

```json
{
  "asset_id": "asset_xxx",
  "tags": [
    {"name": "海边", "source": "manual", "confidence": 1.0}
  ]
}
```

### POST `/api/upload`

用途：上传图片、视频、文档。

表单字段：

- `files`：一个或多个文件。
- `library`：`public` 或 `personal`。

### POST `/api/process-assets`

用途：启动批处理任务，例如视频 OCR、标签补全、文本信号抽取。

### GET `/api/evaluate-retrieval`

用途：检索评测，对比 `clip_only`、`clip_tags`、`agent_v3` 等方案。

## E：前端与演示

E 不直接调用 Python 模块，只调用 HTTP API。前端新增功能前需要确认三件事：

- 接口是否已存在。
- 返回字段是否足够渲染。
- 空数据和失败情况是否有提示。

演示时推荐使用固定查询集，避免现场临时输入过于发散导致展示不可控。

