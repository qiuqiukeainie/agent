# A 组长第一阶段：跨模态检索模型与 Agent 调度架构

## 目标

A 的第一阶段目标不是完成所有多模态理解能力，而是先搭好系统主干：

- CLIP 文本-图像跨模态检索核心。
- Agent 查询调度框架：意图解析、条件拆解、多路召回、融合重排。
- 与 B/C/D/E 的接口边界和项目推进规范。

关键原则：Agent 可以理解用户说了什么，但不能伪造下游能力。比如“和小明的合影”只能被拆成 `people=["小明"]` 这个待解析约束，只有 B 的人脸识别模块提供 `person_id` 后，系统才能真正按“小明”过滤素材。

## A 与其他角色边界

| 角色 | 负责内容 | A 阶段如何对接 |
| --- | --- | --- |
| A | CLIP 检索、Agent 调度、接口规范、项目统筹 | 输出查询计划、调用检索核心、融合重排 |
| B | 图像特征、人脸识别、场景理解 | 向 A 提供 `person_id`、人脸聚类结果、场景标签 |
| C | 音频、字幕 OCR、多模态标签 | 向 A 提供音频文本、字幕文本、统一语义标签 |
| D | Pipeline、Faiss、SQLite、归档与评测 | 向 A 提供向量索引、元数据过滤、评测结果 |
| E | Web、集成、演示和报告 | 消费 A 的 API，展示计划和检索结果 |

## Agent 调度链路

```text
用户查询
  -> 意图解析 intent
  -> 条件拆解 unresolved_conditions
  -> 可执行过滤 executable_filters
  -> 生成 semantic_queries
  -> 多路召回 recall_routes
  -> CLIP 文本-图像召回
  -> 元数据过滤与文件名弱召回
  -> 融合重排
  -> 返回 plan + results
```

## 查询计划示例

查询：“去年冬天和小明在故宫拍的合影”

```json
{
  "raw_query": "去年冬天和小明在故宫拍的合影",
  "intent": "asset_search",
  "semantic_queries": [
    "去年冬天和小明在故宫拍的合影",
    "故宫 合影"
  ],
  "executable_filters": {
    "year": 2025,
    "season": "冬天",
    "kind": "image"
  },
  "unresolved_conditions": {
    "people": ["小明"],
    "locations": ["故宫"],
    "scenes": ["合影"],
    "time_words": ["去年", "冬天"]
  },
  "recall_routes": [
    "clip_text_image",
    "metadata_filter",
    "filename_keyword",
    "deferred_structured_recall"
  ],
  "rerank_policy": "0.82*clip + 0.10*metadata + 0.08*filename; deferred conditions are not scored",
  "downstream_requirements": [
    "B.face_person_index: resolve person names to person_id",
    "B.vlm_scene_tags and C.semantic_tags: provide structured labels",
    "D.sqlite_metadata: provide reliable capture_time metadata"
  ]
}
```

解释：

- `小明` 不是当前阶段可直接匹配的条件，只是待 B 模块解析的人物约束。
- `故宫` 和 `合影` 当前可参与 CLIP 语义召回，但不能当作结构化标签强过滤，除非 B/C 已生成标签。
- `去年`、`冬天` 可以先转成时间过滤条件，但真实项目里仍应以 D 的拍摄时间元数据为准。

## API 契约

### `GET /api/status`

返回模型与调度链路状态。

```json
{
  "vector_model": "clip-vit-base-patch32",
  "asset_count": 10,
  "indexed_count": 10,
  "agent": "planner-agent-v1",
  "routes": ["intent_parse", "condition_split", "clip_recall", "metadata_filter", "fusion_rerank"]
}
```

### `POST /api/upload`

上传图片并建立图像向量。

请求：`multipart/form-data`，字段名为 `files`。

### `GET /api/search?q=...`

返回：

- `plan`：Agent 查询计划。
- `results`：融合重排后的素材列表。

结果字段：

```json
{
  "score": 0.7215,
  "clip_score": 0.8123,
  "metadata_score": 0.5,
  "filename_score": 0.0,
  "metadata_hits": ["2025"],
  "deferred_conditions": {
    "people": ["小明"],
    "locations": ["故宫"],
    "scenes": ["合影"]
  }
}
```

## 第一阶段验收标准

- 能加载 CLIP 并完成图文向量检索。
- 查询 Agent 能输出结构化计划，而不是只把用户文本直接丢给模型。
- 明确区分“当前可执行条件”和“下游待解析条件”。
- API 字段稳定，方便 B/C/D/E 并行开发。
- 项目文档能说明模块边界、数据流和后续迭代路线。
