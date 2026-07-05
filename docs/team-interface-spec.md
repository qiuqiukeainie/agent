# 团队协作与接口规范

本文档是本项目 A/B/C/D/E 五个角色并行开发时共同遵守的协作规范。目标是让每个人的模块可以独立开发、独立调试，并且最终能够稳定集成。

## 1. 协作总原则

### 1.1 统一数据主线

所有模块围绕同一条数据主线协作：

```text
素材文件 -> asset_id -> 元数据 -> 特征/标签/OCR/人脸 -> 检索与展示
```

任何模块的输出都必须能回到一个明确的 `asset_id`。

### 1.2 不互相读取临时文件

组员之间不要通过“我在某个目录放了一个临时 JSON，你去读它”这种方式协作。

推荐顺序：

1. 优先通过 HTTP API 对接。
2. 其次写入统一 SQLite 表。
3. 最后才考虑中间文件，并且必须写入 `data/intermediate/角色名/`。

### 1.3 字段只增不乱改

已经被前端或其他模块使用的字段，不要随便改名。

如果必须修改：

- 在群里说明原因。
- 文档同步更新。
- 保留兼容字段至少一个版本。

### 1.4 先小闭环，再堆功能

每个模块交付时都要有一个最小可运行闭环：

- 能输入什么
- 输出什么
- 怎么验证
- 怎么接入主系统

不要只交一段不能跑的模型代码。

## 2. 目录规范

当前主目录：

```text
F:\agent
```

推荐目录结构：

```text
F:\agent
  app.py
  agent_engine.py
  vector_engine.py
  vector_index.py
  metadata_store.py
  static/
  tools/
  docs/
  data/
    uploads/
    assets.db
    index.json
    eval/
    datasets/
    intermediate/
```

### 2.1 代码目录

| 路径 | 说明 |
| --- | --- |
| `app.py` | HTTP API 服务 |
| `agent_engine.py` | Agent 查询拆解与调度计划 |
| `vector_engine.py` | CLIP 检索、融合重排 |
| `vector_index.py` | Faiss/NumPy 向量索引适配 |
| `metadata_store.py` | SQLite 元数据、标签、OCR、日志 |
| `static/` | Web 前端 |
| `tools/` | 数据导入、评测、批处理脚本 |
| `docs/` | 协作规范、接口、报告说明 |

### 2.2 数据目录

| 路径 | 说明 |
| --- | --- |
| `data/uploads/` | 统一素材目录 |
| `data/assets.db` | SQLite 元数据库 |
| `data/index.json` | 当前向量索引持久化文件 |
| `data/eval/` | 评测报告 |
| `data/datasets/` | 原始数据集，不建议打包提交 |
| `data/intermediate/` | 必要中间文件 |

### 2.3 禁止提交或打包的目录

以下内容不要放进提交包：

```text
.venv/
.model_cache/
__pycache__/
data/datasets/*.zip
data/datasets/coco/val2017/
```

这些内容要么太大，要么可以重建。

## 3. 命名规则

### 3.1 asset_id

素材唯一 ID 统一叫：

```text
asset_id
```

图片文件默认规则：

```text
000000012667.jpg -> asset_id = 000000012667
```

禁止在不同模块中使用这些混乱字段名：

```text
image_id
file_id
pic_id
material_id
```

如果使用外部数据集原始 ID，可以保存为：

```text
source_id
source_dataset
```

### 3.2 person_id

人脸/人物统一使用：

```text
person_id
```

命名格式：

```text
p_0001
p_0002
```

人物别名使用：

```text
alias
```

例如：

```json
{
  "person_id": "p_0001",
  "alias": "小明"
}
```

### 3.3 tag 命名

标签统一小写英文，使用单数名词：

```text
person
dog
cat
bus
building
food
```

不推荐：

```text
People
dogs
漂亮的风景
```

中文标签可以作为 `display_name` 后续补充，但检索和重排优先用英文规范标签。

### 3.4 source 命名

所有自动生成结果都要标明来源：

```text
coco_instances
clip
vlm_blip
vlm_llava
insightface
paddleocr
easyocr
whisper
manual
```

### 3.5 文件命名

Python 文件：

```text
lower_snake_case.py
```

工具脚本：

```text
tools/import_xxx.py
tools/evaluate_xxx.py
tools/rebuild_xxx.py
```

文档：

```text
docs/kebab-case-name.md
```

## 4. 数据库规范

统一数据库：

```text
data/assets.db
```

### 4.1 当前已有表

| 表名 | 说明 |
| --- | --- |
| `assets` | 素材基础信息 |
| `tags` | 标签字典 |
| `asset_tags` | 素材-标签关系 |
| `ocr_texts` | OCR/字幕/ASR 文本 |
| `search_logs` | 搜索日志 |

### 4.2 assets 表核心字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | TEXT | asset_id |
| `filename` | TEXT | 文件名 |
| `path` | TEXT | 文件路径 |
| `kind` | TEXT | image/video/audio |
| `created_at` | TEXT | 创建时间 |
| `width` | INTEGER | 图片宽 |
| `height` | INTEGER | 图片高 |
| `vector_model` | TEXT | 向量模型 |
| `sha256` | TEXT | 文件哈希 |

### 4.3 tags 表

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | INTEGER | 标签 ID |
| `name` | TEXT | 规范标签名 |
| `source` | TEXT | 标签来源 |
| `created_at` | TEXT | 创建时间 |

### 4.4 asset_tags 表

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `asset_id` | TEXT | 素材 ID |
| `tag_id` | INTEGER | 标签 ID |
| `confidence` | REAL | 置信度 |
| `source` | TEXT | 来源 |
| `created_at` | TEXT | 创建时间 |

### 4.5 ocr_texts 表

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `asset_id` | TEXT | 素材 ID |
| `text` | TEXT | OCR/字幕/ASR 文本 |
| `engine` | TEXT | paddleocr/easyocr/whisper/manual |
| `confidence` | REAL | 置信度 |
| `created_at` | TEXT | 写入时间 |

## 5. HTTP API 规范

### 5.1 通用响应规则

成功：

```json
{
  "ok": true,
  "result": {}
}
```

列表：

```json
{
  "assets": []
}
```

失败：

```json
{
  "error": "具体错误原因"
}
```

### 5.2 状态接口

```http
GET /api/status
```

返回示例：

```json
{
  "vector_model": "clip-vit-base-patch32",
  "asset_count": 526,
  "indexed_count": 526,
  "agent": "planner-agent-v1",
  "vector_index": "numpy",
  "routes": ["intent_parse", "condition_split", "clip_recall", "metadata_filter", "fusion_rerank"]
}
```

### 5.3 素材列表

```http
GET /api/assets
```

返回：

```json
{
  "assets": [
    {
      "id": "000000012667",
      "filename": "000000012667.jpg",
      "url": "/uploads/000000012667.jpg",
      "kind": "image",
      "created_at": "2026-06-29T12:00:00",
      "width": 640,
      "height": 480
    }
  ]
}
```

### 5.4 检索接口

```http
GET /api/search?q=a%20photo%20of%20a%20dog
```

返回：

```json
{
  "plan": {
    "raw_query": "a photo of a dog",
    "intent": "asset_search",
    "semantic_queries": ["a photo of a dog"],
    "executable_filters": {"year": null, "season": null, "kind": null},
    "unresolved_conditions": {},
    "recall_routes": ["clip_text_image", "filename_keyword"],
    "rerank_policy": "0.72*clip + 0.12*tag + 0.08*metadata + 0.05*filename + 0.03*ocr",
    "downstream_requirements": []
  },
  "results": [
    {
      "id": "000000029393",
      "filename": "000000029393.jpg",
      "url": "/uploads/000000029393.jpg",
      "score": 0.3112,
      "display_score": 0.98,
      "raw_clip_score": 0.2656,
      "tag_score": 1.0,
      "tag_hits": ["dog"],
      "ocr_score": 0.0,
      "ocr_hits": [],
      "best_prompt": "a photo of a dog"
    }
  ]
}
```

注意：

- `raw_clip_score` 不是概率。
- 页面展示优先使用 `display_score`。
- 结果排序使用融合分 `score`。

### 5.5 OCR 写入接口

```http
POST /api/ocr-text
Content-Type: application/json
```

请求：

```json
{
  "asset_id": "000000012667",
  "text": "SALE 50% OFF",
  "engine": "paddleocr",
  "confidence": 0.93
}
```

返回：

```json
{
  "ok": true,
  "result": {
    "asset_id": "000000012667",
    "engine": "paddleocr",
    "text_length": 12
  }
}
```

### 5.6 统计接口

```http
GET /api/stats
```

返回：

```json
{
  "total_assets": 526,
  "tag_count": 78,
  "asset_tag_count": 1485,
  "ocr_text_count": 0,
  "duplicate_assets": 21,
  "search_log_count": 237,
  "json_index_assets": 526
}
```

## 6. 各角色交付规范

### 6.1 A：算法负责人 / 组长

负责：

- CLIP 检索核心
- Agent 查询调度
- 多路召回与融合重排
- 接口规范维护
- 项目进度统筹

主要文件：

```text
agent_engine.py
vector_engine.py
vector_index.py
docs/team-interface-spec.md
```

交付物：

- `/api/search`
- 查询 plan
- 检索结果
- 评测指标对比
- 每周整合版本

### 6.2 B：视觉算法工程师

负责：

- 图像特征提取
- 人脸检测与聚类
- 场景理解
- VLM 图片描述

建议输出：

```json
{
  "asset_id": "000000012667",
  "tags": [
    {"name": "person", "confidence": 0.91, "source": "vlm_blip"},
    {"name": "building", "confidence": 0.88, "source": "vlm_blip"}
  ],
  "persons": [
    {"person_id": "p_0001", "alias": "小明", "confidence": 0.93}
  ]
}
```

先做标签接入，后做人脸表。

### 6.3 C：多模态融合工程师

负责：

- OCR
- 视频字幕
- Whisper 音频转写
- 多模态标签融合

当前最小接入方式：

```http
POST /api/ocr-text
```

字幕和 ASR 也先按文本写入 `ocr_texts`，`engine` 分别写：

```text
paddleocr
easyocr
whisper
subtitle
```

### 6.4 D：数据与系统工程

负责：

- Pipeline
- SQLite 元数据
- Faiss 向量索引
- 去重
- 评测报表

主要文件：

```text
metadata_store.py
tools/evaluate_coco_retrieval.py
tools/import_coco_tags.py
```

统一评测指标：

- Recall@1
- Recall@5
- Recall@10
- MRR@10

### 6.5 E：全栈与产品负责人

负责：

- Web 页面
- 系统集成
- 演示视频
- PPT 和报告

要求：

- 只调用 HTTP API。
- 不直接读取 Python 内部变量。
- 页面需要展示分数构成、Agent plan、标签/OCR 命中。

## 7. 调试流程

### 7.1 启动服务

```powershell
cd F:\agent
.\.venv\Scripts\python.exe app.py
```

### 7.2 检查基础状态

浏览器打开：

```text
http://127.0.0.1:8000/api/status
http://127.0.0.1:8000/api/stats
```

### 7.3 检查检索

```text
http://127.0.0.1:8000/api/search?q=a%20photo%20of%20a%20dog
```

### 7.4 检查 OCR 写入

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/ocr-text" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"asset_id":"000000012667","text":"banana phone","engine":"paddleocr","confidence":0.93}'
```

### 7.5 跑评测

```powershell
.\.venv\Scripts\python.exe tools\evaluate_coco_retrieval.py --limit 100 --top-k 10
```

## 8. 简单计划分工

### 第 1 阶段：主链路稳定

时间：第 1 周

| 成员 | 任务 | 交付 |
| --- | --- | --- |
| A | CLIP 检索 + Agent plan | `/api/search` 可用 |
| D | SQLite + 索引重建 | `/api/stats` 可用 |
| E | Web 页面 | 能上传、浏览、检索 |
| B | 初步图像标签方案 | 标签字段设计 |
| C | OCR 接口方案 | `/api/ocr-text` 对接 |

### 第 2 阶段：结构化标签增强

时间：第 2 周

| 成员 | 任务 | 交付 |
| --- | --- | --- |
| B | 场景/对象标签生成 | 写入 `asset_tags` |
| C | OCR 批量识别 | 写入 `ocr_texts` |
| D | 评测脚本和报表 | Recall@K 报告 |
| A | 融合重排优化 | 标签/OCR 参与排序 |
| E | 标签/OCR 命中展示 | 页面可解释 |

### 第 3 阶段：人物与归档

时间：第 3 周

| 成员 | 任务 | 交付 |
| --- | --- | --- |
| B | 人脸聚类 | `person_id` 和 alias |
| D | 去重与归档 | 重复素材列表 |
| A | 人物条件接入 Agent | 查询支持 person filter |
| C | 音频/字幕文本 | whisper/subtitle 文本 |
| E | 归档/人物视图 | 页面多视图 |

### 第 4 阶段：答辩与报告

时间：第 4 周

| 成员 | 任务 | 交付 |
| --- | --- | --- |
| A | 架构图、接口说明、总整合 | 技术总述 |
| B | 视觉模块实验 | 模块报告 |
| C | OCR/ASR 融合说明 | 模块报告 |
| D | 指标报表 | Recall/MRR/耗时 |
| E | PPT、演示视频、系统联调 | 答辩材料 |

## 9. 每次提交必须说明

每个成员提交或发包时必须写：

```text
1. 修改文件：
2. 新增接口：
3. 新增字段：
4. 是否需要重建索引：
5. 是否需要更新数据库：
6. 测试命令：
7. 对其他成员的影响：
```

示例：

```text
1. 修改文件：metadata_store.py, tools/import_ocr_text.py
2. 新增接口：POST /api/ocr-text
3. 新增字段：ocr_texts.engine, ocr_texts.confidence
4. 是否需要重建索引：否
5. 是否需要更新数据库：首次启动自动建表
6. 测试命令：python tools/import_ocr_text.py --csv data/ocr_rows.csv
7. 对其他成员的影响：E 可展示 OCR 命中，A 可参与重排
```

## 10. 风险与注意事项

- 不要直接删除 `data/uploads` 中的文件。
- 不要手动编辑 `data/index.json`。
- 不要把 `.venv`、模型缓存、大数据集 zip 打进提交包。
- 不要修改已有 API 字段名。
- 不要让不同模块各自维护一份素材 ID。
- OCR、人脸、标签输出必须带 `confidence` 和 `source`。
- 所有实验结果要保存到 `data/eval/`。

## 11. 当前基线

当前系统基线：

```text
素材数：526
模型：CLIP ViT-B/32
索引后端：NumPy fallback
标签关联：1485
OCR 文本：0
评测：COCO caption 100 queries
Recall@1：0.68
Recall@5：0.96
Recall@10：1.00
MRR@10：0.7927
```

后续任何优化都应尽量和这个基线对比。
