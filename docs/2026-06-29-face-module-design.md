# face_module 人脸模块设计方案

日期：2026-06-29
作者：B（视觉算法工程师）
状态：待审核

## 1. 概述

`face_module` 是素材语义管理系统中 B 角色负责的独立 Python 包，提供人脸检测、特征提取、聚类分组能力。模块只产出结构化数据，不直接操作数据库或 HTTP 接口。

### 核心定位

- **对外**：暴露 `analyze_faces()` 和 `cluster_faces()` 两个函数，A 通过 `from face_module import ...` 调用
- **对内**：`detector.py`（InsightFace 检测+embedding）和 `cluster.py`（聚类逻辑）各自独立
- **边界**：B 模块不写数据库，不自己开端口，不 import A 项目的任何模块

## 2. 目录结构

```text
F:\agent\
  face_module/              ← B 的独立模块
    __init__.py             ← 只暴露 analyze_faces, cluster_faces
    schema.py               ← 数据结构定义
    detector.py             ← InsightFace 检测 + embedding
    cluster.py              ← 聚类逻辑
    tests/
        test_detector.py    ← 单元测试（可 mock 模型）
        test_cluster.py     ← 纯单元测试，不依赖 InsightFace
    README.md               ← 使用说明（给 A）
  app.py
  vector_engine.py          ← A 的代码
  metadata_store.py
  data/
```

### 硬约束

- A 永远不 `from face_module.detector import FaceDetector`，只依赖 `analyze_faces` 和 `cluster_faces` 两个入口
- B 模块内部实现可自由重构，只要两个对外函数签名稳定

## 3. 数据结构（schema.py）

```python
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FaceResult:
    """单张人脸检测结果"""
    asset_id: str                           # 所属素材 id
    face_id: str                            # face_{asset_id}_{index:04d}，全局唯一
    person_id: str = "unknown"              # 聚类后回填，见 PersonCluster
    bbox: list[float] = field(default_factory=list)   # [x1, y1, x2, y2]
    confidence: float = 0.0                 # 检测置信度 [0, 1]
    quality: float = 0.0                    # 人脸质量分 [0, 1]（v1 直接用 confidence）
    source: str = "insightface"             # 模型来源
    embedding: Optional[list[float]] = field(default=None, repr=False)  # 512 维向量


@dataclass
class AssetFaceAnalysis:
    """analyze_faces() 返回值"""
    asset_id: str
    faces: list[FaceResult]
    summary_tags: list[str]                 # 聚类前只含 ["has_face"]
    version: str = "face-module-v1"


@dataclass
class PersonCluster:
    """单个人物的聚类结果"""
    person_id: str                          # person_{index:04d}
    asset_ids: list[str]                    # 该人物出现的素材列表
    face_count: int                         # 关联的脸数
    prototype_face_id: str                  # 代表照的 face_id（quality 最高那张）


@dataclass
class FaceClusterResult:
    """cluster_faces() 返回值"""
    persons: list[PersonCluster]
    faces: list[FaceResult]                 # 已回填 person_id 的脸列表
    asset_tags: list[dict]                  # [{"asset_id": "xxx", "summary_tags": ["has_face","person_0001"]}]
    version: str = "face-module-v1"
```

### 字段约定

| 字段 | 格式/示例 | 说明 |
|------|----------|------|
| `face_id` | `face_000000012667_0001` | 全局唯一，含 asset_id |
| `person_id` | `person_0001` | 聚类后分配，按 cluster 大小降序 |
| `person_id`（占位） | `"unknown"` | 单张检测时的默认值 |
| `bbox` | `[120, 48, 188, 136]` | `[x1, y1, x2, y2]`，与已有接口文档一致 |
| `quality` | 0.0-1.0 | v1 简化实现：直接用 `confidence` 值 |
| `source` | `"insightface"` | 便于后续多模型混用时追溯 |
| `version` | `"face-module-v1"` | 模型升级时 A 可知需重建索引 |

## 4. detector.py 设计

### FaceDetector 类

封装 InsightFace，延迟加载模型。

```python
class FaceDetector:
    def __init__(self, device: str = "auto"):
        """
        device: 'cpu' | 'cuda' | 'auto'（有 GPU 则用 GPU）
        延迟加载：__init__ 只记录配置，不初始化模型
        """

    def analyze(self, asset_path: str, asset_id: str | None = None) -> AssetFaceAnalysis:
        """
        输入：
            asset_path: 图片文件路径
            asset_id: 素材 id（可选，不传则从文件名推导）
        输出：
            AssetFaceAnalysis

        行为：
            - 图片不存在/损坏/格式不支持：抛出 ValueError
            - 无脸：返回 AssetFaceAnalysis(faces=[], summary_tags=[])
            - 有脸：每张脸产生一个 FaceResult，person_id="unknown"
        """
```

### 关键设计点

| 项目 | 决策 |
|------|------|
| 模型 | InsightFace `buffalo_l`（detection + recognition） |
| 加载时机 | 首次调用 `analyze()` 时，不在 `__init__` |
| 无脸 | 正常返回，不抛异常 |
| 文件异常 | 抛出 `ValueError`，包含具体原因 |
| `face_id` 生成 | `face_{asset_id}_{index:04d}`，从 0 开始 |
| `asset_id` 推导 | 传入则用传入值；不传则取 `Path(asset_path).stem` |
| `summary_tags` | 有人脸 → `["has_face"]`；无人脸 → `[]` |
| v1 限制 | 只支持 `.jpg/.jpeg/.png/.webp/.bmp`；视频需 D 先抽帧 |

### quality 计算（v1 简化）

v1 直接使用 `confidence` 值：

```python
quality = confidence  # v1 简化，后续版本可引入大小/姿态/清晰度加权
```

## 5. cluster.py 设计

### FaceClusterer 类

```python
class FaceClusterer:
    def cluster(
        self,
        faces: list[FaceResult],
        method: str = "cosine_threshold",
        threshold: float = 0.5,
    ) -> FaceClusterResult:
        """
        输入：所有素材的所有 FaceResult（含 embedding）
        输出：FaceClusterResult（persons + 回填后的 faces + asset_tags）

        method 选项：
            "cosine_threshold" — v1 唯一实现
            "dbscan"           — v2 预留
        """
```

### 聚类算法（v1）

```
所有 embedding → 两两余弦相似度矩阵
  → 阈值 > 0.5 建立连边
  → 并查集/连通分量分组
  → 每个连通分量 = 一个 person_id
  → person_id 按 cluster 大小降序分配：person_0001 是人最多的组
```

### 代表照选择

每个 `PersonCluster` 取 `quality` 最高的 `FaceResult`，记录其 `face_id` 为 `prototype_face_id`。

### asset_tags 汇总

`cluster_faces()` 自动汇总每个素材的人物标签：

```python
asset_tags = [
    {"asset_id": "000000012667", "summary_tags": ["has_face", "person_0001"]},
    {"asset_id": "000000029393", "summary_tags": ["has_face", "person_0001", "person_0002"]},
]
```

A 可直接遍历写入 `asset_tags` 表。

## 6. 对外 API（__init__.py）

```python
# face_module/__init__.py

_detector = FaceDetector()

def analyze_faces(asset_path: str, asset_id: str | None = None) -> dict:
    """单张素材人脸分析，返回 AssetFaceAnalysis 的字典形式"""
    return _detector.analyze(asset_path, asset_id).to_dict()


def cluster_faces(items: list[dict | AssetFaceAnalysis], threshold: float = 0.5) -> dict:
    """批量聚类，返回 FaceClusterResult 的字典形式"""
    ...
```

A 的调用方式：

```python
from face_module import analyze_faces, cluster_faces
```

## 7. 与 A 项目的集成流程

### A 侧完整调用流程

```python
from face_module import analyze_faces, cluster_faces
from metadata_store import MetadataStore

metadata = MetadataStore("data/assets.db")

# 第一步：逐张分析
analyses = []
for asset in engine.assets:
    if asset.kind == "image":
        analyses.append(analyze_faces(str(asset.path), asset.id))

# 第二步：全局聚类
cluster_result = cluster_faces(analyses)

# 第三步：写入 asset_persons 表（每张脸的实例）
for face in cluster_result["faces"]:
    metadata.upsert_asset_person(
        asset_id=face["asset_id"],
        person_id=face["person_id"],
        face_id=face["face_id"],
        bbox=face["bbox"],
        confidence=face["confidence"],
        source=face["source"],
    )

# 第四步：写入 persons 表（每个人物一条记录）
for person in cluster_result["persons"]:
    metadata.upsert_person(
        person_id=person["person_id"],
        prototype_face_id=person["prototype_face_id"],
        face_count=person["face_count"],
    )

# 第五步：写入 asset_tags 表（summary_tags）
for entry in cluster_result["asset_tags"]:
    metadata.set_asset_tags(
        asset_id=entry["asset_id"],
        tags=entry["summary_tags"],
        source="face_module",
        confidence=1.0,
    )

# 第六步：可选，清理旧结果再写入
metadata.clear_face_results(source="face_module")
```

### A 需要新增的数据库表

**persons 表：**

```sql
CREATE TABLE persons (
    person_id TEXT PRIMARY KEY,           -- person_0001
    prototype_face_id TEXT,               -- 代表照 face_id
    alias TEXT,                           -- "小明"（phase 2 填充）
    face_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
);
```

**asset_persons 表（实际存的是脸实例）：**

```sql
CREATE TABLE asset_persons (
    face_id TEXT PRIMARY KEY,             -- face_000000012667_0001
    asset_id TEXT NOT NULL,
    person_id TEXT NOT NULL,
    bbox TEXT NOT NULL,                   -- JSON: [x1, y1, x2, y2]
    confidence REAL,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(asset_id) REFERENCES assets(id)
);
CREATE INDEX idx_asset_persons_person ON asset_persons(person_id);
```

### A 需要新增的 MetadataStore 方法

| 方法 | 说明 |
|------|------|
| `upsert_asset_person(face_id, asset_id, person_id, bbox, confidence, source)` | 写入/更新一张脸的记录 |
| `upsert_person(person_id, prototype_face_id, face_count)` | 写入/更新一个人物记录 |
| `clear_face_results(source)` | 清理旧分析结果（按 source 删除） |

### 标签写入时机（关键）

| 阶段 | summary_tags |
|------|-------------|
| `analyze_faces()` 后 | 只有 `["has_face"]`（person_id 还是 `"unknown"`） |
| `cluster_faces()` 后 | `["has_face", "person_0001", "person_0002"]`（完整标签）|

⚠️ **A 侧注意事项：**
- 重新分析前需按 `source="face_module"` 清理旧结果，避免残留
- `asset_tags` 里的 person 标签必须在聚类之后写入，不能在 `analyze_faces()` 之后写入

## 8. 验证方案

### 8.1 单元测试（tests/）

| 测试文件 | 测试内容 | 依赖 |
|----------|---------|------|
| `test_cluster.py` | 给定人造 embedding，验证聚类正确性 | 无（纯计算） |
| `test_detector.py` | 用少量样本图验证检测结构正确 | 可 mock，跳过模型下载 |
| `test_schema.py` | 验证序列化/反序列化 | 无 |

### 8.2 管线验证

**第一轮：CelebA 子集**

- 选取 100 个身份，每个 5 张（共 500 张）
- 验证：同身份的人脸被聚到同一个 `person_id` 的比例（聚类纯度）

**第二轮：COCO 现有 526 张**

- 验证：在真实场景图片上检测人脸，无崩溃，bbox 合理

**第三轮：与 A 联调**

- A 调用 `analyze_faces()` → `cluster_faces()` → 写入数据库
- 验证 `/api/search?q=有人脸的照片` 能命中包含 `has_face` 标签的素材

## 9. v1 明确不做的事

| 项目 | 状态 |
|------|------|
| 视频中的人脸检测 | D 抽帧后调 B |
| person_id → alias 映射 | phase 2 |
| 增量聚类/ID 稳定性 | phase 2 |
| VLM/BLIP 场景标签 | 后续独立模块（当前规范中 B 的另一项职责） |
| HTTP API | 不暴露，只通过 Python 函数调用 |
| 直接写数据库 | 不写，由 A 负责 |
| 多模型切换 | 先用 InsightFace，后续可加 FaceNet |

## 10. 依赖

```
insightface>=0.7
numpy
Pillow
```

与 A 现有 `requirements.txt` 兼容（已有 `torch`、`numpy`、`Pillow`）。
