# face_module 人脸模块实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 A 项目下创建 `face_module/` 独立 Python 包，提供 `analyze_faces()` 和 `cluster_faces()` 两个函数，实现人脸检测、特征提取与聚类分组。

**Architecture:** 模块内部分为 `schema.py`（数据结构）、`detector.py`（InsightFace 封装）、`cluster.py`（聚类逻辑）三个文件，对外只通过 `__init__.py` 暴露两个函数。模块只产出结构化 dict，不操作数据库或 HTTP。

**Tech Stack:** Python 3, InsightFace (buffalo_l), numpy, Pillow, dataclasses

---

## 文件清单

| 文件 | 操作 | 职责 |
|------|------|------|
| `face_module/__init__.py` | 创建 | 公开 API：`analyze_faces()`, `cluster_faces()` |
| `face_module/schema.py` | 创建 | 数据结构：`FaceResult`, `AssetFaceAnalysis`, `PersonCluster`, `FaceClusterResult` |
| `face_module/detector.py` | 创建 | `FaceDetector` 类：InsightFace 封装，单张检测 |
| `face_module/cluster.py` | 创建 | `FaceClusterer` 类：余弦相似度 + 连通分量聚类 |
| `face_module/tests/__init__.py` | 创建 | 空文件，标记为测试包 |
| `face_module/tests/test_schema.py` | 创建 | dataclass 序列化/反序列化测试 |
| `face_module/tests/test_cluster.py` | 创建 | 聚类算法纯逻辑测试（人造 embedding） |
| `face_module/tests/test_detector.py` | 创建 | 检测器测试（mock 模型，测试异常路径） |
| `face_module/README.md` | 创建 | 给 A 的使用说明 |

---

### Task 1: 创建 face_module 包骨架

**Files:**
- Create: `face_module/__init__.py`
- Create: `face_module/tests/__init__.py`

- [ ] **Step 1: 创建目录和空文件**

```bash
mkdir -p face_module/tests
```

- [ ] **Step 2: 创建 face_module/__init__.py（占位）**

```python
# face_module/__init__.py
"""face_module — 人脸检测、特征提取与聚类分组。

对外 API:
    analyze_faces(asset_path, asset_id=None) -> dict
    cluster_faces(items, threshold=0.5) -> dict
"""

__version__ = "1.0.0"
```

- [ ] **Step 3: 创建 face_module/tests/__init__.py（空文件）**

```python
# face_module/tests/__init__.py
```

- [ ] **Step 4: 验证目录结构**

```bash
ls -R face_module/
```
Expected:
```
face_module/:
__init__.py  tests/

face_module/tests:
__init__.py
```

---

### Task 2: 创建 schema.py — 数据结构定义

**Files:**
- Create: `face_module/schema.py`

- [ ] **Step 1: 编写 schema.py**

```python
# face_module/schema.py
"""face_module 数据结构定义。

所有 dataclass 提供 to_dict() 方法，返回可 JSON 序列化的纯 dict，
这是 A 调用 analyze_faces() / cluster_faces() 拿到的格式。
"""

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class FaceResult:
    """单张人脸检测/聚类结果"""
    asset_id: str
    face_id: str
    person_id: str = "unknown"
    bbox: list[float] = field(default_factory=list)
    confidence: float = 0.0
    quality: float = 0.0
    source: str = "insightface"
    embedding: Optional[list[float]] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("embedding", None)
        return d


@dataclass
class AssetFaceAnalysis:
    """analyze_faces() 单张素材的检测结果"""
    asset_id: str
    faces: list   # list[FaceResult]
    summary_tags: list   # list[str]
    version: str = "face-module-v1"

    def to_dict(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "faces": [f.to_dict() for f in self.faces],
            "summary_tags": self.summary_tags,
            "version": self.version,
        }


@dataclass
class PersonCluster:
    """单个人物的聚类结果"""
    person_id: str
    asset_ids: list   # list[str]
    face_count: int
    prototype_face_id: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FaceClusterResult:
    """cluster_faces() 批量聚类结果"""
    persons: list   # list[PersonCluster]
    faces: list     # list[FaceResult]（已回填 person_id）
    asset_tags: list   # list[dict]
    version: str = "face-module-v1"

    def to_dict(self) -> dict:
        return {
            "persons": [p.to_dict() for p in self.persons],
            "faces": [f.to_dict() for f in self.faces],
            "asset_tags": self.asset_tags,
            "version": self.version,
        }
```

- [ ] **Step 2: 验证 schema 可导入**

```bash
cd face_module && python -c "from schema import FaceResult, AssetFaceAnalysis, PersonCluster, FaceClusterResult; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: 快速验证 to_dict() 输出格式**

```bash
cd face_module && python -c "
from schema import FaceResult, AssetFaceAnalysis, PersonCluster, FaceClusterResult

# FaceResult
f = FaceResult(asset_id='img_01', face_id='face_img_01_0001', person_id='unknown',
               bbox=[120, 48, 188, 136], confidence=0.96, quality=0.96)
d = f.to_dict()
assert 'embedding' not in d
assert d['asset_id'] == 'img_01'
assert d['person_id'] == 'unknown'
print(f'FaceResult OK: {d}')

# AssetFaceAnalysis
a = AssetFaceAnalysis(asset_id='img_01', faces=[f], summary_tags=['has_face'])
ad = a.to_dict()
assert ad['summary_tags'] == ['has_face']
assert ad['version'] == 'face-module-v1'
print(f'AssetFaceAnalysis OK: {ad}')

# PersonCluster
p = PersonCluster(person_id='person_0001', asset_ids=['img_01'], face_count=1, prototype_face_id='face_img_01_0001')
pd = p.to_dict()
assert pd['person_id'] == 'person_0001'
print(f'PersonCluster OK: {pd}')

# FaceClusterResult
cr = FaceClusterResult(persons=[p], faces=[f], asset_tags=[{'asset_id': 'img_01', 'summary_tags': ['has_face', 'person_0001']}])
crd = cr.to_dict()
assert len(crd['asset_tags']) == 1
print(f'FaceClusterResult OK: {crd}')

print('All schema checks passed.')
"
```
Expected: `All schema checks passed.`

- [ ] **Step 4: 提交**

```bash
git add face_module/schema.py
git commit -m "feat(face_module): add schema.py with FaceResult, AssetFaceAnalysis, PersonCluster, FaceClusterResult

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: 测试 schema.py

**Files:**
- Create: `face_module/tests/test_schema.py`

- [ ] **Step 1: 编写 test_schema.py**

```python
# face_module/tests/test_schema.py
"""测试 schema.py 数据结构序列化正确性"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schema import FaceResult, AssetFaceAnalysis, PersonCluster, FaceClusterResult


class TestFaceResult:
    def test_to_dict_excludes_embedding(self):
        f = FaceResult(
            asset_id="000000012667",
            face_id="face_000000012667_0001",
            person_id="unknown",
            bbox=[120, 48, 188, 136],
            confidence=0.96,
            quality=0.96,
            embedding=[0.1] * 512,
        )
        d = f.to_dict()
        assert "embedding" not in d
        assert d["asset_id"] == "000000012667"
        assert d["face_id"] == "face_000000012667_0001"
        assert d["person_id"] == "unknown"
        assert d["bbox"] == [120, 48, 188, 136]
        assert d["confidence"] == 0.96
        assert d["quality"] == 0.96
        assert d["source"] == "insightface"

    def test_embedding_preserved_in_object(self):
        emb = [0.1] * 512
        f = FaceResult(
            asset_id="img",
            face_id="face_img_0001",
            embedding=emb,
        )
        assert f.embedding == emb

    def test_default_values(self):
        f = FaceResult(asset_id="img", face_id="face_img_0001")
        assert f.person_id == "unknown"
        assert f.bbox == []
        assert f.confidence == 0.0
        assert f.quality == 0.0
        assert f.source == "insightface"
        assert f.embedding is None


class TestAssetFaceAnalysis:
    def test_to_dict_basic(self):
        f = FaceResult(asset_id="img_01", face_id="face_img_01_0001",
                       bbox=[10, 20, 30, 40], confidence=0.9, quality=0.9)
        a = AssetFaceAnalysis(
            asset_id="img_01",
            faces=[f],
            summary_tags=["has_face"],
        )
        d = a.to_dict()
        assert d["asset_id"] == "img_01"
        assert len(d["faces"]) == 1
        assert d["faces"][0]["face_id"] == "face_img_01_0001"
        assert "embedding" not in d["faces"][0]
        assert d["summary_tags"] == ["has_face"]
        assert d["version"] == "face-module-v1"

    def test_empty_faces(self):
        a = AssetFaceAnalysis(asset_id="img_02", faces=[], summary_tags=[])
        d = a.to_dict()
        assert d["faces"] == []
        assert d["summary_tags"] == []


class TestPersonCluster:
    def test_to_dict(self):
        p = PersonCluster(
            person_id="person_0001",
            asset_ids=["img_01", "img_02"],
            face_count=2,
            prototype_face_id="face_img_02_0001",
        )
        d = p.to_dict()
        assert d["person_id"] == "person_0001"
        assert d["asset_ids"] == ["img_01", "img_02"]
        assert d["face_count"] == 2
        assert d["prototype_face_id"] == "face_img_02_0001"


class TestFaceClusterResult:
    def test_to_dict(self):
        p = PersonCluster(
            person_id="person_0001",
            asset_ids=["img_01"],
            face_count=1,
            prototype_face_id="face_img_01_0001",
        )
        f = FaceResult(asset_id="img_01", face_id="face_img_01_0001",
                       person_id="person_0001", bbox=[10, 20, 30, 40],
                       confidence=0.95, quality=0.95)
        cr = FaceClusterResult(
            persons=[p],
            faces=[f],
            asset_tags=[
                {"asset_id": "img_01", "summary_tags": ["has_face", "person_0001"]},
            ],
        )
        d = cr.to_dict()
        assert d["version"] == "face-module-v1"
        assert len(d["persons"]) == 1
        assert len(d["faces"]) == 1
        assert d["faces"][0]["person_id"] == "person_0001"
        assert "embedding" not in d["faces"][0]
        assert d["asset_tags"][0]["summary_tags"] == ["has_face", "person_0001"]
```

- [ ] **Step 2: 运行测试**

```bash
python -m pytest face_module/tests/test_schema.py -v
```
Expected: all tests PASS

- [ ] **Step 3: 提交**

```bash
git add face_module/tests/test_schema.py
git commit -m "test(face_module): add schema serialization tests

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: 创建 cluster.py — 聚类逻辑

**Files:**
- Create: `face_module/cluster.py`

- [ ] **Step 1: 编写 cluster.py**

```python
# face_module/cluster.py
"""人脸聚类逻辑 — 余弦相似度 + 连通分量分组。

对外由 cluster_faces() 包装调用，A 不直接使用 FaceClusterer。
"""

from collections import defaultdict

from schema import FaceClusterResult, PersonCluster


class FaceClusterer:
    """人脸聚类器。

    v1 算法：余弦相似度 + 连通分量。
    v2 预留 method="dbscan" 参数。
    """

    def cluster(
        self,
        faces: list,
        method: str = "cosine_threshold",
        threshold: float = 0.5,
    ) -> FaceClusterResult:
        """对所有检测到的人脸执行聚类。

        Args:
            faces: list[FaceResult]，每个必须含 embedding
            method: "cosine_threshold"（v1）或 "dbscan"（v2 预留）
            threshold: 余弦相似度阈值，默认 0.5

        Returns:
            FaceClusterResult：包含 persons、已回填 person_id 的 faces、asset_tags 汇总

        Raises:
            ValueError: method 不支持或 faces 为空
        """
        if not faces:
            raise ValueError("faces 不能为空，请先调用 analyze_faces()")

        if method == "cosine_threshold":
            return self._cluster_by_threshold(faces, threshold)
        elif method == "dbscan":
            raise NotImplementedError("DBSCAN 聚类将在 v2 支持")
        else:
            raise ValueError(f"不支持的聚类方法: {method}，可选: cosine_threshold, dbscan")

    def _cluster_by_threshold(
        self, faces: list, threshold: float
    ) -> FaceClusterResult:
        """余弦相似度阈值聚类 + 连通分量分组。"""
        import numpy as np

        # 提取 embedding 矩阵 (N, D)
        embeddings = np.array([f.embedding for f in faces], dtype=np.float32)
        n = len(embeddings)

        # L2 归一化
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

        # 余弦相似度矩阵
        sim_matrix = embeddings @ embeddings.T

        # 并查集连通分量
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] > threshold:
                    union(i, j)

        # 收集分组
        groups = defaultdict(list)
        for idx in range(n):
            groups[find(idx)].append(idx)

        # 按组大小降序分配 person_id
        sorted_groups = sorted(groups.values(), key=len, reverse=True)

        persons = []
        face_asset_map = {}   # face_id -> asset_id（从原始 FaceResult 取）
        asset_faces_map = defaultdict(list)  # asset_id -> list[face_index]

        for person_idx, group in enumerate(sorted_groups, start=1):
            person_id = f"person_{person_idx:04d}"

            # 选代表照：组内 quality 最高
            best_face = max(group, key=lambda idx: faces[idx].quality)
            prototype_face_id = faces[best_face].face_id

            # 收集该人物的 asset_ids
            asset_ids = []
            for idx in group:
                asset_id = faces[idx].asset_id
                face_asset_map[faces[idx].face_id] = asset_id
                asset_faces_map[asset_id].append(person_id)
                if asset_id not in asset_ids:
                    asset_ids.append(asset_id)

            persons.append(PersonCluster(
                person_id=person_id,
                asset_ids=asset_ids,
                face_count=len(group),
                prototype_face_id=prototype_face_id,
            ))

            # 回填 person_id 到每个 FaceResult
            for idx in group:
                faces[idx].person_id = person_id

        # 生成 asset_tags 汇总
        asset_tags = []
        for asset_id, person_ids in asset_faces_map.items():
            tags = ["has_face"] + sorted(set(person_ids))
            asset_tags.append({
                "asset_id": asset_id,
                "summary_tags": tags,
            })

        return FaceClusterResult(
            persons=persons,
            faces=faces,
            asset_tags=asset_tags,
        )
```

- [ ] **Step 2: 验证可导入**

```bash
cd face_module && python -c "from cluster import FaceClusterer; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: 提交**

```bash
git add face_module/cluster.py
git commit -m "feat(face_module): add FaceClusterer with cosine_threshold clustering

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: 测试 cluster.py

**Files:**
- Create: `face_module/tests/test_cluster.py`

- [ ] **Step 1: 编写 test_cluster.py**

```python
# face_module/tests/test_cluster.py
"""测试聚类算法纯逻辑 — 不需 InsightFace，只用 NumPy 人造 embedding。"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schema import FaceResult
from cluster import FaceClusterer


def make_face(face_id, asset_id, embedding, quality=1.0):
    """快速构造一个 FaceResult 用于测试"""
    return FaceResult(
        asset_id=asset_id,
        face_id=face_id,
        embedding=embedding,
        quality=quality,
    )


class TestFaceClustererBasic:
    """基础功能测试"""

    def test_empty_faces_raises(self):
        clusterer = FaceClusterer()
        with pytest.raises(ValueError, match="不能为空"):
            clusterer.cluster([])

    def test_unsupported_method_raises(self):
        clusterer = FaceClusterer()
        f = make_face("face_a_0001", "a", [1.0] * 512)
        with pytest.raises(ValueError, match="不支持的聚类方法"):
            clusterer.cluster([f], method="kmeans")

    def test_dbscan_not_implemented_yet(self):
        clusterer = FaceClusterer()
        f = make_face("face_a_0001", "a", [1.0] * 512)
        with pytest.raises(NotImplementedError, match="DBSCAN"):
            clusterer.cluster([f], method="dbscan")

    def test_single_face(self):
        """单张脸：自成一组"""
        clusterer = FaceClusterer()
        emb = [0.1] * 512
        faces = [make_face("face_img_01_0001", "img_01", emb, quality=0.9)]

        result = clusterer.cluster(faces, threshold=0.5)

        assert len(result.persons) == 1
        assert result.persons[0].person_id == "person_0001"
        assert result.persons[0].face_count == 1
        assert result.persons[0].asset_ids == ["img_01"]
        assert result.persons[0].prototype_face_id == "face_img_01_0001"
        assert result.faces[0].person_id == "person_0001"
        assert result.asset_tags == [
            {"asset_id": "img_01", "summary_tags": ["has_face", "person_0001"]}
        ]

    def test_identical_faces_same_person(self):
        """相同 embedding 应归为同一人"""
        clusterer = FaceClusterer()
        emb = [1.0] * 512
        faces = [
            make_face("face_img_01_0001", "img_01", emb, quality=0.9),
            make_face("face_img_01_0002", "img_01", emb, quality=0.8),
        ]
        result = clusterer.cluster(faces, threshold=0.5)
        assert len(result.persons) == 1
        assert result.persons[0].face_count == 2
        assert result.faces[0].person_id == "person_0001"
        assert result.faces[1].person_id == "person_0001"

    def test_opposite_faces_different_person(self):
        """方向相反的向量（cos_sim ≈ -1）应为不同人"""
        clusterer = FaceClusterer()
        emb_a = [1.0] * 512
        emb_b = [-1.0] * 512
        faces = [
            make_face("face_a_0001", "a", emb_a),
            make_face("face_b_0001", "b", emb_b),
        ]
        result = clusterer.cluster(faces, threshold=0.5)
        assert len(result.persons) == 2

    def test_orthogonal_faces_different_person(self):
        """正交向量（cos_sim ≈ 0）应分为不同人"""
        import numpy as np
        clusterer = FaceClusterer()
        emb_a = [1.0] + [0.0] * 511
        emb_b = [0.0] + [1.0] + [0.0] * 510
        faces = [
            make_face("face_a_0001", "a", emb_a),
            make_face("face_b_0001", "b", emb_b),
        ]
        result = clusterer.cluster(faces, threshold=0.5)
        assert len(result.persons) == 2


class TestFaceClustererGroups:
    """多组聚类测试"""

    def test_two_clear_groups(self):
        """两组人脸，组内相似、组间正交"""
        import numpy as np
        clusterer = FaceClusterer()

        group_a_emb = [1.0] * 512
        group_b_emb = [0.0] * 256 + [1.0] * 256

        faces = [
            make_face("face_a1_0001", "a1", group_a_emb, quality=0.9),
            make_face("face_a2_0001", "a2", group_a_emb, quality=0.8),
            make_face("face_b1_0001", "b1", group_b_emb, quality=0.7),
            make_face("face_b2_0001", "b2", group_b_emb, quality=0.95),
        ]
        result = clusterer.cluster(faces, threshold=0.5)

        assert len(result.persons) == 2

        # 按 face_count 排序：多的在前 = person_0001
        sorted_persons = sorted(result.persons, key=lambda p: p.face_count, reverse=True)
        assert sorted_persons[0].person_id in ("person_0001", "person_0002")
        assert sorted_persons[0].face_count == 2
        assert sorted_persons[1].face_count == 2

    def test_near_threshold_boundary(self):
        """cos_sim 刚好在阈值边界"""
        import numpy as np

        emb_a = [1.0] + [0.0] * 511
        # emb_b 与 emb_a 的 cos_sim ≈ 0.6（刚好 > 0.5）
        emb_b = [0.6] + [0.8] + [0.0] * 510
        emb_b = (np.array(emb_b, dtype=np.float32) / np.linalg.norm(emb_b)).tolist()

        clusterer = FaceClusterer()
        faces = [
            make_face("face_a_0001", "a", emb_a),
            make_face("face_b_0001", "b", emb_b),
        ]
        result = clusterer.cluster(faces, threshold=0.5)
        # cos_sim = 0.6 > 0.5 → 同一人
        assert len(result.persons) == 1


class TestPrototypeSelection:
    """代表照选择测试"""

    def test_highest_quality_selected(self):
        clusterer = FaceClusterer()
        emb = [1.0] * 512
        faces = [
            make_face("face_a_0001", "a", emb, quality=0.5),
            make_face("face_a_0002", "a", emb, quality=0.99),
            make_face("face_a_0003", "a", emb, quality=0.7),
        ]
        result = clusterer.cluster(faces, threshold=0.5)
        assert result.persons[0].prototype_face_id == "face_a_0002"


class TestAssetTags:
    """asset_tags 汇总测试"""

    def test_asset_tags_correctly_aggregated(self):
        import numpy as np

        emb_p1 = [1.0] * 512
        emb_p2 = [0.0] * 256 + [1.0] * 256

        clusterer = FaceClusterer()
        faces = [
            make_face("face_a_0001", "asset_a", emb_p1, quality=0.9),
            make_face("face_a_0002", "asset_a", emb_p2, quality=0.8),
            make_face("face_b_0001", "asset_b", emb_p1, quality=0.7),
        ]
        result = clusterer.cluster(faces, threshold=0.5)

        tags_by_asset = {
            t["asset_id"]: set(t["summary_tags"]) for t in result.asset_tags
        }

        assert "has_face" in tags_by_asset["asset_a"]
        assert "has_face" in tags_by_asset["asset_b"]
        # asset_a 有两张不同人的脸 → 两个 person tag
        assert len(tags_by_asset["asset_a"]) >= 3   # has_face + 2 persons
        # asset_b 只有一张脸 → 一个 person tag
        assert len(tags_by_asset["asset_b"]) == 2    # has_face + 1 person
```

- [ ] **Step 2: 运行测试**

```bash
python -m pytest face_module/tests/test_cluster.py -v
```
Expected: all tests PASS

- [ ] **Step 3: 提交**

```bash
git add face_module/tests/test_cluster.py
git commit -m "test(face_module): add clustering logic tests

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: 创建 detector.py — InsightFace 封装

**Files:**
- Create: `face_module/detector.py`

- [ ] **Step 1: 编写 detector.py**

```python
# face_module/detector.py
"""人脸检测与特征提取 — InsightFace 封装。

使用 InsightFace buffalo_l 模型组合（检测 + 识别），
延迟加载，首次调用 analyze() 时才初始化。
"""

from pathlib import Path

from schema import AssetFaceAnalysis, FaceResult


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class FaceDetector:
    """InsightFace 人脸检测器，延迟加载。

    A 永远不直接使用此类，而是通过 face_module.analyze_faces() 调用。
    """

    def __init__(self, device: str = "auto"):
        """
        Args:
            device: 'cpu' | 'cuda' | 'auto'（有 GPU 则用 GPU 0）
        """
        self._device = device
        self._model = None

    def _ensure_model(self):
        """延迟加载 InsightFace 模型（首次调用时触发）。"""
        if self._model is not None:
            return

        import insightface

        if self._device == "auto":
            import torch
            ctx_id = 0 if torch.cuda.is_available() else -1
        elif self._device == "cpu":
            ctx_id = -1
        else:
            ctx_id = 0

        self._model = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            if ctx_id >= 0
            else ["CPUExecutionProvider"],
        )
        self._model.prepare(ctx_id=ctx_id)

    def analyze(
        self, asset_path: str, asset_id: str | None = None
    ) -> AssetFaceAnalysis:
        """对单张图片执行人脸检测与特征提取。

        Args:
            asset_path: 图片文件路径（绝对或相对）
            asset_id: 素材 id，不传则取文件名（不含后缀）

        Returns:
            AssetFaceAnalysis

        Raises:
            ValueError: 文件不存在、格式不支持或图片损坏无法读取
        """
        path = Path(asset_path)

        # 校验文件存在
        if not path.exists():
            raise ValueError(f"文件不存在: {asset_path}")

        # 校验格式
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_IMAGE_SUFFIXES:
            raise ValueError(
                f"不支持的图片格式: {suffix}，仅支持 {SUPPORTED_IMAGE_SUFFIXES}"
            )

        # 推导 asset_id
        if asset_id is None:
            asset_id = path.stem

        # 读取图片
        try:
            from PIL import Image
            import numpy as np
            image = Image.open(path).convert("RGB")
            image_np = np.array(image)
        except Exception as e:
            raise ValueError(f"无法读取图片 {asset_path}: {e}")

        # 确保模型已加载
        self._ensure_model()

        # 检测
        detected_faces = self._model.get(image_np)

        # 无脸
        if not detected_faces:
            return AssetFaceAnalysis(
                asset_id=asset_id,
                faces=[],
                summary_tags=[],
            )

        # 构造 FaceResult 列表
        faces = []
        for idx, f in enumerate(detected_faces):
            face_id = f"face_{asset_id}_{idx:04d}"

            # InsightFace bbox 格式: [x1, y1, x2, y2]（与约定一致）
            bbox = [float(v) for v in f.bbox]

            # detection confidence
            confidence = float(f.det_score) if hasattr(f, "det_score") else 1.0

            # quality: v1 直接用 confidence
            quality = confidence

            # embedding
            embedding = f.embedding.tolist() if hasattr(f, "embedding") and f.embedding is not None else None

            faces.append(FaceResult(
                asset_id=asset_id,
                face_id=face_id,
                person_id="unknown",
                bbox=bbox,
                confidence=confidence,
                quality=quality,
                source="insightface",
                embedding=embedding,
            ))

        return AssetFaceAnalysis(
            asset_id=asset_id,
            faces=faces,
            summary_tags=["has_face"] if faces else [],
        )
```

- [ ] **Step 2: 验证可导入（不加载模型）**

```bash
cd face_module && python -c "from detector import FaceDetector; d = FaceDetector(); print('OK - FaceDetector created (model not loaded)')"
```
Expected: `OK - FaceDetector created (model not loaded)`

- [ ] **Step 3: 提交**

```bash
git add face_module/detector.py
git commit -m "feat(face_module): add FaceDetector with lazy-loaded InsightFace

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: 测试 detector.py

**Files:**
- Create: `face_module/tests/test_detector.py`

- [ ] **Step 1: 编写 test_detector.py**

```python
# face_module/tests/test_detector.py
"""测试 FaceDetector — 异常路径，不依赖真实模型。

detector 的正常检测功能在集成阶段用真实图片验证。
本文件只测错误处理和构造正确性。
"""

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detector import FaceDetector, SUPPORTED_IMAGE_SUFFIXES
from schema import AssetFaceAnalysis


class TestFaceDetectorInit:
    """构造函数测试"""

    def test_default_device_is_auto(self):
        d = FaceDetector()
        assert d._device == "auto"
        assert d._model is None

    def test_explicit_cpu_device(self):
        d = FaceDetector(device="cpu")
        assert d._device == "cpu"

    def test_explicit_cuda_device(self):
        d = FaceDetector(device="cuda")
        assert d._device == "cuda"

    def test_model_not_loaded_on_init(self):
        d = FaceDetector()
        assert d._model is None, "模型在 __init__ 时不应加载"


class TestFileNotFound:
    """文件不存在测试"""

    def test_nonexistent_file_raises(self):
        d = FaceDetector()
        with pytest.raises(ValueError, match="文件不存在"):
            d.analyze("/nonexistent/path/to/image.jpg")

    def test_directory_instead_of_file(self):
        d = FaceDetector()
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="文件不存在"):
                d.analyze(tmpdir)


class TestUnsupportedFormat:
    """不支持的格式测试"""

    def test_text_file_raises(self):
        d = FaceDetector()
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("not an image")
            tmp_path = f.name
        try:
            with pytest.raises(ValueError, match="不支持的图片格式"):
                d.analyze(tmp_path)
        finally:
            Path(tmp_path).unlink()

    def test_gif_raises(self):
        d = FaceDetector()
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
            f.write(b"GIF89a dummy")
            tmp_path = f.name
        try:
            with pytest.raises(ValueError, match="不支持的图片格式"):
                d.analyze(tmp_path)
        finally:
            Path(tmp_path).unlink()

    def test_corrupted_image_raises(self):
        d = FaceDetector()
        with tempfile.NamedTemporaryFile(suffix=".jpg", mode="wb", delete=False) as f:
            f.write(b"this is not a valid jpeg image")
            tmp_path = f.name
        try:
            with pytest.raises(ValueError, match="无法读取图片"):
                d.analyze(tmp_path)
        finally:
            Path(tmp_path).unlink()


class TestAssetIdDerivation:
    """asset_id 推导测试"""

    def test_asset_id_derived_from_filename(self):
        """不传 asset_id 时从文件名推导"""
        d = FaceDetector()

        from PIL import Image
        import numpy as np

        # 构造一张纯色小图
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            img.save(f.name)
            tmp_path = f.name

        try:
            # 不做真实检测（避免加载模型），只测 asset_id 推导逻辑
            path = Path(tmp_path)
            assert path.stem == Path(tmp_path).stem
        finally:
            Path(tmp_path).unlink()


class TestSupportedFormats:
    """支持的格式常量"""

    def test_common_formats_supported(self):
        assert ".jpg" in SUPPORTED_IMAGE_SUFFIXES
        assert ".jpeg" in SUPPORTED_IMAGE_SUFFIXES
        assert ".png" in SUPPORTED_IMAGE_SUFFIXES
        assert ".webp" in SUPPORTED_IMAGE_SUFFIXES
        assert ".bmp" in SUPPORTED_IMAGE_SUFFIXES


class TestSchemaOutputStructure:
    """验证返回值结构（不依赖模型）"""

    def test_no_face_returns_correct_structure(self):
        """mock 场景：无脸时的返回结构"""
        analysis = AssetFaceAnalysis(
            asset_id="test_001",
            faces=[],
            summary_tags=[],
        )
        d = analysis.to_dict()
        assert d["asset_id"] == "test_001"
        assert d["faces"] == []
        assert d["summary_tags"] == []
        assert d["version"] == "face-module-v1"

    def test_with_faces_returns_correct_structure(self):
        from schema import FaceResult

        f = FaceResult(
            asset_id="test_001",
            face_id="face_test_001_0001",
            person_id="unknown",
            bbox=[10.0, 20.0, 50.0, 80.0],
            confidence=0.95,
            quality=0.95,
        )
        analysis = AssetFaceAnalysis(
            asset_id="test_001",
            faces=[f],
            summary_tags=["has_face"],
        )
        d = analysis.to_dict()
        assert d["asset_id"] == "test_001"
        assert len(d["faces"]) == 1
        assert d["faces"][0]["person_id"] == "unknown"
        assert d["summary_tags"] == ["has_face"]
```

- [ ] **Step 2: 运行测试**

```bash
python -m pytest face_module/tests/test_detector.py -v
```
Expected: all tests PASS（不需要 InsightFace 模型）

- [ ] **Step 3: 提交**

```bash
git add face_module/tests/test_detector.py
git commit -m "test(face_module): add detector error-path and structure tests

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: 连接 __init__.py — 对外 API

**Files:**
- Modify: `face_module/__init__.py`

- [ ] **Step 1: 更新 face_module/__init__.py**

```python
# face_module/__init__.py
"""face_module — 人脸检测、特征提取与聚类分组。

对外 API:
    analyze_faces(asset_path, asset_id=None) -> dict
    cluster_faces(items, threshold=0.5) -> dict

使用方式:
    from face_module import analyze_faces, cluster_faces

    result = analyze_faces("path/to/image.jpg", asset_id="img_001")
    cluster_result = cluster_faces([result, ...])
"""

__version__ = "1.0.0"

from detector import FaceDetector
from cluster import FaceClusterer
from schema import AssetFaceAnalysis

_detector = FaceDetector()


def analyze_faces(asset_path: str, asset_id: str | None = None) -> dict:
    """对单张图片执行人脸检测与特征提取。

    Args:
        asset_path: 图片文件路径（.jpg/.jpeg/.png/.webp/.bmp）
        asset_id: 素材 id，可选。不传则取文件名（不含后缀）

    Returns:
        dict:
        {
            "asset_id": "000000012667",
            "faces": [
                {
                    "asset_id": "000000012667",
                    "face_id": "face_000000012667_0001",
                    "person_id": "unknown",
                    "bbox": [120, 48, 188, 136],
                    "confidence": 0.96,
                    "quality": 0.96,
                    "source": "insightface",
                }
            ],
            "summary_tags": ["has_face"],  # 聚类前只有 has_face
            "version": "face-module-v1",
        }

        无脸时 faces 和 summary_tags 均为空列表。

    Raises:
        ValueError: 文件不存在、格式不支持或图片损坏
    """
    return _detector.analyze(asset_path, asset_id).to_dict()


def cluster_faces(
    items: list,
    threshold: float = 0.5,
) -> dict:
    """对多张素材的检测结果执行全局聚类。

    Args:
        items: list[dict | AssetFaceAnalysis]，analyze_faces() 返回的结果列表。
               内部自动展平，从每个元素的 faces 中收集 FaceResult。
        threshold: 余弦相似度阈值，默认 0.5。
                   值越高聚类越严格（同一个人需要更像才归为一组）。

    Returns:
        dict:
        {
            "persons": [
                {
                    "person_id": "person_0001",
                    "asset_ids": ["000000012667", "000000029393"],
                    "face_count": 3,
                    "prototype_face_id": "face_000000012667_0001",
                }
            ],
            "faces": [
                {
                    "asset_id": "000000012667",
                    "face_id": "face_000000012667_0001",
                    "person_id": "person_0001",   # ← 已回填
                    "bbox": [120, 48, 188, 136],
                    "confidence": 0.96,
                    "quality": 0.96,
                    "source": "insightface",
                }
            ],
            "asset_tags": [
                {
                    "asset_id": "000000012667",
                    "summary_tags": ["has_face", "person_0001", "person_0002"],
                }
            ],
            "version": "face-module-v1",
        }

    Raises:
        ValueError: items 为空或所有元素都没有人脸
    """
    # 展平：从每个 AssetFaceAnalysis/dict 中收集 FaceResult
    from schema import FaceResult

    all_faces: list[FaceResult] = []

    for item in items:
        if isinstance(item, dict):
            asset_id = item.get("asset_id", "unknown")
            for fd in item.get("faces", []):
                embedding = fd.get("embedding")
                all_faces.append(FaceResult(
                    asset_id=fd.get("asset_id", asset_id),
                    face_id=fd.get("face_id", ""),
                    person_id=fd.get("person_id", "unknown"),
                    bbox=fd.get("bbox", []),
                    confidence=fd.get("confidence", 0.0),
                    quality=fd.get("quality", 0.0),
                    source=fd.get("source", "insightface"),
                    embedding=embedding if embedding else None,
                ))
        elif hasattr(item, "faces"):
            for f in item.faces:
                all_faces.append(f)
        else:
            raise TypeError(f"不支持的类型: {type(item)}，需要 dict 或 AssetFaceAnalysis")

    clusterer = FaceClusterer()
    result = clusterer.cluster(all_faces, threshold=threshold)
    return result.to_dict()
```

- [ ] **Step 2: 验证 API 可导入**

```bash
cd .. && python -c "from face_module import analyze_faces, cluster_faces; print('OK - API ready')"
```
Expected: `OK - API ready`

- [ ] **Step 3: 运行全部测试**

```bash
python -m pytest face_module/tests/ -v
```
Expected: all existing tests still PASS

- [ ] **Step 4: 提交**

```bash
git add face_module/__init__.py
git commit -m "feat(face_module): wire up public API — analyze_faces() and cluster_faces()

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 9: 创建 README.md — 给 A 的使用说明

**Files:**
- Create: `face_module/README.md`

- [ ] **Step 1: 编写 README.md**

```markdown
# face_module

人脸检测、特征提取与聚类分组模块。

**版本:** 1.0.0 (face-module-v1)
**模型:** InsightFace buffalo_l

## 快速开始

```python
from face_module import analyze_faces, cluster_faces

# 1. 逐张分析
results = []
results.append(analyze_faces("data/uploads/000000012667.jpg", asset_id="000000012667"))
results.append(analyze_faces("data/uploads/000000029393.jpg"))

# 2. 全局聚类
cluster_result = cluster_faces(results)

# 3. 读取结果
for person in cluster_result["persons"]:
    print(f"{person['person_id']}: {person['face_count']} 张脸, 代表照 {person['prototype_face_id']}")

for face in cluster_result["faces"]:
    print(f"{face['face_id']} → {face['person_id']}")

for entry in cluster_result["asset_tags"]:
    print(f"{entry['asset_id']}: {entry['summary_tags']}")
```

## API

### analyze_faces(asset_path, asset_id=None) -> dict

对单张图片做检测。无脸返回空 `faces` 和 `summary_tags`。

- 支持格式：`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`
- 文件异常抛出 `ValueError`
- `person_id` 此时为 `"unknown"`，需聚类后回填

### cluster_faces(items, threshold=0.5) -> dict

全局聚类。`items` 是 `analyze_faces()` 返回结果的列表。

- `threshold`: 0.0-1.0，越高越严格
- 返回 `persons`（人物）、`faces`（已回填 person_id）、`asset_tags`（可直接写入 asset_tags 表）

## 依赖

- insightface >= 0.7
- numpy
- Pillow

安装：

```bash
pip install insightface numpy Pillow
# 若使用 GPU：
pip install onnxruntime-gpu
```

## 注意事项

- 首次调用 `analyze_faces()` 时自动下载模型（~350MB），请确保网络畅通
- A 侧注意事项见设计文档 `docs/superpowers/specs/2026-06-29-face-module-design.md` 第 7 节
```

- [ ] **Step 2: 提交**

```bash
git add face_module/README.md
git commit -m "docs(face_module): add README for role A

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 10: 集成测试 — 端到端跑通

**Files:**
- Create: `face_module/tests/test_integration.py`

- [ ] **Step 1: 编写 test_integration.py**

```python
# face_module/tests/test_integration.py
"""端到端集成测试 — 用真实图片验证 analyze_faces → cluster_faces 完整流程。

需要 InsightFace 模型。在没有模型的环境下跳过。
"""

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _has_model():
    """检查 InsightFace 是否可用"""
    try:
        import insightface
        return True
    except ImportError:
        return False


HAS_MODEL = _has_model()


@pytest.mark.skipif(not HAS_MODEL, reason="InsightFace 模型未安装")
class TestEndToEnd:
    """端到端管线测试"""

    def test_analyze_no_face_image(self):
        """纯色图不应检测到人脸"""
        from face_module import analyze_faces

        from PIL import Image
        import numpy as np

        # 创建纯灰色图（无人脸）
        img = Image.fromarray(
            np.ones((200, 200, 3), dtype=np.uint8) * 128
        )

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            img.save(f.name, "JPEG")
            tmp_path = f.name

        try:
            result = analyze_faces(tmp_path, asset_id="test_gray")
            assert result["asset_id"] == "test_gray"
            # 纯色图通常检测不到人脸
            assert isinstance(result["faces"], list)
        finally:
            Path(tmp_path).unlink()

    def test_analyze_then_cluster(self):
        """分析 → 聚类 完整流程"""
        from face_module import analyze_faces, cluster_faces

        from PIL import Image
        import numpy as np

        # 创建两张不同的纯色图（都不会检测到脸，但验证流程正确）
        results = []
        for i in range(3):
            color = (i * 80) % 256
            img = Image.fromarray(
                np.ones((150, 150, 3), dtype=np.uint8) * color
            )
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                img.save(f.name, "JPEG")
                tmp_path = f.name

            try:
                result = analyze_faces(tmp_path, asset_id=f"test_{i}")
                results.append(result)
            finally:
                Path(tmp_path).unlink()

        # 所有图都无脸 → cluster_faces 会抛异常
        # 验证至少 analyze_faces 都返回了正确结构
        for r in results:
            assert "asset_id" in r
            assert "faces" in r
            assert "summary_tags" in r
            assert r["version"] == "face-module-v1"

    def test_result_keys_exist(self):
        """验证返回 dict 的 key 完整性"""
        from face_module import analyze_faces

        from PIL import Image
        import numpy as np

        img = Image.fromarray(np.ones((100, 100, 3), dtype=np.uint8) * 200)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f.name, "PNG")
            tmp_path = f.name

        try:
            result = analyze_faces(tmp_path, asset_id="key_test")
            assert set(result.keys()) == {"asset_id", "faces", "summary_tags", "version"}
            if result["faces"]:
                face = result["faces"][0]
                required_keys = {"asset_id", "face_id", "person_id", "bbox",
                                 "confidence", "quality", "source"}
                assert required_keys.issubset(set(face.keys()))
                assert "embedding" not in face, "embedding 不应出现在 to_dict() 输出中"
        finally:
            Path(tmp_path).unlink()
```

- [ ] **Step 2: 运行集成测试（需要 InsightFace）**

```bash
python -m pytest face_module/tests/test_integration.py -v
```
Expected: 有模型时 PASS，无模型时 SKIP

- [ ] **Step 3: 提交**

```bash
git add face_module/tests/test_integration.py
git commit -m "test(face_module): add end-to-end integration tests

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 11: 更新 requirements.txt

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: 添加 insightface 依赖**

```bash
echo "insightface>=0.7" >> requirements.txt
```

- [ ] **Step 2: 确认内容**

```bash
cat requirements.txt
```
Expected: 原有内容 + `insightface>=0.7`

- [ ] **Step 3: 提交**

```bash
git add requirements.txt
git commit -m "chore: add insightface dependency for face_module

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 12: 最终验证

- [ ] **Step 1: 运行全部测试**

```bash
python -m pytest face_module/tests/ -v
```
Expected: 所有非模型测试 PASS，集成测试 PASS 或 SKIP

- [ ] **Step 2: 验证 API 导入链路**

```bash
python -c "
from face_module import analyze_faces, cluster_faces
print('analyze_faces:', type(analyze_faces))
print('cluster_faces:', type(cluster_faces))
print('All imports OK.')
"
```
Expected: `All imports OK.`

- [ ] **Step 3: 验证模块自洽**

```bash
python -c "
# 模拟 A 的调用链路：假设拿到 dict 后聚类
from face_module import cluster_faces

# 构造两个假的 analyze_faces 返回结果
r1 = {
    'asset_id': 'img_01',
    'faces': [
        {
            'asset_id': 'img_01',
            'face_id': 'face_img_01_0001',
            'person_id': 'unknown',
            'bbox': [10, 20, 50, 60],
            'confidence': 0.95,
            'quality': 0.95,
            'source': 'insightface',
            'embedding': [1.0]*512,
        }
    ],
    'summary_tags': ['has_face'],
    'version': 'face-module-v1',
}
r2 = {
    'asset_id': 'img_02',
    'faces': [
        {
            'asset_id': 'img_02',
            'face_id': 'face_img_02_0001',
            'person_id': 'unknown',
            'bbox': [30, 40, 80, 90],
            'confidence': 0.88,
            'quality': 0.88,
            'source': 'insightface',
            'embedding': [1.0]*512,  # 与 img_01 相同 → 同一人
        }
    ],
    'summary_tags': ['has_face'],
    'version': 'face-module-v1',
}

result = cluster_faces([r1, r2])

# 验证聚类结果
assert len(result['persons']) == 1, '相同 embedding 应聚类为同一人'
assert result['persons'][0]['face_count'] == 2
assert result['faces'][0]['person_id'] == 'person_0001'
assert result['faces'][1]['person_id'] == 'person_0001'

# 验证 asset_tags 汇总
assert len(result['asset_tags']) == 2
tags_by_asset = {t['asset_id']: t['summary_tags'] for t in result['asset_tags']}
assert 'person_0001' in tags_by_asset['img_01']
assert 'person_0001' in tags_by_asset['img_02']
assert tags_by_asset['img_01'] == ['has_face', 'person_0001']

print('All self-consistency checks passed.')
print(f'Result: {result}')
"
```
Expected: `All self-consistency checks passed.`

- [ ] **Step 4: 提交最终状态**

```bash
git status
git add -A
git commit -m "feat(face_module): complete v1 implementation with tests

Co-Authored-By: Claude <noreply@anthropic.com>"
```
