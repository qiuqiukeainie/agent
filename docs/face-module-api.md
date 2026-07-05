# B 组人脸模块接口规范

本文档定义 B 组人脸识别/聚类模块对 A/D/E 暴露的 Python API，以及当前系统的接入方式。

## 1. 当前模块位置

```text
face_module/
  __init__.py
  detector.py
  cluster.py
  schema.py
```

当前版本：

```text
face-module-v1
InsightFace buffalo_l
```

## 2. 模块职责边界

B 负责：
- 检测图片中的人脸。
- 提取人脸 embedding。
- 将同一个人聚到同一个 `person_id`。
- 输出人脸框、人物组、素材标签。

B 不负责：
- 判断“小明是谁”。
- 维护用户给人物起的名字。
- 直接写 SQLite。
- 直接开 HTTP 服务。

A/D 负责调用 B 的 Python 模块，并把结果写入统一数据库。

## 3. B 暴露的核心 API

```python
from face_module import analyze_faces, cluster_faces

single = analyze_faces("data/uploads/example.jpg", asset_id="example")
result = cluster_faces([single], threshold=0.5)
```

### analyze_faces(asset_path, asset_id=None)

输入：
- `asset_path`：图片路径。
- `asset_id`：素材 ID，不传则默认使用文件名 stem。

输出：

```json
{
  "asset_id": "example",
  "faces": [
    {
      "asset_id": "example",
      "face_id": "face_example_0000",
      "person_id": "unknown",
      "bbox": [120, 48, 188, 136],
      "confidence": 0.96,
      "quality": 0.96,
      "source": "insightface"
    }
  ],
  "summary_tags": ["has_face"],
  "version": "face-module-v1"
}
```

说明：
- 对外 JSON 不返回 embedding，这是合理的，避免数据太大。
- 聚类时内部仍然需要 embedding。

### cluster_faces(items, threshold=0.5)

输入：
- `items`：`analyze_faces()` 的结果列表，或内部 `AssetFaceAnalysis` 对象列表。
- `threshold`：余弦相似度阈值，越高越严格。

输出：

```json
{
  "persons": [
    {
      "person_id": "person_0001",
      "asset_ids": ["example"],
      "face_count": 1,
      "prototype_face_id": "face_example_0000"
    }
  ],
  "faces": [
    {
      "asset_id": "example",
      "face_id": "face_example_0000",
      "person_id": "person_0001",
      "bbox": [120, 48, 188, 136],
      "confidence": 0.96,
      "quality": 0.96,
      "source": "insightface"
    }
  ],
  "asset_tags": [
    {
      "asset_id": "example",
      "summary_tags": ["has_face", "person_0001"]
    }
  ],
  "version": "face-module-v1"
}
```

## 4. 当前 A/D 接入方式

系统通过脚本统一调度：

```powershell
.\.venv\Scripts\python.exe tools\build_person_clusters.py --library personal --backend auto
```

参数：
- `--backend auto`：优先使用 B 的 `face_module`，不可用时回退旧的 Haar+CLIP 简易方案。
- `--backend face_module`：强制使用 B 模块，失败就报错。
- `--backend haar`：强制使用旧方案。
- `--threshold`：聚类阈值。默认 `face_module=0.5`，`haar=0.78`。
- `--min-quality`：过滤低质量人脸。

注意：
- B 的 `analyze_faces()` 对外 dict 不带 embedding；A/D 在批量聚类时会直接调用 `FaceDetector().analyze()` 获取内部对象，保证 `cluster_faces()` 能拿到 embedding。
- 这不改变 B 的公开 API，只是 A/D 的集成适配。

## 5. 写入数据库的最终结构

A/D 会把 B 的结果写入：

| 表 | 用途 |
| --- | --- |
| `persons` | 人物组，例如 `person_0001` |
| `asset_faces` | 每张素材里的人脸框与 person_id |
| `asset_tags` | `has_face`、`person_0001` 等标签 |

前端人物面板读取：

```http
GET /api/persons?library=personal
```

详情页读取单张素材的人脸命中：

```http
GET /api/asset?id=<asset_id>
```

## 6. 命名机制

初始状态只做聚类：

```json
{
  "person_id": "person_0001",
  "display_name": null
}
```

用户后续可以命名：

```json
{
  "person_id": "person_0001",
  "display_name": "小明"
}
```

命名只改变显示名，不改变底层 `person_id`。

## 7. 环境依赖

B 模块需要：

```text
insightface
onnxruntime
numpy
Pillow
```

CPU 环境安装：

```powershell
.\.venv\Scripts\python.exe -m pip install insightface onnxruntime
```

首次运行 InsightFace 会下载 `buffalo_l` 模型，体积较大，需要网络。
