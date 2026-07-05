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
    print(f"{face['face_id']} -> {face['person_id']}")

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
- A 侧注意事项见设计文档
