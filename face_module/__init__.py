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

from typing import Optional

from .detector import FaceDetector
from .cluster import FaceClusterer
from .schema import FaceResult

_detector = FaceDetector()


def analyze_faces(asset_path: str, asset_id: Optional[str] = None) -> dict:
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
    all_faces: list = []

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
            raise TypeError(
                "不支持的类型: {}，需要 dict 或 AssetFaceAnalysis".format(type(item))
            )

    clusterer = FaceClusterer()
    result = clusterer.cluster(all_faces, threshold=threshold)
    return result.to_dict()
