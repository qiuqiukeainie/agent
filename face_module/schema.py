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
