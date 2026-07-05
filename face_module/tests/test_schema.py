# face_module/tests/test_schema.py
"""测试 schema.py 数据结构序列化正确性"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from face_module.schema import FaceResult, AssetFaceAnalysis, PersonCluster, FaceClusterResult


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
        f = FaceResult(
            asset_id="img_01", face_id="face_img_01_0001",
            bbox=[10, 20, 30, 40], confidence=0.9, quality=0.9,
        )
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
        f = FaceResult(
            asset_id="img_01", face_id="face_img_01_0001",
            person_id="person_0001", bbox=[10, 20, 30, 40],
            confidence=0.95, quality=0.95,
        )
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
