# face_module/tests/test_cluster.py
"""测试聚类算法纯逻辑 — 不需 InsightFace，只用 NumPy 人造 embedding。"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from face_module.schema import FaceResult
from face_module.cluster import FaceClusterer


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
        clusterer = FaceClusterer()

        group_a_emb = [1.0] + [0.0] * 511   # 第一个维度
        group_b_emb = [0.0] + [1.0] + [0.0] * 510  # 第二个维度，正交

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
        assert sorted_persons[0].face_count == 2
        assert sorted_persons[1].face_count == 2

    def test_near_threshold_boundary(self):
        """cos_sim 刚好在阈值边界"""
        import numpy as np

        emb_a = [1.0] + [0.0] * 511
        # emb_b 与 emb_a 的 cos_sim ≈ 0.6（刚好 > 0.5）
        emb_b_raw = [0.6] + [0.8] + [0.0] * 510
        emb_b = (np.array(emb_b_raw, dtype=np.float32) / np.linalg.norm(emb_b_raw)).tolist()

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
        emb_p1 = [1.0] + [0.0] * 511   # 第一个维度
        emb_p2 = [0.0] + [1.0] + [0.0] * 510  # 第二个维度，正交

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
