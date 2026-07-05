# face_module/tests/test_integration.py
"""端到端集成测试 — 用真实图片验证 analyze_faces -> cluster_faces 完整流程。

需要 InsightFace 模型。在没有模型的环境下跳过。
"""

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def _has_model():
    """检查 InsightFace 是否可用"""
    try:
        import insightface  # noqa: F401
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

        img = Image.fromarray(
            np.ones((200, 200, 3), dtype=np.uint8) * 128
        )

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            img.save(f.name, "JPEG")
            tmp_path = f.name

        try:
            result = analyze_faces(tmp_path, asset_id="test_gray")
            assert result["asset_id"] == "test_gray"
            assert isinstance(result["faces"], list)
        finally:
            Path(tmp_path).unlink()

    def test_analyze_then_cluster(self):
        """分析 -> 聚类 完整流程"""
        from face_module import analyze_faces, cluster_faces

        from PIL import Image
        import numpy as np

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


class TestSelfConsistency:
    """不依赖模型的完整链路自洽测试"""

    def test_analyze_cluster_roundtrip(self):
        """模拟 A 的完整调用链路"""
        from face_module import cluster_faces

        r1 = {
            "asset_id": "img_01",
            "faces": [
                {
                    "asset_id": "img_01",
                    "face_id": "face_img_01_0001",
                    "person_id": "unknown",
                    "bbox": [10, 20, 50, 60],
                    "confidence": 0.95,
                    "quality": 0.95,
                    "source": "insightface",
                    "embedding": [1.0] * 512,
                }
            ],
            "summary_tags": ["has_face"],
            "version": "face-module-v1",
        }
        r2 = {
            "asset_id": "img_02",
            "faces": [
                {
                    "asset_id": "img_02",
                    "face_id": "face_img_02_0001",
                    "person_id": "unknown",
                    "bbox": [30, 40, 80, 90],
                    "confidence": 0.88,
                    "quality": 0.88,
                    "source": "insightface",
                    "embedding": [1.0] * 512,  # 与 img_01 相同 -> 同一人
                }
            ],
            "summary_tags": ["has_face"],
            "version": "face-module-v1",
        }

        result = cluster_faces([r1, r2])

        assert len(result["persons"]) == 1, "相同 embedding 应聚类为同一人"
        assert result["persons"][0]["face_count"] == 2
        assert result["faces"][0]["person_id"] == "person_0001"
        assert result["faces"][1]["person_id"] == "person_0001"

        assert len(result["asset_tags"]) == 2
        tags_by_asset = {t["asset_id"]: t["summary_tags"] for t in result["asset_tags"]}
        assert "person_0001" in tags_by_asset["img_01"]
        assert "person_0001" in tags_by_asset["img_02"]

    def test_no_face_items_raises(self):
        """所有图都无脸时 cluster_faces 应抛异常"""
        from face_module import cluster_faces

        r1 = {
            "asset_id": "img_01",
            "faces": [],
            "summary_tags": [],
            "version": "face-module-v1",
        }
        with pytest.raises(ValueError, match="不能为空"):
            cluster_faces([r1])
