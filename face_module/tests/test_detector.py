# face_module/tests/test_detector.py
"""测试 FaceDetector — 异常路径，不依赖真实模型。

detector 的正常检测功能在集成阶段用真实图片验证。
本文件只测错误处理和构造正确性。
"""

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from face_module.detector import FaceDetector, SUPPORTED_IMAGE_SUFFIXES
from face_module.schema import AssetFaceAnalysis, FaceResult


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
            with pytest.raises(ValueError, match="路径不是文件"):
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
