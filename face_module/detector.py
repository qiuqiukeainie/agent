# face_module/detector.py
"""人脸检测与特征提取 — InsightFace 封装。

使用 InsightFace buffalo_l 模型组合（检测 + 识别），
延迟加载，首次调用 analyze() 时才初始化。
"""

from pathlib import Path
from typing import Optional

from .schema import AssetFaceAnalysis, FaceResult


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
            try:
                import torch
                ctx_id = 0 if torch.cuda.is_available() else -1
            except ImportError:
                ctx_id = -1  # torch 未安装 → CPU
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
        self, asset_path: str, asset_id: Optional[str] = None
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

        # 校验文件存在且是文件（非目录）
        if not path.exists():
            raise ValueError(f"文件不存在: {asset_path}")
        if not path.is_file():
            raise ValueError(f"路径不是文件: {asset_path}")

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
            image = Image.open(path).convert("RGB")
            image_np = __import__("numpy").array(image)
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
            embedding = (
                f.embedding.tolist()
                if hasattr(f, "embedding") and f.embedding is not None
                else None
            )

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
