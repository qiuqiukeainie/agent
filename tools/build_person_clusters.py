from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vector_engine import SUPPORTED_IMAGE_TYPES, SUPPORTED_VIDEO_TYPES, VectorEngine, extract_video_frames, normalize


def main() -> None:
    parser = argparse.ArgumentParser(description="Build minimal person clusters for the personal library.")
    parser.add_argument("--data-dir", default="data", help="Project data directory.")
    parser.add_argument("--library", default="personal", choices=["personal", "public"], help="Library to process.")
    parser.add_argument("--threshold", type=float, default=None, help="Cosine threshold. Defaults: face_module=0.5, haar=0.78.")
    parser.add_argument("--min-quality", type=float, default=0.18, help="Skip weak face detections below this quality.")
    parser.add_argument("--max-video-frames", type=int, default=6, help="Video frames to inspect.")
    parser.add_argument(
        "--min-cluster-faces",
        type=int,
        default=2,
        help="Drop unnamed clusters with fewer faces. Keeps named clusters from previous runs.",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "face_module", "haar"],
        help="auto prefers B group's face_module and falls back to the legacy Haar pipeline.",
    )
    args = parser.parse_args()

    engine = VectorEngine(args.data_dir)
    ignored_asset_ids = set(engine.load_person_ignored_assets())
    assets = [asset for asset in engine.assets if asset.library == args.library]

    if args.backend in {"auto", "face_module"}:
        try:
            result = build_with_face_module(engine, assets, ignored_asset_ids, args)
            print(
                f"backend=face_module library={args.library} assets={len(assets)} "
                f"persons={result['persons']} faces={result['faces']}"
            )
            print(engine.library_stats())
            return
        except Exception as exc:
            if args.backend == "face_module":
                raise
            print(f"face_module unavailable, fallback to haar backend: {exc}")

    result = build_with_haar(engine, assets, ignored_asset_ids, args)
    print(
        f"backend=haar library={args.library} assets={len(assets)} "
        f"persons={result['persons']} faces={result['faces']}"
    )
    print(engine.library_stats())


def build_with_face_module(engine: VectorEngine, assets: list, ignored_asset_ids: set[str], args: argparse.Namespace) -> dict:
    from face_module import cluster_faces
    from face_module.detector import FaceDetector

    detector = FaceDetector()
    analysis_items = []
    for asset in assets:
        if asset.id in ignored_asset_ids or asset.kind != "image":
            continue
        path = resolve_asset_path(asset.path)
        if not path.exists() or path.suffix.lower() not in SUPPORTED_IMAGE_TYPES:
            continue
        analysis = detector.analyze(str(path), asset_id=asset.id)
        if analysis.faces:
            analysis_items.append(analysis)

    engine.metadata.delete_asset_tags_by_source("person_cluster")
    if not analysis_items:
        engine.metadata.replace_person_clusters([], [], library=args.library)
        return {"persons": 0, "faces": 0}

    threshold = 0.5 if args.threshold is None else args.threshold
    cluster_result = cluster_faces(analysis_items, threshold=threshold)
    face_rows = normalize_face_module_faces(cluster_result.get("faces", []), args.min_quality)
    asset_persons = build_asset_person_map(face_rows)
    apply_person_merge_rules(face_rows, asset_persons, engine.load_person_merge_rules())
    persons = build_person_rows(face_rows, confidence=0.92)
    persons, face_rows, asset_persons = filter_unstable_person_clusters(
        persons,
        face_rows,
        asset_persons,
        engine.metadata.named_person_ids(args.library),
        args.min_cluster_faces,
    )
    write_person_tags(engine, asset_persons, confidence=0.92)
    engine.metadata.replace_person_clusters(persons, face_rows, library=args.library)
    return {"persons": len(persons), "faces": len(face_rows)}


def build_with_haar(engine: VectorEngine, assets: list, ignored_asset_ids: set[str], args: argparse.Namespace) -> dict:
    detector = HaarFaceDetector()
    engine.metadata.delete_asset_tags_by_source("person_cluster")
    face_rows = []
    clusters: list[dict] = []
    asset_persons: dict[str, set[str]] = {}

    for asset in assets:
        if asset.id in ignored_asset_ids:
            continue
        path = resolve_asset_path(asset.path)
        if not path.exists():
            continue

        images = load_asset_images(path, asset.kind, args.max_video_frames)
        asset_face_index = 0
        person_tags = []
        for image in images:
            for bbox, confidence, quality in detector.detect(image):
                if quality < args.min_quality:
                    continue
                crop = crop_face(image, bbox)
                embedding = engine.encoder.encode_image(crop)
                threshold = 0.78 if args.threshold is None else args.threshold
                person_id = assign_cluster(clusters, embedding, threshold)
                asset_face_index += 1
                face_id = f"face_{asset.id}_{asset_face_index:04d}"
                person_tags.append(person_id)
                face_rows.append(
                    {
                        "face_id": face_id,
                        "asset_id": asset.id,
                        "person_id": person_id,
                        "bbox": bbox,
                        "confidence": confidence,
                        "quality": quality,
                        "source": "opencv_haar_clip",
                    }
                )

        if person_tags:
            asset_persons.setdefault(asset.id, set()).update(person_tags)

    apply_person_merge_rules(face_rows, asset_persons, engine.load_person_merge_rules())

    persons, person_id_map = renumber_persons(face_rows, confidence=0.86)
    asset_persons = {
        asset_id: {person_id_map[item] for item in temporary_person_ids if item in person_id_map}
        for asset_id, temporary_person_ids in asset_persons.items()
    }
    persons, face_rows, asset_persons = filter_unstable_person_clusters(
        persons,
        face_rows,
        asset_persons,
        engine.metadata.named_person_ids(args.library),
        args.min_cluster_faces,
    )
    write_person_tags(engine, asset_persons, confidence=0.86)
    engine.metadata.replace_person_clusters(persons, face_rows, library=args.library)
    return {"persons": len(persons), "faces": len(face_rows)}


def resolve_asset_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def normalize_face_module_faces(faces: list[dict], min_quality: float) -> list[dict]:
    rows = []
    for face in faces:
        quality = float(face.get("quality") or 0.0)
        if quality < min_quality:
            continue
        bbox = [int(round(float(value))) for value in face.get("bbox", [])[:4]]
        if len(bbox) != 4:
            continue
        rows.append(
            {
                "face_id": str(face.get("face_id") or f"face_{face.get('asset_id', 'unknown')}_{len(rows) + 1:04d}"),
                "asset_id": str(face.get("asset_id") or ""),
                "person_id": str(face.get("person_id") or "unknown"),
                "bbox": bbox,
                "confidence": float(face.get("confidence") or 0.0),
                "quality": quality,
                "source": str(face.get("source") or "insightface"),
            }
        )
    return rows


def build_asset_person_map(face_rows: list[dict]) -> dict[str, set[str]]:
    asset_persons: dict[str, set[str]] = {}
    for face in face_rows:
        asset_persons.setdefault(face["asset_id"], set()).add(face["person_id"])
    return asset_persons


def build_person_rows(face_rows: list[dict], confidence: float) -> list[dict]:
    persons = []
    for person_id in sorted({face["person_id"] for face in face_rows}):
        person_faces = [face for face in face_rows if face["person_id"] == person_id]
        prototype = max(person_faces, key=lambda item: item.get("quality", 0.0)) if person_faces else None
        persons.append(
            {
                "person_id": person_id,
                "alias": None,
                "display_name": None,
                "face_count": len(person_faces),
                "prototype_face_id": prototype["face_id"] if prototype else None,
                "confidence": confidence,
            }
        )
    return persons


def filter_unstable_person_clusters(
    persons: list[dict],
    face_rows: list[dict],
    asset_persons: dict[str, set[str]],
    named_person_ids: set[str],
    min_cluster_faces: int,
) -> tuple[list[dict], list[dict], dict[str, set[str]]]:
    if min_cluster_faces <= 1:
        return persons, face_rows, asset_persons
    stable_person_ids = {
        person["person_id"]
        for person in persons
        if int(person.get("face_count", 0)) >= min_cluster_faces or person["person_id"] in named_person_ids
    }
    persons = [person for person in persons if person["person_id"] in stable_person_ids]
    face_rows = [face for face in face_rows if face["person_id"] in stable_person_ids]
    asset_persons = {
        asset_id: {person_id for person_id in person_ids if person_id in stable_person_ids}
        for asset_id, person_ids in asset_persons.items()
    }
    asset_persons = {asset_id: person_ids for asset_id, person_ids in asset_persons.items() if person_ids}
    return persons, face_rows, asset_persons


def renumber_persons(face_rows: list[dict], confidence: float) -> tuple[list[dict], dict[str, str]]:
    person_id_map = {}
    active_person_ids = sorted({face["person_id"] for face in face_rows})
    for index, old_id in enumerate(active_person_ids, start=1):
        person_id_map[old_id] = f"person_{index:04d}"
    for face in face_rows:
        face["person_id"] = person_id_map.get(face["person_id"], face["person_id"])
    return build_person_rows(face_rows, confidence), person_id_map


def write_person_tags(engine: VectorEngine, asset_persons: dict[str, set[str]], confidence: float) -> None:
    for asset_id, person_ids in asset_persons.items():
        engine.set_asset_tags(
            asset_id,
            sorted({"has_face", *person_ids}) if person_ids else [],
            source="person_cluster",
            confidence=confidence,
            replace_source=True,
        )


class HaarFaceDetector:
    def __init__(self) -> None:
        import cv2

        self.cv2 = cv2
        front_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        profile_path = Path(cv2.data.haarcascades) / "haarcascade_profileface.xml"
        self.front_detector = cv2.CascadeClassifier(str(front_path))
        self.profile_detector = cv2.CascadeClassifier(str(profile_path))
        if self.front_detector.empty():
            raise RuntimeError(f"failed to load Haar cascade: {front_path}")

    def detect(self, image: Image.Image) -> list[tuple[list[int], float, float]]:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        original_width, original_height = rgb.size
        resize_scale = min(1.0, 1400.0 / max(original_width, original_height))
        small = rgb.resize((int(original_width * resize_scale), int(original_height * resize_scale))) if resize_scale < 1 else rgb
        arr = np.asarray(small)
        gray = self.cv2.cvtColor(arr, self.cv2.COLOR_RGB2GRAY)
        candidates = []
        candidates.extend(self.detect_with(self.front_detector, gray, flipped=False))
        candidates.extend(self.detect_with(self.front_detector, gray, flipped=True))
        if not self.profile_detector.empty():
            candidates.extend(self.detect_with(self.profile_detector, gray, flipped=False))
            candidates.extend(self.detect_with(self.profile_detector, gray, flipped=True))
        faces = non_max_suppression(candidates, iou_threshold=0.35, containment_threshold=0.62)
        rows = []
        small_width, small_height = small.size
        for x, y, w, h in faces:
            x1s = max(int(x), 0)
            y1s = max(int(y), 0)
            x2s = min(int(x + w), small_width)
            y2s = min(int(y + h), small_height)
            x1 = int(x1s / resize_scale)
            y1 = int(y1s / resize_scale)
            x2 = int(x2s / resize_scale)
            y2 = int(y2s / resize_scale)
            face_area = max((x2 - x1) * (y2 - y1), 1)
            image_area = max(original_width * original_height, 1)
            quality = min(max(face_area / image_area * 18.0, 0.05), 1.0)
            rows.append(([x1, y1, x2, y2], 0.86, quality))
        return rows

    def detect_with(self, detector, gray: np.ndarray, flipped: bool) -> list[list[int]]:
        source = self.cv2.flip(gray, 1) if flipped else gray
        rows = detector.detectMultiScale(
            source,
            scaleFactor=1.06,
            minNeighbors=4,
            minSize=(32, 32),
        )
        width = gray.shape[1]
        result = []
        for x, y, w, h in rows:
            if flipped:
                x = width - x - w
            result.append([int(x), int(y), int(w), int(h)])
        return result


def load_asset_images(path: Path, kind: str, max_video_frames: int) -> list[Image.Image]:
    suffix = path.suffix.lower()
    if kind == "image" and suffix in SUPPORTED_IMAGE_TYPES:
        with Image.open(path) as image:
            return [ImageOps.exif_transpose(image).convert("RGB")]
    if kind == "video" and suffix in SUPPORTED_VIDEO_TYPES:
        return extract_video_frames(path, max_video_frames)
    return []


def crop_face(image: Image.Image, bbox: list[int]) -> Image.Image:
    x1, y1, x2, y2 = bbox
    width, height = image.size
    pad_x = int((x2 - x1) * 0.18)
    pad_y = int((y2 - y1) * 0.22)
    crop_box = (
        max(x1 - pad_x, 0),
        max(y1 - pad_y, 0),
        min(x2 + pad_x, width),
        min(y2 + pad_y, height),
    )
    return image.crop(crop_box).resize((224, 224))


def non_max_suppression(
    boxes: list[list[int]],
    iou_threshold: float,
    containment_threshold: float,
) -> list[list[int]]:
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda item: item[2] * item[3], reverse=True)
    kept: list[list[int]] = []
    for box in boxes:
        if all(
            iou_xywh(box, existing) < iou_threshold
            and containment_xywh(box, existing) < containment_threshold
            for existing in kept
        ):
            kept.append(box)
    return kept


def containment_xywh(left: list[int], right: list[int]) -> float:
    lx1, ly1, lw, lh = left
    rx1, ry1, rw, rh = right
    lx2, ly2 = lx1 + lw, ly1 + lh
    rx2, ry2 = rx1 + rw, ry1 + rh
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    inter = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
    smaller = max(min(lw * lh, rw * rh), 1)
    return inter / smaller


def iou_xywh(left: list[int], right: list[int]) -> float:
    lx1, ly1, lw, lh = left
    rx1, ry1, rw, rh = right
    lx2, ly2 = lx1 + lw, ly1 + lh
    rx2, ry2 = rx1 + rw, ry1 + rh
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    inter = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
    union = lw * lh + rw * rh - inter
    return inter / union if union else 0.0


def assign_cluster(clusters: list[dict], embedding: np.ndarray, threshold: float) -> str:
    embedding = normalize(embedding.astype(np.float32))
    if not clusters:
        clusters.append({"person_id": "tmp_0001", "centroid": embedding, "count": 1})
        return "tmp_0001"

    scored = [
        (float(np.dot(embedding, cluster["centroid"])), cluster)
        for cluster in clusters
    ]
    score, cluster = max(scored, key=lambda item: item[0])
    if score >= threshold:
        count = cluster["count"] + 1
        cluster["centroid"] = normalize((cluster["centroid"] * cluster["count"] + embedding) / count)
        cluster["count"] = count
        return cluster["person_id"]

    person_id = f"tmp_{len(clusters) + 1:04d}"
    clusters.append({"person_id": person_id, "centroid": embedding, "count": 1})
    return person_id


def apply_person_merge_rules(face_rows: list[dict], asset_persons: dict[str, set[str]], rules: list[dict]) -> None:
    parent: dict[str, str] = {}

    def find(person_id: str) -> str:
        parent.setdefault(person_id, person_id)
        while parent[person_id] != person_id:
            parent[person_id] = parent[parent[person_id]]
            person_id = parent[person_id]
        return person_id

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for face in face_rows:
        find(face["person_id"])

    for rule in rules:
        asset_ids = rule.get("asset_ids", [])
        person_ids = sorted(
            {
                person_id
                for asset_id in asset_ids
                for person_id in asset_persons.get(str(asset_id), set())
            }
        )
        if len(person_ids) < 2:
            continue
        anchor = person_ids[0]
        for person_id in person_ids[1:]:
            union(anchor, person_id)

    for face in face_rows:
        face["person_id"] = find(face["person_id"])

    for asset_id, person_ids in list(asset_persons.items()):
        asset_persons[asset_id] = {find(person_id) for person_id in person_ids}


if __name__ == "__main__":
    main()
