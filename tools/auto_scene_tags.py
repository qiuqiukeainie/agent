from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vector_engine import VectorEngine, cosine_similarity


TAG_PROMPTS = {
    "cat": ["a cat", "a pet cat"],
    "dog": ["a dog", "a pet dog"],
    "bird": ["a bird", "an animal bird"],
    "animal": ["an animal", "wildlife or pet animal"],
    "person": ["a person", "people"],
    "walking": ["a person walking", "people walking"],
    "dance": ["people dancing", "a dance performance"],
    "sports": ["sports activity", "people playing sports"],
    "basketball": ["basketball game", "a person playing basketball"],
    "football": ["football soccer game", "people playing football"],
    "city": ["city street", "urban scene"],
    "street": ["street scene", "road in a city"],
    "traffic": ["traffic on a road", "vehicles in traffic"],
    "train": ["train", "railway train"],
    "bus": ["bus", "public bus"],
    "bicycle": ["bicycle", "person riding bicycle"],
    "car": ["car", "vehicle"],
    "waterfall": ["waterfall", "falling water in nature"],
    "ocean": ["ocean waves", "sea waves"],
    "mountain": ["mountain landscape", "mountains"],
    "nature": ["nature landscape", "outdoor natural scene"],
    "grass": ["grass field", "green grass"],
    "sky": ["blue sky", "sky outdoors"],
    "food": ["food", "meal"],
    "cooking": ["cooking food", "food preparation"],
    "coffee": ["coffee cup", "coffee drink"],
    "fruit": ["fruit", "fresh fruit"],
    "office": ["office room", "workplace"],
    "classroom": ["classroom", "students in classroom"],
    "library": ["library", "bookshelves in a library"],
    "museum": ["museum interior", "exhibition room"],
    "stage": ["stage performance", "performance on stage"],
    "night": ["night scene", "dark night"],
    "snow": ["snow", "snowy scene"],
    "building": ["building architecture", "architecture"],
    "flower": ["flower", "flowers"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate lightweight CLIP zero-shot scene tags for assets.")
    parser.add_argument("--data-dir", default="data", help="Project data directory.")
    parser.add_argument("--kind", choices=["all", "image", "video"], default="all", help="Assets to tag.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum assets to process; 0 means all.")
    parser.add_argument("--top-k", type=int, default=3, help="Maximum tags per asset.")
    parser.add_argument("--threshold", type=float, default=0.28, help="Minimum CLIP score for a tag.")
    parser.add_argument("--margin", type=float, default=0.025, help="Keep tags close enough to the best score.")
    parser.add_argument("--keep-existing-source", action="store_true", help="Do not clear previous zero-shot tags first.")
    args = parser.parse_args()

    engine = VectorEngine(args.data_dir)
    if not args.keep_existing_source:
        deleted = engine.metadata.delete_asset_tags_by_source("clip_zero_shot_scene")
        print(f"cleared_previous_zero_shot_tags={deleted}")
    tag_vectors = build_tag_vectors(engine)
    processed = 0
    written = 0

    for asset in engine.assets:
        if args.kind != "all" and asset.kind != args.kind:
            continue
        if args.limit and processed >= args.limit:
            break
        processed += 1

        vector = np.array(asset.vector, dtype=np.float32)
        scored = []
        for tag, prompt_vectors in tag_vectors.items():
            score = max(cosine_similarity(vector, prompt_vector) for prompt_vector in prompt_vectors)
            scored.append((score, tag))
        scored.sort(reverse=True)

        best_score = scored[0][0] if scored else 0.0
        selected = [
            tag
            for score, tag in scored[: args.top_k]
            if score >= args.threshold and score >= best_score - args.margin
        ]
        if asset.kind == "video":
            selected.append("video")
        if selected:
            confidence = min(max(scored[0][0], 0.0), 1.0)
            engine.set_asset_tags(asset.id, selected, source="clip_zero_shot_scene", confidence=confidence)
            written += 1

    print(f"processed={processed} tagged_assets={written}")
    print(engine.library_stats())


def build_tag_vectors(engine: VectorEngine) -> dict[str, list[np.ndarray]]:
    tag_vectors = {}
    for tag, prompts in TAG_PROMPTS.items():
        tag_vectors[tag] = [engine.encoder.encode_text(prompt) for prompt in prompts]
    return tag_vectors


if __name__ == "__main__":
    main()
