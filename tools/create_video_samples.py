from __future__ import annotations

from pathlib import Path

import av
import numpy as np


def main() -> None:
    out_dir = Path("data/video_samples")
    out_dir.mkdir(parents=True, exist_ok=True)
    make_video(out_dir / "red_square_motion.mp4", theme="red")
    make_video(out_dir / "blue_sky_grass_motion.mp4", theme="landscape")
    make_video(out_dir / "yellow_ball_motion.mp4", theme="ball")
    make_video(out_dir / "green_block_motion.mp4", theme="green")
    print(f"created samples in {out_dir}")


def make_video(path: Path, theme: str, width: int = 320, height: int = 240, fps: int = 12, seconds: int = 5) -> None:
    container = av.open(str(path), mode="w")
    stream = container.add_stream(preferred_codec(), rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {"preset": "veryfast", "crf": "23"}

    for index in range(fps * seconds):
        frame_array = render_frame(index, width, height, theme)
        frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()


def preferred_codec() -> str:
    for codec in ("libx264", "h264", "mpeg4"):
        try:
            av.codec.Codec(codec, "w")
            return codec
        except Exception:
            continue
    return "mpeg4"


def render_frame(index: int, width: int, height: int, theme: str) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    if theme == "landscape":
        frame[: height // 2, :, :] = np.array([80, 160, 235], dtype=np.uint8)
        frame[height // 2 :, :, :] = np.array([70, 170, 80], dtype=np.uint8)
        x = 20 + (index * 4) % (width - 80)
        frame[height // 2 - 20 : height // 2 + 20, x : x + 60, :] = np.array([255, 255, 255], dtype=np.uint8)
        return frame

    frame[:, :, :] = np.array([30, 30, 35], dtype=np.uint8)
    if theme == "green":
        frame[:, :, :] = np.array([20, 55, 30], dtype=np.uint8)
        size = 64
        x = 10 + (index * 4) % (width - size - 20)
        y = height // 2 - size // 2
        frame[y : y + size, x : x + size, :] = np.array([50, 220, 90], dtype=np.uint8)
        return frame
    if theme == "ball":
        frame[:, :, :] = np.array([35, 45, 80], dtype=np.uint8)
        radius = 30
        cx = 40 + (index * 5) % (width - 80)
        cy = height // 2
        yy, xx = np.ogrid[:height, :width]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        frame[mask] = np.array([240, 210, 40], dtype=np.uint8)
        return frame
    size = 58
    x = 10 + (index * 5) % (width - size - 20)
    y = height // 2 - size // 2
    frame[y : y + size, x : x + size, :] = np.array([230, 50, 45], dtype=np.uint8)
    return frame


if __name__ == "__main__":
    main()
