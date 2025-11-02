from __future__ import annotations

import argparse
import glob
import os
from typing import List

import cv2

from .api import WB86Pipeline


def _iter_images(path: str) -> List[str]:
    if os.path.isdir(path):
        files = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            files.extend(glob.glob(os.path.join(path, ext)))
        return sorted(files)
    return [path]


def run_images(pipeline: WB86Pipeline, input_path: str, out_dir: str, visualize: bool):
    for p in _iter_images(input_path):
        try:
            pipeline.infer_image(p, out_dir=out_dir, visualize=visualize)
        except Exception as e:
            print(f"[WARN] Failed {p}: {e}")


def run_video(pipeline: WB86Pipeline, video_path: str, out_dir: str, visualize: bool):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        base = f"frame_{idx:06d}"
        img_path = os.path.join(out_dir, base + ".jpg")
        cv2.imwrite(img_path, frame)
        pipeline.infer_image(img_path, out_dir=out_dir, visualize=visualize)
        idx += 1
    cap.release()


def run_webcam(pipeline: WB86Pipeline, cam_index: int, out_dir: str, visualize: bool):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(cam_index)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        base = f"webcam_{idx:06d}"
        img_path = os.path.join(out_dir, base + ".jpg")
        cv2.imwrite(img_path, frame)
        pipeline.infer_image(img_path, out_dir=out_dir, visualize=visualize)
        idx += 1
    cap.release()


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("wb86")
    ap.add_argument("--input", type=str, help="image file or directory")
    ap.add_argument("--video", type=str, default=None, help="video file path")
    ap.add_argument("--webcam", type=int, default=None, help="webcam index (e.g., 0)")
    ap.add_argument("--config", type=str, default=None, help="path to YAML config")
    ap.add_argument("--out", type=str, default="out", help="output directory")
    ap.add_argument("--visualize", action="store_true", help="save visualization overlays")
    return ap


def main():
    ap = build_argparser()
    args = ap.parse_args()
    pipe = WB86Pipeline(config_path=args.config)
    if args.video:
        run_video(pipe, args.video, args.out, args.visualize)
    elif args.webcam is not None:
        run_webcam(pipe, args.webcam, args.out, args.visualize)
    elif args.input:
        run_images(pipe, args.input, args.out, args.visualize)
    else:
        print("Provide --input, --video, or --webcam")


if __name__ == "__main__":
    main()

