#!/usr/bin/env python3
import argparse
from pathlib import Path
from PIL import Image

# Soft import Real-ESRGAN
_HAS_REALESRGAN = False
try:
    from realesrgan import RealESRGAN
    _HAS_REALESRGAN = True
except Exception:
    _HAS_REALESRGAN = False

def upscale_bicubic(img: Image.Image, scale: int) -> Image.Image:
    return img.resize((img.width * scale, img.height * scale), resample=Image.BICUBIC)

def infer_realesrgan(img: Image.Image, scale: int, device: str = "cuda"):
    if not _HAS_REALESRGAN:
        return None
    try:
        import torch
        dev = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
        model_scale = 4
        model = RealESRGAN(dev, model_scale)
        model.load_weights(RealESRGAN.weights_path("RealESRGAN_x4plus"))
        out = model.predict(img)
        if scale != model_scale:
            out = out.resize((img.width * scale, img.height * scale), Image.LANCZOS)
        return out
    except Exception as e:
        print(f"[WARN] Real-ESRGAN failed: {e}")
        return None

def save_image(img: Image.Image, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() in [".jpg", ".jpeg"]:
        img.save(out_path, quality=95, subsampling=0)
    else:
        img.save(out_path)

def process_image(path: Path, out_dir: Path, scale: int, prefer: str, device: str):
    img = Image.open(path).convert("RGB")
    out_img = None
    if prefer == "realesrgan":
        out_img = infer_realesrgan(img, scale, device=device)
        if out_img is None:
            print(f"[INFO] Fallback to bicubic for {path.name}")
            out_img = upscale_bicubic(img, scale)
    else:
        out_img = upscale_bicubic(img, scale)
    out_path = out_dir / path.name
    save_image(out_img, out_path)
    print(f"[OK] {path} -> {out_path}")

def main():
    ap = argparse.ArgumentParser(description="High-quality SR: prefer Real-ESRGAN, fallback Bicubic")
    ap.add_argument("--input", "-i", required=True, help="Image or directory")
    ap.add_argument("--output", "-o", required=True, help="Output directory")
    ap.add_argument("--scale", "-s", type=int, default=4, choices=[2,3,4], help="Upscale factor")
    ap.add_argument("--prefer", choices=["realesrgan", "bicubic"], default="realesrgan")
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    exts = {".png",".jpg",".jpeg",".bmp",".webp"}

    if in_path.is_dir():
        files = [p for p in in_path.iterdir() if p.suffix.lower() in exts]
        if not files:
            print("[WARN] No images found.")
        for p in files:
            process_image(p, out_dir, args.scale, args.prefer, args.device)
    else:
        if in_path.suffix.lower() not in exts:
            raise SystemExit("[ERROR] Input must be an image file")
        process_image(in_path, out_dir, args.scale, args.prefer, args.device)

if __name__ == "__main__":
    main()
