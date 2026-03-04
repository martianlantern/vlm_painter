#!/usr/bin/env -S PYTHONUNBUFFERED=1 uv run --env-file .env --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "fire>=0.7.1",
#   "torch>=2.0.0",
#   "torchvision>=0.15.0",
#   "Pillow>=10.0.0",
#   "numpy",
#   "datasets>=3.0.0",
# ]
# ///
import json
import os
import random
import fire
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from stroke_renderer import render_strokes, render_to_image, load_meta_brushes

SYSTEM_PROMPT = """You are a painting AI. Given an image, generate brush strokes to recreate it.
Each stroke has 8 parameters: x, y, width, height, rotation, R, G, B
- x, y: center position (0.0 to 1.0)
- width, height: stroke size (0.01 to 0.5)
- rotation: angle (0.0 to 1.0, where 1.0 = 180 degrees)
- R, G, B: color (0.0 to 1.0)

Output exactly 16 strokes, one per line, as comma-separated values."""

USER_PROMPT = "Generate 16 strokes to paint this image:"


def random_stroke():
    x = random.uniform(0.05, 0.95)
    y = random.uniform(0.05, 0.95)
    w = random.uniform(0.02, 0.35)
    h = random.uniform(0.02, 0.35)
    rot = random.uniform(0.0, 1.0)
    r = random.uniform(0.0, 1.0)
    g = random.uniform(0.0, 1.0)
    b = random.uniform(0.0, 1.0)
    return [x, y, w, h, rot, r, g, b]


def strokes_to_text(strokes):
    lines = []
    for s in strokes:
        lines.append(",".join(f"{v:.2f}" for v in s))
    return "\n".join(lines)


def generate_random_painting(num_strokes=16, canvas_h=256, canvas_w=256, device="cuda"):
    strokes = [random_stroke() for _ in range(num_strokes)]
    params = torch.tensor(strokes, dtype=torch.float32).to(device)
    img = render_to_image(params, canvas_h, canvas_w, device)
    return strokes, img


def color_guided_strokes(target_img, num_strokes=16):
    arr = np.array(target_img).astype(np.float32) / 255.0
    h, w, c = arr.shape
    strokes = []
    for _ in range(num_strokes):
        px = random.randint(0, w - 1)
        py = random.randint(0, h - 1)
        patch_size = random.randint(8, 64)
        y1 = max(0, py - patch_size // 2)
        y2 = min(h, py + patch_size // 2)
        x1 = max(0, px - patch_size // 2)
        x2 = min(w, px + patch_size // 2)
        patch = arr[y1:y2, x1:x2]
        avg_color = patch.mean(axis=(0, 1))

        x_norm = px / w
        y_norm = py / h
        sw = random.uniform(0.02, 0.25)
        sh = random.uniform(0.02, 0.25)
        rot = random.uniform(0.0, 1.0)
        strokes.append([x_norm, y_norm, sw, sh, rot, float(avg_color[0]), float(avg_color[1]), float(avg_color[2])])
    return strokes


def main(
    output_dir: str = "data/sft",
    num_random: int = 500,
    num_celeba: int = 1500,
    num_strokes: int = 16,
    canvas_size: int = 256,
    device: str = "cuda",
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/images", exist_ok=True)

    meta_brushes = load_meta_brushes(device)
    samples = []

    print(f"Generating {num_random} random painting examples...")
    for i in range(num_random):
        strokes, img = generate_random_painting(num_strokes, canvas_size, canvas_size, device)
        img_name = f"random_{i:05d}.png"
        img.save(f"{output_dir}/images/{img_name}")
        samples.append({
            "image": f"images/{img_name}",
            "strokes": strokes,
            "text": strokes_to_text(strokes),
            "source": "random",
        })
        if (i + 1) % 100 == 0:
            print(f"  Random: {i+1}/{num_random}")

    print(f"\nGenerating {num_celeba} CelebA-guided examples...")
    from datasets import load_dataset
    ds = load_dataset("nielsr/CelebA-faces", split="train", streaming=True)

    celeba_count = 0
    for idx, example in enumerate(ds):
        if celeba_count >= num_celeba:
            break
        try:
            target = example["image"].resize((canvas_size, canvas_size))
            strokes = color_guided_strokes(target, num_strokes)
            params = torch.tensor(strokes, dtype=torch.float32).to(device)
            rendered = render_to_image(params, canvas_size, canvas_size, device)

            img_name = f"celeba_{celeba_count:05d}.png"
            target_name = f"celeba_{celeba_count:05d}_target.png"
            rendered.save(f"{output_dir}/images/{img_name}")
            target.save(f"{output_dir}/images/{target_name}")

            samples.append({
                "image": f"images/{img_name}",
                "target": f"images/{target_name}",
                "strokes": strokes,
                "text": strokes_to_text(strokes),
                "source": "celeba",
            })
            celeba_count += 1
            if celeba_count % 100 == 0:
                print(f"  CelebA: {celeba_count}/{num_celeba}")
        except Exception as e:
            print(f"  Skipping CelebA {idx}: {e}")
            continue

    random.shuffle(samples)

    conversations = []
    for s in samples:
        conversations.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": os.path.join(output_dir, s["image"])},
                        {"type": "text", "text": USER_PROMPT},
                    ],
                },
                {"role": "assistant", "content": s["text"]},
            ],
            "source": s["source"],
        })

    split_idx = int(len(conversations) * 0.95)
    train_data = conversations[:split_idx]
    val_data = conversations[split_idx:]

    with open(f"{output_dir}/train.json", "w") as f:
        json.dump(train_data, f)
    with open(f"{output_dir}/val.json", "w") as f:
        json.dump(val_data, f)

    print(f"\nDone! Total: {len(samples)} samples")
    print(f"  Train: {len(train_data)} | Val: {len(val_data)}")
    print(f"  Saved to {output_dir}/")


if __name__ == "__main__":
    fire.Fire(main)
