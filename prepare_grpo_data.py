#!/usr/bin/env -S PYTHONUNBUFFERED=1 uv run --env-file .env --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "fire>=0.7.1",
#   "datasets>=3.0.0",
#   "Pillow>=10.0.0",
#   "python-dotenv",
# ]
# ///
import os
import json
import fire
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()


SYSTEM_PROMPT = """You are a painting AI. Given an image, generate brush strokes to recreate it.
Each stroke has 8 parameters: x, y, width, height, rotation, R, G, B
- x, y: center position (0.0 to 1.0)
- width, height: stroke size (0.01 to 0.5)
- rotation: angle (0.0 to 1.0, where 1.0 = 180 degrees)
- R, G, B: color (0.0 to 1.0)

Output exactly 16 strokes, one per line, as comma-separated values."""

USER_PROMPT = "Generate 16 strokes to paint this image:"


def main(
    output_dir: str = "data/grpo",
    num_images: int = 500,
    canvas_size: int = 256,
):
    os.makedirs(f"{output_dir}/images", exist_ok=True)

    from datasets import load_dataset

    print(f"Downloading {num_images} CelebA images...")
    ds = load_dataset("nielsr/CelebA-faces", split="train", streaming=True)

    samples = []
    for i, example in enumerate(ds):
        if i >= num_images:
            break
        img = example["image"].resize((canvas_size, canvas_size))
        img_name = f"celeba_{i:05d}.png"
        img_path = f"{output_dir}/images/{img_name}"
        img.save(img_path)

        samples.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image", "image": os.path.abspath(img_path)},
                    {"type": "text", "text": USER_PROMPT},
                ]},
            ],
            "image_path": os.path.abspath(img_path),
        })

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{num_images}")

    with open(f"{output_dir}/prompts.json", "w") as f:
        json.dump(samples, f)

    print(f"Done! {len(samples)} prompts saved to {output_dir}/")


if __name__ == "__main__":
    fire.Fire(main)
