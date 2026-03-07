#!/usr/bin/env -S PYTHONUNBUFFERED=1 uv run --env-file .env --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "fire>=0.7.1",
#   "openai>=1.0.0",
#   "Pillow>=10.0.0",
#   "torch>=2.0.0",
#   "datasets>=3.0.0",
#   "numpy",
# ]
# ///
import json
import base64
import re
import os
import fire
from io import BytesIO
from pathlib import Path
from openai import OpenAI
from datasets import load_dataset
from PIL import Image

SYSTEM_PROMPT = """You are an image painter. Given an image your task is to generate brush strokes to paint an image similar to it.
Each stroke has 8 parameters: x, y, width, height, rotation, R, G, B
- x, y: center position (0.0 to 1.0)
- width, height: stroke size (0.1 to 0.5)
- rotation: angle (0.0 to 1.0, where 1.0 = 180 degrees)
- R, G, B: color (0.0 to 1.0)

Output strokes one per line, as comma-separated values."""

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def encode_pil_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def call_vlm(client, model, user_prompt, image, temperature=0.7, max_tokens=1500):
    if isinstance(image, str):
        b64 = encode_image(image)
    elif isinstance(image, Image.Image):
        b64 = encode_pil_image(image)
    else:
        print(f"Undefined image type: {type(image)}")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content

def extract_strokes(text):
    strokes = []
    for line in text.strip().split("\n"):
        line = line.strip().strip(",").strip('"').strip("'")
        if not line:
            continue
        nums = re.findall(r"[-+]?\d*\.?\d+", line)
        if len(nums) >= 8:
            vals = [float(n) for n in nums[:8]]
            strokes.append(vals)
    return strokes


def validate_strokes(strokes):
    for s in strokes:
        x, y, w, h, rot, r, g, b = s
        if all(0 <= v <= 1 for v in [x, y, rot, r, g, b]) and 0.001 <= w <= 0.6 and 0.001 <= h <= 0.6:
            return False
    return True

def main(
    model: str = "checkpoints/sft_merged/",
    dataset_id: str = "darshanmakwana/vlm_painter_sft",
    api_base: str = "http://localhost:8000/v1",
):
    client = OpenAI(base_url=api_base, api_key="dummy")
    test_data = load_dataset(dataset_id)["test"]
    sample = test_data[0]
    image = sample["image"]
    num_strokes = 89
    user_prompt = f"Generate {num_strokes} strokes to paint this image:"
    result = call_vlm(client, model, user_prompt, image)
    strokes = extract_strokes(result)
    if len(strokes) != num_strokes:
        print(f"Prompted {num_strokes} strokes, got {len(strokes)} strokes")
    else:
        if not validate_strokes(strokes):
            print(f"Invalid strokes params")

if __name__ == "__main__":
    fire.Fire(main)