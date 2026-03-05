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
from tqdm.auto import tqdm
from stroke_renderer import render_strokes, render_to_image, load_meta_brushes
from datasets import DatasetDict, Dataset

def random_stroke():
    x = random.uniform(0.05, 0.95)
    y = random.uniform(0.05, 0.95)
    w = random.uniform(0.1, 0.5)
    h = random.uniform(0.1, 0.5)
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

def generate_random_painting(num_strokes=25, canvas_h=256, canvas_w=256, device="cuda"):
    strokes = [random_stroke() for _ in range(num_strokes)]
    params = torch.tensor(strokes, dtype=torch.float32).to(device)
    img = render_to_image(params, canvas_h, canvas_w, device)
    return strokes, img

def main(
    num_train_samples: int = 1000,
    num_test_samples: int = 50,
    canvas_size: int = 256,
    device: str = "cuda"
):
    train_ds = {
        "image": [],
        "strokes": [],
        "num_strokes": []
    }
    for i in tqdm(range(num_train_samples)):
        num_strokes = random.randint(25, 100)
        strokes, img = generate_random_painting(
            num_strokes=num_strokes,
            canvas_h=canvas_size,
            canvas_w=canvas_size,
            device=device
        )
        train_ds["image"].append(img)
        train_ds["strokes"].append(strokes_to_text(strokes))
        train_ds["num_strokes"].append(len(strokes))

    test_ds = {
        "image": [],
        "strokes": [],
        "num_strokes": []
    }
    for i in tqdm(range(num_test_samples)):
        num_strokes = random.randint(25, 100)
        strokes, img = generate_random_painting(
            num_strokes=num_strokes,
            canvas_h=canvas_size,
            canvas_w=canvas_size,
            device=device
        )
        test_ds["image"].append(img)
        test_ds["strokes"].append(strokes_to_text(strokes))
        test_ds["num_strokes"].append(len(strokes))

    ds = DatasetDict({
        "train": Dataset.from_dict(train_ds),
        "test": Dataset.from_dict(test_ds)
    })
    ds.push_to_hub("darshanmakwana/vlm_painter_sft")

if __name__ == "__main__":
    fire.Fire(main)