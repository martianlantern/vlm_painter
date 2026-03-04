#!/usr/bin/env -S PYTHONUNBUFFERED=1 uv run --env-file .env --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "fire>=0.7.1",
#   "torch>=2.0.0",
#   "transformers @ git+https://github.com/huggingface/transformers.git",
#   "peft>=0.18.0",
#   "accelerate>=1.0.0",
#   "Pillow>=10.0.0",
#   "qwen-vl-utils>=0.0.14",
#   "numpy",
#   "python-dotenv",
# ]
# ///
import os
import re
import fire
import torch
import numpy as np
from PIL import Image
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


def extract_strokes(text):
    strokes = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        nums = re.findall(r"[-+]?\d*\.?\d+", line)
        if len(nums) >= 8:
            vals = [float(n) for n in nums[:8]]
            if all(0 <= v <= 1.5 for v in vals):
                strokes.append(vals)
    return strokes


def main(
    model_name: str = "Qwen/Qwen3.5-0.8B",
    adapter_path: str = "checkpoints/sft/final",
    image_dir: str = "samples",
    output_dir: str = "results/sft_eval",
    num_images: int = 5,
    num_samples: int = 3,
    device: str = "0",
):
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    os.makedirs(output_dir, exist_ok=True)

    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info
    from stroke_renderer import render_to_image

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    base_model = AutoModelForImageTextToText.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    images = sorted(Path(image_dir).glob("*.png"))[:num_images]
    total_valid = 0
    total_runs = 0

    for img_path in images:
        print(f"\n--- {img_path.name} ---")
        for s_idx in range(num_samples):
            msgs = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image", "image": str(img_path)},
                    {"type": "text", "text": USER_PROMPT},
                ]},
            ]
            text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            img_inputs, _ = process_vision_info(msgs)

            inputs = processor(text=[text], images=img_inputs, padding=True, return_tensors="pt").to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=600,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                )

            generated = output_ids[0][inputs["input_ids"].shape[1]:]
            output_text = processor.tokenizer.decode(generated, skip_special_tokens=True)

            think_end = output_text.find("</think>")
            if think_end >= 0:
                output_text = output_text[think_end + len("</think>"):].strip()

            strokes = extract_strokes(output_text)
            valid = [s for s in strokes if all(0 <= v <= 1 for v in s) and all(0.001 <= s[j] <= 0.6 for j in [2, 3])]

            total_runs += 1
            total_valid += len(valid)

            print(f"  Sample {s_idx}: extracted={len(strokes)} valid={len(valid)}")
            print(f"    Output: {output_text[:200]}")

            if valid:
                params = torch.tensor(valid, dtype=torch.float32).cuda()
                rendered = render_to_image(params)
                out_name = f"{img_path.stem}_s{s_idx}.png"
                rendered.save(os.path.join(output_dir, out_name))

                original = Image.open(img_path).resize((256, 256))
                side_by_side = Image.new("RGB", (512, 256))
                side_by_side.paste(original, (0, 0))
                side_by_side.paste(rendered, (256, 0))
                side_by_side.save(os.path.join(output_dir, f"{img_path.stem}_s{s_idx}_compare.png"))

    print(f"\n{'='*40}")
    print(f"Total runs: {total_runs}")
    print(f"Avg valid strokes: {total_valid / max(total_runs, 1):.1f}")
    print(f"Results saved to {output_dir}/")


if __name__ == "__main__":
    fire.Fire(main)
