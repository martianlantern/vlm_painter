#!/usr/bin/env -S PYTHONUNBUFFERED=1 uv run --env-file .env --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "fire>=0.7.1",
#   "openai>=1.0.0",
#   "Pillow>=10.0.0",
#   "torch>=2.0.0",
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

PROMPTS = {
    "csv_basic": """You are a painting AI. Given an image, generate brush strokes to recreate it.
Each stroke has 8 parameters: x, y, width, height, rotation, R, G, B
- x, y: center position (0.0 to 1.0)
- width, height: stroke size (0.01 to 0.5)
- rotation: angle (0.0 to 1.0, where 1.0 = 180 degrees)
- R, G, B: color (0.0 to 1.0)

Output exactly 16 strokes, one per line, as comma-separated values.
Example:
0.50,0.30,0.20,0.05,0.00,0.80,0.60,0.40
0.25,0.70,0.10,0.15,0.50,0.10,0.20,0.30

Generate 16 strokes to paint this image:""",

    "csv_short": """Paint this image using brush strokes. Output 16 strokes as CSV lines.
Format: x,y,w,h,rotation,r,g,b (all values 0.0-1.0, w/h range 0.01-0.5)
Example: 0.50,0.30,0.20,0.05,0.00,0.80,0.60,0.40""",

    "json_strokes": """You are a painting AI. Analyze this image and generate brush strokes to recreate it.
Output a JSON array of 16 stroke objects. Each stroke: {"x":float,"y":float,"w":float,"h":float,"r":float,"R":float,"G":float,"B":float}
- x,y: position 0-1, w,h: size 0.01-0.5, r: rotation 0-1, R,G,B: color 0-1
Output ONLY the JSON array, nothing else.""",

    "tagged_csv": """<task>Paint the given image using brush strokes.</task>
<format>Each line: x,y,w,h,rotation,r,g,b (values 0.0-1.0, w/h: 0.01-0.5)</format>
<instructions>Generate exactly 16 strokes to recreate the image. Output ONLY the stroke lines.</instructions>
<strokes>""",

    "thinking_csv": """Look at this image carefully. I want you to paint it using brush strokes.

Think about the dominant colors, the face position, and key features. Then output exactly 16 brush strokes.

Each stroke is: x,y,width,height,rotation,R,G,B
- x,y: center (0-1), width,height: size (0.01-0.5), rotation (0-1), R,G,B: color (0-1)

First describe what you see briefly, then output the strokes after "STROKES:" on separate lines.""",
}


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def call_vlm(client, model, prompt, image_path, temperature=0.7, max_tokens=1500):
    b64 = encode_image(image_path)
    resp = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
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
    valid = []
    for s in strokes:
        x, y, w, h, rot, r, g, b = s
        if all(0 <= v <= 1 for v in [x, y, rot, r, g, b]) and 0.001 <= w <= 0.6 and 0.001 <= h <= 0.6:
            valid.append(s)
    return valid


def main(
    image_dir: str = "samples",
    api_base: str = "http://localhost:8000/v1",
    model: str = "Qwen/Qwen3.5-0.8B",
    output_dir: str = "results/prompt_experiment",
    num_images: int = 3,
    samples_per_prompt: int = 3,
    temperature: float = 0.7,
):
    os.makedirs(output_dir, exist_ok=True)
    client = OpenAI(base_url=api_base, api_key="dummy")
    images = sorted(Path(image_dir).glob("*.png"))[:num_images]

    all_results = {}

    for prompt_name, prompt_text in PROMPTS.items():
        print(f"\n{'='*60}")
        print(f"Testing prompt: {prompt_name}")
        print(f"{'='*60}")
        results = []

        for img_path in images:
            for sample_idx in range(samples_per_prompt):
                print(f"  Image: {img_path.name} | Sample {sample_idx+1}/{samples_per_prompt}")
                try:
                    raw = call_vlm(client, model, prompt_text, str(img_path), temperature)
                    strokes = extract_strokes(raw)
                    valid = validate_strokes(strokes)

                    result = {
                        "image": img_path.name,
                        "sample": sample_idx,
                        "raw_output": raw,
                        "num_extracted": len(strokes),
                        "num_valid": len(valid),
                        "strokes": valid,
                    }
                    results.append(result)
                    print(f"    Extracted: {len(strokes)} | Valid: {len(valid)}")

                    if valid:
                        import torch
                        from stroke_renderer import render_to_image
                        params = torch.tensor(valid, dtype=torch.float32).cuda()
                        rendered = render_to_image(params)
                        out_name = f"{prompt_name}_{img_path.stem}_s{sample_idx}.png"
                        rendered.save(os.path.join(output_dir, out_name))
                        print(f"    Saved: {out_name}")
                except Exception as e:
                    print(f"    ERROR: {e}")
                    results.append({"image": img_path.name, "sample": sample_idx, "error": str(e)})

        all_results[prompt_name] = results

    summary = {}
    for prompt_name, results in all_results.items():
        valid_counts = [r.get("num_valid", 0) for r in results]
        extracted_counts = [r.get("num_extracted", 0) for r in results]
        total = len(results)
        has_any = sum(1 for v in valid_counts if v > 0)
        summary[prompt_name] = {
            "total_runs": total,
            "runs_with_valid_strokes": has_any,
            "avg_valid_strokes": sum(valid_counts) / max(total, 1),
            "avg_extracted_strokes": sum(extracted_counts) / max(total, 1),
            "max_valid_strokes": max(valid_counts) if valid_counts else 0,
        }

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, s in summary.items():
        print(f"\n{name}:")
        for k, v in s.items():
            print(f"  {k}: {v}")

    with open(os.path.join(output_dir, "experiment_results.json"), "w") as f:
        json.dump({"summary": summary, "details": {k: [
            {kk: vv for kk, vv in r.items() if kk != "raw_output"} for r in v
        ] for k, v in all_results.items()}}, f, indent=2)

    with open(os.path.join(output_dir, "raw_outputs.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    fire.Fire(main)
