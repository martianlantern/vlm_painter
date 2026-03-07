#!/usr/bin/env -S PYTHONUNBUFFERED=1 uv run --env-file .env --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "fire>=0.7.1",
#   "torch>=2.0.0",
#   "transformers @ git+https://github.com/huggingface/transformers.git",
#   "accelerate>=1.0.0",
#   "wandb",
#   "Pillow>=10.0.0",
#   "qwen-vl-utils>=0.0.14",
#   "numpy",
#   "python-dotenv",
#   "torchvision",
# ]
# ///
import os, re, math, sys
import fire, torch, numpy as np, wandb
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SYSTEM_PROMPT = """You are a painting AI. Given an image, generate brush strokes to recreate it.
Each stroke has 8 parameters: x, y, width, height, rotation, R, G, B
- x, y: center position (0.0 to 1.0)
- width, height: stroke size (0.01 to 0.5)
- rotation: angle (0.0 to 1.0, where 1.0 = 180 degrees)
- R, G, B: color (0.0 to 1.0)

Output exactly 16 strokes, one per line, as comma-separated values."""

USER_PROMPT = "Generate 16 strokes to paint this image:"


def extract_strokes(text):
    think_end = text.find("</think>")
    if think_end >= 0:
        text = text[think_end + len("</think>"):].strip()
    strokes = []
    for line in text.strip().split("\n"):
        nums = re.findall(r"[-+]?\d*\.?\d+", line.strip())
        if len(nums) >= 8:
            strokes.append([float(n) for n in nums[:8]])
    return strokes


def clamp_strokes(strokes):
    return [[
        max(0, min(1, s[0])), max(0, min(1, s[1])),
        max(0.01, min(0.5, s[2])), max(0.01, min(0.5, s[3])),
        max(0, min(1, s[4])),
        max(0, min(1, s[5])), max(0, min(1, s[6])), max(0, min(1, s[7])),
    ] for s in strokes]


def main(
    model_path: str = "checkpoints/sft_merged",
    image_dir: str = "data/grpo/images",
    output_dir: str = "results/sft_eval_v2",
    num_images: int = 50,
    num_samples: int = 2,
    temperature: float = 0.7,
    device: str = "0",
    wandb_project: str = "vlm-painter",
    run_name: str = "sft-eval-v2",
):
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    os.makedirs(output_dir, exist_ok=True)

    from transformers import AutoProcessor, AutoModelForImageTextToText
    from qwen_vl_utils import process_vision_info
    from stroke_renderer import render_strokes, render_to_image, load_meta_brushes

    wandb.init(project=wandb_project, name=run_name, config={
        "model": model_path, "num_images": num_images,
        "temperature": temperature, "num_samples": num_samples,
    })

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model.eval()

    meta_brushes = load_meta_brushes(device="cuda")
    images = sorted(Path(image_dir).glob("*.png"))[:num_images]

    all_metrics = []
    comparison_images = []

    for img_idx, img_path in enumerate(images):
        best = {"reward": -1, "rendered": None}

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
            inputs = processor(
                text=[text], images=img_inputs, padding=True, return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=600,
                    do_sample=True, temperature=temperature, top_p=0.9,
                    pad_token_id=processor.tokenizer.eos_token_id,
                )

            output_text = processor.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
            )
            raw = extract_strokes(output_text)
            valid = clamp_strokes(raw[:16])

            if not valid:
                all_metrics.append({
                    "valid_strokes": 0, "mse": 1.0, "psnr": 0,
                    "coverage": 0, "color_dist": 1, "reward": 0,
                })
                continue

            params = torch.tensor(valid, dtype=torch.float32).cuda()
            canvas = render_strokes(params, 256, 256, device="cuda", meta_brushes=meta_brushes)
            target = Image.open(img_path).convert("RGB").resize((256, 256))
            tt = torch.from_numpy(np.array(target)).float().permute(2, 0, 1).unsqueeze(0).cuda() / 255.0

            mse = ((canvas - tt) ** 2).mean().item()
            coverage = (canvas < 0.99).any(dim=1, keepdim=True).float().mean().item()
            cm, tm = canvas.mean(dim=(2, 3)), tt.mean(dim=(2, 3))
            color_dist = ((cm - tm) ** 2).sum().sqrt().item()

            reward = (
                min(len(valid), 16) / 16 * 0.15 +
                max(0, 1 - math.sqrt(mse) * 3) * 0.50 +
                max(0, 1 - color_dist * 2) * 0.15 +
                min(coverage * 3, 1) * 0.20
            )

            m = {
                "valid_strokes": len(valid), "mse": mse,
                "psnr": -10 * math.log10(mse + 1e-8),
                "coverage": coverage, "color_dist": color_dist, "reward": reward,
            }
            all_metrics.append(m)

            if reward > best["reward"]:
                best = {"reward": reward, "rendered": render_to_image(params)}

        if best["rendered"]:
            target = Image.open(img_path).convert("RGB").resize((256, 256))
            comp = Image.new("RGB", (512, 256))
            comp.paste(target, (0, 0))
            comp.paste(best["rendered"], (256, 0))
            comp.save(f"{output_dir}/{img_path.stem}_compare.png")
            comparison_images.append(comp)
            if img_idx < 20:
                wandb.log({
                    f"comparisons/img_{img_idx:02d}": wandb.Image(comp),
                })

        print(
            f"[{img_idx+1}/{len(images)}] {img_path.name}: "
            f"valid={all_metrics[-1].get('valid_strokes', 0)} reward={best['reward']:.3f}"
        )

    avg = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
    print("\n=== SFT Evaluation Summary ===")
    for k, v in avg.items():
        print(f"  {k}: {v:.4f}")
    wandb.log({f"summary/{k}": v for k, v in avg.items()})

    if comparison_images:
        n = min(16, len(comparison_images))
        grid = Image.new("RGB", (512 * 4, 256 * math.ceil(n / 4)))
        for i in range(n):
            grid.paste(comparison_images[i], ((i % 4) * 512, (i // 4) * 256))
        grid.save(f"{output_dir}/grid.png")
        wandb.log({"summary/grid": wandb.Image(grid)})

    wandb.finish()
    print(f"Results saved to {output_dir}/")


if __name__ == "__main__":
    fire.Fire(main)
