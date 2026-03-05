#!/usr/bin/env -S PYTHONUNBUFFERED=1 uv run --env-file .env --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "fire>=0.7.1",
#   "torch>=2.0.0",
#   "transformers @ git+https://github.com/huggingface/transformers.git",
#   "peft>=0.18.0",
#   "accelerate>=1.0.0",
#   "wandb",
#   "Pillow>=10.0.0",
#   "qwen-vl-utils>=0.0.14",
#   "python-dotenv",
#   "numpy",
# ]
# ///
import json, os, re, math, random
import fire, torch, torch.nn.functional as F, numpy as np, wandb
from PIL import Image
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


def compute_reward(text, target_path, meta_brushes, device="cuda"):
    from stroke_renderer import render_strokes
    raw = extract_strokes(text)
    n_strokes = len(raw)

    if n_strokes < 10:
        return -0.5

    format_score = min(n_strokes, 16) / 16.0

    valid = clamp_strokes(raw[:16])
    params_list = valid
    params = torch.tensor(valid, dtype=torch.float32).to(device)
    canvas = render_strokes(params, 256, 256, device=device, meta_brushes=meta_brushes)

    target = Image.open(target_path).convert("RGB").resize((256, 256))
    target_t = torch.from_numpy(np.array(target)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    target_t = target_t.to(device)

    painted_mask = (canvas < 0.99).any(dim=1, keepdim=True).float()
    coverage = painted_mask.mean().item()
    coverage_score = min(coverage * 3.0, 1.0)

    if coverage > 0.01:
        painted_pixels = painted_mask.expand_as(canvas)
        c_masked = canvas * painted_pixels
        t_masked = target_t * painted_pixels
        n_painted = painted_pixels.sum().clamp(min=1)
        local_mse = ((c_masked - t_masked) ** 2).sum() / n_painted
        local_score = max(0, 1.0 - local_mse.item() * 3.0)
    else:
        local_score = 0.0

    c_mean = canvas.mean(dim=(2, 3))
    t_mean = target_t.mean(dim=(2, 3))
    global_color_dist = ((c_mean - t_mean) ** 2).sum().sqrt().item()
    color_score = max(0, 1.0 - global_color_dist * 2.0)

    positions = torch.tensor([[s[0], s[1]] for s in params_list], dtype=torch.float32)
    if len(positions) > 1:
        pos_std = positions.std(dim=0).mean().item()
        spread_score = min(pos_std * 4.0, 1.0)
    else:
        spread_score = 0.0

    colors = torch.tensor([[s[5], s[6], s[7]] for s in params_list], dtype=torch.float32)
    if len(colors) > 1:
        color_std = colors.std(dim=0).mean().item()
        diversity_score = min(color_std * 4.0, 1.0)
    else:
        diversity_score = 0.0

    reward = (
        format_score * 0.15 +
        local_score * 0.25 +
        color_score * 0.15 +
        coverage_score * 0.20 +
        spread_score * 0.10 +
        diversity_score * 0.15
    )
    return reward


def build_messages(image_path):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": USER_PROMPT},
        ]},
    ]


@torch.no_grad()
def generate_batch(model, processor, image_paths, n_gen=4, max_tokens=500, temperature=0.7):
    from qwen_vl_utils import process_vision_info
    all_texts, all_images = [], []
    for img_path in image_paths:
        msgs = build_messages(img_path)
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        img_inputs, _ = process_vision_info(msgs)
        for _ in range(n_gen):
            all_texts.append(text)
            all_images.extend(img_inputs)

    inputs = processor(
        text=all_texts, images=all_images, padding=True,
        truncation=True, max_length=700, return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        **inputs, max_new_tokens=max_tokens,
        temperature=temperature, do_sample=True, top_p=0.9,
        pad_token_id=processor.tokenizer.eos_token_id,
    )

    prompt_len = inputs["input_ids"].shape[1]
    completions = []
    for i in range(len(all_texts)):
        text = processor.decode(outputs[i][prompt_len:], skip_special_tokens=True)
        completions.append(text)

    result = []
    for i in range(len(image_paths)):
        result.append(completions[i * n_gen : (i + 1) * n_gen])
    return result


def get_completion_logprobs(model, processor, image_path, completion_text):
    from qwen_vl_utils import process_vision_info
    msgs = build_messages(image_path)
    msgs.append({"role": "assistant", "content": completion_text})
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    img_inputs, _ = process_vision_info(msgs[:2])
    inputs = processor(
        text=[text], images=img_inputs, padding=True,
        truncation=True, max_length=1200, return_tensors="pt",
    ).to(model.device)

    labels = inputs["input_ids"].clone()
    marker = "<|im_start|>assistant\n"
    marker_ids = processor.tokenizer.encode(marker, add_special_tokens=False)
    ids = labels[0].tolist()
    for j in range(len(ids) - len(marker_ids)):
        if ids[j:j+len(marker_ids)] == marker_ids:
            labels[0, :j+len(marker_ids)] = -100
            break

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model(**inputs).logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    per_token_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    ).view(shift_labels.shape)

    mask = (shift_labels != -100).float()
    n_tokens = mask.sum()
    if n_tokens == 0:
        return torch.tensor(0.0, device=model.device), 0
    avg_nll = (per_token_loss * mask).sum() / n_tokens
    return avg_nll, int(n_tokens.item())


def main(
    model_path: str = "checkpoints/sft_merged",
    data_path: str = "data/grpo/prompts.json",
    output_dir: str = "checkpoints/grpo_v4",
    batch_size: int = 4,
    grad_accum: int = 4,
    lr: float = 1e-6,
    lora_r: int = 16,
    lora_alpha: int = 32,
    num_generations: int = 6,
    max_steps: int = 200,
    warmup_steps: int = 10,
    save_interval: int = 25,
    log_image_interval: int = 5,
    kl_coeff: float = 0.2,
    device: str = "0",
    wandb_project: str = "vlm-painter",
    run_name: str = "grpo-v4",
):
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/samples", exist_ok=True)

    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import LoraConfig, get_peft_model
    from stroke_renderer import render_to_image, load_meta_brushes

    wandb.init(project=wandb_project, name=run_name, config={
        "model": model_path, "lr": lr, "lora_r": lora_r,
        "num_generations": num_generations, "batch_size": batch_size,
        "kl_coeff": kl_coeff, "grad_accum": grad_accum,
    })

    with open(data_path) as f:
        all_prompts = json.load(f)

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"

    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda")

    ref_model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda")
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    lora_config = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01, betas=(0.9, 0.99),
    )

    def lr_lambda(step):
        if step < warmup_steps:
            return max(step / warmup_steps, 0.1)
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        return max(0.5 * (1 + math.cos(math.pi * progress)), 0.1)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    meta_brushes = load_meta_brushes(device="cuda")
    global_step = 0
    best_running_reward = -1.0
    reward_history = []

    for epoch in range(10):
        random.shuffle(all_prompts)
        for batch_start in range(0, len(all_prompts), batch_size):
            if global_step >= max_steps:
                break

            batch = all_prompts[batch_start:batch_start + batch_size]
            if len(batch) < batch_size:
                continue

            image_paths = [s["image_path"] for s in batch]

            model.eval()
            completions = generate_batch(model, processor, image_paths, n_gen=num_generations)
            model.train()

            batch_rewards = []
            for i, (img_path, comps) in enumerate(zip(image_paths, completions)):
                rewards = [compute_reward(c, img_path, meta_brushes) for c in comps]
                batch_rewards.append(rewards)

            good_pairs = []
            for i in range(len(batch)):
                group = batch_rewards[i]
                group_mean = np.mean(group)
                group_std = np.std(group) + 1e-8
                best_idx = int(np.argmax(group))
                if group[best_idx] > 0:
                    adv = (group[best_idx] - group_mean) / group_std
                    adv = max(adv, 0.01)
                    good_pairs.append((i, best_idx, adv, group[best_idx]))

            if not good_pairs:
                print(f"Step {global_step:4d} | SKIP", flush=True)
                global_step += 1
                continue

            reward_history.extend([r for rlist in batch_rewards for r in rlist])

            optimizer.zero_grad()
            total_policy_loss = 0.0
            total_kl = 0.0
            n_updates = 0

            for img_idx, gen_idx, advantage, reward in good_pairs:
                comp = completions[img_idx][gen_idx]

                policy_nll, n_tok = get_completion_logprobs(
                    model, processor, image_paths[img_idx], comp
                )
                if n_tok == 0:
                    continue

                with torch.no_grad():
                    ref_nll, _ = get_completion_logprobs(
                        ref_model, processor, image_paths[img_idx], comp
                    )

                kl = (ref_nll - policy_nll).clamp(min=0)

                loss = (policy_nll * advantage + kl_coeff * kl) / grad_accum
                loss.backward()
                total_policy_loss += policy_nll.item() * advantage
                total_kl += kl.item()
                n_updates += 1

            if n_updates > 0 and (global_step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            all_rewards_flat = [r for rlist in batch_rewards for r in rlist]
            best_rewards = [max(r) for r in batch_rewards]
            mean_r = np.mean(all_rewards_flat)
            mean_best = np.mean(best_rewards)
            running = np.mean(reward_history[-200:])
            if running > best_running_reward:
                best_running_reward = running

            wandb.log({
                "reward/mean": mean_r,
                "reward/best": mean_best,
                "reward/running": running,
                "reward/best_running": best_running_reward,
                "reward/good_ratio": len(good_pairs) / batch_size,
                "train/loss": total_policy_loss / max(n_updates, 1),
                "train/kl": total_kl / max(n_updates, 1),
                "train/lr": scheduler.get_last_lr()[0],
            }, step=global_step)

            print(f"Step {global_step:4d} | Loss: {total_policy_loss/max(n_updates,1):.4f} | "
                  f"R: {mean_r:.3f} (best: {mean_best:.3f}) | "
                  f"KL: {total_kl/max(n_updates,1):.4f} | "
                  f"Good: {len(good_pairs)}/{batch_size} | "
                  f"Running: {running:.3f} | LR: {scheduler.get_last_lr()[0]:.2e}", flush=True)

            if global_step % log_image_interval == 0:
                for i in range(min(2, len(batch))):
                    best_idx = int(np.argmax(batch_rewards[i]))
                    best = completions[i][best_idx]
                    raw = extract_strokes(best)
                    if raw:
                        valid = clamp_strokes(raw[:16])
                        params = torch.tensor(valid, dtype=torch.float32).cuda()
                        rendered = render_to_image(params)
                        target = Image.open(image_paths[i]).resize((256, 256))
                        comp_img = Image.new("RGB", (512, 256))
                        comp_img.paste(target, (0, 0))
                        comp_img.paste(rendered, (256, 0))
                        comp_img.save(f"{output_dir}/samples/step_{global_step:04d}_img{i}.png")
                        wandb.log({f"samples/img_{i}": wandb.Image(comp_img)}, step=global_step)

            if global_step > 0 and global_step % save_interval == 0:
                model.save_pretrained(f"{output_dir}/step_{global_step}")
                print(f"  Saved: step_{global_step}", flush=True)

            global_step += 1

        if global_step >= max_steps:
            break

    model.save_pretrained(f"{output_dir}/final")
    processor.save_pretrained(f"{output_dir}/final")
    wandb.finish()
    print(f"\nGRPO v4 complete! Model saved to {output_dir}/final", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
