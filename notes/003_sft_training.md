---
title: sft_training_v1
description: SFT training of Qwen3.5-0.8B for brush stroke format generation. Achieved 15/16 avg valid strokes.
related notes: [[001_qwen3.5_vllm_server_setup.md]], [[002_prompt_format_experiment.md]]
---

## Goal
Train Qwen3.5-0.8B to reliably output brush strokes in CSV format (x,y,w,h,rotation,R,G,B).

## Data Generation
- 1000 samples (950 train, 50 val)
- 300 random stroke paintings + 700 CelebA-guided strokes
- CelebA-guided: sample random positions on face, extract avg patch color
- Each sample: rendered image → 16 strokes in CSV format

## Training Config
- QLoRA: 4-bit NF4, lora_r=64, lora_alpha=128
- Target modules: q/k/v/o/gate/up/down_proj
- Batch: 2 × 8 grad_accum = effective 16
- LR: 1e-4, cosine schedule, warmup 20 steps
- Epochs: 3 (180 steps total)
- ~40 min on A100 80GB

## Results
- Train loss: 0.8+ → 0.6864
- Eval loss: 0.7123
- **Pre-SFT: ~8.5 avg valid strokes, ~50% format compliance**
- **Post-SFT: 15.0 avg valid strokes, 100% format compliance**
- Model correctly generates 15/16 strokes in valid CSV format
- Values are in correct ranges (0-1 for all params, 0.01-0.5 for w/h)
- Colors show diversity and rough correspondence to image content

## Key Observations
- SFT successfully teaches format compliance
- Strokes don't semantically match the target images yet (expected - that's for GRPO)
- The model avoids degenerate outputs (no more monotonic incrementing)

## Artifacts
- Adapter: `checkpoints/sft/final/`
- Eval outputs: `results/sft_eval/`
- wandb: https://wandb.ai/martianlantern/vlm-painter/runs/sm4yugnn

## Next Steps
- GRPO training with reward = image similarity between rendered strokes and target
