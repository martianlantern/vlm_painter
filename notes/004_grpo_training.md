---
title: grpo_rl_training_v1
description: GRPO RL training of Qwen3.5-0.8B to paint CelebA faces using brush strokes. Custom reward combining format, pixel similarity, and color matching.
related notes: [[001_qwen3.5_vllm_server_setup.md]], [[002_prompt_format_experiment.md]], [[003_sft_training.md]]
---

## Goal
Train the SFT'd Qwen3.5-0.8B model to generate brush strokes that visually match target CelebA images using GRPO reinforcement learning.

## Architecture
- **Generation**: vLLM server serving SFT-merged model for fast (~2.6s/completion) multimodal generation
- **Training**: Separate PyTorch model with LoRA on bf16 base model
- **Reward**: Custom multi-component reward function

## Reward Function
```
total_reward = format_score * 0.3 + pixel_score * 0.5 + color_score * 0.2
```
- **Format score (30%)**: fraction of valid strokes (16 = 1.0)
- **Pixel score (50%)**: `max(0, 1 - sqrt(MSE) * 3)` between rendered canvas and target
- **Color score (20%)**: `max(0, 1 - color_distance * 2)` mean color similarity

## Training Config
- Model: SFT-merged Qwen3.5-0.8B (bf16, no quantization)
- LoRA: r=32, alpha=64, 1.48% trainable params
- LR: 5e-6 cosine, AdamW
- Batch: 4 images × 4 generations = 16 completions/step
- Grad accum: 4

## Setup Challenges
1. **Triton PassManager bug**: fla kernels incompatible with Triton 3.2.0 on this CUDA setup
   - Solution: Uninstall fla, use torch fallback (slower but works)
2. **causal-conv1d**: Compilation fails due to CUDA symbol mismatch with torch 2.6
   - Solution: Skip it, use without fast path
3. **device_map="auto"**: Hangs indefinitely on get_peft_model
   - Solution: Use explicit `.to("cuda")`
4. **vLLM tokenizer**: Merged model has new `TokenizersBackend` class
   - Solution: Copy original tokenizer files from HuggingFace

## Early Results
- Baseline reward: ~0.281
- Step 5: reward ~0.288
- Step 10: reward ~0.285 (slow early improvement expected)
- Sample images logged to wandb at each step

## wandb Runs
- GRPO: https://wandb.ai/martianlantern/vlm-painter/runs/3vw8172h
- SFT: https://wandb.ai/martianlantern/vlm-painter/runs/sm4yugnn

## Files
- `train_grpo.py`: GRPO training script
- `train_sft.py`: SFT training script
- `stroke_renderer.py`: Standalone stroke rendering module
- `generate_sft_data.py`: SFT data generation
- `prepare_grpo_data.py`: GRPO prompt preparation
- `experiment_prompts.py`: Phase 1 prompt experiments
- `eval_sft.py`: SFT evaluation
- `merge_sft_adapter.py`: LoRA adapter merging
- `serve_qwen35vl.py`: vLLM server script

## Next Steps
- Monitor GRPO training for reward improvement
- If rewards plateau, try increasing num_generations or adjusting reward weights
- Evaluate final model on held-out CelebA images
