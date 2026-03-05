---
title: vllm_serving_setup
description: Serving Qwen3.5-0.8B base and SFT-merged model via vLLM on separate ports
related notes: [[003_sft_training.md]]
---

## Goal
Serve both the base Qwen3.5-0.8B and the SFT fine-tuned (merged) model via vLLM for inference.

## Setup

- **Port 8000**: Base model `Qwen/Qwen3.5-0.8B`
- **Port 8001**: Merged SFT model `checkpoints/sft_merged`
- GPU: Single A100 80GB, each server uses 0.45 gpu_memory_utilization
- Both use `--trust-remote-code`, prefix caching enabled, max_model_len=8000

## Commands

```bash
# Base model
./serve_qwen35vl.py --port 8000 --gpu_memory_utilization 0.45 --model "Qwen/Qwen3.5-0.8B"

# Merged SFT model
./serve_qwen35vl.py --port 8001 --gpu_memory_utilization 0.45 --model "checkpoints/sft_merged"
```

## LoRA Serving Issue

vLLM currently has a **known bug** with LoRA adapters on Qwen3-VL/Qwen3.5 models:
- GitHub issues: #28186, #27962
- Error: `IndexError: list index out of range` in `column_parallel_linear.py` when activating LoRA
- Root cause: vLLM's LoRA layer handling for merged column-parallel projections (QKV, gate-up) doesn't match the PEFT adapter format for this architecture
- **Workaround**: Use merged model weights instead (which we already have at `checkpoints/sft_merged`)

## Fixes Applied

1. Patched `tokenizer_config.json` in merged model: changed `"tokenizer_class": "TokenizersBackend"` to `"Qwen2Tokenizer"` (vLLM's transformers doesn't recognize the newer class)
2. Updated `merge_sft_adapter.py` to auto-fix this tokenizer class issue on future merges
3. Set `lora_dropout` to 0.0 in adapter_config.json (vLLM requirement, though LoRA serving doesn't work yet anyway)

## Verification

Both servers respond to `/v1/models`:
- Port 8000: model id `Qwen/Qwen3.5-0.8B`
- Port 8001: model id `checkpoints/sft_merged`
