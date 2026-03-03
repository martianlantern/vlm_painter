#!/usr/bin/env -S PYTHONUNBUFFERED=1 uv run --prerelease=allow --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "vllm @ git+https://github.com/vllm-project/vllm.git",
#   "transformers @ git+https://github.com/huggingface/transformers.git",
#   "fire>=0.7.1",
#   "qwen-vl-utils>=0.0.14",
# ]
# ///
import sys

def serve(
    model: str = "Qwen/Qwen3.5-0.8B",
    host: str = "0.0.0.0",
    port: int = 8000,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.9,
    dtype: str = "bfloat16",
    enable_prefix_caching: bool = True,
    extra: list[str] | None = None,
    device: str = "0",
):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    argv = [
        "vllm", "serve", model,
        "--host", host,
        "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--dtype", dtype,
        "--limit-mm-per-prompt", '{"image": 1, "video": 0}',
    ]
    if enable_prefix_caching:
        argv.append("--enable-prefix-caching")
    if extra:
        argv.extend(extra)

    sys.argv = argv
    from vllm.entrypoints.cli.main import main
    main()

if __name__ == "__main__":
    import fire
    fire.Fire(serve)
