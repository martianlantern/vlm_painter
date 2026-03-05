#!/usr/bin/env -S PYTHONUNBUFFERED=1 uv run --env-file .env --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "fire>=0.7.1",
#   "torch>=2.0.0",
#   "transformers @ git+https://github.com/huggingface/transformers.git",
#   "peft>=0.18.0",
#   "accelerate>=1.0.0",
#   "pillow>=12.1.1",
#   "torchvision>=0.25.0",
# ]
# ///
import os
import fire
import torch

def main(
    base_model: str = "Qwen/Qwen3.5-0.8B",
    adapter_path: str = "checkpoints/sft/final",
    output_path: str = "checkpoints/sft_merged",
    device: str = "0",
):
    os.environ["CUDA_VISIBLE_DEVICES"] = device

    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import PeftModel

    print(f"Loading base model: {base_model}")
    model = AutoModelForImageTextToText.from_pretrained(
        base_model, dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True,
    )

    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("Merging adapter...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {output_path}")
    model.save_pretrained(output_path)

    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    processor.save_pretrained(output_path)

    import json
    tc_path = os.path.join(output_path, "tokenizer_config.json")
    with open(tc_path) as f:
        tc = json.load(f)
    if tc.get("tokenizer_class") == "TokenizersBackend":
        tc["tokenizer_class"] = "Qwen2Tokenizer"
        with open(tc_path, "w") as f:
            json.dump(tc, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    fire.Fire(main)
