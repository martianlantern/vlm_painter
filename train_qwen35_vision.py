#!/usr/bin/env -S PYTHONUNBUFFERED=1 uv run --env-file .env --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "bitsandbytes>=0.49.2",
#     "torch==2.8.0",
#     "torchvision>=0.23.0",
#     "transformers==5.2.0",
#     "triton>=3.4.0",
#     "trl==0.22.2",
#     "unsloth>=2026.3.3",
#     "unsloth-zoo>=2026.3.1",
#     "wandb>=0.25.0",
#     "xformers==0.0.32.post2",
#     "httpx>=0.27.0",
# ]
# ///
from unsloth import FastVisionModel
import subprocess
import signal
import tempfile
import torch
import time
import copy
import json
import os
import shutil

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen3.5-0.8B",
    load_in_4bit = False, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True, # False if not finetuning vision layers
    finetune_language_layers   = True, # False if not finetuning language layers
    finetune_attention_modules = True, # False if not finetuning attention layers
    finetune_mlp_modules       = True, # False if not finetuning MLP layers

    r = 16,           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 16,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
)

from datasets import load_dataset
dataset = load_dataset("unsloth/LaTeX_OCR", split = "train")
test_dataset = load_dataset("unsloth/LaTeX_OCR", split = "test")

def convert_to_conversation(sample):
    conversation = [
        { "role": "user",
          "content" : [
            {"type" : "text",  "text"  : "Write the LaTeX representation for this image."},
            {"type" : "image", "image" : sample["image"]} ]
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : sample["text"]} ]
        },
    ]
    return { "messages" : conversation }
converted_dataset = [convert_to_conversation(sample) for sample in dataset]
converted_test_dataset = [convert_to_conversation(sample) for sample in test_dataset]

from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback

FastVisionModel.for_training(model) # Enable for training!

import asyncio
import base64
import httpx

def pil_to_base64(img):
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

async def vllm_request(client, msg):
    image = None
    text = None
    for item in msg["content"]:
        if item["type"] == "text":
            text = item["text"]
        elif item["type"] == "image":
            image = pil_to_base64(item["image"])

    payload = {
        "model": "dummy",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image}"}
                },
            ],
        }],
        "max_tokens": 256,
    }

    r = await client.post("/v1/chat/completions", json=payload)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

async def run_parallel_inference(messages, concurrency=32):

    async with httpx.AsyncClient(
        base_url="http://localhost:8000",
        timeout=None,
    ) as client:

        sem = asyncio.Semaphore(concurrency)

        async def limited(msg):
            async with sem:
                return await vllm_request(client, msg)

        tasks = [asyncio.create_task(limited(m["messages"][0])) for m in messages]

        results = []
        with tqdm(total=len(tasks), desc="vLLM inference") as pbar:
            for fut in asyncio.as_completed(tasks):
                res = await fut
                results.append(res)
                pbar.update(1)

        return results

class PredictionCallback(TrainerCallback):
    def __init__(self):
        pass

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 50 == 1:

            model = kwargs["model"]
            tmpdir = tempfile.mkdtemp(prefix="tmp_model_ckpt")
            
            start_time = time.time()
            merged_model = copy.deepcopy(model)
            merged_model = merged_model.merge_and_unload()
            merged_model.save_pretrained(
                tmpdir,
                safe_serialization=True,
                max_shard_size="10GB",
            )
            tokenizer.save_pretrained(tmpdir)
            tc_path = os.path.join(tmpdir, "tokenizer_config.json")
            with open(tc_path) as f:
                tc = json.load(f)
            if tc.get("tokenizer_class") == "TokenizersBackend":
                tc["tokenizer_class"] = "Qwen2Tokenizer"
                with open(tc_path, "w") as f:
                    json.dump(tc, f, indent=2)
            print("Time taken to save model shards: ", time.time() - start_time, "s")

            start_time = time.time()
            proc = subprocess.Popen([
                "./serve_qwen35vl.py",
                "--model", tmpdir,
                "--port", "8000",
                "--gpu_memory_utilization", "0.3",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid)

            while True:
                try:
                    requests.get("http://localhost:8000/health")
                    break
                except:
                    time.sleep(1)
            print("Time taken for vLLM server to start: ", time.time() - start_time, "s")

            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    
            shutil.rmtree(tmpdir, ignore_errors=True)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer), # Must use!
    train_dataset = converted_dataset,
    eval_dataset = converted_test_dataset,
    callbacks=[PredictionCallback()],
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # max_steps = 30,
        num_train_epochs = 1, # Set this instead of max_steps for full training runs
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "wandb",     # For Weights and Biases
        # evals
        eval_strategy="no",
        eval_steps=-1,
        eval_on_start=False,

        # You MUST put the below items for vision finetuning:
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        max_length = 2048,
    ),
)

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer.can_return_loss = True
trainer_stats = trainer.train()

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
