#!/usr/bin/env -S PYTHONUNBUFFERED=1 uv run --env-file .env --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "fire>=0.7.1",
#   "torch>=2.0.0",
#   "transformers @ git+https://github.com/huggingface/transformers.git",
#   "trl>=0.29.0",
#   "peft>=0.18.0",
#   "accelerate>=1.0.0",
#   "datasets>=3.0.0",
#   "wandb",
#   "Pillow>=10.0.0",
#   "qwen-vl-utils>=0.0.14",
#   "causal-conv1d>=1.6.0",
# ]
# ///
import os
import fire
import torch
import wandb
from PIL import Image

SYSTEM_PROMPT = """You are an image painter. Given an image your task is to generate brush strokes to paint an image similar to it.
Each stroke has 8 parameters: x, y, width, height, rotation, R, G, B
- x, y: center position (0.0 to 1.0)
- width, height: stroke size (0.1 to 0.5)
- rotation: angle (0.0 to 1.0, where 1.0 = 180 degrees)
- R, G, B: color (0.0 to 1.0)

Output strokes one per line, as comma-separated values."""


def main(
    model_name: str = "Qwen/Qwen3.5-0.8B",
    dataset_id: str = "darshanmakwana/vlm_painter_sft",
    output_dir: str = "checkpoints/sft",
    epochs: int = 3,
    batch_size: int = 2,
    grad_accum: int = 8,
    lr: float = 1e-4,
    lora_r: int = 64,
    lora_alpha: int = 128,
    max_seq_len: int = 5000,
    device: str = "0",
    wandb_project: str = "vlm-painter",
    run_name: str = "sft-v1",
):
    os.environ["CUDA_VISIBLE_DEVICES"] = device

    from transformers import AutoProcessor, AutoModelForImageTextToText, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model
    from qwen_vl_utils import process_vision_info
    from datasets import load_dataset

    wandb.init(project=wandb_project, name=run_name, config={
        "model": model_name, "epochs": epochs, "lr": lr,
        "lora_r": lora_r, "batch_size": batch_size * grad_accum,
    })

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    processor.tokenizer.padding_side = "right"

    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda")

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    ds = load_dataset(dataset_id)
    train_split = ds["train"]
    test_split = ds["test"]

    def build_messages(example):
        user_prompt = f"Generate {example['num_strokes']} strokes to paint this image:"
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image", "image": example["image"]},
            ]},
            {"role": "assistant", "content": example["strokes"]},
        ]

    def preprocess(split):
        samples = []
        for example in split:
            msgs = build_messages(example)
            text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            img_inputs, _ = process_vision_info(msgs)

            inputs = processor(
                text=[text],
                images=img_inputs if img_inputs else None,
                padding=False,
                truncation=True,
                max_length=max_seq_len,
                return_tensors="pt",
            )

            input_ids = inputs["input_ids"].squeeze(0)
            labels = input_ids.clone()

            assistant_marker = "<|im_start|>assistant\n"
            marker_tokens = processor.tokenizer.encode(assistant_marker, add_special_tokens=False)
            marker_len = len(marker_tokens)
            ids_list = input_ids.tolist()

            for i in range(len(ids_list) - marker_len):
                if ids_list[i:i+marker_len] == marker_tokens:
                    labels[:i+marker_len] = -100
                    break

            sample = {"input_ids": input_ids, "labels": labels, "attention_mask": inputs["attention_mask"].squeeze(0)}
            if "pixel_values" in inputs:
                sample["pixel_values"] = inputs["pixel_values"].squeeze(0)
            if "image_grid_thw" in inputs:
                sample["image_grid_thw"] = inputs["image_grid_thw"].squeeze(0)
            samples.append(sample)
        return samples

    print("Preprocessing train data...")
    train_data = preprocess(train_split)
    print(max([sample["input_ids"].shape[-1] for sample in train_data]))
    print("Preprocessing test data...")
    val_data = preprocess(test_split)
    print(max([sample["input_ids"].shape[-1] for sample in val_data]))
    print(f"Train: {len(train_data)} | Val: {len(val_data)}")

    def collate_fn(batch):
        max_len = max(b["input_ids"].shape[0] for b in batch)
        pad_id = processor.tokenizer.pad_token_id or 0

        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

        for i, b in enumerate(batch):
            seq_len = b["input_ids"].shape[0]
            input_ids[i, :seq_len] = b["input_ids"]
            labels[i, :seq_len] = b["labels"]
            attention_mask[i, :seq_len] = b["attention_mask"]

        result = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

        if "pixel_values" in batch[0]:
            result["pixel_values"] = torch.cat([b["pixel_values"] for b in batch], dim=0)
        if "image_grid_thw" in batch[0]:
            result["image_grid_thw"] = torch.cat([b["image_grid_thw"].unsqueeze(0) if b["image_grid_thw"].dim() == 1 else b["image_grid_thw"] for b in batch], dim=0)

        return result

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        bf16=True,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        report_to="wandb",
        remove_unused_columns=False,
        dataloader_num_workers=8,
        gradient_checkpointing=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=collate_fn,
    )

    trainer.train()
    trainer.save_model(f"{output_dir}/final")
    processor.save_pretrained(f"{output_dir}/final")
    wandb.finish()
    print(f"\nSFT training complete! Model saved to {output_dir}/final")

if __name__ == "__main__":
    fire.Fire(main)
