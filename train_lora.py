#!/usr/bin/env python3
"""LoRA SFT for Qwen2.5-1.5B — optimized for Kaggle A100 (bf16) with CPU fallback."""

from __future__ import annotations

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Kaggle T4 x2: tránh hang DDP / multi-GPU với Trainer đơn giản
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_lora")

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def messages_to_text(tokenizer, messages: List[Dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def build_dataset(tokenizer, rows: List[Dict[str, Any]], max_length: int) -> Dataset:
    def _map(example):
        text = messages_to_text(tokenizer, example["messages"])
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    ds = Dataset.from_list(rows)
    return ds.map(_map, remove_columns=ds.column_names, desc="Tokenize")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "train.jsonl")
    parser.add_argument("--val", type=Path, default=ROOT / "data" / "val.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "lora")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=0, help="0 = auto (4 on A100, 1 on CPU)")
    parser.add_argument("--grad-accum", type=int, default=0, help="0 = auto")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-train-samples", type=int, default=0, help="0 = all")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    args = parser.parse_args()

    if not args.train.exists():
        raise SystemExit(f"Missing {args.train}. Run: python generate_dataset.py --n 1600")

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    # A100 prefers bf16; other GPUs use fp16
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = use_cuda and not use_bf16

    if args.batch_size <= 0:
        args.batch_size = 4 if use_cuda else 1
    if args.grad_accum <= 0:
        args.grad_accum = 4 if use_cuda else 8

    logger.info(
        "Device=%s cuda=%s bf16=%s fp16=%s batch=%s accum=%s model=%s",
        device,
        use_cuda,
        use_bf16,
        use_fp16,
        args.batch_size,
        args.grad_accum,
        args.model,
    )
    if use_cuda:
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)
    load_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    # device_map="auto" can confuse PEFT on some Kaggle stacks — use explicit .cuda()
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    if use_cuda:
        model = model.cuda()
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_rows = load_jsonl(args.train)
    val_rows = load_jsonl(args.val) if args.val.exists() else []
    if args.max_train_samples and args.max_train_samples > 0:
        train_rows = train_rows[: args.max_train_samples]

    train_ds = build_dataset(tokenizer, train_rows, args.max_seq_length)
    eval_ds = build_dataset(tokenizer, val_rows, args.max_seq_length) if val_rows else None

    args.out.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.out / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.03,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=args.save_steps if eval_ds is not None else None,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=True,
        report_to=[],
        dataloader_num_workers=0,
        remove_unused_columns=False,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        ddp_find_unused_parameters=False,
    )

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, return_tensors="pt")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    logger.info("Starting training (%d samples)...", len(train_rows))
    trainer.train()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    meta = {
        "base_model": args.model,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "train_samples": len(train_rows),
        "max_seq_length": args.max_seq_length,
        "device": device,
        "bf16": use_bf16,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
    }
    (args.out / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Saved LoRA adapter → %s", args.out)


if __name__ == "__main__":
    main()
