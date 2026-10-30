"""LoRA fine-tuning of a chat model on assistant dialogs.

The loop is plain PyTorch on purpose: every step (batching, loss, accumulation, schedule,
evaluation, saving) is visible here, with no trainer framework in between.

  python -m aurora_assistant.train --config configs/assistant/tiny-qwen.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .format import tokenize_for_training


@dataclass
class TrainConfig:
    base_model: str                              # Hugging Face id or local path
    out_dir: str
    train_file: str = "data/assistant/train.jsonl"
    eval_file: str = "data/assistant/eval.jsonl"
    extra_train_files: list[str] = field(default_factory=lambda: ["data/assistant/feedback.jsonl"])  # corrections from the chat
    trust_remote_code: bool = False              # Kimi / Moonlight models ship their own model code
    dtype: str = "bfloat16"                      # bfloat16 on modern GPUs, float32 on CPU
    max_len: int = 2048
    # LoRA: small trainable matrices next to frozen weights. r = their rank (capacity).
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] | str = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    epochs: int = 2
    batch_size: int = 4
    grad_accum: int = 4
    lr: float = 2e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    eval_every: int = 50                         # optimizer steps
    max_eval_dialogs: int = 64
    gradient_checkpointing: bool = True          # trades compute for memory on big models
    seed: int = 0

    @classmethod
    def load(cls, path: str) -> "TrainConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**data)


def device_of(dtype: str) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if dtype == "bfloat16":
        print("No GPU found: training in float32 on CPU")
    return "cpu"


def load_dialogs(path: str, tokenizer, max_len: int) -> list[tuple[list[int], list[int]]]:
    """JSONL with {"messages": [...]} -> (input ids, labels). Too long dialogs are skipped, and counted."""
    out, skipped = [], 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        example = tokenize_for_training(tokenizer, json.loads(line)["messages"], max_len)
        if example is None:
            skipped += 1
        else:
            out.append(example)
    print(f"{path}: {len(out)} dialogs" + (f", {skipped} skipped as longer than {max_len} tokens" if skipped else ""))
    return out


def collate(batch, pad_id: int, device: str) -> dict[str, torch.Tensor]:
    """Right-pad to the longest dialog in the batch. Padding never contributes to the loss (label -100)."""
    width = max(len(ids) for ids, _ in batch)
    input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    attention = torch.zeros((len(batch), width), dtype=torch.long)
    for i, (ids, lab) in enumerate(batch):
        input_ids[i, :len(ids)] = torch.tensor(ids)
        labels[i, :len(lab)] = torch.tensor(lab)
        attention[i, :len(ids)] = 1
    return {"input_ids": input_ids.to(device), "labels": labels.to(device), "attention_mask": attention.to(device)}


def masked_loss(model, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, int]:
    """Next-token loss on the tokens the model must learn, and how many there are.

    The usual way multiplies every position by the whole vocabulary (150k+ columns) and only then
    drops the masked ones. Here the decoder runs as usual, but the output layer is applied only to the
    positions that count, typically 5-15% of the dialog. Same loss, a fraction of the memory.
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    hidden = base.get_decoder()(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
    targets = batch["labels"][:, 1:]               # position t predicts token t+1
    keep = targets != -100
    logits = base.get_output_embeddings()(hidden[:, :-1][keep]).float()
    return F.cross_entropy(logits, targets[keep]), int(keep.sum())


def lr_factor(step: int, warmup: int, total: int) -> float:
    """Linear warmup, then cosine decay to 10% of the peak."""
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def eval_loss(model, data, cfg: TrainConfig, pad_id: int, device: str) -> float:
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(data), cfg.batch_size):
        loss, n = masked_loss(model, collate(data[i:i + cfg.batch_size], pad_id, device))
        total += loss.item() * n
        count += n
    model.train()
    return total / max(1, count)


def load_model(cfg: TrainConfig, device: str):
    dtype = torch.bfloat16 if cfg.dtype == "bfloat16" and device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=cfg.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, dtype=dtype, trust_remote_code=cfg.trust_remote_code)
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()  # lets gradients reach LoRA layers through checkpointed blocks
    lora = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                      target_modules=cfg.target_modules, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"{cfg.base_model}: {total / 1e6:.0f}M parameters, training {trainable / 1e6:.2f}M ({trainable / total:.2%}) in LoRA adapters")
    return model, tokenizer


def train(cfg: TrainConfig) -> dict:
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = device_of(cfg.dtype)
    model, tokenizer = load_model(cfg, device)
    train_data = load_dialogs(cfg.train_file, tokenizer, cfg.max_len)
    for extra in cfg.extra_train_files:
        if Path(extra).exists():
            train_data += load_dialogs(extra, tokenizer, cfg.max_len)
    eval_data = load_dialogs(cfg.eval_file, tokenizer, cfg.max_len)[: cfg.max_eval_dialogs]

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(train_data) / (cfg.batch_size * cfg.grad_accum))
    total_steps = steps_per_epoch * cfg.epochs
    warmup = max(1, int(total_steps * cfg.warmup_ratio))
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_config.json").write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")
    log = (out / "log.jsonl").open("a", encoding="utf-8")

    def record(**rec):
        log.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log.flush()
        print("  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True)

    best = eval_loss(model, eval_data, cfg, tokenizer.pad_token_id, device)
    record(step=0, eval_loss=best)
    step, t0 = 0, time.time()
    model.train()
    for epoch in range(cfg.epochs):
        order = list(range(len(train_data)))
        random.shuffle(order)
        batches = [order[i:i + cfg.batch_size] for i in range(0, len(order), cfg.batch_size)]
        for b, idx in enumerate(batches):
            batch = collate([train_data[i] for i in idx], tokenizer.pad_token_id, device)
            loss = masked_loss(model, batch)[0] / cfg.grad_accum
            loss.backward()
            if (b + 1) % cfg.grad_accum and b + 1 < len(batches):
                continue  # keep accumulating gradients
            for g in optimizer.param_groups:
                g["lr"] = cfg.lr * lr_factor(step, warmup, total_steps)
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % 5 == 0:
                record(step=step, epoch=epoch, loss=loss.item() * cfg.grad_accum, lr=optimizer.param_groups[0]["lr"],
                       elapsed_s=round(time.time() - t0))
            if step % cfg.eval_every == 0 or step == total_steps:
                current = eval_loss(model, eval_data, cfg, tokenizer.pad_token_id, device)
                record(step=step, eval_loss=current)
                if current < best:
                    best = current
                    model.save_pretrained(out / "adapter")       # only the LoRA weights: megabytes, not gigabytes
                    tokenizer.save_pretrained(out / "adapter")
    log.close()
    if not (out / "adapter").exists():  # never improved: still save, so the run is inspectable
        model.save_pretrained(out / "adapter")
        tokenizer.save_pretrained(out / "adapter")
    return {"steps": step, "best_eval_loss": best, "adapter": str(out / "adapter")}


def main():
    ap = argparse.ArgumentParser(description="LoRA fine-tuning on assistant dialogs")
    ap.add_argument("--config", required=True)
    print(train(TrainConfig.load(ap.parse_args().config)))


if __name__ == "__main__":
    main()
