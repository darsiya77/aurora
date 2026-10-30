"""Pretraining: next-token prediction on the processed corpus.

Single device:   python -m aurora_llm.train --config configs/tiny.yaml
Several GPUs:    torchrun --nproc_per_node=8 -m aurora_llm.train --config configs/base.yaml
Resume:          add --resume (continues from <out_dir>/last.pt)
"""
from __future__ import annotations

import argparse
import math
import os
import time

import torch
import torch.distributed as dist

from .config import Config, load_config
from .data.dataset import TokenStream
from .model import GPT
from .train_utils import Logger, autocast, load_checkpoint, lr_at, pick_device, pick_dtype, save_checkpoint


@torch.no_grad()
def evaluate(model, stream: TokenStream, cfg: Config, device: str, dtype, batches: int) -> float:
    model.eval()
    gen = torch.Generator().manual_seed(0)  # the same windows every time, so numbers are comparable
    losses = []
    for _ in range(batches):
        x, y = stream.batch(cfg.train.batch_size, cfg.model.max_seq_len, device, gen)
        with autocast(device, dtype):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train(cfg: Config, resume: bool = False) -> dict:
    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if ddp:
        dist.init_process_group(backend="nccl")
        rank, local_rank, world = dist.get_rank(), int(os.environ["LOCAL_RANK"]), dist.get_world_size()
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        rank, world = 0, 1
        device = pick_device(cfg.train.device)
    master = rank == 0
    t = cfg.train
    torch.manual_seed(t.seed + rank)
    dtype = pick_dtype(t.dtype, device.split(":")[0])
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_data = TokenStream(cfg.data.processed_dir, "train")
    val_data = TokenStream(cfg.data.processed_dir, "val")
    if train_data.meta["vocab_size"] != cfg.model.vocab_size:
        raise ValueError(f"Tokenizer has {train_data.meta['vocab_size']} tokens, config says {cfg.model.vocab_size}")

    step, best = 0, math.inf
    ckpt_path = os.path.join(t.out_dir, "last.pt")
    if resume and os.path.exists(ckpt_path):
        model, _, state = load_checkpoint(ckpt_path, device)
        step, best = state["step"], state["extra"].get("best_val", math.inf)
    else:
        state = None
        model = GPT(cfg.model).to(device)
    optimizer = model.optimizer(t.lr, t.weight_decay, (t.beta1, t.beta2), device.split(":")[0])
    if state and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    scaler = torch.amp.GradScaler(enabled=dtype == torch.float16)
    raw_model = model
    if t.compile:
        model = torch.compile(model)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    log = Logger(t.out_dir, enabled=master)
    tokens_per_step = t.batch_size * t.grad_accum * cfg.model.max_seq_len * world
    if master:
        print(f"model: {raw_model.num_params() / 1e6:.2f}M parameters (without embeddings), "
              f"{raw_model.num_params(False) / 1e6:.2f}M total")
        print(f"device={device} dtype={dtype} tokens/step={tokens_per_step:,} "
              f"plan={t.max_steps * tokens_per_step / 1e6:.1f}M tokens, corpus={len(train_data) / 1e6:.1f}M tokens")

    model.train()
    x, y = train_data.batch(t.batch_size, cfg.model.max_seq_len, device)
    t_last = time.time()
    while step < t.max_steps:
        lr = lr_at(step, t.lr, t.min_lr, t.warmup_steps, t.max_steps)
        for g in optimizer.param_groups:
            g["lr"] = lr
        loss_sum = 0.0
        for micro in range(t.grad_accum):
            if ddp:
                model.require_backward_grad_sync = micro == t.grad_accum - 1
            with autocast(device, dtype):
                _, loss = model(x, y)
                loss = loss / t.grad_accum
            x, y = train_data.batch(t.batch_size, cfg.model.max_seq_len, device)  # prefetch the next batch
            scaler.scale(loss).backward()
            loss_sum += loss.item()
        scaler.unscale_(optimizer)
        gnorm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), t.grad_clip).item()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if master and step % t.log_every == 0:
            dt = time.time() - t_last
            t_last = time.time()
            log.log(step=step, loss=loss_sum, lr=lr, grad_norm=gnorm, tok_per_s=round(tokens_per_step * t.log_every / dt))
        if master and (step % t.eval_every == 0 or step == t.max_steps):
            val = evaluate(raw_model, val_data, cfg, device, dtype, t.eval_batches)
            log.log(step=step, val_loss=val, val_ppl=math.exp(min(val, 20)))
            if val < best:
                best = val
                save_checkpoint(os.path.join(t.out_dir, "best.pt"), raw_model, cfg, step, extra={"best_val": best})
        if master and (step % t.save_every == 0 or step == t.max_steps):
            save_checkpoint(ckpt_path, raw_model, cfg, step, optimizer, extra={"best_val": best})

    if ddp:
        dist.destroy_process_group()
    return {"step": step, "best_val": best}


def main():
    ap = argparse.ArgumentParser(description="Pretrain the model")
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-steps", type=int, help="override train.max_steps")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.max_steps:
        cfg.train.max_steps = a.max_steps
    print(train(cfg, a.resume))


if __name__ == "__main__":
    main()
