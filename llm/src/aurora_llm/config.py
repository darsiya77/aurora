"""Configuration: one YAML file describes the model, the data and the training run."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4          # fewer K/V heads than query heads = grouped-query attention
    ffn_mult: float = 2.6667     # SwiGLU hidden size = ffn_mult * d_model, rounded to 64
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    dropout: float = 0.0
    tie_embeddings: bool = True

    def __post_init__(self):
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def ffn_dim(self) -> int:
        return int(round(self.ffn_mult * self.d_model / 64)) * 64


@dataclass
class DataConfig:
    processed_dir: str = "data/processed"
    tokenizer_path: str = "data/tokenizer/tokenizer.json"


@dataclass
class TrainConfig:
    out_dir: str = "runs/default"
    seed: int = 1337
    batch_size: int = 8              # sequences per micro-batch per device
    grad_accum: int = 1              # micro-batches per optimizer step
    max_steps: int = 1000
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_every: int = 200
    eval_batches: int = 20
    log_every: int = 10
    save_every: int = 500
    dtype: str = "auto"              # auto | float32 | bfloat16 | float16
    compile: bool = False
    device: str = "auto"             # auto | cpu | cuda | mps


@dataclass
class SFTConfig:
    base_checkpoint: str = "runs/default/last.pt"
    train_file: str = "data/sft/train.jsonl"
    val_file: str = ""
    out_dir: str = "runs/sft"
    epochs: int = 3
    batch_size: int = 8
    lr: float = 5e-5
    warmup_steps: int = 20
    max_seq_len: int = 0             # 0 = use the model's max_seq_len


@dataclass
class Config:
    name: str = "default"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def _build(cls, values: dict | None):
    values = values or {}
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**values)


def config_from_dict(d: dict) -> Config:
    return Config(
        name=d.get("name", "default"),
        model=_build(ModelConfig, d.get("model")),
        data=_build(DataConfig, d.get("data")),
        train=_build(TrainConfig, d.get("train")),
        sft=_build(SFTConfig, d.get("sft")),
    )


def load_config(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as f:
        return config_from_dict(yaml.safe_load(f) or {})
