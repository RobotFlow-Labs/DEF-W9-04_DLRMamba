from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tomllib


@dataclass(slots=True)
class ModelConfig:
    num_classes: int = 8
    in_channels: int = 3
    fusion_channels: int = 32
    embed_dim: int = 64
    num_blocks: int = 3
    rank_ratio: float = 0.5
    state_dim: int = 64


@dataclass(slots=True)
class TrainConfig:
    lr: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 5e-4
    batch_size: int = 8
    epochs: int = 300
    lambda_task: float = 1.0
    lambda_svd: float = 0.5
    lambda_state: float = 0.1
    lambda_feat: float = 1.5
    precision: str = "bf16"
    gradient_clip: float = 1.0
    warmup_ratio: float = 0.05
    min_lr: float = 1e-6
    seed: int = 42
    log_interval: int = 10
    val_interval: int = 1
    resume: str = ""


@dataclass(slots=True)
class DataConfig:
    image_size: int = 640
    num_workers: int = 4
    pin_memory: bool = True
    train_root: str = ""
    val_root: str = ""
    test_root: str = ""


@dataclass(slots=True)
class CheckpointConfig:
    output_dir: str = "/mnt/artifacts-datai/checkpoints/dlrmamba"
    log_dir: str = "/mnt/artifacts-datai/logs/dlrmamba"
    tensorboard_dir: str = "/mnt/artifacts-datai/tensorboard/dlrmamba"
    save_every_n_epochs: int = 10
    keep_top_k: int = 2
    metric: str = "val_loss"
    mode: str = "min"


@dataclass(slots=True)
class EarlyStoppingConfig:
    enabled: bool = True
    patience: int = 20
    min_delta: float = 1e-4


@dataclass(slots=True)
class InferConfig:
    conf_threshold: float = 0.25
    topk: int = 300


@dataclass(slots=True)
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    infer: InferConfig = field(default_factory=InferConfig)


def _load_dataclass(dc_cls, payload: dict) -> object:
    kwargs = {k: v for k, v in payload.items() if k in dc_cls.__dataclass_fields__}
    return dc_cls(**kwargs)


def load_config(path: str | Path) -> AppConfig:
    p = Path(path)
    with p.open("rb") as f:
        payload = tomllib.load(f)

    return AppConfig(
        model=_load_dataclass(ModelConfig, payload.get("model", {})),
        train=_load_dataclass(TrainConfig, payload.get("train", {})),
        data=_load_dataclass(DataConfig, payload.get("data", {})),
        checkpoint=_load_dataclass(CheckpointConfig, payload.get("checkpoint", {})),
        early_stopping=_load_dataclass(EarlyStoppingConfig, payload.get("early_stopping", {})),
        infer=_load_dataclass(InferConfig, payload.get("infer", {})),
    )
