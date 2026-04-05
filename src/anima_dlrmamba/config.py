from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(slots=True)
class DataConfig:
    image_size: int = 640
    num_workers: int = 2
    train_root: str = ""
    val_root: str = ""


@dataclass(slots=True)
class InferConfig:
    conf_threshold: float = 0.25
    topk: int = 300


@dataclass(slots=True)
class AppConfig:
    model: ModelConfig
    train: TrainConfig
    data: DataConfig
    infer: InferConfig


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
        infer=_load_dataclass(InferConfig, payload.get("infer", {})),
    )
