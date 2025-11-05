"""Poor man's configuration system."""

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal
import warnings

import torch
import yaml

import ttlm.model
import ttlm.tokenizer.ascii
import ttlm.tokenizer.bpe

HIDDEN_DIM_DIVISOR = 64
FF_DIM_MULTIPLIER = 4


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""

    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = False
    shuffle: bool = True


@dataclass
class TokenizerConfig:
    """Configuration for tokenizer."""

    module: type[ttlm.tokenizer.base.Tokenizer] = ttlm.tokenizer.bpe.BPETokenizer
    # module: type[ttlm.tokenizer.base.Tokenizer] = ttlm.tokenizer.ascii.AsciiTokenizer


@dataclass
class ModelConfig:
    """Configuration for the model architecture."""

    module: type[torch.nn.Module] = ttlm.model.Model
    hidden_dim: int = 128
    num_layers: int | None = None
    num_heads: int | None = None
    ff_dim: int | None = None
    dropout: float = 0.1
    num_parameters: int | None = None


@dataclass
class OptimizerConfig:
    """Configuration for the optimizer."""

    name: Literal["adamw"] = "adamw"
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    """Configuration for the learning rate scheduler."""

    warmup_steps_ratio: float = 0.01
    min_lr_ratio: float = 0.1
    num_cycles: float = 0.5


@dataclass
class PreTrainingConfig:
    """Top-level configuration for a training run."""

    experiment: str = "default"
    ckpt_path: str | None = None

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)

    epochs: int = 30
    device: Literal["cuda", "cpu", "mps"] = "mps"
    dtype: torch.dtype = torch.float32
    max_steps: int | float = float("inf")
    val_check_interval: int = 2048
    max_flops: int | None = None

    def __post_init__(self) -> None:
        """Validate and setup."""
        if self.ckpt_path is None:
            self.ckpt_path = f"logs/pretrain/{self.experiment}"
        if self.model.num_layers is None:
            self.model.num_layers = max(
                2, self.model.hidden_dim // HIDDEN_DIM_DIVISOR)
        if self.model.num_heads is None:
            self.model.num_heads = max(
                1, self.model.hidden_dim // HIDDEN_DIM_DIVISOR)
            while (
                self.model.hidden_dim % self.model.num_heads != 0
                and self.model.num_heads > 1
            ):
                self.model.num_heads -= 1
        if self.model.ff_dim is None:
            self.model.ff_dim = self.model.hidden_dim * FF_DIM_MULTIPLIER

        os.makedirs(self.ckpt_path, exist_ok=True)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        data = self.to_dict()
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        data = asdict(self)
        data["dtype"] = str(
            self.dtype
        )  # torch type is too annoying to serialize automatically
        return data

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PreTrainingConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.unsafe_load(f)
        data["data"] = DataConfig(**data["data"])
        data["model"] = ModelConfig(**data["model"])
        data["optimizer"] = OptimizerConfig(**data["optimizer"])
        data["scheduler"] = SchedulerConfig(**data["scheduler"])
        data["tokenizer"] = TokenizerConfig(**data["tokenizer"])
        if "dtype" in data and isinstance(data["dtype"], str):
            dtype_str = data["dtype"].replace("torch.", "")
            data["dtype"] = getattr(torch, dtype_str)
        return cls(**data)
