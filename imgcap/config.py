"""Training configuration.

Config values are resolved in this order (highest priority last):
    1. dataclass defaults below
    2. a YAML config file, if `--config path/to.yaml` is given
    3. explicit CLI flags (e.g. `--lr 1e-4`)

So: CLI overrides YAML, YAML overrides the built-in defaults.
"""
from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, fields
from typing import Optional

import torch
import yaml


@dataclass
class TrainConfig:
    # ── Data ────────────────────────────────────────────────────────────
    images_dir: str = "data/flickr30k_images"
    captions_csv: str = "data/flickr30k_images/results.csv"
    # Where prepared artifacts (split csv + vocab) are cached. If these
    # already exist, `prepare_data` skips regenerating them.
    split_csv_path: str = "artifacts/flickr30_split.csv"
    vocab_stoi_path: str = "artifacts/vocab_stoi.pkl"
    vocab_itos_path: str = "artifacts/vocab_itos.pkl"

    # ── Split (must sum to 1.0) ────────────────────────────────────────
    train_split: float = 0.70
    val_split: float = 0.20
    test_split: float = 0.10

    # ── Sample cap (None = full dataset) ───────────────────────────────
    cap: Optional[int] = None  # set e.g. 1000 for quick debug runs
    num_workers: int = 4

    # ── Model ───────────────────────────────────────────────────────────
    feature_size: int = 2048
    lstm_hidden_size: int = 1024
    num_layers: int = 2
    embedding_dim: int = 1024
    attention_dim: int = 512
    dropout: float = 0.5

    # ── Optimisation ────────────────────────────────────────────────────
    batch_size: int = 128
    epochs: int = 100
    lr: float = 4e-4
    encoder_lr: float = 1e-5
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    teacher_forcing_ratio: float = 0.80
    grad_clip: float = 5.0

    # ── Warmup & Scheduler ──────────────────────────────────────────────
    warmup_epochs: int = 3            # encoder frozen for first N epochs
    lr_scheduler: str = "plateau"     # "cosine" | "plateau"

    # ── Stage 2: SCST fine-tuning ───────────────────────────────────────
    # None = cross-entropy only (equivalent to the old v7 notebook).
    # An int N = train with cross-entropy until epoch N, then switch to
    # SCST for the remaining epochs (equivalent to the v8 notebook).
    scst_start_epoch: Optional[int] = None

    # ── Eval ────────────────────────────────────────────────────────────
    eval_every: int = 1               # run val metrics every N epochs
    best_metric: str = "cider"        # "val_loss" | "bleu_1".."bleu_4" | "cider"

    # ── Checkpointing ───────────────────────────────────────────────────
    save_dir: str = "checkpoints"
    save_every: int = 5
    resume_checkpoint: Optional[str] = None      # path -> resume (weights+opt+sched+epoch)
    pretrained_checkpoint: Optional[str] = None  # path -> weights only, fresh training

    # ── Logging (optional integrations) ─────────────────────────────────
    use_wandb: bool = False
    wandb_project: str = "imgcap-flickr30k"
    wandb_run_name: str = "resnet50-lstm-attn"
    log_every: int = 1                # log batch metrics every N steps

    use_hf_hub: bool = False
    hf_repo_id: Optional[str] = None
    hf_private_repo: bool = False

    # ── Misc ────────────────────────────────────────────────────────────
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        assert abs(self.train_split + self.val_split + self.test_split - 1.0) < 1e-6, (
            "train_split + val_split + test_split must equal 1.0"
        )

    def summary(self) -> str:
        lines = [
            f"Device          : {self.device}",
            f"Epochs          : {self.epochs}  |  Batch size : {self.batch_size}",
            f"LR (decoder)    : {self.lr}  |  LR (encoder): {self.encoder_lr}",
            f"Grad clip       : {self.grad_clip}",
            f"Warmup epochs   : {self.warmup_epochs}",
            f"SCST start epoch: {self.scst_start_epoch or 'disabled (cross-entropy only)'}",
            f"Best metric     : {self.best_metric}",
            f"Resume ckpt     : {self.resume_checkpoint}",
            f"Pretrained ckpt : {self.pretrained_checkpoint}",
        ]
        return "\n".join(lines)


def _add_dataclass_args(parser: argparse.ArgumentParser) -> None:
    """Auto-generate `--field-name` CLI flags from TrainConfig's fields.

    Every flag defaults to argparse.SUPPRESS so that only flags the user
    actually passed show up in the parsed namespace — that's what lets us
    layer CLI > YAML > dataclass-defaults cleanly.
    """
    def _str2bool(v):
        return str(v).lower() in ("1", "true", "yes", "y")

    for f in fields(TrainConfig):
        flag = "--" + f.name.replace("_", "-")
        kwargs = {"dest": f.name, "default": argparse.SUPPRESS}

        # `from __future__ import annotations` makes f.type a plain string,
        # so match on substrings rather than the actual type objects.
        type_str = str(f.type)
        if "bool" in type_str:
            kwargs["type"] = _str2bool
        elif "int" in type_str:
            kwargs["type"] = int
        elif "float" in type_str:
            kwargs["type"] = float
        else:
            kwargs["type"] = str

        parser.add_argument(flag, **kwargs)


def build_arg_parser(description: str = "Image captioning") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a YAML config file. CLI flags override values in it.",
    )
    _add_dataclass_args(parser)
    return parser


def load_config(argv: Optional[list] = None,
                 parser: Optional[argparse.ArgumentParser] = None) -> TrainConfig:
    """Resolve a TrainConfig from defaults -> YAML file -> CLI flags."""
    parser = parser or build_arg_parser()
    args = parser.parse_args(argv)
    overrides = vars(args).copy()
    config_path = overrides.pop("config", None)

    values = {}
    if config_path:
        with open(config_path, "r") as fh:
            yaml_values = yaml.safe_load(fh) or {}
        valid_fields = {f.name for f in fields(TrainConfig)}
        unknown = set(yaml_values) - valid_fields
        if unknown:
            raise ValueError(f"Unknown key(s) in {config_path}: {sorted(unknown)}")
        values.update(yaml_values)

    # CLI flags (only ones explicitly passed, thanks to argparse.SUPPRESS)
    values.update(overrides)

    return TrainConfig(**values)


def save_config(cfg: TrainConfig, path: str) -> None:
    with open(path, "w") as fh:
        yaml.safe_dump(dataclasses.asdict(cfg), fh, sort_keys=False)
