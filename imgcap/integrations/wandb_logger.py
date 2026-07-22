"""Thin optional W&B wrapper.

Every call site in the engine/train code accepts `logger=None` and simply
skips logging when it is. This wrapper exists so `train.py` only has to
construct one object and pass it around, instead of littering `if cfg.use_wandb`
checks through the training loops.
"""
import os
from dataclasses import asdict


class WandbLogger:
    def __init__(self, cfg, model=None, criterion=None):
        import wandb  # imported lazily so wandb isn't a hard dependency

        self._wandb = wandb

        api_key = os.environ.get("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key)

        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            config=asdict(cfg),
            tags=["resnet50", "lstm", "attention", "flickr30k"],
            resume="allow",
        )
        if model is not None and criterion is not None:
            wandb.watch(model, criterion, log="gradients", log_freq=cfg.log_every)

        print("W&B run :", wandb.run.name)
        print("Run URL :", wandb.run.url)

    def log(self, metrics: dict, step: int) -> None:
        self._wandb.log(metrics, step=step)

    def finish(self) -> None:
        self._wandb.finish()


def build_logger(cfg, model=None, criterion=None):
    """Returns a WandbLogger if cfg.use_wandb, else None.

    Downstream code just needs to check `logger is not None`.
    """
    if not cfg.use_wandb:
        return None
    return WandbLogger(cfg, model=model, criterion=criterion)
