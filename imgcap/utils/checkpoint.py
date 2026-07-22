import os
from dataclasses import asdict
from typing import Optional, Tuple

import torch


def save_checkpoint(path, epoch, model, optimizer, scheduler, best_score, cfg) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_score": best_score,
        "cfg": asdict(cfg),
    }, path)
    print(f"  \u2713 Checkpoint saved -> {path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None,
                     resume: bool = True, device: str = "cpu") -> Tuple[int, Optional[float]]:
    """
    resume=True  -> restore model + optimizer + scheduler + epoch (continue training)
    resume=False -> restore model weights only (pretrained transfer, fresh training)
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"  \u2713 Model weights loaded from {path}")

    start_epoch = 1
    best_score = None

    if resume:
        if optimizer is not None and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            print("  \u2713 Optimizer state restored")
        if scheduler is not None and "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
            print("  \u2713 Scheduler state restored")
        start_epoch = ckpt.get("epoch", 0) + 1
        best_score = ckpt.get("best_score", None)
        print(f"  \u2713 Resuming from epoch {start_epoch}")

    return start_epoch, best_score
