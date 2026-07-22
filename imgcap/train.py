"""Main training entrypoint.

Runs cross-entropy training and, if `cfg.scst_start_epoch` is set, switches
to SCST fine-tuning from that epoch onward -- all in one continuous loop
with shared checkpointing/scheduling/eval.

Usage:
    python -m imgcap.train --config configs/default.yaml
    python -m imgcap.train --config configs/default.yaml --lr 1e-4 --epochs 50
    python -m imgcap.train --images-dir /data/flickr30k --captions-csv /data/results.csv
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from imgcap.config import TrainConfig, load_config
from imgcap.data.loaders import build_loaders
from imgcap.engine.evaluate import eval_one_epoch, extract_metric, is_better
from imgcap.engine.train_scst import train_one_epoch_scst
from imgcap.engine.train_xent import train_one_epoch
from imgcap.integrations.hf_hub import build_uploader
from imgcap.integrations.wandb_logger import build_logger
from imgcap.models.imgcap import ImgCap
from imgcap.utils.checkpoint import load_checkpoint, save_checkpoint
from imgcap.utils.references import collect_references_per_image
from imgcap.utils.seed import seed_everything


def build_model_optimizer_scheduler(cfg: TrainConfig, vocab_size: int, device: torch.device):
    model = ImgCap(
        feature_size=cfg.feature_size,
        lstm_hidden_size=cfg.lstm_hidden_size,
        num_layers=cfg.num_layers,
        vocab_size=vocab_size,
        embedding_dim=cfg.embedding_dim,
        attention_dim=cfg.attention_dim,
        dropout=cfg.dropout,
    ).to(device)

    # Encoder starts fully frozen (see ResNet50.__init__); unfreeze the
    # later conv blocks so the backbone can adapt slightly to captioning.
    for name, param in model.encoder.ResNet50.named_parameters():
        param.requires_grad = "layer3" in name or "layer4" in name

    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.encoder.named_parameters() if p.requires_grad],
         "lr": cfg.encoder_lr},
        {"params": model.decoder.parameters(), "lr": cfg.lr},
    ], weight_decay=cfg.weight_decay)

    if cfg.lr_scheduler == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer, T_max=cfg.epochs - cfg.warmup_epochs, eta_min=cfg.lr * 0.01,
        )
    else:
        scheduler = ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3,  # watching cider
        )

    return model, optimizer, scheduler


def main(cfg: TrainConfig | None = None) -> None:
    cfg = cfg or load_config()
    print(cfg.summary())

    seed_everything(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    device = torch.device(cfg.device)

    uploader = build_uploader(cfg)
    on_artifact_saved = (lambda path, name: uploader.upload(path, name)) if uploader else None

    train_loader, val_loader, test_loader, vocab_size, pad_idx, train_ds = build_loaders(
        cfg, on_artifact_saved=on_artifact_saved,
    )

    model, optimizer, scheduler = build_model_optimizer_scheduler(cfg, vocab_size, device)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=cfg.label_smoothing)
    scaler = GradScaler()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params    : {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    start_epoch = 1
    best_score = None

    if cfg.pretrained_checkpoint and os.path.isfile(cfg.pretrained_checkpoint):
        print(f"\nLoading pretrained weights: {cfg.pretrained_checkpoint}")
        load_checkpoint(cfg.pretrained_checkpoint, model, resume=False, device=cfg.device)

    if cfg.resume_checkpoint and os.path.isfile(cfg.resume_checkpoint):
        print(f"\nResuming from checkpoint: {cfg.resume_checkpoint}")
        start_epoch, best_score = load_checkpoint(
            cfg.resume_checkpoint, model, optimizer, scheduler,
            resume=True, device=cfg.device,
        )

    logger = build_logger(cfg, model=model, criterion=criterion)
    global_step = 0

    # Pre-build the training-set reference dict once if we'll need it for SCST.
    train_refs_dict = None
    if cfg.scst_start_epoch is not None:
        train_refs_dict = collect_references_per_image(cfg.split_csv_path, "train", train_ds)

    for epoch in range(start_epoch, cfg.epochs + 1):
        tf_ratio = max(
            0.50,
            cfg.teacher_forcing_ratio - (epoch - 1) / max(cfg.epochs - 1, 1) * 0.30,
        )

        print(f"\n{'=' * 60}")
        print(f"  Epoch {epoch}/{cfg.epochs}")
        print(f"{'=' * 60}")

        current_lr = optimizer.param_groups[0]["lr"]
        use_scst = cfg.scst_start_epoch is not None and epoch >= cfg.scst_start_epoch

        if not use_scst:
            train_loss, train_acc, global_step = train_one_epoch(
                model, train_loader, optimizer, criterion, scaler, device,
                epoch, cfg, global_step, tf_ratio, pad_idx, logger=logger,
            )
            print(f"  Train  -- loss: {train_loss:.4f}  acc: {train_acc:.4f}  lr: {current_lr:.6f}")
            log_dict = {"train/epoch_loss": train_loss, "train/epoch_acc": train_acc,
                        "train/lr": current_lr, "epoch": epoch}
        else:
            train_loss, global_step = train_one_epoch_scst(
                model, train_loader, optimizer, scaler, device,
                epoch, cfg, global_step, train_ds, cfg.split_csv_path,
                refs_dict=train_refs_dict, logger=logger,
            )
            print(f"  Train  -- loss: {train_loss:.4f}  lr: {current_lr:.6f}  [SCST]")
            log_dict = {"train/epoch_loss": train_loss, "train/lr": current_lr, "epoch": epoch}

        if logger is not None:
            logger.log(log_dict, step=global_step)

        if isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        # ReduceLROnPlateau is stepped after validation below.

        if epoch % cfg.eval_every == 0:
            val_loss, val_acc, bleu_scores, cider_score, refs, cands = eval_one_epoch(
                model, val_loader, criterion, device, train_ds,
                cfg, global_step, cfg.split_csv_path, split="val", logger=logger,
            )
            print(f"  Val    -- loss: {val_loss:.4f}  acc: {val_acc:.4f}")
            print(f"  BLEU   -- 1: {bleu_scores['bleu_1']:.4f}  2: {bleu_scores['bleu_2']:.4f}"
                  f"  3: {bleu_scores['bleu_3']:.4f}  4: {bleu_scores['bleu_4']:.4f}")
            print(f"  CIDEr  -- {cider_score:.4f}")

            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(cider_score)

            current_score = extract_metric(val_loss, bleu_scores, cider_score, cfg.best_metric)
            if is_better(current_score, best_score, cfg.best_metric):
                best_score = current_score
                best_path = os.path.join(cfg.save_dir, "best_model.pt")
                save_checkpoint(best_path, epoch, model, optimizer, scheduler, best_score, cfg)
                if uploader:
                    uploader.upload(best_path, "best_model.pt")
                if logger is not None:
                    logger.log({"val/best_" + cfg.best_metric: best_score}, step=global_step)
                print(f"  \u2605 New best {cfg.best_metric}: {best_score:.4f}")

        if epoch % cfg.save_every == 0:
            periodic_path = os.path.join(cfg.save_dir, f"epoch_{epoch:03d}.pt")
            save_checkpoint(periodic_path, epoch, model, optimizer, scheduler, best_score, cfg)
            if uploader:
                uploader.upload(periodic_path, f"epoch_{epoch:03d}.pt")

    print(f"\n{'=' * 60}")
    print("  Training complete!")
    print(f"  Best {cfg.best_metric}: {best_score}")
    print(f"{'=' * 60}")

    if logger is not None:
        logger.finish()


if __name__ == "__main__":
    main()
