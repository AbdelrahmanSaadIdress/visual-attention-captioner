import torch.nn as nn
from torch.cuda.amp import autocast
from tqdm import tqdm

from imgcap.metrics.eval_metrics import token_accuracy


def train_one_epoch(model, loader, optimizer, criterion, scaler, device,
                     epoch, cfg, global_step, tf_ratio, pad_idx, logger=None):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"[Train] Epoch {epoch}", leave=False)

    for batch_idx, (images, captions, _) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        captions = captions.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast():
            logits = model(images, captions, tf_ratio)  # (B, T, V)

            # Shift: predict token t+1 from token t.
            logits_ = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            targets_ = captions[:, 1:].contiguous().view(-1)
            loss = criterion(logits_, targets_)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        acc = token_accuracy(logits[:, :-1, :], captions[:, 1:], pad_idx)
        total_loss += loss.item()
        total_acc += acc
        n_batches += 1
        global_step += 1

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.4f}")

        if logger is not None and batch_idx % cfg.log_every == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            logger.log({
                "train/batch_loss": loss.item(),
                "train/batch_acc": acc,
                "train/lr": current_lr,
            }, step=global_step)

    avg_loss = total_loss / n_batches
    avg_acc = total_acc / n_batches
    return avg_loss, avg_acc, global_step
