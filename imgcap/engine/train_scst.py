import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from tqdm import tqdm

from imgcap.losses.scst import scst_loss
from imgcap.utils.references import collect_references_per_image


def train_one_epoch_scst(model, loader, optimizer, scaler, device,
                          epoch, cfg, global_step, vocab, split_csv_path,
                          refs_dict=None, logger=None):
    """One epoch of SCST fine-tuning.

    `refs_dict` (image_id -> list of reference token lists) can be passed
    in to avoid recomputing it every epoch; if omitted it's rebuilt here.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    if refs_dict is None:
        refs_dict = collect_references_per_image(split_csv_path, "train", vocab)

    pbar = tqdm(loader, desc=f"[SCST] Epoch {epoch}", leave=False)

    for batch_idx, (images, captions, image_ids) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        captions = captions.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast():
            features = model.encoder(images)

            sampled_ids, log_probs = model.decoder.sample_with_log_probs(
                features, vocab.vocab_stoi
            )

            with torch.no_grad():
                greedy_ids = model.decoder.greedy_rollout(
                    features, vocab.vocab_stoi["<sos>"]
                )

        batch_refs = [refs_dict.get(img_id, [[]]) for img_id in image_ids]

        loss = scst_loss(log_probs, sampled_ids, greedy_ids, batch_refs, vocab.vocab_itos)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1
        global_step += 1

        pbar.set_postfix(loss=f"{loss.item():.4f}")
        if logger is not None:
            logger.log({"scst/batch_loss": loss.item()}, step=global_step)

    return total_loss / n_batches, global_step
