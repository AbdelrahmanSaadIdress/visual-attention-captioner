import cv2
import torch
from torch.cuda.amp import autocast
from torchvision import transforms as T
from tqdm import tqdm

from imgcap.data.loaders import IMAGENET_MEAN, IMAGENET_STD
from imgcap.generation.search import greedy_search
from imgcap.metrics.eval_metrics import eval_BLEU, eval_CIDEr, token_accuracy
from imgcap.utils.references import collect_references_per_image

_SPECIAL_TOKENS = {"<sos>", "<eos>", "<pad>", "<EN>"}


def eval_one_epoch(model, loader, criterion, device, vocab, cfg,
                    global_step, split_csv_path, split="val",
                    decode_fn=greedy_search, logger=None):
    """Two-pass evaluation.

    Pass 1 -- batch loop (images/captions tensors from the loader)
        -> loss and token accuracy (teacher forcing = 1.0), same as training.

    Pass 2 -- per unique image
        -> generate one hypothesis per image (default: greedy; pass
           `decode_fn=beam_search` for beam decoding at eval time)
        -> compute BLEU / CIDEr against all reference captions for that image.
    """
    model.eval()

    # ── Pass 1: loss & accuracy ──────────────────────────────────────
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"[{split.capitalize()}] Loss", leave=False)
    with torch.no_grad():
        for images, captions, _ in pbar:
            images = images.to(device, non_blocking=True)
            captions = captions.to(device, non_blocking=True)

            with autocast():
                logits = model(images, captions, teacher_forcing_ratio=1.0)
                logits_ = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
                targets_ = captions[:, 1:].contiguous().view(-1)
                loss = criterion(logits_, targets_)

            acc = token_accuracy(logits[:, :-1, :], captions[:, 1:], vocab.vocab_stoi["<pad>"])
            total_loss += loss.item()
            total_acc += acc
            n_batches += 1

    avg_loss = total_loss / n_batches
    avg_acc = total_acc / n_batches

    # ── Pass 2: BLEU & CIDEr with all references ─────────────────────
    refs_dict = collect_references_per_image(split_csv_path, split, vocab)
    unique_images = list(refs_dict.keys())

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    all_hyps = []   # List[List[str]]       -- 1 hypothesis per image
    all_refs = []   # List[List[List[str]]] -- N references per image

    batch_size = cfg.batch_size
    with torch.no_grad():
        for start in tqdm(range(0, len(unique_images), batch_size),
                           desc=f"[{split.capitalize()}] Decode", leave=False):

            batch_ids = unique_images[start:start + batch_size]

            batch_images = []
            for img_id in batch_ids:
                img_path = f"{cfg.images_dir}/{img_id}"
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                batch_images.append(transform(img))

            batch_tensor = torch.stack(batch_images).to(device)

            hypotheses = decode_fn(model, batch_tensor, vocab, device=device)

            for i, img_id in enumerate(batch_ids):
                hyp = [w for w in hypotheses[i] if w not in _SPECIAL_TOKENS]
                all_hyps.append(hyp)
                all_refs.append(refs_dict[img_id])

    bleu_scores = eval_BLEU(all_hyps, all_refs)
    cider_score, _ = eval_CIDEr(all_hyps, all_refs)

    metrics = {
        f"{split}/loss": avg_loss,
        f"{split}/acc": avg_acc,
        f"{split}/bleu_1": bleu_scores["bleu_1"],
        f"{split}/bleu_2": bleu_scores["bleu_2"],
        f"{split}/bleu_3": bleu_scores["bleu_3"],
        f"{split}/bleu_4": bleu_scores["bleu_4"],
        f"{split}/cider": cider_score,
    }
    if logger is not None:
        logger.log(metrics, step=global_step)

    return avg_loss, avg_acc, bleu_scores, cider_score, all_refs, all_hyps


def is_better(new_score, best_score, metric: str) -> bool:
    if best_score is None:
        return True
    if metric == "val_loss":
        return new_score < best_score  # lower loss is better
    return new_score > best_score      # higher BLEU/CIDEr is better


def extract_metric(avg_loss, bleu_scores, cider_score, metric: str):
    if metric == "val_loss":
        return avg_loss
    if metric == "cider":
        return cider_score
    return bleu_scores[metric]  # "bleu_1" .. "bleu_4"
