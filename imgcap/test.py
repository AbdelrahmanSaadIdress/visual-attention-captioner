"""Standalone test-set evaluation.

Loads a trained checkpoint and reports loss/accuracy/BLEU/CIDEr on the
held-out test split.

Usage:
    python -m imgcap.test --config configs/default.yaml \
        --checkpoint checkpoints/best_model.pt
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from imgcap.config import build_arg_parser, load_config
from imgcap.data.loaders import build_loaders
from imgcap.engine.evaluate import eval_one_epoch
from imgcap.models.imgcap import ImgCap
from imgcap.utils.checkpoint import load_checkpoint
from imgcap.utils.seed import seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = build_arg_parser(description="Evaluate an ImgCap checkpoint on the test split")
    parser.add_argument("--checkpoint", type=str, required=True,
                         help="Path to a .pt checkpoint saved by imgcap.train")
    parser.add_argument("--num-examples", type=int, default=10,
                         help="How many sample predictions to print")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config(parser=parser)

    seed_everything(cfg.seed)
    device = torch.device(cfg.device)

    train_loader, val_loader, test_loader, vocab_size, pad_idx, train_ds = build_loaders(cfg)

    model = ImgCap(
        feature_size=cfg.feature_size,
        lstm_hidden_size=cfg.lstm_hidden_size,
        num_layers=cfg.num_layers,
        vocab_size=vocab_size,
        embedding_dim=cfg.embedding_dim,
        attention_dim=cfg.attention_dim,
        dropout=cfg.dropout,
    ).to(device)

    load_checkpoint(args.checkpoint, model, resume=False, device=cfg.device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=cfg.label_smoothing)

    test_loss, test_acc, test_bleu, test_cider, refs, cands = eval_one_epoch(
        model, test_loader, criterion, device, train_ds,
        cfg, global_step=0, split_csv_path=cfg.split_csv_path, split="test",
    )

    print(f"\n{'=' * 60}")
    print("  TEST RESULTS")
    print(f"  Loss   : {test_loss:.4f}  |  Acc    : {test_acc:.4f}")
    print(f"  BLEU-1 : {test_bleu['bleu_1']:.4f}  |  BLEU-2 : {test_bleu['bleu_2']:.4f}")
    print(f"  BLEU-3 : {test_bleu['bleu_3']:.4f}  |  BLEU-4 : {test_bleu['bleu_4']:.4f}")
    print(f"  CIDEr  : {test_cider:.4f}")
    print(f"{'=' * 60}")

    for i, (ref, cand) in enumerate(zip(refs, cands)):
        print(cand, " ===> ", ref)
        print("=" * 80)
        if i + 1 >= args.num_examples:
            break


if __name__ == "__main__":
    main()
