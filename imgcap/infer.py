"""Caption a single image with a trained checkpoint.

Usage:
    python -m imgcap.infer \
        --checkpoint checkpoints/best_model.pt \
        --vocab-stoi-path artifacts/vocab_stoi.pkl \
        --vocab-itos-path artifacts/vocab_itos.pkl \
        --image path/to/image.jpg \
        --beam-width 5
"""
from __future__ import annotations

import argparse
import pickle

import torch
from PIL import Image
from torchvision import transforms as T

from imgcap.config import TrainConfig, build_arg_parser, load_config
from imgcap.data.loaders import IMAGENET_MEAN, IMAGENET_STD
from imgcap.generation.search import beam_search
from imgcap.models.imgcap import ImgCap
from imgcap.utils.checkpoint import load_checkpoint

_SPECIAL_TOKENS = {"<sos>", "<eos>", "<pad>", "<EN>"}


class _VocabView:
    """Minimal object exposing `.vocab_stoi` / `.vocab_itos`, matching what
    `beam_search` / `greedy_search` expect (normally a Flickr30 dataset)."""

    def __init__(self, vocab_stoi_path: str, vocab_itos_path: str):
        with open(vocab_stoi_path, "rb") as f:
            self.vocab_stoi = pickle.load(f)
        with open(vocab_itos_path, "rb") as f:
            self.vocab_itos = pickle.load(f)


def caption_image(image_path: str, model, vocab, device, beam_width: int = 5,
                   show: bool = False) -> str:
    model.eval()
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    with torch.no_grad():
        hypotheses = beam_search(model, img_tensor, vocab, device=device, beam_width=beam_width)

    caption_tokens = [w for w in hypotheses[0] if w not in _SPECIAL_TOKENS]
    caption = " ".join(caption_tokens)

    if show:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Caption: {caption}", fontsize=13, wrap=True)
        plt.tight_layout()
        plt.show()

    return caption


def build_parser() -> argparse.ArgumentParser:
    parser = build_arg_parser(description="Caption a single image")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--show", action="store_true", help="Display the image with matplotlib")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg: TrainConfig = load_config(parser=parser)

    device = torch.device(cfg.device)
    vocab = _VocabView(cfg.vocab_stoi_path, cfg.vocab_itos_path)
    vocab_size = len(vocab.vocab_stoi)

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

    caption = caption_image(
        image_path=args.image, model=model, vocab=vocab, device=device,
        beam_width=args.beam_width, show=args.show,
    )
    print("Generated Caption:", caption)


if __name__ == "__main__":
    main()
